"""Generation metrics via RAGAS.

Four metrics, two per half of the pipeline:

    faithfulness         - are the answer's claims entailed by the retrieved
                           context? This is the hallucination number.
    answer_relevancy     - does the answer actually address the question?
    context_precision    - are the retrieved passages relevant, and ranked with
                           the relevant ones first?
    context_recall       - did retrieval find everything the reference answer
                           needed?

Two deliberate choices about what powers the judge:

  * The judge LLM is the configured grader model, not the generator. Scoring an
    answer with the same model that wrote it inflates the result; using a
    different (and cheaper) model is both more honest and cheaper to run.
  * The embedding model behind answer_relevancy is this project's own local
    encoder, wrapped for LangChain. RAGAS otherwise reaches for OpenAI
    embeddings by default, which would mean the eval needed a second vendor's
    key to score a run that never used it.

RAGAS metrics that need a judge only run when a key is available; the retrieval
half of the evaluation (retrieval_eval.py) needs no key at all.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ._ragas_compat import install as _install_compat

_install_compat()  # must precede any ragas import

from ..config import Settings, settings as default_settings  # noqa: E402

log = logging.getLogger(__name__)


class LocalEmbeddings:
    """Adapt this project's dense encoder to LangChain's Embeddings interface.

    Implements the two methods RAGAS calls. Subclassing langchain_core's
    Embeddings is unnecessary - it is a protocol in practice, and duck typing
    avoids a hard import of a class whose location has moved between versions.
    """

    def __init__(self, settings: Settings) -> None:
        from ..retrieval.encoders import get_encoders

        self._dense, _ = get_encoders(settings)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._dense.encode_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._dense.encode_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def _build_metrics(judge, embeddings) -> list:
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    return [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
    ]


# RAGAS metric names -> the names used in the README table.
_RENAME = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "response_relevancy": "answer_relevancy",
    "semantic_similarity": "answer_similarity",
    "llm_context_precision_with_reference": "context_precision",
    "context_precision": "context_precision",
    "context_recall": "context_recall",
    "llm_context_recall": "context_recall",
}


def score_with_ragas(
    rows: list[dict],
    settings: Settings | None = None,
    *,
    judge_provider: str | None = None,
    judge_model: str | None = None,
) -> dict[str, float | None]:
    """Score collected answers. Returns {} if no judge model is available."""
    settings = settings or default_settings
    usable = [r for r in rows if r.get("answer") and r.get("contexts")]
    if not usable:
        log.warning("nothing to score: every row was missing an answer or contexts")
        return {}

    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.run_config import RunConfig

        from ..llm import build_chat_model, resolve_provider

        provider = resolve_provider(judge_provider or settings.grader_provider)
        primary_name = judge_model or settings.grader_model
        judge_chat = build_chat_model(provider, primary_name, max_tokens=1024)

        # NOT wrapped in with_fallbacks(): tried it, and RAGAS reaches into the
        # model object for a `.temperature` attribute that a RunnableWithFallbacks
        # does not expose ("RunnableWithFallbacks object has no field
        # temperature"), which fails every single judging call rather than
        # falling back. Confirmed the wrapper constructs fine and that a plain
        # .invoke() through it recovers correctly - the break is specific to
        # whatever attribute access RAGAS's own executor does internally, and
        # was only caught by running a real judged evaluation, not by
        # constructing the object or calling it directly. Left as a single
        # model until that is fixed properly; the graph's own generation and
        # grading calls (call_with_fallback, in llm.py) are unaffected.
        judge = LangchainLLMWrapper(judge_chat)
        embeddings = LocalEmbeddings(settings)

        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=r["question"],
                    response=r["answer"],
                    retrieved_contexts=list(r["contexts"]),
                    reference=r.get("reference") or "",
                )
                for r in usable
            ]
        )

        log.info(
            "scoring %d answers with RAGAS (judge=%s)",
            len(usable),
            judge_model or settings.grader_model,
        )
        # RAGAS defaults to 16 concurrent judge calls (RunConfig.max_workers).
        # Free-tier Gemini caps flash-lite class models at 15 requests/MINUTE
        # (a separate, tighter limit than the daily cap), so the default fired
        # past it instantly: 461 429s in three minutes on one run here, with
        # judgements that timed out counted as failures rather than zeros (see
        # _summarise) but still missing from the result. Capping concurrency
        # to comfortably under the per-minute limit trades wall-clock time for
        # actually finishing.
        result = evaluate(
            dataset=dataset,
            metrics=_build_metrics(judge, embeddings),
            show_progress=True,
            raise_exceptions=False,
            run_config=RunConfig(max_workers=4, max_wait=90, timeout=240),
        )
    except Exception as exc:
        log.error("RAGAS scoring failed: %s", exc)
        return {}

    return _summarise(result, len(usable))


def _summarise(result, n_samples: int) -> dict[str, float | None]:
    """Reduce a RAGAS result to per-metric means, separating failures from zeros.

    RAGAS is run with raise_exceptions=False so one bad judge response cannot
    abort a paid run. The cost is that a failed judgement becomes NaN, and
    naively averaging turns it into a low score that is indistinguishable from
    a genuine one. A metric where the judge failed is not a metric of zero - so
    NaNs are counted and reported under "<metric>_failed" rather than folded
    into the average.
    """
    try:
        frame = result.to_pandas()
    except Exception as exc:  # pragma: no cover - depends on ragas internals
        log.warning("could not read RAGAS results: %s", exc)
        return {}

    scores: dict[str, float | None] = {}
    for column in frame.columns:
        name = _RENAME.get(column)
        if name is None:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        failed = int(series.isna().sum())
        valid = series.dropna()

        scores[name] = round(float(valid.mean()), 4) if len(valid) else None
        if failed:
            scores[f"{name}_failed"] = failed
            log.warning(
                "%s: the judge returned nothing usable for %d/%d samples",
                name, failed, n_samples,
            )
    return scores


def abstention_metrics(rows: list[dict]) -> dict[str, Any]:
    """How well the pipeline declines questions the corpus cannot answer.

    RAGAS has nothing to say about this - it scores answers, and the right
    behaviour on an out-of-scope question is to produce no answer at all. A
    pipeline that confidently answers everything can post good faithfulness
    scores on the answerable set while being useless in practice, so the
    negatives are tracked separately.
    """
    out_of_scope = [r for r in rows if not r.get("answerable")]
    answerable = [r for r in rows if r.get("answerable")]

    correct_abstentions = sum(
        r.get("status") == "insufficient_context" for r in out_of_scope
    )
    wrongly_refused = sum(
        r.get("status") == "insufficient_context" for r in answerable
    )

    return {
        "out_of_scope_total": len(out_of_scope),
        "correct_abstentions": correct_abstentions,
        "abstention_rate": (
            round(correct_abstentions / len(out_of_scope), 4) if out_of_scope else None
        ),
        "answerable_total": len(answerable),
        "wrongly_refused": wrongly_refused,
        "false_refusal_rate": (
            round(wrongly_refused / len(answerable), 4) if answerable else None
        ),
        "ungrounded_answers": sum(not r.get("grounded", True) for r in rows),
        "mean_attempts": (
            round(sum(r.get("attempts", 1) for r in rows) / len(rows), 2) if rows else None
        ),
    }
