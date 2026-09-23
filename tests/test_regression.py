"""Quality gate: fail the build when answer quality regresses.

Two tiers, and the split is the point.

  Tier 1 - retrieval regression (no marker, no key, free)
      Runs on every commit. Asserts hit@5 and MRR stay above a floor that the
      committed baseline already clears. Most regressions - a chunking change,
      a prefetch tweak, a model swap - show up here first, and catching them
      costs nothing.

  Tier 2 - generation regression (@pytest.mark.eval, needs a key, costs money)
      Runs faithfulness and answer relevancy through DeepEval on a small fixed
      sample. Deselected by default:

          pytest                    # tier 1 only
          pytest -m eval            # tier 2 as well

Thresholds are floors, not targets. They sit below the measured baseline so the
gate catches real degradation rather than normal judge variance; the numbers in
BASELINE come from evals_out/ and should be updated when a change legitimately
moves them.
"""

from __future__ import annotations

import os

import pytest

from askthedocs.config import settings
from askthedocs.evals import golden
from askthedocs.evals.retrieval_eval import run_retrieval_eval
from askthedocs.llm import PROVIDERS

# Floors sit below the committed baseline (evals_out/), so the gate catches real
# degradation rather than noise. Measured baseline for hybrid-without-rerank on
# bge-small: hit@5 0.975, hit@10 1.000, mrr 0.918, ndcg@10 0.931.
RETRIEVAL_FLOORS = {
    "hit@5": 0.90,
    "hit@10": 0.95,
    "mrr": 0.82,
    "ndcg@10": 0.85,
}

GENERATION_FLOORS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
}

# Tier 2 runs on a fixed slice so cost is predictable.
GENERATION_SAMPLE = 8


def _has_key() -> bool:
    spec = PROVIDERS.get(settings.provider)
    return bool(spec and (not spec.needs_key or (spec.env_key and os.getenv(spec.env_key))))


@pytest.fixture(scope="module")
def golden_items():
    try:
        return golden.load()
    except FileNotFoundError:
        pytest.skip("no golden set; run `python -m askthedocs.evals.build_golden`")


# The retrieval gate is meaningless against a handful of chunks: every metric
# collapses to zero and the failure looks like a regression rather than a
# missing corpus. Other tests in this suite seed tiny collections, so require a
# corpus large enough to actually be the real index.
MIN_CORPUS_CHUNKS = 500


@pytest.fixture(scope="module")
def real_settings():
    """Settings pointed at the real index rather than the test scratch store.

    conftest redirects writes to a temporary directory so tests cannot corrupt a
    real index; this gate needs read access to the actual corpus, so it points
    back explicitly. Against a Qdrant server (QDRANT_URL) there is no
    redirection and this is a no-op.
    """
    from conftest import real_store_path

    return settings.variant(local_path=real_store_path())


@pytest.fixture(scope="module")
def indexed_or_skip(real_settings):
    from askthedocs.retrieval import store

    try:
        size = store.collection_size(real_settings)
    except Exception as exc:
        pytest.skip(f"Qdrant unreachable: {exc}")

    if size < MIN_CORPUS_CHUNKS:
        pytest.skip(
            f"collection '{real_settings.collection}' holds {size} chunks, fewer "
            f"than the {MIN_CORPUS_CHUNKS} this gate needs. Build it with "
            f"`askthedocs index --recreate`."
        )


# --------------------------------------------------------------------------- #
# Tier 1 - free, runs on every commit
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def agentic_retrieval(golden_items, indexed_or_skip, real_settings):
    """The shipped retrieval configuration: hybrid, no cross-encoder."""
    return run_retrieval_eval(
        real_settings.variant(use_hybrid=True, use_reranker=False),
        golden_items,
        config_name="agentic",
        progress=False,
    )


@pytest.mark.parametrize("metric,floor", sorted(RETRIEVAL_FLOORS.items()))
def test_retrieval_above_floor(agentic_retrieval, metric, floor):
    actual = agentic_retrieval.metrics.get(metric)
    assert actual is not None, f"{metric} was not computed"
    assert actual >= floor, (
        f"{metric} regressed: {actual:.3f} < {floor:.3f} floor. "
        f"Re-run `askthedocs eval` and check what changed in retrieval."
    )


def test_hybrid_beats_dense_on_ranking(golden_items, indexed_or_skip, real_settings):
    """The justification for carrying a sparse arm at all.

    Note *which* metric is asserted. On this corpus dense retrieval alone
    already reaches hit@5 = 1.000, so hybrid cannot improve coverage - it
    actually gives up a little (0.975) because fusion admits BM25 candidates
    that push a relevant page out of the top five. What it buys is **ranking**:
    hit@1 0.750 -> 0.875 and MRR 0.860 -> 0.918.

    That is the trade worth making here, because only the top few passages reach
    the prompt and their order decides what the model leans on. Asserting hit@5
    instead would fail while the sparse arm was doing exactly its job - which is
    what this test originally did, and what the real numbers corrected.

    If ranking ever stops improving, the honest move is to drop the sparse arm
    rather than keep the claim in the README.
    """
    dense = run_retrieval_eval(
        real_settings.variant(use_hybrid=False, use_reranker=False),
        golden_items,
        config_name="dense",
        progress=False,
    )
    hybrid = run_retrieval_eval(
        real_settings.variant(use_hybrid=True, use_reranker=False),
        golden_items,
        config_name="hybrid",
        progress=False,
    )
    assert hybrid.metrics["mrr"] > dense.metrics["mrr"], (
        f"hybrid MRR {hybrid.metrics['mrr']:.3f} did not beat dense "
        f"{dense.metrics['mrr']:.3f} - the sparse arm is not earning its place"
    )
    assert hybrid.metrics["hit@1"] >= dense.metrics["hit@1"], (
        f"hybrid hit@1 {hybrid.metrics['hit@1']:.3f} fell below dense "
        f"{dense.metrics['hit@1']:.3f}"
    )
    # Coverage may dip slightly; a large drop would mean fusion is crowding out
    # relevant pages rather than reordering them.
    assert hybrid.metrics["hit@5"] >= dense.metrics["hit@5"] - 0.05, (
        f"hybrid hit@5 {hybrid.metrics['hit@5']:.3f} is well below dense "
        f"{dense.metrics['hit@5']:.3f} - fusion is losing relevant pages"
    )


def test_reranking_is_off_by_default_because_it_did_not_help():
    """Pins the conclusion the ablation reached.

    The cross-encoder lowered MRR on this corpus (0.918 -> 0.871) while costing
    roughly 59x the retrieval latency, because dense retrieval alone already
    reaches hit@5 = 1.0 and leaves it nothing to fix. The default was changed to
    match that finding, and this test fails if someone silently turns it back on
    without re-running the ablation - which would make the README's reasoning
    wrong.

    If a future corpus *does* justify reranking, re-run `askthedocs eval`, flip
    the default, and update this test to assert the improvement instead.
    """
    assert settings.reranker == "none"
    assert not settings.reranking_enabled


# --------------------------------------------------------------------------- #
# Eval sampling
# --------------------------------------------------------------------------- #
def test_small_samples_stay_representative(golden_items):
    """`--limit N` must be a miniature of the full set, not its first N rows.

    The golden set is grouped by doc set with the out-of-scope questions last,
    so a head slice yields one doc set and zero negatives - abstention then
    scores over an empty set and the run silently measures a third of the
    corpus. This is the regression test for that.
    """
    from askthedocs.evals.golden import sample

    for limit in (4, 8, 12, 20):
        picked = sample(golden_items, limit)
        assert len(picked) == limit

        doc_sets = {i.doc_set for i in picked if i.answerable}
        assert len(doc_sets) >= 2, (
            f"limit={limit} covered only {doc_sets} - a head slice, not a sample"
        )
        assert any(not i.answerable for i in picked), (
            f"limit={limit} contains no out-of-scope question, so abstention "
            f"would be measured over an empty set"
        )


def test_sampling_is_deterministic(golden_items):
    """Repeat runs must be comparable, so selection cannot be random."""
    from askthedocs.evals.golden import sample

    first = [i.id for i in sample(golden_items, 10)]
    second = [i.id for i in sample(golden_items, 10)]
    assert first == second


def test_no_limit_returns_everything(golden_items):
    from askthedocs.evals.golden import sample

    assert sample(golden_items, None) == golden_items
    assert sample(golden_items, 10_000) == golden_items


def test_golden_set_is_well_formed(golden_items):
    assert len(golden_items) >= 30, "the golden set should hold at least 30 questions"
    assert any(not i.answerable for i in golden_items), (
        "the golden set needs out-of-scope questions, or abstention is unmeasured"
    )
    ids = [i.id for i in golden_items]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    for item in golden_items:
        assert item.question.strip(), f"{item.id} has an empty question"
        if item.answerable:
            assert item.reference_answer.strip(), f"{item.id} has no reference answer"
            assert item.relevant_urls, f"{item.id} has no relevance labels"


# --------------------------------------------------------------------------- #
# Tier 2 - needs a judge model, costs money
# --------------------------------------------------------------------------- #
@pytest.mark.eval
@pytest.mark.skipif(not _has_key(), reason="no LLM API key configured")
@pytest.mark.parametrize("metric_name", sorted(GENERATION_FLOORS))
def test_generation_above_floor(golden_items, indexed_or_skip, real_settings, metric_name):
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    from askthedocs.evals.deepeval_model import LangChainJudge
    from askthedocs.graph.build import answer_question

    sample = [i for i in golden_items if i.answerable][:GENERATION_SAMPLE]
    judge = LangChainJudge(real_settings)
    threshold = GENERATION_FLOORS[metric_name]

    metric = (
        FaithfulnessMetric(threshold=threshold, model=judge, async_mode=False)
        if metric_name == "faithfulness"
        else AnswerRelevancyMetric(threshold=threshold, model=judge, async_mode=False)
    )

    scores, failures = [], []
    for item in sample:
        result = answer_question(item.question, settings=real_settings)
        if result.status != "answered" or not result.contexts:
            continue  # a correct abstention is not a generation failure
        metric.measure(
            LLMTestCase(
                input=item.question,
                actual_output=result.answer,
                retrieval_context=list(result.contexts),
                expected_output=item.reference_answer,
            )
        )
        scores.append(metric.score)
        if metric.score < threshold:
            failures.append(f"  {item.id}: {metric.score:.2f} - {metric.reason}")

    assert scores, "no answerable question produced an answer to score"
    mean = sum(scores) / len(scores)
    assert mean >= threshold, (
        f"mean {metric_name} {mean:.3f} < {threshold} over {len(scores)} questions\n"
        + "\n".join(failures)
    )
