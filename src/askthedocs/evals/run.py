"""Evaluation driver.

Produces the numbers in the README, in two independent halves:

  Retrieval (offline, free, deterministic)
      Ablates the retrieval stack - dense only, +sparse/RRF, +cross-encoder -
      against page-level relevance labels. Needs no LLM, so it runs in CI and
      costs nothing. This is the half that proves hybrid search and reranking
      earn their complexity.

  Generation (needs an LLM judge, costs money)
      Runs the naive and agentic graphs end to end and scores the answers with
      RAGAS: faithfulness, answer relevancy, context precision, context recall.

The halves are separate commands on purpose. Retrieval regressions are the
common case and you want to catch them on every commit; paying for a judge on
every commit is how eval harnesses end up switched off.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..config import (
    EVAL_RESULTS_DIR,
    REPO_ROOT,
    Settings,
    settings as base_settings,
)
from ..logging_setup import setup as setup_logging
from ..retrieval import store
from . import golden
from .retrieval_eval import RetrievalReport, format_table, run_retrieval_eval

log = logging.getLogger(__name__)


# Retrieval ablation: each step adds exactly one mechanism to the one before.
#
# The rerank row names its model explicitly rather than relying on the default.
# The shipped default is RERANKER_MODEL=none (the ablation is why), so
# use_reranker=True alone would enable a reranker that does not exist and the
# row would silently duplicate the one above it.
RETRIEVAL_CONFIGS: dict[str, dict] = {
    "dense only (naive)": dict(use_hybrid=False, use_reranker=False,
                               use_self_correction=False),
    "+ sparse / RRF fusion": dict(use_hybrid=True, use_reranker=False,
                                  use_self_correction=False),
    "+ cross-encoder rerank": dict(use_hybrid=True, use_reranker=True,
                                   reranker="bge-base",
                                   use_self_correction=False),
}

# End-to-end configurations scored by RAGAS.
GENERATION_CONFIGS: dict[str, dict] = {
    "naive": dict(use_hybrid=False, use_reranker=False, use_self_correction=False,
                  max_retries=0),
    "agentic": dict(use_hybrid=True, use_reranker=True, use_self_correction=True),
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _write(payload: dict, name: str) -> Path:
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", path)
    return path


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def run_retrieval_suite(
    settings: Settings | None = None,
    limit: int | None = None,
    configs: list[str] | None = None,
) -> list[RetrievalReport]:
    items = golden.sample(golden.load(), limit)

    settings = settings or base_settings
    reports: list[RetrievalReport] = []

    for name, overrides in RETRIEVAL_CONFIGS.items():
        if configs and name not in configs:
            continue
        log.info("scoring retrieval config: %s", name)
        started = time.perf_counter()
        report = run_retrieval_eval(
            settings.variant(**overrides), items, config_name=name
        )
        log.info(
            "  %s: hit@5=%.3f mrr=%.3f ndcg@10=%.3f (%.1fs)",
            name,
            report.metrics.get("hit@5", 0),
            report.metrics.get("mrr", 0),
            report.metrics.get("ndcg@10", 0),
            time.perf_counter() - started,
        )
        reports.append(report)

    table = format_table(reports)
    print("\n" + table + "\n")

    _write(
        {
            "kind": "retrieval",
            "timestamp": _stamp(),
            "embedding": settings.embedding,
            "reranker": settings.reranker,
            "golden_stats": golden.stats(items),
            "results": [r.row() for r in reports],
            "markdown_table": table,
        },
        f"retrieval-{settings.embedding}-{_stamp()}.json",
    )
    return reports


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
class QuotaExhausted(RuntimeError):
    """The provider refused on quota, so the run cannot be completed."""


# Provider-agnostic markers for "you are out of quota", as opposed to a
# transient rate limit. The SDKs retry the latter internally; a daily cap will
# never clear, so retrying just burns wall-clock and hides the real cause.
_QUOTA_MARKERS = (
    "resource_exhausted",
    "exceeded your current quota",
    "insufficient_quota",
    "quota exceeded",
    "billing",
    "credit balance is too low",
)


def _is_quota_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


def collect_answers(
    settings: Settings,
    items: list[golden.GoldenItem],
    config_name: str,
) -> list[dict]:
    """Run the graph over the golden set and capture what RAGAS needs.

    A full run is roughly ``len(items) x 4`` model calls per configuration -
    query analysis, grading, generation and the groundedness check - which is
    enough to exhaust a free tier. Quota failures abort immediately with the
    arithmetic, rather than letting the SDK retry a daily cap that will not
    clear for hours.
    """
    from ..graph.build import answer_question
    from ..observability import tracing

    rows: list[dict] = []
    for n, item in enumerate(items, 1):
        try:
            result = answer_question(
                item.question,
                settings=settings,
                callbacks=tracing.callbacks(settings.langfuse_enabled),
            )
        except Exception as exc:
            if _is_quota_error(exc):
                raise QuotaExhausted(
                    f"{settings.provider} refused on quota after {n - 1} of "
                    f"{len(items)} questions.\n"
                    f"A full run needs roughly {len(items) * 4 * 2} model calls "
                    f"({len(items)} questions x ~4 calls x 2 configs).\n"
                    f"Options: raise the provider quota; point GRADER_MODEL and "
                    f"LLM_MODEL at a model with more headroom; run a smaller "
                    f"slice with `askthedocs eval --limit 5`; or set "
                    f"LLM_PROVIDER=ollama for an unlimited local model."
                ) from exc
            raise
        # A graph-level error surfaces as status="error" rather than raising,
        # so quota failures inside a node land here instead.
        if result.status == "error" and _is_quota_error(RuntimeError(result.answer)):
            raise QuotaExhausted(
                f"{settings.provider} refused on quota after {n - 1} of "
                f"{len(items)} questions: {result.answer[:200]}"
            )

        rows.append(
            {
                "id": item.id,
                "question": item.question,
                "answer": result.answer,
                "contexts": result.contexts,
                "reference": item.reference_answer,
                "answerable": item.answerable,
                "status": result.status,
                "grounded": result.grounded,
                "attempts": result.attempts,
                "latency_ms": result.latency_ms,
            }
        )
        if n % 5 == 0:
            log.info("  %s: %d/%d answered", config_name, n, len(items))

    tracing.flush()
    return rows


def run_generation_suite(
    settings: Settings | None = None,
    limit: int | None = None,
    configs: list[str] | None = None,
) -> dict:
    from .ragas_eval import abstention_metrics, score_with_ragas

    items = golden.sample(golden.load(), limit)

    settings = settings or base_settings
    answerable = [i for i in items if i.answerable]

    results: dict[str, dict] = {}
    for name, overrides in GENERATION_CONFIGS.items():
        if configs and name not in configs:
            continue
        log.info("running generation config: %s (%d questions)", name, len(items))
        cfg = settings.variant(**overrides)

        rows = collect_answers(cfg, items, name)
        scored_rows = [r for r in rows if r["answerable"]]

        results[name] = {
            "ragas": score_with_ragas(scored_rows, settings=cfg),
            "abstention": abstention_metrics(rows),
            "latency_p50_ms": sorted(r["latency_ms"] for r in rows)[len(rows) // 2],
            "n": len(scored_rows),
            "rows": rows,
        }

    payload = {
        "kind": "generation",
        "timestamp": _stamp(),
        "embedding": settings.embedding,
        "reranker": settings.reranker,
        "generator_model": settings.model,
        "judge_model": settings.grader_model,
        "n_answerable": len(answerable),
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                    for k, v in results.items()},
    }
    _write({**payload, "detail": results}, f"generation-{_stamp()}.json")

    print("\n" + format_generation_table(payload) + "\n")
    return payload


def format_generation_table(payload: dict) -> str:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    header = "| Configuration | " + " | ".join(
        m.replace("_", " ").title() for m in metrics
    ) + " | Correct abstentions | p50 latency |"
    divider = "|---|" + "|".join("---" for _ in metrics) + "|---|---|"

    rows = []
    for name, result in payload.get("results", {}).items():
        cells = [
            f"{result['ragas'].get(m):.3f}" if result["ragas"].get(m) is not None else "-"
            for m in metrics
        ]
        abst = result.get("abstention", {})
        cells.append(
            f"{abst.get('correct_abstentions', 0)}/{abst.get('out_of_scope_total', 0)}"
        )
        cells.append(f"{result.get('latency_p50_ms', 0):.0f} ms")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")

    return "\n".join([header, divider, *rows])


README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"


def _read_the_table(reports: list[RetrievalReport]) -> str:
    """Write the honest reading of the ablation, including bad news.

    Generated rather than hand-written so it cannot drift from the numbers
    directly above it. A README that praises a component the table shows to be
    useless is worse than no commentary at all.
    """
    by_name = {r.config: r.metrics for r in reports}
    dense = by_name.get("dense only (naive)")
    hybrid = by_name.get("+ sparse / RRF fusion")
    rerank = by_name.get("+ cross-encoder rerank")
    if not (dense and hybrid):
        return ""

    d1, h1 = dense.get("hit@1", 0), hybrid.get("hit@1", 0)
    dm, hm = dense.get("mrr", 0), hybrid.get("mrr", 0)

    lines = ["\n**Reading this table.**\n"]
    if h1 > d1 or hm > dm:
        lines.append(
            f"- Adding the BM25 arm moves hit@1 from {d1:.3f} to {h1:.3f} and MRR "
            f"from {dm:.3f} to {hm:.3f}. Fusion is not finding pages dense "
            f"retrieval missed — it is putting the right page **first**, which is "
            f"what matters when only the top few reach the prompt."
        )
    else:
        lines.append(
            f"- The BM25 arm did not improve ranking here (hit@1 {d1:.3f} → "
            f"{h1:.3f}, MRR {dm:.3f} → {hm:.3f})."
        )

    if rerank:
        r1, rm = rerank.get("hit@1", 0), rerank.get("mrr", 0)
        r5, h5 = rerank.get("hit@5", 0), hybrid.get("hit@5", 0)
        r_ms, h_ms = rerank.get("p50_ms", 0), hybrid.get("p50_ms", 1)
        if rm < hm:
            lines.append(
                f"- **The cross-encoder does not earn its place on this corpus.** "
                f"It lowers MRR from {hm:.3f} to {rm:.3f} and hit@1 from {h1:.3f} "
                f"to {r1:.3f}, while costing {r_ms / max(h_ms, 1):.0f}× the latency "
                f"({h_ms:.0f} ms → {r_ms:.0f} ms on CPU). It does reach hit@5 "
                f"{r5:.3f} against {h5:.3f}, but {r_ms / 1000:.1f} s per query is "
                f"not a trade worth making for that."
            )
            lines.append(
                f"- The likely reason is headroom. Dense retrieval alone already "
                f"reaches hit@5 {dense.get('hit@5', 0):.3f} here, so there is almost "
                f"nothing left for a reranker to fix. Cross-encoders earn their cost "
                f"on larger, noisier collections where the top 20 contain real "
                f"distractors — and on hardware where they are not running on a "
                f"laptop CPU."
            )
            lines.append(
                "- So **reranking ships disabled by default** (`RERANKER_MODEL=none`) "
                "and stays one flag away. Leaving it on would mean claiming a "
                "component this measurement does not support."
            )
        else:
            lines.append(
                f"- The cross-encoder improves MRR from {hm:.3f} to {rm:.3f}, at "
                f"{r_ms / max(h_ms, 1):.0f}× the retrieval latency."
            )

    lines.append(
        "\n_This is what the eval harness is for. The ablation was built to "
        "justify the architecture, and on this corpus it declined to justify one "
        "third of it._"
    )
    return "\n".join(lines) + "\n"


def write_readme_section(
    retrieval: list[RetrievalReport],
    generation: dict | None = None,
    corpus: dict | None = None,
) -> None:
    """Replace the README's results block with freshly measured numbers.

    The table is generated rather than typed so it cannot drift away from what
    the harness actually produced - a README with hand-copied metrics is one
    refactor away from being a false claim.
    """
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        return

    body = readme.read_text(encoding="utf-8")
    if README_START not in body or README_END not in body:
        log.warning("README results markers missing; leaving it alone")
        return

    parts = ["### Retrieval\n"]
    if corpus:
        parts.append(
            f"Measured on {corpus['chunks']:,} chunks from {corpus['pages']} "
            f"documentation pages, against {corpus['questions']} hand-labelled "
            f"questions. No LLM involved — these numbers are deterministic and "
            f"cost nothing to reproduce.\n"
        )
    parts.append(format_table(retrieval))
    parts.append(
        "\n`hit@k` is the share of questions with a correct page in the top k — "
        "the ceiling on what generation can get right. `mrr` rewards ranking the "
        "answer first. `p50_ms` is retrieval latency only, excluding generation.\n"
    )
    parts.append(_read_the_table(retrieval))

    parts.append("\n### Generation\n")
    questions = (corpus or {}).get("questions", 44)
    if generation and generation.get("results"):
        parts.append(
            f"Judged by `{generation.get('judge_model')}`, generating with "
            f"`{generation.get('generator_model')}`, over "
            f"{generation.get('n_answerable')} answerable questions.\n"
        )
        parts.append(format_generation_table(generation))
    else:
        parts.append(
            "_Not yet measured._ Unlike the retrieval half, these metrics need "
            "an LLM to judge every answer, so they need an API key **and enough "
            "quota to finish**. A full run is roughly "
            f"{questions * 4 * 2} model calls ({questions} questions x ~4 calls "
            "x 2 configurations), before RAGAS adds its own judging on top.\n\n"
            "```bash\n"
            "export ANTHROPIC_API_KEY=sk-ant-...   # or GOOGLE_API_KEY / OPENAI_API_KEY\n"
            "askthedocs eval\n"
            "askthedocs eval --limit 5             # a cheap slice to smoke-test it\n"
            "```\n\n"
            "Free tiers tend to be the binding constraint rather than cost. "
            "Gemini's free tier caps `flash-lite` at 20 requests per day, which "
            "this run exhausts in about five questions; the harness detects a "
            "quota refusal and stops with the arithmetic rather than retrying a "
            "cap that will not clear for hours.\n\n"
            "**The retrieval numbers above are complete, deterministic, and "
            "need no key.**\n"
        )

    block = f"{README_START}\n\n" + "\n".join(parts) + f"\n\n{README_END}"
    start = body.index(README_START)
    end = body.index(README_END) + len(README_END)
    readme.write_text(body[:start] + block + body[end:], encoding="utf-8")
    log.info("updated the README results section")


def corpus_stats(items, settings: Settings | None = None) -> dict:
    """Corpus size as *indexed*, not as scraped.

    data/raw holds more pages than the index contains, because
    INDEX_MAX_PAGES_PER_SET caps how many are taken per doc set. Reporting the
    on-disk count would overstate the corpus the metrics were measured against.
    """
    from ..config import RAW_DIR
    from ..ingest.chunk import _select_pages

    settings = settings or base_settings
    pages = 0
    if RAW_DIR.exists():
        for directory in RAW_DIR.iterdir():
            if directory.is_dir():
                pages += len(_select_pages(directory, settings.max_pages_per_set))
    try:
        chunks = store.collection_size(settings)
    except Exception:
        chunks = 0
    return {"pages": pages, "chunks": chunks, "questions": len(items)}


def run_comparison(
    configs: list[str] | None = None,
    settings: Settings | None = None,
    limit: int | None = None,
    write_readme: bool = True,
) -> None:
    """Entry point for `askthedocs eval`: retrieval always, generation if keyed."""
    setup_logging()
    settings = settings or base_settings
    retrieval = run_retrieval_suite(settings=settings, limit=limit)

    import os

    from ..llm import PROVIDERS

    generation = None
    spec = PROVIDERS.get(base_settings.provider)
    if spec and spec.needs_key and spec.env_key and not os.getenv(spec.env_key):
        log.warning(
            "Skipping generation metrics: %s is not set, so there is no judge "
            "model. The retrieval metrics above are complete and need no key.",
            spec.env_key,
        )
    else:
        try:
            generation = run_generation_suite(
                settings=settings, limit=limit, configs=configs
            )
        except QuotaExhausted as exc:
            # Not a crash: the retrieval half is complete and worth keeping, so
            # report why generation is missing and still write the README.
            log.error("Generation metrics could not be measured.\n%s", exc)
            generation = None

    if write_readme:
        write_readme_section(retrieval, generation, corpus_stats(golden.load(), settings))


def main(argv: list[str] | None = None) -> int:
    """`python -m askthedocs.evals.run`.

    Takes arguments rather than ignoring them: without this, `--help` silently
    started a full evaluation, which on a metered API is an expensive way to
    find out what the flags are.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="askthedocs.evals.run",
        description=(
            "Score the retrieval stack (free, offline) and, when a key with "
            "enough quota is configured, the generation stack via RAGAS."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m askthedocs.evals.run                 # everything\n"
            "  python -m askthedocs.evals.run --limit 5       # cheap slice\n"
            "  python -m askthedocs.evals.run --retrieval-only\n"
        ),
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="only score the first N golden questions")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="generation configs to run (naive, agentic)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="skip the LLM-judged half entirely")
    parser.add_argument("--no-readme", action="store_true",
                        help="do not rewrite the README results section")
    args = parser.parse_args(argv)

    setup_logging()
    settings = base_settings

    if args.retrieval_only:
        reports = run_retrieval_suite(
            settings=settings, limit=args.limit, configs=args.configs
        )
        if not args.no_readme:
            write_readme_section(
                reports, None, corpus_stats(golden.load(), settings)
            )
        return 0

    run_comparison(
        configs=args.configs,
        settings=settings,
        limit=args.limit,
        write_readme=not args.no_readme,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
