"""Retrieval metrics - no LLM, no API cost, fully deterministic.

This is the half of the evaluation that isolates retrieval from generation. If
the right passage never reaches the prompt, no amount of prompt engineering will
produce a correct answer, so measuring retrieval on its own tells you which half
of the pipeline to fix.

Metrics, all computed against page-level relevance labels from the golden set:

  hit_rate@k  - fraction of questions with at least one relevant page in top k.
                The ceiling on what generation can possibly get right.
  recall@k    - fraction of a question's relevant pages that appear in top k.
  precision@k - fraction of the top k that are relevant. Falls as k grows;
                matters because irrelevant passages actively mislead the model.
  mrr         - 1/rank of the first relevant page, averaged. Rewards putting the
                answer first, which is what reranking is for.
  ndcg@k      - rank-discounted gain; the standard IR summary number.

Comparing configurations on these numbers is how the README justifies hybrid
search and reranking rather than asserting they help.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from ..config import Settings
from ..retrieval import store
from ..retrieval.encoders import get_reranker
from ..retrieval.schema import Hit
from .golden import GoldenItem

log = logging.getLogger(__name__)


@dataclass
class QueryOutcome:
    question_id: str
    ranked_urls: list[str]
    relevant_urls: set[str]
    latency_ms: float

    def first_relevant_rank(self) -> int | None:
        for i, url in enumerate(self.ranked_urls, 1):
            if url in self.relevant_urls:
                return i
        return None

    def hit_at(self, k: int) -> float:
        return float(any(u in self.relevant_urls for u in self.ranked_urls[:k]))

    def recall_at(self, k: int) -> float:
        if not self.relevant_urls:
            return 0.0
        found = {u for u in self.ranked_urls[:k] if u in self.relevant_urls}
        return len(found) / len(self.relevant_urls)

    def precision_at(self, k: int) -> float:
        window = self.ranked_urls[:k]
        if not window:
            return 0.0
        return sum(u in self.relevant_urls for u in window) / len(window)

    def ndcg_at(self, k: int) -> float:
        # Binary gains, so DCG is a sum of 1/log2(rank+1) over relevant hits.
        dcg = sum(
            1.0 / math.log2(i + 1)
            for i, url in enumerate(self.ranked_urls[:k], 1)
            if url in self.relevant_urls
        )
        ideal_n = min(len(self.relevant_urls), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
        return dcg / idcg if idcg else 0.0


@dataclass
class RetrievalReport:
    config: str
    embedding: str
    reranker: str
    n_questions: int
    metrics: dict[str, float] = field(default_factory=dict)
    outcomes: list[QueryOutcome] = field(default_factory=list)

    def row(self) -> dict:
        return {
            "config": self.config,
            "embedding": self.embedding,
            "reranker": self.reranker,
            "n": self.n_questions,
            **self.metrics,
        }


def _dedupe_urls(hits: list[Hit]) -> list[str]:
    """Rank pages, not chunks.

    Several chunks from the same page routinely occupy the top of a chunk-level
    ranking. Collapsing to first-occurrence order turns that into a page
    ranking, which is what the labels are expressed in - otherwise one page
    filling the top 5 would look like five separate correct results.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for hit in hits:
        if hit.source_url and hit.source_url not in seen:
            seen.add(hit.source_url)
            ordered.append(hit.source_url)
    return ordered


def run_retrieval_eval(
    settings: Settings,
    items: list[GoldenItem],
    *,
    config_name: str,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    progress: bool = True,
) -> RetrievalReport:
    """Score one retrieval configuration over the answerable golden questions."""
    scored = [i for i in items if i.answerable and i.relevant_urls]
    outcomes: list[QueryOutcome] = []

    reranker = get_reranker(settings) if settings.reranking_enabled else None

    for n, item in enumerate(scored, 1):
        started = time.perf_counter()
        trace = store.search(settings, item.question)
        hits = trace.hits

        if reranker is not None and hits:
            scores = reranker.score(item.question, [h.text for h in hits])
            for hit, score in zip(hits, scores):
                hit.rerank_score = score
            hits = sorted(hits, key=lambda h: h.rerank_score or 0.0, reverse=True)

        outcomes.append(
            QueryOutcome(
                question_id=item.id,
                ranked_urls=_dedupe_urls(hits),
                relevant_urls=set(item.relevant_urls),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        )
        if progress and n % 10 == 0:
            log.info("  %s: %d/%d", config_name, n, len(scored))

    metrics: dict[str, float] = {}
    if outcomes:
        for k in ks:
            metrics[f"hit@{k}"] = sum(o.hit_at(k) for o in outcomes) / len(outcomes)
            metrics[f"recall@{k}"] = sum(o.recall_at(k) for o in outcomes) / len(outcomes)
        metrics["precision@5"] = sum(o.precision_at(5) for o in outcomes) / len(outcomes)
        metrics["mrr"] = sum(
            1.0 / r if (r := o.first_relevant_rank()) else 0.0 for o in outcomes
        ) / len(outcomes)
        metrics["ndcg@10"] = sum(o.ndcg_at(10) for o in outcomes) / len(outcomes)
        latencies = sorted(o.latency_ms for o in outcomes)
        metrics["p50_ms"] = latencies[len(latencies) // 2]
        metrics["p95_ms"] = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]

    return RetrievalReport(
        config=config_name,
        embedding=settings.embedding,
        reranker=settings.reranker if settings.reranking_enabled else "none",
        n_questions=len(outcomes),
        metrics={k: round(v, 4) for k, v in metrics.items()},
        outcomes=outcomes,
    )


def format_table(reports: list[RetrievalReport], columns: list[str] | None = None) -> str:
    """Markdown table, ready to paste into the README."""
    if not reports:
        return "_no results_"
    columns = columns or [
        "hit@1", "hit@3", "hit@5", "recall@5", "mrr", "ndcg@10", "p50_ms",
    ]

    header = "| Configuration | Embedding | " + " | ".join(columns) + " |"
    divider = "|---|---|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for report in reports:
        cells = []
        for col in columns:
            value = report.metrics.get(col)
            if value is None:
                cells.append("-")
            elif col.endswith("_ms"):
                cells.append(f"{value:.0f}")
            else:
                cells.append(f"{value:.3f}")
        rows.append(
            f"| {report.config} | {report.embedding} | " + " | ".join(cells) + " |"
        )
    return "\n".join([header, divider, *rows])
