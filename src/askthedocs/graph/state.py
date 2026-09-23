"""Typed state for the retrieval graph.

Every node reads and writes this one dict. Keeping the retrieval artefacts
(hits, scores, rewrites) in state rather than hiding them in closures is what
makes the run inspectable: the API returns the same trace the graph built, and
the UI renders it, so a reviewer can see *why* an answer came out the way it
did instead of taking it on faith.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from ..retrieval.schema import Hit

Status = Literal["answered", "insufficient_context", "error"]


def _append(left: list, right: list) -> list:
    """Reducer for accumulating lists across nodes and retry loops."""
    return (left or []) + (right or [])


class GraphState(TypedDict, total=False):
    # --- input ---------------------------------------------------------
    question: str            # what the user actually asked, never mutated
    doc_sets: list[str]      # optional corpus filter

    # --- query analysis -------------------------------------------------
    query: str               # what we actually search with, may be rewritten
    intent: str              # lookup | howto | comparison | out_of_scope
    rewrites: Annotated[list[str], _append]
    attempts: int

    # --- retrieval ------------------------------------------------------
    hits: list[Hit]
    retrieval_mode: str

    # --- grading --------------------------------------------------------
    relevant: bool
    grade_score: float
    grade_reason: str

    # --- generation -----------------------------------------------------
    answer: str
    citations: list[dict]

    # --- groundedness ---------------------------------------------------
    grounded: bool
    groundedness_reason: str
    regenerated: bool

    # --- bookkeeping ----------------------------------------------------
    status: Status
    error: str
    events: Annotated[list[dict], _append]


@dataclass
class RunResult:
    """What a caller gets back from one graph run."""

    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    status: Status = "answered"
    grounded: bool = True
    relevant: bool = True
    attempts: int = 1
    rewrites: list[str] = field(default_factory=list)
    retrieval_mode: str = "hybrid"
    events: list[dict] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    config_name: str = "agentic"

    def summary(self) -> str:
        bits = [
            f"status={self.status}",
            f"mode={self.retrieval_mode}",
            f"attempts={self.attempts}",
            f"grounded={self.grounded}",
            f"{self.latency_ms:.0f}ms",
        ]
        return "  ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "status": self.status,
            "grounded": self.grounded,
            "relevant": self.relevant,
            "attempts": self.attempts,
            "rewrites": self.rewrites,
            "retrieval_mode": self.retrieval_mode,
            "events": self.events,
            "latency_ms": round(self.latency_ms, 1),
            "config": self.config_name,
        }


def initial_state(question: str, doc_sets: list[str] | None = None) -> GraphState:
    return GraphState(
        question=question,
        query=question,
        doc_sets=doc_sets or [],
        attempts=0,
        rewrites=[],
        hits=[],
        events=[],
        status="answered",
        grounded=True,
        relevant=True,
        regenerated=False,
    )
