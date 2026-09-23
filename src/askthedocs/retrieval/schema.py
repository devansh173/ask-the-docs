"""Shared data shapes for the retrieval layer."""

from __future__ import annotations

from dataclasses import dataclass, field

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


@dataclass
class Chunk:
    """A slice of a source document, ready to index."""

    chunk_id: str
    text: str
    source_url: str
    title: str
    section: str
    doc_set: str
    position: int

    def payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_url": self.source_url,
            "title": self.title,
            "section": self.section,
            "doc_set": self.doc_set,
            "position": self.position,
        }


@dataclass
class Hit:
    """A retrieved chunk plus whatever scores it has picked up so far."""

    chunk_id: str
    text: str
    source_url: str
    title: str
    section: str
    doc_set: str
    retrieval_score: float
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.retrieval_score

    @classmethod
    def from_point(cls, point) -> "Hit":
        p = point.payload or {}
        return cls(
            chunk_id=p.get("chunk_id", str(point.id)),
            text=p.get("text", ""),
            source_url=p.get("source_url", ""),
            title=p.get("title", ""),
            section=p.get("section", ""),
            doc_set=p.get("doc_set", ""),
            retrieval_score=float(point.score) if point.score is not None else 0.0,
        )

    def citation(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "section": self.section,
            "source_url": self.source_url,
            "score": round(self.score, 4),
        }


@dataclass
class RetrievalTrace:
    """What the retrieval stage did, for tracing and for the UI."""

    query: str
    mode: str = "hybrid"
    dense_candidates: int = 0
    sparse_candidates: int = 0
    fused: int = 0
    reranked: int = 0
    latency_ms: float = 0.0
    hits: list[Hit] = field(default_factory=list)
