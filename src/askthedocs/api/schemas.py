"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Corpus = Literal["docs", "uploads"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)

    # --- generation --------------------------------------------------------
    provider: str | None = Field(
        default=None, description="anthropic | google_genai | openai | ollama"
    )
    model: str | None = None
    api_key: str | None = Field(
        default=None,
        description=(
            "Optional. Used for this request only - never stored, logged or traced. "
            "Falls back to the server's environment when omitted."
        ),
    )

    # --- retrieval ---------------------------------------------------------
    embedding: str | None = Field(
        default=None, description="Embedding model slug; must already be indexed."
    )
    reranker: str | None = Field(default=None, description="Reranker slug, or 'none'.")
    corpus: Corpus = "docs"
    workspace: str | None = Field(
        default=None, description="Upload workspace, when corpus='uploads'."
    )
    doc_sets: list[str] = Field(default_factory=list)
    config: Literal["agentic", "naive"] = "agentic"

    def redacted(self) -> dict[str, Any]:
        """A log-safe view. The key never leaves this object."""
        data = self.model_dump(exclude={"api_key"})
        data["api_key"] = "<supplied>" if self.api_key else None
        return data


class Citation(BaseModel):
    chunk_id: str
    title: str
    section: str
    source_url: str
    score: float
    # Every retrieved passage is returned, in the order the model saw them, so
    # an in-text marker [n] always resolves to source n. This says whether the
    # answer actually pointed at it.
    cited: bool = True


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    status: str
    grounded: bool
    relevant: bool = True
    attempts: int = 1
    rewrites: list[str] = Field(default_factory=list)
    retrieval_mode: str = "hybrid"
    events: list[dict] = Field(default_factory=list)
    latency_ms: float
    config: str


class ProviderInfo(BaseModel):
    id: str
    label: str
    models: list[str]
    default_model: str
    needs_key: bool
    key_in_env: bool
    notes: str = ""


class EmbeddingInfo(BaseModel):
    slug: str
    label: str
    model: str
    dim: int
    size_mb: int
    backend: str
    notes: str = ""
    indexed_chunks: int = 0
    available: bool = True
    unavailable_reason: str = ""


class RerankerInfo(BaseModel):
    slug: str
    label: str
    model: str
    size_mb: int
    backend: str
    notes: str = ""
    available: bool = True


class ModelCatalog(BaseModel):
    embeddings: list[EmbeddingInfo]
    rerankers: list[RerankerInfo]
    default_embedding: str
    default_reranker: str


class UploadedDocInfo(BaseModel):
    doc_id: str
    filename: str
    title: str
    chars: int
    chunks: int
    pages: int | None = None
    uploaded_at: str


class UploadResponse(BaseModel):
    workspace: str
    embedding: str
    documents: list[UploadedDocInfo]
    chunks_indexed: int
    total_chunks: int
    elapsed_ms: float


class WorkspaceInfo(BaseModel):
    workspace: str
    embedding: str
    total_chunks: int
    documents: list[UploadedDocInfo] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    collection: str
    indexed_chunks: int
    embedding: str
    embedding_model: str
    embedding_dim: int
    sparse_model: str
    reranker: str
    reranker_model: str
    profile: str
    qdrant: str
    langfuse: bool
    default_provider: str
    corpus_pages: int = 0
