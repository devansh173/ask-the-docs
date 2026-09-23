"""Central configuration.

Everything tunable lives here so the eval harness and the API can construct
alternate configurations - naive vs agentic, one embedding model vs another -
without touching call sites.

Two registries matter:

  EMBEDDINGS  the dense models a user can pick from in the UI. Each one has a
              different dimensionality, so each gets its own Qdrant collection;
              you cannot query a 384-dim index with a 768-dim vector.
  RERANKERS   the cross-encoders. Independent of the embedding choice, because
              reranking happens after retrieval and does not touch the index.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
GOLDEN_DIR = DATA_DIR / "golden"
UPLOAD_DIR = DATA_DIR / "uploads"
EVAL_RESULTS_DIR = REPO_ROOT / "evals_out"


def _default_local_qdrant() -> Path:
    """Where embedded Qdrant keeps its SQLite store.

    Deliberately *outside* the repository by default. Embedded Qdrant commits to
    SQLite once per point (see qdrant_client/local/persistence.py), so indexing
    thousands of chunks is thousands of separate commits. If that file sits in a
    synced folder - OneDrive, Dropbox, iCloud Drive - the sync client re-uploads
    the whole growing store after every commit and ingestion slows to a crawl.
    That is not hypothetical: it is where this project was developed, and it
    took indexing from minutes to hours.

    Keeping the index in a cache directory also stops a large, fully rebuildable
    artefact being synced to the user's cloud storage on every re-index.

    Override with QDRANT_LOCAL_PATH to put it anywhere, including back in the repo.
    """
    override = os.getenv("QDRANT_LOCAL_PATH")
    if override:
        return Path(override).expanduser()

    base = (
        os.getenv("LOCALAPPDATA")          # Windows
        or os.getenv("XDG_CACHE_HOME")     # Linux
        or str(Path.home() / ".cache")     # macOS / fallback
    )
    return Path(base) / "askthedocs" / "qdrant"


QDRANT_LOCAL_PATH = _default_local_qdrant()

Backend = Literal["onnx", "torch"]


# --------------------------------------------------------------------------- #
# Embedding models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmbeddingSpec:
    slug: str
    model: str
    dim: int
    size_mb: int
    label: str
    backend: Backend = "onnx"
    # Asymmetric models want an instruction on the query side only.
    query_prefix: str = ""
    doc_prefix: str = ""
    notes: str = ""
    max_tokens: int = 512

    @property
    def collection_suffix(self) -> str:
        return self.slug


EMBEDDINGS: dict[str, EmbeddingSpec] = {
    "bge-small": EmbeddingSpec(
        slug="bge-small",
        model="BAAI/bge-small-en-v1.5",
        dim=384,
        size_mb=67,
        label="BGE Small v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
        notes="Default. Fastest to index, strong for its size on English text.",
    ),
    "minilm": EmbeddingSpec(
        slug="minilm",
        model="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        size_mb=90,
        label="MiniLM-L6-v2",
        notes="The 2022-era default, and what v1 of this project used. Kept as a baseline to measure against.",
    ),
    "bge-base": EmbeddingSpec(
        slug="bge-base",
        model="BAAI/bge-base-en-v1.5",
        dim=768,
        size_mb=210,
        label="BGE Base v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
        notes="Roughly 3x the index size of bge-small for a modest accuracy gain.",
    ),
    "nomic": EmbeddingSpec(
        slug="nomic",
        model="nomic-ai/nomic-embed-text-v1.5",
        dim=768,
        size_mb=520,
        label="Nomic Embed v1.5",
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
        max_tokens=8192,
        notes="8k context, so long sections survive without being split as aggressively.",
    ),
    "mxbai": EmbeddingSpec(
        slug="mxbai",
        model="mixedbread-ai/mxbai-embed-large-v1",
        dim=1024,
        size_mb=640,
        label="mxbai Embed Large",
        query_prefix="Represent this sentence for searching relevant passages: ",
        notes="Best English retrieval quality available here through ONNX.",
    ),
    "e5-multilingual": EmbeddingSpec(
        slug="e5-multilingual",
        model="intfloat/multilingual-e5-large",
        dim=1024,
        size_mb=2240,
        label="Multilingual E5 Large",
        query_prefix="query: ",
        doc_prefix="passage: ",
        notes="Use when the documents are not in English. Large download.",
    ),
    "qwen3": EmbeddingSpec(
        slug="qwen3",
        model="Qwen/Qwen3-Embedding-0.6B",
        dim=1024,
        size_mb=2400,
        label="Qwen3 Embedding 0.6B",
        backend="torch",
        query_prefix=(
            "Instruct: Given a technical question, retrieve documentation "
            "passages that answer it\nQuery: "
        ),
        max_tokens=8192,
        notes="Leads open multilingual retrieval benchmarks. Needs torch (requirements-quality.txt).",
    ),
}

DEFAULT_EMBEDDING = "bge-small"


# --------------------------------------------------------------------------- #
# Rerankers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RerankerSpec:
    slug: str
    model: str
    size_mb: int
    label: str
    backend: Backend = "onnx"
    notes: str = ""


RERANKERS: dict[str, RerankerSpec] = {
    "none": RerankerSpec("none", "", 0, "No reranking",
                         notes="Fusion order goes straight to the model."),
    "minilm-l6": RerankerSpec("minilm-l6", "Xenova/ms-marco-MiniLM-L-6-v2", 80,
                              "MS-Marco MiniLM L6",
                              notes="Very fast. Noticeably weaker than the BGE rerankers."),
    "bge-base": RerankerSpec("bge-base", "BAAI/bge-reranker-base", 1040,
                             "BGE Reranker Base",
                             notes="Best quality-per-millisecond of the cross-encoders here."),
    "jina-v2": RerankerSpec("jina-v2", "jinaai/jina-reranker-v2-base-multilingual", 1110,
                            "Jina Reranker v2",
                            notes="Multilingual, comparable quality to BGE base."),
    "bge-v2-m3": RerankerSpec("bge-v2-m3", "BAAI/bge-reranker-v2-m3", 2300,
                              "BGE Reranker v2-m3", backend="torch",
                              notes="Strongest option. Needs torch."),
}

# Reranking is OFF by default, and that is a measured decision rather than a
# convenience one. On this corpus the ablation in evals_out/ shows the
# cross-encoder *lowering* MRR (0.918 -> 0.871) while costing ~59x the retrieval
# latency on CPU, because dense retrieval alone already reaches hit@5 = 1.0 and
# leaves the reranker nothing to fix. It stays one flag away for corpora with
# real distractors, or for hardware that is not a laptop CPU.
DEFAULT_RERANKER = "none"

SPARSE_MODEL = "Qdrant/bm25"


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
ModelProfile = Literal["lite", "quality"]

PROFILE_PRESETS: dict[str, tuple[str, str]] = {
    # profile -> (embedding slug, reranker slug)
    "lite": ("bge-small", "none"),
    "reranked": ("bge-small", "bge-base"),
    "quality": ("qwen3", "bge-v2-m3"),
}


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw and raw.strip() else default


def collection_name(base: str, embedding_slug: str) -> str:
    """One collection per embedding model.

    Vector dimensionality is fixed when a collection is created, so a 768-dim
    model cannot query an index built with a 384-dim one. Encoding the model
    into the name makes that a non-issue instead of a confusing runtime error,
    and lets several models coexist so they can be compared.
    """
    safe = re.sub(r"[^a-z0-9]+", "_", embedding_slug.lower()).strip("_")
    return f"{base}__{safe}"


# --------------------------------------------------------------------------- #
# Runtime settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    # --- models ------------------------------------------------------------
    embedding: str = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING)
    reranker: str = os.getenv("RERANKER_MODEL", DEFAULT_RERANKER)

    # --- storage -----------------------------------------------------------
    collection_base: str = os.getenv("QDRANT_COLLECTION", "askthedocs")
    qdrant_url: str | None = os.getenv("QDRANT_URL") or None
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    # Embedded-mode storage directory. Part of Settings rather than a module
    # constant so a caller can point at a different store without reimporting -
    # which is what lets the test suite write to a scratch directory while the
    # CI retrieval gate still reads the real index.
    local_path: Path = QDRANT_LOCAL_PATH

    chunk_size: int = _env_int("CHUNK_SIZE", 600)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 100)
    # Pages taken per doc set at index time, highest-priority first. The scraper
    # keeps more than this on disk; indexing everything produced ~12k chunks and
    # a 95-minute cold start, which is a poor first experience for anyone
    # cloning the repo. Set 0 for no limit.
    max_pages_per_set: int = _env_int("INDEX_MAX_PAGES_PER_SET", 25)

    # --- retrieval ---------------------------------------------------------
    dense_prefetch: int = _env_int("DENSE_PREFETCH", 40)
    sparse_prefetch: int = _env_int("SPARSE_PREFETCH", 40)
    fusion_limit: int = _env_int("FUSION_LIMIT", 20)
    top_k: int = _env_int("TOP_K", 5)

    # --- graph behaviour ---------------------------------------------------
    use_reranker: bool = _env_bool("USE_RERANKER", False)
    use_hybrid: bool = _env_bool("USE_HYBRID", True)
    use_self_correction: bool = _env_bool("USE_SELF_CORRECTION", True)
    max_retries: int = _env_int("MAX_RETRIES", 2)
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))

    # --- generation --------------------------------------------------------
    provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    model: str = os.getenv("LLM_MODEL", "claude-opus-5")
    grader_provider: str = os.getenv("GRADER_PROVIDER", "") or os.getenv(
        "LLM_PROVIDER", "anthropic"
    )
    grader_model: str = os.getenv("GRADER_MODEL", "claude-sonnet-5")
    max_tokens: int = _env_int("LLM_MAX_TOKENS", 2048)
    # How many sibling models to fall back through on a quota refusal, on top
    # of the configured one. Free-tier quotas are metered per model, so a
    # provider with several models effectively multiplies its daily allowance
    # by this many. 0 disables fallback and restores the old single-model
    # behaviour.
    #
    # Set to 3 (chain length 4) rather than a smaller number because building
    # this project's eval harness against Gemini's free tier found that which
    # models are actually available shifts during a single day of ordinary
    # testing - a chain of 2 models was not enough headroom in practice.
    max_fallbacks: int = _env_int("LLM_MAX_FALLBACKS", 3)

    # --- observability -----------------------------------------------------
    langfuse_enabled: bool = _env_bool("LANGFUSE_ENABLED", True)

    # ----------------------------------------------------------------------
    @property
    def embedding_spec(self) -> EmbeddingSpec:
        if self.embedding not in EMBEDDINGS:
            raise ValueError(
                f"Unknown embedding {self.embedding!r}. "
                f"Expected one of {sorted(EMBEDDINGS)}."
            )
        return EMBEDDINGS[self.embedding]

    @property
    def reranker_spec(self) -> RerankerSpec:
        if self.reranker not in RERANKERS:
            raise ValueError(
                f"Unknown reranker {self.reranker!r}. "
                f"Expected one of {sorted(RERANKERS)}."
            )
        return RERANKERS[self.reranker]

    @property
    def collection(self) -> str:
        return collection_name(self.collection_base, self.embedding)

    @property
    def reranking_enabled(self) -> bool:
        return self.use_reranker and self.reranker != "none"

    @property
    def profile(self) -> str:
        """Which preset this matches, for display. 'custom' if neither."""
        for name, (emb, rer) in PROFILE_PRESETS.items():
            if self.embedding == emb and self.reranker == rer:
                return name
        return "custom"

    def variant(self, **overrides) -> "Settings":
        """A copy with fields replaced - used by the eval harness and the API."""
        return replace(self, **overrides)

    def with_profile(self, profile: str) -> "Settings":
        if profile not in PROFILE_PRESETS:
            raise ValueError(f"Unknown profile {profile!r}")
        embedding, reranker = PROFILE_PRESETS[profile]
        return self.variant(embedding=embedding, reranker=reranker)


settings = Settings()


# Named configurations the eval harness scores against each other.
def naive_config(base: Settings | None = None) -> Settings:
    """Single-pass dense retrieval: no sparse arm, no rerank, no correction."""
    base = base or settings
    return base.variant(
        use_hybrid=False,
        use_reranker=False,
        use_self_correction=False,
        max_retries=0,
    )


def agentic_config(base: Settings | None = None) -> Settings:
    """The full pipeline."""
    base = base or settings
    return base.variant(
        use_hybrid=True,
        use_reranker=True,
        use_self_correction=True,
    )
