"""Encoder layer: dense, sparse and cross-encoder models behind one interface.

Two backends implement the same protocols:

  * "onnx"  - fastembed, quantised CPU inference, no torch dependency.
  * "torch" - sentence-transformers, for models fastembed does not carry.

Models are cached per model name rather than per profile, so switching the
embedding in the UI and switching back does not reload weights. The cross-encoder
can be released explicitly (``release_reranker``), which matters on small
machines: the largest reranker here is ~2.3 GB and the largest embedder ~2.4 GB,
and this was built on a box with 7.3 GB total.
"""

from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass
from typing import Protocol, Sequence

from ..config import SPARSE_MODEL, EmbeddingSpec, RerankerSpec, Settings

log = logging.getLogger(__name__)

# Bulk-encoding batch size. Deliberately NOT using fastembed's parallel= option:
# it forks worker processes, and on Windows (spawn) each one reloads the ONNX
# model, which measured *slower* than the single-process path. ONNX Runtime's
# own intra-op threading made no difference either - embedding cost here is
# dominated by sequence length, not by core count, because attention is O(n^2):
# measured 254 chunks/s at 50 chars, 45 at 200, 10 at 800, and 2 on real
# code-heavy chunks that tokenise near the model's 512-token limit.
# That is why chunk_size matters more than any threading knob.
BULK_BATCH = 64


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class DenseEncoder(Protocol):
    dim: int

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def encode_query(self, text: str) -> list[float]: ...


class SparseEncoder(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...
    def encode_query(self, text: str) -> SparseVector: ...


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


# --------------------------------------------------------------------------- #
# ONNX backend (fastembed)
# --------------------------------------------------------------------------- #
class OnnxDenseEncoder:
    def __init__(self, spec: EmbeddingSpec) -> None:
        from fastembed import TextEmbedding

        self._spec = spec
        self.dim = spec.dim
        log.info("loading dense model %s (onnx, %d MB)", spec.model, spec.size_mb)
        self._model = TextEmbedding(model_name=spec.model)

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [self._spec.doc_prefix + t for t in texts]
        return [
            v.tolist()
            for v in self._model.embed(prefixed, batch_size=BULK_BATCH)
        ]

    def encode_query(self, text: str) -> list[float]:
        # No parallelism for a single query - spawning workers costs more than
        # the one forward pass it would save.
        prefixed = self._spec.query_prefix + text
        return next(iter(self._model.query_embed(prefixed))).tolist()


class OnnxSparseEncoder:
    def __init__(self) -> None:
        from fastembed import SparseTextEmbedding

        log.info("loading sparse model %s", SPARSE_MODEL)
        self._model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    @staticmethod
    def _convert(emb) -> SparseVector:
        return SparseVector(
            indices=[int(i) for i in emb.indices],
            values=[float(v) for v in emb.values],
        )

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [
            self._convert(e)
            for e in self._model.embed(list(texts), batch_size=BULK_BATCH)
        ]

    def encode_query(self, text: str) -> SparseVector:
        # BM25 scores the query side differently from the document side (no term
        # saturation), which is why fastembed exposes a separate query_embed.
        return self._convert(next(iter(self._model.query_embed(text))))


class OnnxReranker:
    def __init__(self, spec: RerankerSpec) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        log.info("loading reranker %s (onnx, %d MB)", spec.model, spec.size_mb)
        self._model = TextCrossEncoder(model_name=spec.model)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        return [float(s) for s in self._model.rerank(query, list(documents))]


# --------------------------------------------------------------------------- #
# Torch backend (sentence-transformers)
# --------------------------------------------------------------------------- #
class TorchDenseEncoder:
    def __init__(self, spec: EmbeddingSpec) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"{spec.label} needs torch. Install it with:\n"
                f"    pip install -r requirements-quality.txt"
            ) from exc

        self._spec = spec
        self.dim = spec.dim
        log.info("loading dense model %s (torch/cpu)", spec.model)
        self._model = SentenceTransformer(spec.model, device="cpu")

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([self._spec.doc_prefix + t for t in texts])

    def encode_query(self, text: str) -> list[float]:
        return self._encode([self._spec.query_prefix + text])[0]


class TorchReranker:
    def __init__(self, spec: RerankerSpec) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"{spec.label} needs torch. Install it with:\n"
                f"    pip install -r requirements-quality.txt"
            ) from exc

        log.info("loading reranker %s (torch/cpu)", spec.model)
        self._model = CrossEncoder(spec.model, device="cpu", max_length=512)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, d) for d in documents]
        return [float(s) for s in self._model.predict(pairs, batch_size=8)]


class NullReranker:
    """Used when reranking is switched off, so callers need no special case."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return [0.0] * len(documents)


# --------------------------------------------------------------------------- #
# Lazy registry
# --------------------------------------------------------------------------- #
class EncoderRegistry:
    """Process-wide lazy singletons, keyed by model name.

    Thread-safe because uvicorn serves requests from a thread pool; two
    concurrent first-requests would otherwise each load a copy of the weights.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._dense: dict[str, DenseEncoder] = {}
        self._sparse: SparseEncoder | None = None
        self._reranker: dict[str, Reranker] = {}

    def dense(self, spec: EmbeddingSpec) -> DenseEncoder:
        with self._lock:
            if spec.slug not in self._dense:
                cls = OnnxDenseEncoder if spec.backend == "onnx" else TorchDenseEncoder
                self._dense[spec.slug] = cls(spec)
            return self._dense[spec.slug]

    def sparse(self) -> SparseEncoder:
        # BM25 is ~10 MB and independent of the dense model; one instance serves
        # every collection.
        with self._lock:
            if self._sparse is None:
                self._sparse = OnnxSparseEncoder()
            return self._sparse

    def reranker(self, spec: RerankerSpec) -> Reranker:
        with self._lock:
            if spec.slug not in self._reranker:
                if not spec.model:
                    self._reranker[spec.slug] = NullReranker()
                else:
                    cls = OnnxReranker if spec.backend == "onnx" else TorchReranker
                    self._reranker[spec.slug] = cls(spec)
            return self._reranker[spec.slug]

    def release_reranker(self, spec: RerankerSpec) -> None:
        """Drop a cross-encoder's weights.

        Worth calling for the large torch reranker, which is ~2.3 GB and cannot
        sit alongside a 2.4 GB embedder on a small machine.
        """
        with self._lock:
            if self._reranker.pop(spec.slug, None) is not None:
                gc.collect()
                log.info("released reranker %s", spec.slug)

    def loaded(self) -> dict[str, list[str]]:
        with self._lock:
            return {
                "dense": sorted(self._dense),
                "sparse": ["bm25"] if self._sparse else [],
                "rerankers": sorted(self._reranker),
            }


REGISTRY = EncoderRegistry()


def get_encoders(settings: Settings) -> tuple[DenseEncoder, SparseEncoder]:
    return REGISTRY.dense(settings.embedding_spec), REGISTRY.sparse()


def get_reranker(settings: Settings) -> Reranker:
    return REGISTRY.reranker(settings.reranker_spec)
