"""Qdrant access: collection setup, indexing, and hybrid search.

Hybrid search is a single Query API call. Two prefetches run in parallel
server-side - one over the dense vector, one over the BM25 sparse vector - and
their *ranks* are fused with Reciprocal Rank Fusion. Fusing ranks rather than
scores is the point: dense cosine similarity lives in [-1, 1] and BM25 scores
are unbounded, so any weighted sum of the raw numbers is arbitrary.

Two deployment shapes, same code path:

  * ``QDRANT_URL`` set  -> a real Qdrant server (Docker or Qdrant Cloud).
  * ``QDRANT_URL`` unset -> embedded local mode against ``data/qdrant``.

Local mode is the default so the project runs on a laptop with no Docker and no
signup. One known divergence: on the sparse arm, local mode returns points whose
dot product is exactly 0.0 where the server's indexed path drops them. Qdrant's
RRF scores a result at 1/(k + rank) with k=2 and zero-based ranks, so a
zero-score point at the tail of a 40-candidate prefetch contributes 1/41, about
5% of the 1/2 a top-ranked hit contributes. It is tail noise, not a change to
the head of the ranking. Pointing QDRANT_URL at a server removes it entirely.
"""

from __future__ import annotations

import atexit
import logging
import time
import uuid
from typing import Sequence

from qdrant_client import QdrantClient, models

from ..config import Settings
from .encoders import SparseVector, get_encoders
from .schema import DENSE_VECTOR, SPARSE_VECTOR, Chunk, Hit, RetrievalTrace

log = logging.getLogger(__name__)

_CLIENTS: dict[str, QdrantClient] = {}


@atexit.register
def _close_clients() -> None:
    """Close local-mode clients before interpreter teardown.

    QdrantClient.__del__ tries to import during shutdown and raises a confusing
    ImportError if the local storage is still open. Closing here pre-empts it.
    """
    for client in _CLIENTS.values():
        try:
            client.close()
        except Exception:  # teardown is best effort
            pass
    _CLIENTS.clear()


def get_client(settings: Settings) -> QdrantClient:
    """One client per target. Local mode holds a file lock, so it must be shared."""
    key = settings.qdrant_url or f"local:{settings.local_path}"
    if key not in _CLIENTS:
        if settings.qdrant_url:
            log.info("connecting to Qdrant server at %s", settings.qdrant_url)
            _CLIENTS[key] = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=60,
            )
        else:
            settings.local_path.mkdir(parents=True, exist_ok=True)
            log.info("using embedded Qdrant at %s", settings.local_path)
            _CLIENTS[key] = QdrantClient(path=str(settings.local_path))
    return _CLIENTS[key]


def is_local(settings: Settings) -> bool:
    return not settings.qdrant_url


def ensure_collection(
    settings: Settings, dim: int, *, recreate: bool = False
) -> QdrantClient:
    client = get_client(settings)
    name = settings.collection

    if recreate and client.collection_exists(name):
        log.warning("dropping existing collection %s", name)
        client.delete_collection(name)

    if not client.collection_exists(name):
        log.info("creating collection %s (dim=%d)", name, dim)
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    # BM25 term weights from fastembed carry no IDF component -
                    # Qdrant computes IDF from corpus statistics at query time.
                    # Without this modifier the sparse arm degrades to raw TF.
                    modifier=models.Modifier.IDF
                )
            },
        )
        # Payload indexes so metadata filters stay cheap as the corpus grows.
        # Local mode scans payloads anyway and warns if you ask for an index.
        if not is_local(settings):
            for field_name in ("doc_set", "source_url"):
                client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
    return client


def collection_size(settings: Settings) -> int:
    client = get_client(settings)
    if not client.collection_exists(settings.collection):
        return 0
    return client.count(settings.collection, exact=True).count


def index_chunks(
    settings: Settings,
    chunks: Sequence[Chunk],
    *,
    # Embedding dominates the cost and both encoders batch internally, so larger
    # batches mean fewer round trips without raising peak memory much.
    batch_size: int = 128,
    recreate: bool = False,
    progress=None,
) -> int:
    """Embed and upsert chunks. Returns the number of points written."""
    dense_enc, sparse_enc = get_encoders(settings)
    client = ensure_collection(settings, dense_enc.dim, recreate=recreate)

    written = 0
    for start in range(0, len(chunks), batch_size):
        batch = list(chunks[start : start + batch_size])
        texts = [c.text for c in batch]

        dense_vecs = dense_enc.encode_documents(texts)
        sparse_vecs = sparse_enc.encode_documents(texts)

        points = [
            models.PointStruct(
                # Qdrant ids must be uuid or int; derive a stable uuid from the
                # chunk id so re-ingesting the same corpus updates in place.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector={
                    DENSE_VECTOR: dense,
                    SPARSE_VECTOR: models.SparseVector(
                        indices=sparse.indices, values=sparse.values
                    ),
                },
                payload=chunk.payload(),
            )
            for chunk, dense, sparse in zip(batch, dense_vecs, sparse_vecs)
        ]
        # wait=True only means anything against a server, where it blocks until
        # the write is applied. Embedded mode is synchronous already, and each
        # upsert there persists the whole store - so asking for many small
        # confirmed writes is what makes local ingestion crawl. Large batches,
        # and no per-batch wait locally.
        client.upsert(
            collection_name=settings.collection,
            points=points,
            wait=not is_local(settings),
        )
        written += len(points)
        if progress is not None:
            progress(written, len(chunks))

    return written


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def _filter_for(doc_sets: Sequence[str] | None) -> models.Filter | None:
    if not doc_sets:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="doc_set", match=models.MatchAny(any=list(doc_sets))
            )
        ]
    )


def search(
    settings: Settings,
    query: str,
    *,
    limit: int | None = None,
    doc_sets: Sequence[str] | None = None,
) -> RetrievalTrace:
    """Hybrid (dense + BM25, RRF-fused) or dense-only, per settings.use_hybrid."""
    started = time.perf_counter()
    limit = limit or settings.fusion_limit
    client = get_client(settings)
    dense_enc, sparse_enc = get_encoders(settings)
    query_filter = _filter_for(doc_sets)

    dense_vec = dense_enc.encode_query(query)

    if not settings.use_hybrid:
        response = client.query_points(
            collection_name=settings.collection,
            query=dense_vec,
            using=DENSE_VECTOR,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        hits = [Hit.from_point(p) for p in response.points]
        return RetrievalTrace(
            query=query,
            mode="dense",
            dense_candidates=len(hits),
            fused=len(hits),
            latency_ms=(time.perf_counter() - started) * 1000,
            hits=hits,
        )

    sparse_vec: SparseVector = sparse_enc.encode_query(query)

    response = client.query_points(
        collection_name=settings.collection,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using=DENSE_VECTOR,
                limit=settings.dense_prefetch,
                filter=query_filter,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices, values=sparse_vec.values
                ),
                using=SPARSE_VECTOR,
                limit=settings.sparse_prefetch,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )

    hits = [Hit.from_point(p) for p in response.points]
    return RetrievalTrace(
        query=query,
        mode="hybrid",
        dense_candidates=settings.dense_prefetch,
        sparse_candidates=settings.sparse_prefetch,
        fused=len(hits),
        latency_ms=(time.perf_counter() - started) * 1000,
        hits=hits,
    )


def search_arm(
    settings: Settings, query: str, arm: str, limit: int = 10
) -> list[Hit]:
    """Single-arm search. Used by the sanity-check script to show dense and
    sparse retrieval disagreeing, which is the whole reason to fuse them."""
    client = get_client(settings)
    dense_enc, sparse_enc = get_encoders(settings)

    if arm == "dense":
        response = client.query_points(
            collection_name=settings.collection,
            query=dense_enc.encode_query(query),
            using=DENSE_VECTOR,
            limit=limit,
            with_payload=True,
        )
    elif arm == "sparse":
        sv = sparse_enc.encode_query(query)
        response = client.query_points(
            collection_name=settings.collection,
            query=models.SparseVector(indices=sv.indices, values=sv.values),
            using=SPARSE_VECTOR,
            limit=limit,
            with_payload=True,
        )
    else:
        raise ValueError(f"unknown arm {arm!r}")

    hits = [Hit.from_point(p) for p in response.points]
    if arm == "sparse" and is_local(settings):
        # See module docstring: local mode keeps zero-score sparse matches.
        hits = [h for h in hits if h.retrieval_score > 0.0]
    return hits
