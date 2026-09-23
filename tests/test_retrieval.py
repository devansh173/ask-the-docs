"""Retrieval tests. No LLM, no network, no API key - these run in CI on every commit."""

from __future__ import annotations

import pytest

from askthedocs.config import (
    EMBEDDINGS,
    RERANKERS,
    Settings,
    agentic_config,
    naive_config,
    settings,
)
from askthedocs.ingest.chunk import _is_noise, _parse_frontmatter
from askthedocs.ingest.sources import SOURCES_BY_NAME
from askthedocs.retrieval import store
from askthedocs.retrieval.schema import Chunk

COLLECTION = "pytest_retrieval"

CORPUS = [
    (
        "IDF modifier",
        "When you store BM25 sparse vectors in Qdrant you must set "
        "modifier=Modifier.IDF on the sparse vector params. Qdrant then computes "
        "inverse document frequency from corpus statistics at query time.",
    ),
    (
        "RRF fusion",
        "Reciprocal Rank Fusion merges two ranked lists by summing 1/(k+rank) "
        "for every document. Qdrant exposes it through FusionQuery with Fusion.RRF "
        "on the Query API.",
    ),
    (
        "Prefetch",
        "A prefetch runs a nested search whose candidates are passed to the outer "
        "query. Hybrid search uses one prefetch for the dense vector and another "
        "for the sparse vector.",
    ),
    (
        "StateGraph",
        "LangGraph's StateGraph takes a typed state schema. Nodes read and write "
        "that state, and conditional edges route between them based on its contents.",
    ),
    (
        "Cross encoder",
        "A cross-encoder scores a query and a document together in one forward "
        "pass. It is more accurate than bi-encoder cosine similarity and far "
        "slower, so it runs over a shortlist rather than the whole corpus.",
    ),
    (
        "Prompt caching",
        "Prompt caching matches on an exact prefix. Any byte change anywhere in "
        "the prefix invalidates every cache entry after it.",
    ),
]


@pytest.fixture(scope="module")
def indexed() -> Settings:
    cfg = settings.variant(collection_base=COLLECTION)
    chunks = [
        Chunk(
            chunk_id=f"t{i}",
            text=f"Test docs > {name}\n\n{body}",
            source_url=f"https://example.test/{i}",
            title=name,
            section=name,
            doc_set="test",
            position=i,
        )
        for i, (name, body) in enumerate(CORPUS)
    ]
    store.index_chunks(cfg, chunks, recreate=True)
    return cfg


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_embedding_registry_is_self_consistent():
    for slug, spec in EMBEDDINGS.items():
        assert spec.slug == slug
        assert spec.dim > 0
        assert spec.backend in {"onnx", "torch"}
        assert spec.model, f"{slug} has no model name"


def test_every_embedding_gets_its_own_collection():
    """Dimensionality is fixed per collection, so the names must not collide."""
    names = {settings.variant(embedding=s).collection for s in EMBEDDINGS}
    assert len(names) == len(EMBEDDINGS)


def test_reranker_registry_is_self_consistent():
    for slug, spec in RERANKERS.items():
        assert spec.slug == slug
        assert spec.backend in {"onnx", "torch"}
    assert RERANKERS["none"].model == "", "the 'none' reranker must have no model"


def test_none_reranker_disables_reranking():
    assert not settings.variant(reranker="none").reranking_enabled
    assert settings.variant(reranker="bge-base", use_reranker=True).reranking_enabled


def test_naive_config_disables_every_mechanism():
    cfg = naive_config(settings)
    assert not cfg.use_hybrid
    assert not cfg.use_reranker
    assert not cfg.use_self_correction
    assert cfg.max_retries == 0


def test_agentic_config_enables_them():
    cfg = agentic_config(settings)
    assert cfg.use_hybrid and cfg.use_reranker and cfg.use_self_correction


def test_source_filters_exclude_disallowed_paths():
    claude = SOURCES_BY_NAME["claude-api"]
    # robots.txt disallows /api/
    assert not claude.matches("https://platform.claude.com/docs/en/api/messages")
    assert claude.matches("https://platform.claude.com/docs/en/build-with-claude/x")
    # Other locales must not leak in alongside English.
    assert not claude.matches("https://platform.claude.com/docs/ja/home")


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def test_frontmatter_roundtrip():
    meta, body = _parse_frontmatter(
        '---\nurl: https://x.test/a\ntitle: "A Title"\ndoc_set: qdrant\n---\n\nBody here.'
    )
    assert meta["url"] == "https://x.test/a"
    assert meta["title"] == "A Title"
    assert body.startswith("Body here")


def test_noise_filter_drops_fragments_keeps_prose():
    assert _is_noise("Too short.")
    assert not _is_noise(
        "Reciprocal Rank Fusion merges ranked lists by summing reciprocal ranks, "
        "which avoids comparing dense and sparse scores that live on different "
        "scales and would otherwise need arbitrary weighting."
    )


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def test_indexing_writes_every_chunk(indexed):
    assert store.collection_size(indexed) == len(CORPUS)


def test_dense_search_returns_ranked_hits(indexed):
    hits = store.search_arm(indexed, "what is a cross encoder reranker", "dense", limit=3)
    assert hits
    assert hits[0].title == "Cross encoder"
    assert all(
        a.retrieval_score >= b.retrieval_score for a, b in zip(hits, hits[1:])
    ), "dense hits must come back sorted"


def test_sparse_arm_matches_exact_identifiers(indexed):
    """BM25 is in the stack for terms a dense model blurs together."""
    hits = store.search_arm(indexed, "Modifier.IDF", "sparse", limit=3)
    assert hits, "BM25 should match the literal identifier"
    assert hits[0].title == "IDF modifier"
    assert all(h.retrieval_score > 0 for h in hits), "zero-score matches must be dropped"


def test_hybrid_returns_fused_ranking(indexed):
    trace = store.search(indexed.variant(use_hybrid=True), "how does RRF fusion work")
    assert trace.mode == "hybrid"
    assert trace.hits
    assert trace.hits[0].title == "RRF fusion"
    assert trace.dense_candidates > 0 and trace.sparse_candidates > 0


def test_dense_only_config_reports_dense_mode(indexed):
    trace = store.search(indexed.variant(use_hybrid=False), "how does RRF fusion work")
    assert trace.mode == "dense"
    assert trace.sparse_candidates == 0


def test_hybrid_recovers_a_rare_identifier_dense_alone_ranks_lower(indexed):
    """The case that justifies the sparse arm.

    A bare identifier carries little semantic signal, so the bi-encoder spreads
    it across anything topically adjacent. BM25 matches the literal token, and
    fusion pulls it back to the top.
    """
    query = "Modifier.IDF"
    dense = store.search(indexed.variant(use_hybrid=False), query, limit=5).hits
    hybrid = store.search(indexed.variant(use_hybrid=True), query, limit=5).hits

    def rank_of(hits, title):
        return next((i for i, h in enumerate(hits, 1) if h.title == title), 99)

    assert rank_of(hybrid, "IDF modifier") <= rank_of(dense, "IDF modifier")
    assert rank_of(hybrid, "IDF modifier") == 1


def test_doc_set_filter_restricts_results(indexed):
    assert store.search(indexed, "fusion", doc_sets=["test"]).hits
    assert not store.search(indexed, "fusion", doc_sets=["nonexistent"]).hits


def test_payload_survives_the_roundtrip(indexed):
    hit = store.search(indexed, "prompt caching prefix").hits[0]
    assert hit.title == "Prompt caching"
    assert hit.source_url.startswith("https://example.test/")
    assert hit.doc_set == "test"
    assert hit.chunk_id
