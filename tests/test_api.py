"""API surface tests.

The lifespan handler warms the encoders and opens Qdrant, so these use
TestClient as a context manager to exercise it exactly as uvicorn would.

A small corpus is indexed first. Without it every /api/ask call short-circuits
on "the index is empty", which would mask the provider and key validation these
tests are actually about. No LLM is called: a request without a key must fail
with a 400, and proving that is the point rather than a limitation.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from askthedocs.api.main import app
from askthedocs.config import settings
from askthedocs.retrieval import store
from askthedocs.retrieval.schema import Chunk

SEED = [
    ("RRF fusion", "Reciprocal Rank Fusion merges ranked lists by summing "
                   "1/(k + rank) for each document, with k=2 in Qdrant."),
    ("IDF modifier", "Set modifier=Modifier.IDF on sparse vector params so "
                     "Qdrant applies inverse document frequency."),
]


@pytest.fixture(scope="module", autouse=True)
def seeded_index():
    chunks = [
        Chunk(
            chunk_id=f"api{i}",
            text=f"Test docs > {name}\n\n{body}",
            source_url=f"https://example.test/{i}",
            title=name,
            section=name,
            doc_set="claude-api",
            position=i,
        )
        for i, (name, body) in enumerate(SEED)
    ]
    store.index_chunks(settings, chunks, recreate=True)
    return settings


@pytest.fixture(scope="module")
def client(seeded_index):
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
def test_health_reports_the_active_configuration(client):
    body = client.get("/api/health").json()

    assert body["status"] in {"ok", "empty", "qdrant_unreachable"}
    assert body["embedding"] and body["embedding_model"]
    assert body["embedding_dim"] > 0
    assert body["sparse_model"]
    assert body["reranker"]
    assert isinstance(body["indexed_chunks"], int)
    # Surfacing the chunk count is what makes "deployed but never indexed"
    # diagnosable from outside the container.
    assert "qdrant" in body


def test_health_sees_the_seeded_chunks(client):
    assert client.get("/api/health").json()["indexed_chunks"] == len(SEED)


# --------------------------------------------------------------------------- #
# Catalogs
# --------------------------------------------------------------------------- #
def test_providers_endpoint_drives_the_ui_picker(client):
    providers = client.get("/api/providers").json()
    assert providers
    ids = {p["id"] for p in providers}
    assert {"anthropic", "google_genai", "openai", "ollama"} <= ids
    for provider in providers:
        assert provider["models"]
        assert provider["default_model"] in provider["models"]


def test_provider_response_never_includes_a_key(client):
    body = client.get("/api/providers").text
    assert "sk-" not in body
    assert "api_key" not in body


def test_models_endpoint_lists_embeddings_with_index_sizes(client):
    catalog = client.get("/api/models").json()

    assert catalog["embeddings"] and catalog["rerankers"]
    slugs = {e["slug"] for e in catalog["embeddings"]}
    assert catalog["default_embedding"] in slugs

    active = next(e for e in catalog["embeddings"] if e["slug"] == settings.embedding)
    assert active["indexed_chunks"] == len(SEED), (
        "the UI relies on this count to disable un-indexed models"
    )
    for embedding in catalog["embeddings"]:
        assert embedding["dim"] > 0
        assert embedding["backend"] in {"onnx", "torch"}


def test_rerankers_include_a_none_option(client):
    catalog = client.get("/api/models").json()
    assert "none" in {r["slug"] for r in catalog["rerankers"]}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_missing_question_is_a_422(client):
    assert client.post("/api/ask", json={}).status_code == 422


def test_too_short_question_is_a_422(client):
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422


def test_unknown_embedding_is_rejected(client):
    response = client.post(
        "/api/ask",
        json={"question": "how does RRF fusion work?", "embedding": "not-a-model"},
    )
    assert response.status_code == 400
    assert "Unknown embedding" in response.json()["detail"]


def test_unindexed_embedding_says_how_to_fix_it(client):
    """Selecting a model with no index must explain itself, not return nothing."""
    response = client.post(
        "/api/ask",
        json={"question": "how does RRF fusion work?", "embedding": "bge-base"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "index" in detail.lower()
    assert "--embedding bge-base" in detail


def test_unknown_provider_is_a_400_not_a_500(client):
    response = client.post(
        "/api/ask",
        json={"question": "how does RRF fusion work?", "provider": "not-a-provider"},
    )
    assert response.status_code == 400
    assert "Unknown provider" in response.json()["detail"]


def test_missing_key_is_a_400_with_a_useful_message(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    response = client.post(
        "/api/ask",
        json={"question": "how does RRF fusion work?", "provider": "anthropic"},
    )
    assert response.status_code == 400
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #
def _upload(client, name: str, content: bytes, **form):
    return client.post(
        "/api/upload",
        files=[("files", (name, io.BytesIO(content), "application/octet-stream"))],
        data={"workspace": "pytest", "embedding": settings.embedding, **form},
    )


def test_markdown_upload_is_chunked_and_indexed(client):
    body = (
        "# Retrieval notes\n\n"
        "## Fusion\n\n"
        + "Reciprocal rank fusion combines two ranked lists by rank rather than "
        "by score, which avoids having to weight incomparable scales. " * 6
        + "\n\n## Reranking\n\n"
        + "A cross-encoder reads the query and the passage together in a single "
        "forward pass, which is more accurate and much slower. " * 6
    ).encode()

    response = _upload(client, "notes.md", body)
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["chunks_indexed"] > 0
    assert payload["total_chunks"] >= payload["chunks_indexed"]
    assert payload["documents"][0]["filename"] == "notes.md"
    assert payload["documents"][0]["title"] == "Retrieval notes"


def test_uploaded_documents_are_listed(client):
    info = client.get(
        f"/api/workspace?workspace=pytest&embedding={settings.embedding}"
    ).json()
    assert info["total_chunks"] > 0
    assert any(d["filename"] == "notes.md" for d in info["documents"])


def test_uploads_live_in_a_separate_collection_from_the_docs(client):
    """A user's files must not leak into the shared documentation index."""
    assert client.get("/api/health").json()["indexed_chunks"] == len(SEED)


def test_unsupported_file_type_is_rejected_with_a_readable_message(client):
    response = _upload(client, "archive.zip", b"PK\x03\x04 not a document")
    assert response.status_code == 400
    assert "unsupported file type" in response.json()["detail"].lower()


def test_empty_file_is_rejected(client):
    response = _upload(client, "empty.txt", b"")
    assert response.status_code == 400


def test_asking_an_empty_workspace_explains_itself(client):
    response = client.post(
        "/api/ask",
        json={
            "question": "what do my documents say about fusion?",
            "corpus": "uploads",
            "workspace": "never-used",
        },
    )
    assert response.status_code == 400
    assert "workspace" in response.json()["detail"].lower()


def test_workspace_can_be_cleared(client):
    cleared = client.delete(
        f"/api/workspace?workspace=pytest&embedding={settings.embedding}"
    ).json()
    assert cleared["cleared"] is True

    info = client.get(
        f"/api/workspace?workspace=pytest&embedding={settings.embedding}"
    ).json()
    assert info["total_chunks"] == 0
    assert info["documents"] == []


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #
def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Ask the Docs" in response.text


def test_static_assets_are_served(client):
    for path in ("/static/app.js", "/static/styles.css"):
        assert client.get(path).status_code == 200, f"{path} is not being served"


def test_openapi_schema_builds(client):
    """A schema that fails to build means a response model is malformed."""
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/ask", "/api/health", "/api/upload", "/api/models"} <= set(paths)
