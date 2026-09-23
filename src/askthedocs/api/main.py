"""FastAPI application.

Serves the JSON API and the static chat UI from one process, so the whole thing
is a single `askthedocs serve` with no separate frontend build step.

Two things here are worth knowing before reading the handlers:

**API keys.** A key supplied by the browser is used to construct the chat model
for that request and is then dropped. It is excluded from request logging (see
AskRequest.redacted), never written to disk, and never attached to a Langfuse
trace. The alternative - storing keys server-side - would make this demo a
credential store, which is not something a portfolio project should be.

**Embedding choice is not free.** Each embedding model has its own collection
because vector dimensionality is fixed at creation. Switching models in the UI
therefore switches collections, and if that collection was never indexed the API
says so explicitly instead of returning zero results and looking broken.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import (
    DEFAULT_EMBEDDING,
    DEFAULT_RERANKER,
    EMBEDDINGS,
    RAW_DIR,
    RERANKERS,
    SPARSE_MODEL,
    naive_config,
    settings as base_settings,
)
from ..config import REPO_ROOT
from ..graph.build import answer_question, config_name, get_graph
from ..graph.state import initial_state
from ..ingest import upload as upload_ingest
from ..llm import LLMConfigError, build_bundle, provider_catalog
from ..logging_setup import setup as setup_logging
from ..observability import tracing
from ..retrieval import store
from .schemas import (
    AskRequest,
    AskResponse,
    EmbeddingInfo,
    HealthResponse,
    ModelCatalog,
    ProviderInfo,
    RerankerInfo,
    UploadResponse,
    WorkspaceInfo,
)

log = logging.getLogger(__name__)

FRONTEND_DIR = REPO_ROOT / "frontend"

# Uploaded-document metadata, per workspace. Deliberately in-process: the
# chunks themselves live in Qdrant, and this is only what the UI needs to list
# files. A restart loses the listing, not the index.
_WORKSPACES: dict[str, list[dict]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    spec = base_settings.embedding_spec
    log.info(
        "starting: embedding=%s (%dd) reranker=%s",
        spec.model, spec.dim, base_settings.reranker_spec.label,
    )
    try:
        count = store.collection_size(base_settings)
        log.info("collection '%s' holds %d chunks", base_settings.collection, count)
        if count == 0:
            log.warning("collection is empty - run `askthedocs index --recreate`")
    except Exception as exc:
        log.warning("could not reach Qdrant at startup: %s", exc)

    # Warm the encoders so the first user request is not a 20 second model load.
    await asyncio.to_thread(_warm)
    yield
    tracing.flush()


def _warm() -> None:
    try:
        from ..retrieval.encoders import get_encoders

        dense, sparse = get_encoders(base_settings)
        dense.encode_query("warmup")
        sparse.encode_query("warmup")
        log.info("encoders warm")
    except Exception as exc:
        log.warning("encoder warmup failed: %s", exc)


app = FastAPI(
    title="Ask the Docs",
    version="2.0.0",
    description="Agentic RAG over documentation, with your own documents and models",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Settings resolution
# --------------------------------------------------------------------------- #
def _resolve(request: AskRequest):
    """Turn a request's model/corpus choices into a Settings object."""
    settings = base_settings

    if request.embedding:
        if request.embedding not in EMBEDDINGS:
            raise HTTPException(400, f"Unknown embedding '{request.embedding}'")
        settings = settings.variant(embedding=request.embedding)

    if request.reranker:
        if request.reranker not in RERANKERS:
            raise HTTPException(400, f"Unknown reranker '{request.reranker}'")
        settings = settings.variant(
            reranker=request.reranker, use_reranker=request.reranker != "none"
        )

    if request.corpus == "uploads":
        workspace = request.workspace or "default"
        settings = settings.variant(
            collection_base=f"upload_{upload_ingest._safe(workspace)}"
        )

    if request.config == "naive":
        settings = naive_config(settings)
    return settings


def _require_index(settings, *, corpus: str) -> None:
    """Fail loudly when the selected collection holds nothing."""
    try:
        count = store.collection_size(settings)
    except Exception as exc:
        raise HTTPException(503, f"Qdrant is unreachable: {exc}") from exc

    if count:
        return

    if corpus == "uploads":
        raise HTTPException(
            400,
            "No documents in this workspace yet for the selected embedding model. "
            "Upload a file, or switch back to the embedding you indexed with.",
        )
    raise HTTPException(
        400,
        f"The '{settings.embedding_spec.label}' index is empty. Build it with:\n"
        f"    askthedocs index --recreate --embedding {settings.embedding}",
    )


# --------------------------------------------------------------------------- #
# Health and catalogs
# --------------------------------------------------------------------------- #
@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    spec = base_settings.embedding_spec
    try:
        count = await asyncio.to_thread(store.collection_size, base_settings)
        status = "ok" if count else "empty"
    except Exception as exc:
        log.warning("health check could not reach Qdrant: %s", exc)
        count, status = 0, "qdrant_unreachable"

    # Pages actually *indexed*, not everything sitting in data/raw. The scraper
    # keeps more than INDEX_MAX_PAGES_PER_SET takes, so the on-disk count would
    # overstate what the collection can answer from.
    pages = 0
    if RAW_DIR.exists():
        from ..ingest.chunk import _select_pages

        for directory in RAW_DIR.iterdir():
            if directory.is_dir():
                pages += len(_select_pages(directory, base_settings.max_pages_per_set))

    return HealthResponse(
        status=status,
        collection=base_settings.collection,
        indexed_chunks=count,
        embedding=spec.slug,
        embedding_model=spec.model,
        embedding_dim=spec.dim,
        sparse_model=SPARSE_MODEL,
        reranker=base_settings.reranker_spec.slug,
        reranker_model=base_settings.reranker_spec.model or "-",
        profile=base_settings.profile,
        qdrant=base_settings.qdrant_url or "embedded (local mode)",
        langfuse=bool(tracing.callbacks(base_settings.langfuse_enabled)),
        default_provider=base_settings.provider,
        corpus_pages=pages,
    )


@app.get("/api/providers", response_model=list[ProviderInfo])
async def providers() -> list[ProviderInfo]:
    return [ProviderInfo(**p) for p in provider_catalog()]


@app.get("/api/models", response_model=ModelCatalog)
async def models() -> ModelCatalog:
    """Embeddings and rerankers, with how much is indexed for each.

    The chunk count is what lets the UI grey out a model the user has not
    indexed yet, instead of letting them select it and get no results.
    """
    def _count(slug: str) -> int:
        try:
            return store.collection_size(base_settings.variant(embedding=slug))
        except Exception:
            return 0

    torch_available = _torch_available()

    embeddings = []
    for spec in EMBEDDINGS.values():
        usable = spec.backend == "onnx" or torch_available
        embeddings.append(
            EmbeddingInfo(
                slug=spec.slug,
                label=spec.label,
                model=spec.model,
                dim=spec.dim,
                size_mb=spec.size_mb,
                backend=spec.backend,
                notes=spec.notes,
                indexed_chunks=await asyncio.to_thread(_count, spec.slug),
                available=usable,
                unavailable_reason=(
                    "" if usable else "needs torch (pip install -r requirements-quality.txt)"
                ),
            )
        )

    rerankers = [
        RerankerInfo(
            slug=spec.slug,
            label=spec.label,
            model=spec.model or "-",
            size_mb=spec.size_mb,
            backend=spec.backend,
            notes=spec.notes,
            available=spec.backend == "onnx" or torch_available,
        )
        for spec in RERANKERS.values()
    ]

    return ModelCatalog(
        embeddings=embeddings,
        rerankers=rerankers,
        default_embedding=DEFAULT_EMBEDDING,
        default_reranker=DEFAULT_RERANKER,
    )


def _torch_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #
@app.post("/api/upload", response_model=UploadResponse)
async def upload(
    files: list[UploadFile] = File(...),
    workspace: str = Form("default"),
    embedding: str = Form(DEFAULT_EMBEDDING),
) -> UploadResponse:
    if embedding not in EMBEDDINGS:
        raise HTTPException(400, f"Unknown embedding '{embedding}'")

    settings = base_settings.variant(embedding=embedding)
    payloads: list[tuple[str, bytes]] = []
    for item in files:
        payloads.append((item.filename or "untitled", await item.read()))

    started = time.perf_counter()
    try:
        docs, written = await asyncio.to_thread(
            upload_ingest.ingest_files, settings, workspace, payloads
        )
    except upload_ingest.UploadError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log.exception("upload failed")
        raise HTTPException(500, f"Indexing failed: {exc}") from exc

    key = f"{upload_ingest._safe(workspace)}:{embedding}"
    existing = {d["doc_id"]: d for d in _WORKSPACES.get(key, [])}
    for doc in docs:
        existing[doc.doc_id] = doc.to_dict()
    _WORKSPACES[key] = list(existing.values())

    return UploadResponse(
        workspace=workspace,
        embedding=embedding,
        documents=_WORKSPACES[key],
        chunks_indexed=written,
        total_chunks=await asyncio.to_thread(
            upload_ingest.workspace_size, settings, workspace
        ),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )


@app.get("/api/workspace", response_model=WorkspaceInfo)
async def workspace_info(
    workspace: str = "default", embedding: str = DEFAULT_EMBEDDING
) -> WorkspaceInfo:
    if embedding not in EMBEDDINGS:
        raise HTTPException(400, f"Unknown embedding '{embedding}'")
    settings = base_settings.variant(embedding=embedding)
    key = f"{upload_ingest._safe(workspace)}:{embedding}"
    return WorkspaceInfo(
        workspace=workspace,
        embedding=embedding,
        total_chunks=await asyncio.to_thread(
            upload_ingest.workspace_size, settings, workspace
        ),
        documents=_WORKSPACES.get(key, []),
    )


@app.delete("/api/workspace")
async def clear_workspace(
    workspace: str = "default", embedding: str = DEFAULT_EMBEDDING
) -> dict:
    if embedding not in EMBEDDINGS:
        raise HTTPException(400, f"Unknown embedding '{embedding}'")
    settings = base_settings.variant(embedding=embedding)
    removed = await asyncio.to_thread(
        upload_ingest.clear_workspace, settings, workspace
    )
    _WORKSPACES.pop(f"{upload_ingest._safe(workspace)}:{embedding}", None)
    return {"cleared": removed, "workspace": workspace, "embedding": embedding}


# --------------------------------------------------------------------------- #
# Ask
# --------------------------------------------------------------------------- #
@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    log.info("ask: %s", json.dumps(request.redacted())[:400])
    settings = _resolve(request)
    _require_index(settings, corpus=request.corpus)

    try:
        result = await asyncio.to_thread(
            answer_question,
            request.question,
            settings=settings,
            doc_sets=request.doc_sets or None,
            provider=request.provider,
            model=request.model,
            api_key=request.api_key,
            callbacks=tracing.callbacks(settings.langfuse_enabled),
        )
    except LLMConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result.status == "error":
        raise HTTPException(status_code=502, detail=result.answer)

    return AskResponse(**result.to_dict())


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/ask/stream")
async def ask_stream(request: AskRequest) -> StreamingResponse:
    """Stream node progress, then the answer token by token.

    The graph makes several LLM calls (analysis, grading, generation,
    groundedness). Only the generation node's tokens are streamed as answer
    text; the rest surface as progress events so the UI can show what the
    pipeline is doing instead of a spinner.
    """
    log.info("ask/stream: %s", json.dumps(request.redacted())[:400])
    settings = _resolve(request)
    _require_index(settings, corpus=request.corpus)

    try:
        llms = build_bundle(
            settings,
            provider=request.provider,
            model=request.model,
            api_key=request.api_key,
        )
    except LLMConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    graph = get_graph(settings, llms)

    async def stream():
        config = {
            "recursion_limit": 40,
            "callbacks": tracing.callbacks(settings.langfuse_enabled),
            "metadata": {
                "config": config_name(settings),
                "embedding": settings.embedding,
                "corpus": request.corpus,
                "provider": llms.provider,
            },
        }
        state = initial_state(request.question, request.doc_sets or None)
        final: dict = {}
        answered = False

        yield _sse("start", {
            "config": config_name(settings),
            "provider": llms.provider,
            "model": llms.generator_model,
            "embedding": settings.embedding_spec.label,
            "reranker": settings.reranker_spec.label,
            "corpus": request.corpus,
        })
        try:
            async for mode, payload in graph.astream(
                state, config=config, stream_mode=["updates", "messages"]
            ):
                if mode == "messages":
                    chunk, meta = payload
                    if meta.get("langgraph_node") != "generate":
                        continue
                    text = getattr(chunk, "content", "")
                    if isinstance(text, list):
                        text = "".join(
                            b.get("text", "") for b in text if isinstance(b, dict)
                        )
                    if text:
                        answered = True
                        yield _sse("token", {"text": text})

                elif mode == "updates":
                    for node, update in payload.items():
                        final.update(update or {})
                        for event in (update or {}).get("events", []):
                            yield _sse("node", event)
                        # A regeneration replaces the answer; tell the UI to
                        # clear what it has streamed so far.
                        if node == "mark_regenerate":
                            yield _sse("reset", {"reason": "ungrounded"})

        except Exception as exc:
            log.exception("stream failed")
            yield _sse("error", {"detail": str(exc)})
            return

        hits = final.get("hits") or []
        citations = final.get("citations") or [h.citation() for h in hits]
        yield _sse(
            "done",
            {
                "answer": final.get("answer", ""),
                "citations": citations,
                "status": final.get("status", "answered"),
                "grounded": final.get("grounded", True),
                "relevant": final.get("relevant", True),
                "attempts": final.get("attempts", 1),
                "rewrites": final.get("rewrites", []),
                "retrieval_mode": final.get("retrieval_mode", "hybrid"),
                "streamed": answered,
                "config": config_name(settings),
            },
        )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")
