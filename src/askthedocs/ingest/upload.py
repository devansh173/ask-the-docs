"""Ingest user-uploaded documents.

Uploads go into their own Qdrant collection, separate from the built-in
documentation corpus, so a user's private files never mix into the shared index
and can be dropped wholesale when they are done.

The collection name encodes both the workspace and the embedding model
(``upload_<workspace>__<embedding>``) because vector dimensionality is fixed at
collection creation: a 768-dim model cannot query an index built with a 384-dim
one. Switching embedding in the UI therefore means re-indexing the uploads,
which the API reports rather than failing mysteriously.

Supported inputs are PDF, plain text, Markdown and HTML. Everything is converted
to markdown and then run through exactly the same heading-aware chunker the
documentation corpus uses, so retrieval behaves identically on both.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import UPLOAD_DIR, Settings, collection_name
from ..retrieval.schema import Chunk

log = logging.getLogger(__name__)

# Extensions we can actually parse. Anything else is rejected with a clear
# message rather than being indexed as mojibake.
SUPPORTED = {".pdf", ".txt", ".md", ".markdown", ".html", ".htm", ".rst"}

MAX_FILE_BYTES = 20 * 1024 * 1024   # 20 MB per file
MAX_TOTAL_CHARS = 4_000_000         # ~1M tokens of text per workspace


class UploadError(ValueError):
    """Raised for input problems that should reach the user as a 400."""


@dataclass
class UploadedDoc:
    doc_id: str
    filename: str
    title: str
    chars: int
    chunks: int
    pages: int | None = None
    uploaded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict:
        return asdict(self)


def upload_collection(settings: Settings, workspace: str) -> str:
    """Collection name for a workspace under the current embedding model."""
    return collection_name(f"upload_{_safe(workspace)}", settings.embedding)


def _safe(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return cleaned[:40] or "default"


def workspace_dir(workspace: str) -> Path:
    path = UPLOAD_DIR / _safe(workspace)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _clean(text: str) -> str:
    # PDF extraction routinely yields NUL bytes, soft hyphens and ligatures that
    # survive into chunks and quietly damage both embeddings and BM25 tokens.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "").replace("­", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_pdf(data: bytes) -> tuple[str, int]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise UploadError(f"could not read the PDF: {exc}") from exc

    if reader.is_encrypted:
        # An encrypted PDF often "opens" but extracts nothing, which would
        # otherwise look like a blank document rather than a locked one.
        try:
            reader.decrypt("")
        except Exception as exc:
            raise UploadError("this PDF is password protected") from exc

    pages: list[str] = []
    for n, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            log.warning("page %d failed to extract: %s", n, exc)
            text = ""
        if text.strip():
            # Keep a page marker so citations can point at a page number.
            pages.append(f"\n\n## Page {n}\n\n{text}")

    if not pages:
        raise UploadError(
            "no selectable text found - this looks like a scanned PDF. "
            "Run OCR on it first, then upload the searchable version."
        )
    return _clean("".join(pages)), len(reader.pages)


def _parse_html(data: bytes) -> str:
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    soup = BeautifulSoup(data.decode("utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    main = soup.select_one("main") or soup.select_one("article") or soup.body or soup
    return _clean(markdownify(str(main), heading_style="ATX"))


def parse_file(filename: str, data: bytes) -> tuple[str, int | None]:
    """Return (markdown, page_count). Raises UploadError on bad input."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise UploadError(
            f"{filename}: unsupported file type '{suffix or 'none'}'. "
            f"Supported: {', '.join(sorted(SUPPORTED))}"
        )
    if not data:
        raise UploadError(f"{filename} is empty")
    if len(data) > MAX_FILE_BYTES:
        raise UploadError(
            f"{filename} is {len(data) // 1024 // 1024} MB; the limit is "
            f"{MAX_FILE_BYTES // 1024 // 1024} MB"
        )

    if suffix == ".pdf":
        return _parse_pdf(data)
    if suffix in {".html", ".htm"}:
        return _parse_html(data), None

    text = _clean(data.decode("utf-8", errors="replace"))
    if not text:
        raise UploadError(f"{filename} contained no readable text")
    return text, None


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def _title_from(filename: str, markdown: str) -> str:
    heading = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    if heading and len(heading.group(1)) < 120:
        return heading.group(1).strip()
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip() or filename


def build_chunks(
    settings: Settings,
    filename: str,
    markdown: str,
    doc_id: str,
) -> list[Chunk]:
    """Chunk an uploaded document with the same splitter the corpus uses."""
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    title = _title_from(filename, markdown)
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    body_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    try:
        sections = header_splitter.split_text(markdown)
    except Exception:
        sections = []
    if not sections:
        sections = [type("Doc", (), {"page_content": markdown, "metadata": {}})()]

    chunks: list[Chunk] = []
    position = 0
    for section in sections:
        meta = getattr(section, "metadata", {}) or {}
        section_path = " > ".join(
            meta[k] for k in ("h1", "h2", "h3") if meta.get(k)
        )
        breadcrumb = " > ".join(p for p in (title, section_path) if p)

        for piece in body_splitter.split_text(section.page_content):
            if len(piece.strip()) < 80:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{position}",
                    text=f"{breadcrumb}\n\n{piece.strip()}",
                    # No public URL for an upload; the UI renders these as
                    # filename + section rather than as a link.
                    source_url="",
                    title=title,
                    section=section_path,
                    doc_set=f"upload:{filename}",
                    position=position,
                )
            )
            position += 1
    return chunks


def ingest_files(
    settings: Settings,
    workspace: str,
    files: list[tuple[str, bytes]],
    *,
    progress=None,
) -> tuple[list[UploadedDoc], int]:
    """Parse, chunk and index uploaded files. Returns (docs, chunks written)."""
    from ..retrieval import store

    if not files:
        raise UploadError("no files were uploaded")

    collection = upload_collection(settings, workspace)
    docs: list[UploadedDoc] = []
    all_chunks: list[Chunk] = []
    total_chars = 0

    for filename, data in files:
        markdown, pages = parse_file(filename, data)
        total_chars += len(markdown)
        if total_chars > MAX_TOTAL_CHARS:
            raise UploadError(
                "that is more text than one workspace holds "
                f"(limit ~{MAX_TOTAL_CHARS // 1000}k characters). "
                "Upload fewer documents, or clear the workspace first."
            )

        # Content-addressed id: re-uploading the same file updates in place
        # instead of duplicating it in the index.
        doc_id = hashlib.sha1(
            f"{filename}:{hashlib.sha1(data).hexdigest()}".encode()
        ).hexdigest()[:16]

        chunks = build_chunks(settings, filename, markdown, doc_id)
        if not chunks:
            raise UploadError(f"{filename} produced no indexable text")

        all_chunks.extend(chunks)
        docs.append(
            UploadedDoc(
                doc_id=doc_id,
                filename=filename,
                title=_title_from(filename, markdown),
                chars=len(markdown),
                chunks=len(chunks),
                pages=pages,
            )
        )

    target = settings.variant(collection_base=f"upload_{_safe(workspace)}")
    written = store.index_chunks(target, all_chunks, progress=progress)
    log.info(
        "workspace %s: indexed %d chunks from %d file(s) into %s",
        workspace, written, len(docs), collection,
    )
    return docs, written


def clear_workspace(settings: Settings, workspace: str) -> bool:
    """Drop a workspace's collection. Returns whether anything was removed."""
    from ..retrieval import store

    target = settings.variant(collection_base=f"upload_{_safe(workspace)}")
    client = store.get_client(target)
    if client.collection_exists(target.collection):
        client.delete_collection(target.collection)
        log.info("dropped upload collection %s", target.collection)
        return True
    return False


def workspace_size(settings: Settings, workspace: str) -> int:
    from ..retrieval import store

    target = settings.variant(collection_base=f"upload_{_safe(workspace)}")
    try:
        return store.collection_size(target)
    except Exception:
        return 0
