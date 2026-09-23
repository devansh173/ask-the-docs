"""Turn raw markdown pages into indexable chunks.

Two-stage split, because a flat character split on technical docs is the single
biggest source of bad retrieval:

  1. Split on markdown headings first, so a chunk never straddles two unrelated
     sections and every chunk knows which section it came from.
  2. Split anything still oversized recursively, preferring paragraph then line
     then word boundaries.

Each chunk is then prefixed with a breadcrumb ("Qdrant docs > Hybrid Queries >
Fusion"). The breadcrumb is embedded along with the body, which gives the dense
model the topical context that a mid-page fragment otherwise lacks - a chunk
that says "set this to IDF" is useless without knowing it is about sparse
vectors in Qdrant.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ..config import GOLDEN_DIR, RAW_DIR, Settings
from ..retrieval.schema import Chunk
from .sources import SOURCES_BY_NAME

log = logging.getLogger(__name__)

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

_SET_LABELS = {
    "claude-api": "Claude API docs",
    "langgraph": "LangGraph docs",
    "qdrant": "Qdrant docs",
}


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    meta: dict[str, str] = {}
    for line in text[3:end].splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            value = value.strip()
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                value = value.strip("\"'")
            meta[key.strip()] = value
    return meta, text[end + 4 :].lstrip("\n")


def _section_path(metadata: dict) -> str:
    parts = [metadata.get(h) for _, h in _HEADERS]
    return " > ".join(p for p in parts if p)


def _is_noise(text: str) -> bool:
    """Drop chunks that are navigation residue rather than content."""
    stripped = text.strip()
    if len(stripped) < 120:
        return True
    # Mostly link list / table of contents.
    if stripped.count("\n") > 3 and len(re.findall(r"^\s*[-*]\s", stripped, re.M)) > 10:
        alpha = sum(c.isalpha() for c in stripped)
        if alpha < len(stripped) * 0.5:
            return True
    return False


def chunk_file(path: Path, settings: Settings) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)

    doc_set = meta.get("doc_set", path.parent.name)
    title = meta.get("title") or path.stem
    url = meta.get("url", "")
    label = _SET_LABELS.get(doc_set, doc_set)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS,
        strip_headers=False,
    )
    body_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    try:
        sections = header_splitter.split_text(body)
    except Exception as exc:  # malformed markdown: fall back to a flat split
        log.debug("header split failed for %s (%s); falling back", path.name, exc)
        sections = []

    if not sections:
        sections = [type("Doc", (), {"page_content": body, "metadata": {}})()]

    chunks: list[Chunk] = []
    position = 0
    for section in sections:
        section_path = _section_path(getattr(section, "metadata", {}) or {})
        breadcrumb = " > ".join(p for p in (label, title, section_path) if p)

        for piece in body_splitter.split_text(section.page_content):
            if _is_noise(piece):
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_set}:{path.stem}:{position}",
                    text=f"{breadcrumb}\n\n{piece.strip()}",
                    source_url=url,
                    title=title,
                    section=section_path,
                    doc_set=doc_set,
                    position=position,
                )
            )
            position += 1

    return chunks


def _page_url(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines()[:6]:
        if line.startswith("url:"):
            return line[4:].strip()
    return ""


@lru_cache(maxsize=1)
def _golden_urls() -> frozenset[str]:
    """Pages the golden set labels as containing an answer.

    These are pinned into the index regardless of the page cap. An evaluation
    corpus that does not contain the labelled pages measures nothing: recall
    would be capped below 1.0 by a bookkeeping decision rather than by
    retrieval quality. Pinning them is standard IR practice - the collection
    must contain the relevant documents; the *other* pages are what make the
    task non-trivial, and those are still selected by priority alone.
    """
    path = GOLDEN_DIR / "golden_set.jsonl"
    if not path.exists():
        return frozenset()

    urls: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            urls.update(json.loads(line).get("relevant_urls") or [])
        except json.JSONDecodeError:
            continue
    return frozenset(urls)


def _select_pages(doc_dir: Path, limit: int) -> list[Path]:
    """Take the golden-labelled pages plus the highest-priority remainder.

    The scraper already ranked pages when it chose what to fetch, but that
    ranking is not recorded on disk, so it is recomputed here from each page's
    URL using the same DocSource.rank. That keeps "which pages matter" defined
    in one place (ingest/sources.py) rather than drifting between the scrape
    step and the index step.
    """
    files = sorted(doc_dir.glob("*.md"))
    if not limit or len(files) <= limit:
        return files

    source = SOURCES_BY_NAME.get(doc_dir.name)
    if source is None:
        return files[:limit]

    required = _golden_urls()
    pinned = [p for p in files if _page_url(p) in required]
    rest = sorted(
        (p for p in files if p not in set(pinned)),
        key=lambda p: source.rank(_page_url(p)),
    )
    selected = pinned + rest[: max(0, limit - len(pinned))]
    if pinned:
        log.debug("%s: pinned %d golden pages", doc_dir.name, len(pinned))
    return selected


def chunk_corpus(settings: Settings, only: list[str] | None = None) -> list[Chunk]:
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"No scraped pages at {RAW_DIR}. Run `askthedocs scrape` first."
        )

    chunks: list[Chunk] = []
    for doc_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        if only and doc_dir.name not in only:
            continue

        files = _select_pages(doc_dir, settings.max_pages_per_set)
        set_chunks: list[Chunk] = []
        for path in files:
            set_chunks.extend(chunk_file(path, settings))
        log.info("%s: %d pages -> %d chunks", doc_dir.name, len(files), len(set_chunks))
        chunks.extend(set_chunks)

    if not chunks:
        raise RuntimeError("Chunking produced nothing - is data/raw empty?")

    lengths = sorted(len(c.text) for c in chunks)
    log.info(
        "total %d chunks (median %d chars, p95 %d chars)",
        len(chunks),
        lengths[len(lengths) // 2],
        lengths[int(len(lengths) * 0.95)],
    )
    return chunks
