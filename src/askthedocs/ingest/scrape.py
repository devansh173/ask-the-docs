"""Fetch documentation pages and normalise them to markdown on disk.

Two extraction paths:

  * Mintlify-backed sites (platform.claude.com, docs.langchain.com) serve a
    clean markdown rendering at ``<url>.md``. Use it - it is already free of
    navigation chrome, which is the main source of junk chunks in doc RAG.
  * Everything else is fetched as HTML, reduced to its main content element,
    and converted with markdownify.

Pages land in ``data/raw/<doc_set>/<slug>.md`` with YAML frontmatter carrying
the canonical URL and title, so chunking never has to re-fetch anything and the
whole pipeline is reproducible offline.
"""

from __future__ import annotations

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

from ..config import RAW_DIR
from .sources import SOURCES, DocSource

log = logging.getLogger(__name__)

USER_AGENT = "askthedocs-indexer/2.0 (portfolio RAG project; +https://github.com/)"
REQUEST_DELAY = 0.25  # seconds between requests to one host

# Content containers, most specific first.
_MAIN_SELECTORS = ("main article", "article", "main", "div.content", "body")

_DROP_TAGS = ("script", "style", "nav", "header", "footer", "aside", "noscript", "svg")


@dataclass
class Page:
    doc_set: str
    url: str
    title: str
    markdown: str

    def slug(self) -> str:
        path = re.sub(r"^https?://", "", self.url).rstrip("/")
        return re.sub(r"[^a-zA-Z0-9]+", "-", path).strip("-")[:120]


# --------------------------------------------------------------------------- #
# Sitemap
# --------------------------------------------------------------------------- #
def fetch_sitemap_urls(client: httpx.Client, source: DocSource) -> list[str]:
    response = client.get(source.sitemap)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    urls: list[str] = []
    for loc in root.iter():
        if loc.tag.endswith("}loc") or loc.tag == "loc":
            if loc.text:
                urls.append(loc.text.strip())

    matched = [u for u in urls if source.matches(u)]
    # Priority tier first, then shallowest, so concept pages win the cap.
    matched = sorted(set(matched), key=source.rank)
    log.info(
        "%s: %d/%d sitemap urls matched, keeping %d",
        source.name,
        len(matched),
        len(urls),
        min(len(matched), source.max_pages),
    )
    return matched[: source.max_pages]


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _strip_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block, body = text[3:end], text[end + 4 :]

    meta: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, body.lstrip("\n")


def _clean_markdown(text: str) -> str:
    """Strip MDX scaffolding while keeping the prose and code inside it.

    Mintlify pages are MDX: the content is wrapped in components like
    <CodeGroup>, <Note>, <Steps>, <ParamField>. MDX components are capitalised
    by convention and raw HTML is not, so matching on a leading capital removes
    the scaffolding generically instead of chasing a list of tag names that
    changes whenever the docs team adds a component.
    """
    # Opening/closing/self-closing MDX component tags (capitalised names only).
    text = re.sub(r"</?[A-Z][A-Za-z0-9]*(\s[^>]*?)?/?>", "", text)

    # Fence info strings carry theme JSON: ```python Google theme={...} -> ```python
    def _fence(match: re.Match) -> str:
        info = match.group(1).strip()
        lang = info.split()[0] if info else ""
        lang = "" if lang.startswith(("{", "theme")) else lang
        return f"```{lang}"

    text = re.sub(r"^```([^\n]*)$", _fence, text, flags=re.MULTILINE)

    # Mintlify emits {/* comments */} and bare {" "} spacers.
    text = re.sub(r"\{/\*.*?\*/\}", "", text, flags=re.DOTALL)
    text = re.sub(r'\{"\s*"\}', " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse blank runs
    text = re.sub(r"[ \t]+\n", "\n", text)        # trailing whitespace
    return text.strip()


def extract_markdown_page(client: httpx.Client, url: str, doc_set: str) -> Page | None:
    response = client.get(url + ".md")
    if response.status_code != 200:
        log.debug("md endpoint missing for %s (%s)", url, response.status_code)
        return None

    meta, body = _strip_frontmatter(response.text)

    # LangChain prepends a documentation-index banner to every .md page.
    body = re.sub(r"^(> .*\n)+\n?", "", body)

    title = meta.get("title") or ""
    if not title:
        heading = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        title = heading.group(1).strip() if heading else url.rstrip("/").split("/")[-1]

    return Page(doc_set=doc_set, url=meta.get("url", url), title=title,
                markdown=_clean_markdown(body))


def extract_html_page(client: httpx.Client, url: str, doc_set: str) -> Page | None:
    response = client.get(url)
    if response.status_code != 200:
        log.debug("skipping %s (%s)", url, response.status_code)
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else url

    for tag in soup(list(_DROP_TAGS)):
        tag.decompose()

    main = next((soup.select_one(sel) for sel in _MAIN_SELECTORS
                 if soup.select_one(sel)), None)
    if main is None:
        return None

    markdown = markdownify(str(main), heading_style="ATX", strip=["a"])
    return Page(doc_set=doc_set, url=url, title=title,
                markdown=_clean_markdown(markdown))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def write_page(page: Page) -> Path:
    out_dir = RAW_DIR / page.doc_set
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{page.slug()}.md"
    frontmatter = (
        "---\n"
        f"url: {page.url}\n"
        f"title: {json.dumps(page.title)}\n"
        f"doc_set: {page.doc_set}\n"
        "---\n\n"
    )
    path.write_text(frontmatter + page.markdown, encoding="utf-8")
    return path


def scrape_source(source: DocSource, *, min_chars: int = 400) -> list[Page]:
    headers = {"User-Agent": USER_AGENT}
    pages: list[Page] = []

    with httpx.Client(
        headers=headers, timeout=30.0, follow_redirects=True
    ) as client:
        urls = fetch_sitemap_urls(client, source)

        for i, url in enumerate(urls, 1):
            try:
                page = None
                if source.markdown_suffix:
                    page = extract_markdown_page(client, url, source.name)
                if page is None:
                    page = extract_html_page(client, url, source.name)
            except httpx.HTTPError as exc:
                log.warning("fetch failed for %s: %s", url, exc)
                page = None

            if page is None:
                continue
            if len(page.markdown) < min_chars:
                log.debug("skipping thin page %s (%d chars)", url, len(page.markdown))
                continue

            write_page(page)
            pages.append(page)
            if i % 20 == 0:
                log.info("  %s: %d/%d fetched", source.name, len(pages), len(urls))
            time.sleep(REQUEST_DELAY)

    log.info("%s: wrote %d pages", source.name, len(pages))
    return pages


def scrape_all(only: list[str] | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in SOURCES:
        if only and source.name not in only:
            continue
        counts[source.name] = len(scrape_source(source))
    return counts
