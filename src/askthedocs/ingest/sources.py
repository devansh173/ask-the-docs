"""Declarative corpus definition.

The corpus is deliberately self-referential: the demo answers questions about
the stack it is built on (Claude, LangGraph, Qdrant). That makes the golden set
easy to write accurately and gives a reviewer questions they can sanity-check
without domain knowledge.

Each source resolves its page list from the site's own sitemap rather than by
crawling links, which keeps the fetch polite and bounded. All three sites permit
this: docs.claude.com disallows only /api/, docs.langchain.com signals
"ai-train=yes, ai-input=yes", and qdrant.tech sets no restrictions.

Page selection matters more than page count. Each source caps how many pages it
takes, so *which* pages survive the cap decides what the index can answer. The
`priority` patterns put reference and concept pages ahead of third-party
integration stubs - a page explaining how RRF fusion works is worth more than
fifty "use Qdrant with <vendor>" pages that mostly repeat a config snippet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DocSource:
    name: str
    sitemap: str
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    # Earlier patterns win the page cap. Anything unmatched sorts last.
    priority: tuple[str, ...] = ()
    max_pages: int = 120
    # Mintlify-backed sites serve a clean markdown rendering at <url>.md.
    # Everything else falls back to HTML extraction.
    markdown_suffix: bool = False
    description: str = ""

    def matches(self, url: str) -> bool:
        if not any(re.search(p, url) for p in self.include):
            return False
        return not any(re.search(p, url) for p in self.exclude)

    def rank(self, url: str) -> tuple[int, int, str]:
        """Sort key: priority tier, then shallower paths, then alphabetical."""
        tier = next(
            (i for i, p in enumerate(self.priority) if re.search(p, url)),
            len(self.priority),
        )
        return (tier, url.count("/"), url)


SOURCES: tuple[DocSource, ...] = (
    DocSource(
        name="claude-api",
        description="Claude API / Anthropic platform documentation",
        sitemap="https://platform.claude.com/sitemap.xml",
        # The sitemap carries 13 locales; /docs/en/ pins it to English.
        include=(r"platform\.claude\.com/docs/en/",),
        exclude=(
            r"/api/",           # disallowed by robots.txt
            r"/release-notes",
            r"/resources/",
            r"/about-claude/model-deprecations",
        ),
        priority=(
            r"/build-with-claude/",
            r"/agents-and-tools/",
            r"/test-and-evaluate/",
            r"/about-claude/models/",
            r"/get-started",
        ),
        max_pages=120,
        markdown_suffix=True,
    ),
    DocSource(
        name="langgraph",
        description="LangGraph and LangChain (Python) documentation",
        sitemap="https://docs.langchain.com/sitemap.xml",
        include=(
            r"docs\.langchain\.com/oss/python/langgraph/",
            r"docs\.langchain\.com/oss/python/langchain/",
        ),
        exclude=(
            r"/reference/",
            r"/changelog",
            # Generative-UI and vendor UI kits: a large, self-similar section
            # that would otherwise crowd out the graph documentation.
            r"/langchain/frontend/",
            r"/get-help",
            r"/install",
        ),
        priority=(
            r"/langgraph/",      # the part this project actually uses
            r"/langchain/agents",
            r"/langchain/middleware/",
            r"/langchain/(messages|models|tools|streaming|structured-output)",
        ),
        max_pages=120,
        markdown_suffix=True,
    ),
    DocSource(
        name="qdrant",
        description="Qdrant vector database documentation",
        sitemap="https://qdrant.tech/sitemap.xml",
        include=(r"qdrant\.tech/documentation/",),
        exclude=(
            # Managed-offering and ops pages: not what the demo answers about.
            r"/documentation/cloud",
            r"/documentation/private-cloud/",
            r"/documentation/hybrid-cloud/",
            r"/documentation/managed-cloud/",
            r"/documentation/release-notes",
            r"/documentation/edge/",
            r"/documentation/capacity-planning",
            r"/documentation/deploy",
            # One-page-per-vendor sections that repeat the same snippet.
            r"/documentation/embeddings/",
            r"/documentation/data-management/",
            r"/documentation/data-synchronization/",
            r"/documentation/platforms/",
            r"/documentation/agentic-tools/",
            r"/documentation/datasets",
            r"/documentation/ecosystem",
        ),
        priority=(
            r"/documentation/concepts/",
            r"/documentation/search/",
            r"/documentation/guides/",
            r"/documentation/tutorials",
            r"/documentation/(quickstart|overview|interfaces)",
            r"/documentation/frameworks/(langchain|langgraph)",
        ),
        max_pages=120,
        markdown_suffix=False,  # static site, HTML only
    ),
)


SOURCES_BY_NAME = {s.name: s for s in SOURCES}
