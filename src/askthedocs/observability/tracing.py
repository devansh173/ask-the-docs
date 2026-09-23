"""Langfuse tracing.

Every LLM call and every LangGraph node shows up as a span, with latency, token
counts and cost attached. That is what turns "the answer was wrong" into "the
rerank node put the right passage at position 7, and the grader passed it
anyway" - which is the difference between a demo and something you can debug.

Tracing is strictly optional. If the keys are absent, or Langfuse is
unreachable, ``callbacks()`` returns an empty list and the pipeline runs exactly
as before. An observability layer that can take the app down is worse than no
observability layer.

A note on self-hosting: Langfuse v3+ wants 4+ CPUs and 16 GiB of RAM
(ClickHouse alone asks for 8 GiB and fails to start under 4). This project was
built on a 7.3 GB laptop, so the default is Langfuse Cloud's free tier;
``docker-compose.langfuse.yml`` in the repo root is there for machines that can
actually hold the stack.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)

_WARNED = False


def _configured() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@lru_cache(maxsize=1)
def _handler():
    """Build the LangChain callback handler once, or return None."""
    global _WARNED
    if not _configured():
        if not _WARNED:
            log.info("Langfuse not configured (no keys); tracing disabled")
            _WARNED = True
        return None
    try:
        from langfuse.langchain import CallbackHandler

        # The SDK reads LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST from the
        # environment; HOST defaults to cloud.langfuse.com.
        handler = CallbackHandler()
        log.info(
            "Langfuse tracing enabled (host=%s)",
            os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        return handler
    except Exception as exc:
        log.warning("Langfuse unavailable, continuing untraced: %s", exc)
        return None


def callbacks(enabled: bool = True) -> list[Any]:
    if not enabled:
        return []
    handler = _handler()
    return [handler] if handler else []


def flush() -> None:
    """Drain the buffer. Worth calling before a short-lived process exits."""
    if not _configured():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:
        log.debug("Langfuse flush failed: %s", exc)


def score_run(trace_id: str | None, name: str, value: float, comment: str = "") -> None:
    """Attach an eval score to a trace, so RAGAS numbers land next to the run."""
    if not trace_id or not _configured():
        return
    try:
        from langfuse import get_client

        get_client().create_score(
            trace_id=trace_id, name=name, value=value, comment=comment or None
        )
    except Exception as exc:
        log.debug("Langfuse score failed: %s", exc)
