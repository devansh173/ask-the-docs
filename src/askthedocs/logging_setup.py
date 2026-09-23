"""Logging configuration.

The HTTP clients underneath huggingface_hub and the Anthropic/OpenAI SDKs log
every request at INFO, which buries our own messages under a wall of CDN
redirects. Quiet them and keep the application's own logger readable.
"""

from __future__ import annotations

import logging
import os

_NOISY = (
    "httpx",
    "httpcore",
    "httpx2",
    "huggingface_hub",
    "filelock",
    "urllib3",
    "anthropic",
    "openai",
    "langfuse",
    "opentelemetry",
)

_CONFIGURED = False


def setup(level: int | str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = level or os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

    # fastembed's downloader prints tqdm bars to stderr; keep them for ingestion
    # but silence the symlink warning Windows always emits.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    _CONFIGURED = True
