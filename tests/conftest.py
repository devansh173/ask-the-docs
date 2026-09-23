"""Test session setup.

Embedded Qdrant takes an exclusive file lock on its storage directory, so a test
run would otherwise fail outright whenever the dev server or an indexing job is
using the real one - "Storage folder ... is already accessed by another instance".
Pointing the tests at their own temporary directory makes the suite independent
of whatever else is running, and guarantees a test can never write into a real
index.

The environment variable has to be set before ``askthedocs.config`` is imported,
because that module reads it at import time. conftest.py is imported before any
test module, which is what makes this work.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_TEST_STORE = Path(tempfile.gettempdir()) / "askthedocs-pytest-qdrant"

# Start from a clean store so a crashed previous run cannot leave a stale lock
# or half-written collection behind.
shutil.rmtree(_TEST_STORE, ignore_errors=True)
_TEST_STORE.mkdir(parents=True, exist_ok=True)

# Remember where the real index lives before redirecting writes, so the
# retrieval gate can still read it (see REAL_STORE below).
_REAL_STORE = os.environ.get("QDRANT_LOCAL_PATH")

os.environ["QDRANT_LOCAL_PATH"] = str(_TEST_STORE)
# Never let a stray QDRANT_URL in the developer's shell point the suite at a
# real server.
os.environ.pop("QDRANT_URL", None)
os.environ.pop("QDRANT_API_KEY", None)
# Tracing in tests would emit spans to a real Langfuse project.
os.environ["LANGFUSE_ENABLED"] = "false"

import pytest  # noqa: E402

from askthedocs.config import _default_local_qdrant  # noqa: E402


def real_store_path():
    """Where the developer's or CI's actual index lives.

    conftest points every test at a scratch directory so nothing can write into
    a real index. But the retrieval regression gate has to *read* the real one -
    it scores the shipped corpus - and if it silently read the empty scratch
    store it would skip in CI and the "quality gate" would be decorative.
    """
    if _REAL_STORE:
        return Path(_REAL_STORE)
    saved = os.environ.pop("QDRANT_LOCAL_PATH")
    try:
        return _default_local_qdrant()
    finally:
        os.environ["QDRANT_LOCAL_PATH"] = saved


@pytest.fixture(scope="session", autouse=True)
def _cleanup_store():
    yield
    from askthedocs.retrieval import store

    store._close_clients()
    shutil.rmtree(_TEST_STORE, ignore_errors=True)


def pytest_report_header(config) -> str:
    return f"askthedocs: test qdrant store at {_TEST_STORE}"
