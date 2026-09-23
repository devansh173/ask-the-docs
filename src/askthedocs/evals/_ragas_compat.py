"""Compatibility shim: import this before importing ragas.

ragas 0.4.3 does this at module import time::

    from langchain_community.chat_models.vertexai import ChatVertexAI
    from langchain_community.llms import VertexAI

langchain-community is being sunset, and 0.4.x no longer ships those modules,
so importing ragas raises ModuleNotFoundError on a current LangChain 1.x
install. Pinning langchain-community back to 0.3.x is not an option - that line
requires langchain-core <1.0, which conflicts with langchain 1.4 / langgraph 1.2.

The two symbols are used in exactly one place: a list of model classes that
support n-completions, consulted through ``isinstance``::

    MULTIPLE_COMPLETION_SUPPORTED = [OpenAI, ChatOpenAI, ..., ChatVertexAI, VertexAI]

Substituting empty placeholder classes makes those isinstance checks return
False, which is the correct answer here - this project never routes RAGAS
through Vertex AI. Nothing else in ragas touches them.

Scope note: the stub is only registered if the real module is genuinely absent,
so installing langchain-google-vertexai later takes precedence automatically.
"""

from __future__ import annotations

import logging
import sys
import types

log = logging.getLogger(__name__)

_INSTALLED = False


def _make_stub(module_name: str, class_names: list[str]) -> types.ModuleType:
    module = types.ModuleType(module_name)
    for name in class_names:
        # A distinct class per name so isinstance() is well-defined and False.
        setattr(module, name, type(name, (), {"__doc__": "ragas compat stub"}))
    module.__doc__ = "Placeholder installed by askthedocs.evals._ragas_compat"
    return module


def install() -> None:
    """Register stubs for the langchain-community modules ragas expects."""
    global _INSTALLED
    if _INSTALLED:
        return

    try:
        import langchain_community.chat_models.vertexai  # noqa: F401

        _INSTALLED = True
        return  # the real thing is present; leave it alone
    except Exception:
        pass

    chat_module = "langchain_community.chat_models.vertexai"
    if chat_module not in sys.modules:
        sys.modules[chat_module] = _make_stub(chat_module, ["ChatVertexAI"])

    # ragas also imports VertexAI from langchain_community.llms, which does
    # exist as a package but no longer exports that name.
    try:
        from langchain_community import llms as community_llms

        if not hasattr(community_llms, "VertexAI"):
            community_llms.VertexAI = type(
                "VertexAI", (), {"__doc__": "ragas compat stub"}
            )
    except Exception as exc:
        log.debug("could not patch langchain_community.llms: %s", exc)

    log.debug("installed langchain-community compatibility stubs for ragas")
    _INSTALLED = True
