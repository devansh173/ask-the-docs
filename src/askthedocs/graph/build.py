"""Assemble the LangGraph.

Two topologies are built from the same nodes, selected by settings. This is
deliberate: the "before" number in the README's eval table has to come from a
real baseline running through the same code, not from a separate script that
might differ in a dozen small ways.

    naive (baseline)
        START -> retrieve -> truncate -> generate -> END

    agentic (full pipeline)
                     +--------------------------------+
                     v                                |
        START -> analyze_query -> retrieve -> rerank -> grade_documents
                                                          |
                          +-------------------------------+
                          |            |                  |
                     (sufficient)  (retry left)      (exhausted /
                          |            |              out of scope)
                          v            v                  v
                      generate <- rewrite_query        give_up -> END
                          |
                          v
                 check_groundedness --(grounded)--> END
                          |
                     (ungrounded, first time)
                          v
                    mark_regenerate -> generate

The groundedness loop runs at most once: mark_regenerate sets a flag that the
router reads, so a model that keeps hallucinating returns its answer with
grounded=False rather than looping forever.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..config import Settings, settings as default_settings
from ..llm import LLMBundle, build_bundle
from . import nodes
from .state import GraphState, RunResult, initial_state

log = logging.getLogger(__name__)

_GRAPH_CACHE: dict[tuple, Any] = {}


def _bind(fn, deps: nodes.Deps):
    """Wrap a (state, deps) node into the (state) -> dict LangGraph expects."""
    return partial(fn, deps=deps)


def build_graph(settings: Settings, llms: LLMBundle):
    deps = nodes.Deps(settings=settings, llms=llms)
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", _bind(nodes.retrieve, deps))
    graph.add_node("generate", _bind(nodes.generate, deps))

    if not settings.use_self_correction:
        # ---- baseline: retrieve, cut to top-k, answer --------------------
        graph.add_node("truncate", _bind(nodes.truncate, deps))
        graph.add_edge(START, "retrieve")
        if settings.use_reranker:
            graph.add_node("rerank", _bind(nodes.rerank, deps))
            graph.add_edge("retrieve", "rerank")
            graph.add_edge("rerank", "generate")
        else:
            graph.add_edge("retrieve", "truncate")
            graph.add_edge("truncate", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    # ---- full agentic pipeline -------------------------------------------
    graph.add_node("analyze_query", _bind(nodes.analyze_query, deps))
    graph.add_node("grade_documents", _bind(nodes.grade_documents, deps))
    graph.add_node("rewrite_query", _bind(nodes.rewrite_query, deps))
    graph.add_node("check_groundedness", _bind(nodes.check_groundedness, deps))
    graph.add_node("mark_regenerate", _bind(nodes.mark_regenerate, deps))
    graph.add_node("give_up", _bind(nodes.give_up, deps))

    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "retrieve")

    if settings.use_reranker:
        graph.add_node("rerank", _bind(nodes.rerank, deps))
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "grade_documents")
    else:
        graph.add_node("truncate", _bind(nodes.truncate, deps))
        graph.add_edge("retrieve", "truncate")
        graph.add_edge("truncate", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        partial(nodes.route_after_grade, settings=settings),
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "give_up": "give_up",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "check_groundedness")
    graph.add_conditional_edges(
        "check_groundedness",
        partial(nodes.route_after_groundedness, settings=settings),
        {"__end__": END, "mark_regenerate": "mark_regenerate"},
    )
    graph.add_edge("mark_regenerate", "generate")
    graph.add_edge("give_up", END)

    return graph.compile()


def get_graph(settings: Settings, llms: LLMBundle):
    """Reuse a compiled graph for an identical configuration.

    The key includes ``id(llms.generator)``, which looks redundant next to the
    provider and model names but is not: the nodes close over the concrete
    model objects, and those objects carry the API key. Keying only on provider
    and model name would let a request that supplied its own key reuse a graph
    built around a *different* caller's model instance - i.e. answer one user's
    question with another user's credentials. Identity in the key makes that
    impossible; the cost is that per-request keys simply miss the cache, and
    compiling a graph is cheap.
    """
    key = (
        settings.use_self_correction,
        settings.use_reranker,
        settings.use_hybrid,
        settings.max_retries,
        settings.top_k,
        settings.profile,
        llms.provider,
        llms.generator_model,
        llms.grader_model,
        id(llms.generator),
    )
    if key not in _GRAPH_CACHE:
        _GRAPH_CACHE[key] = build_graph(settings, llms)
        if len(_GRAPH_CACHE) > 16:  # bounded: keys include per-request models
            _GRAPH_CACHE.pop(next(iter(_GRAPH_CACHE)))
    return _GRAPH_CACHE[key]


def config_name(settings: Settings) -> str:
    if settings.use_self_correction:
        return "agentic"
    return "naive+rerank" if settings.use_reranker else "naive"


def answer_question(
    question: str,
    *,
    settings: Settings | None = None,
    llms: LLMBundle | None = None,
    doc_sets: list[str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    callbacks: list | None = None,
) -> RunResult:
    """Run one question through the graph and normalise the output."""
    settings = settings or default_settings
    llms = llms or build_bundle(
        settings, provider=provider, model=model, api_key=api_key
    )
    graph = get_graph(settings, llms)

    started = time.perf_counter()
    config: dict[str, Any] = {"recursion_limit": 40}
    if callbacks:
        config["callbacks"] = callbacks
        config["metadata"] = {
            "config": config_name(settings),
            "profile": settings.profile,
            "provider": llms.provider,
        }

    try:
        final = graph.invoke(initial_state(question, doc_sets), config=config)
    except Exception as exc:
        log.exception("graph run failed")
        return RunResult(
            question=question,
            answer=f"The pipeline failed before it could answer: {exc}",
            status="error",
            latency_ms=(time.perf_counter() - started) * 1000,
            config_name=config_name(settings),
        )

    hits = final.get("hits") or []
    return RunResult(
        question=question,
        answer=final.get("answer", ""),
        citations=final.get("citations", []),
        status=final.get("status", "answered"),
        grounded=final.get("grounded", True),
        relevant=final.get("relevant", True),
        attempts=final.get("attempts", 1),
        rewrites=final.get("rewrites", []),
        retrieval_mode=final.get("retrieval_mode", "hybrid"),
        events=final.get("events", []),
        contexts=[h.text for h in hits],
        latency_ms=(time.perf_counter() - started) * 1000,
        config_name=config_name(settings),
    )
