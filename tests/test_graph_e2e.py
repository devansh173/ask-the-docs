"""End-to-end graph execution against stub models.

Compiling a graph proves the edges are wired; it does not prove the nodes can
actually be *called*. The nodes are bound with functools.partial, and LangGraph
inspects a node's signature to decide what to pass it, so a binding mistake only
shows up at invoke time. These tests run whole graphs with scripted models and
a small real index, which exercises retrieval, reranking, routing and the retry
loop without touching a paid API.
"""

from __future__ import annotations

import json

import pytest

from askthedocs.config import naive_config, settings
from askthedocs.graph.build import answer_question
from askthedocs.llm import LLMBundle
from askthedocs.retrieval import store
from askthedocs.retrieval.schema import Chunk

COLLECTION = "pytest_e2e"

CORPUS = [
    ("RRF fusion", "Reciprocal Rank Fusion merges ranked lists by summing 1/(k + rank) "
                   "for each document. Qdrant uses k=2 by default and zero-based ranks."),
    ("IDF modifier", "Set modifier=Modifier.IDF on sparse vector params so Qdrant "
                     "applies inverse document frequency from corpus statistics."),
    ("StateGraph", "A StateGraph takes a typed state schema. Nodes read and write that "
                   "state and conditional edges route between them."),
    ("Cross encoder", "A cross-encoder scores a query and document jointly in one "
                      "forward pass, more accurately but far more slowly than a bi-encoder."),
    ("Prompt caching", "Prompt caching matches on an exact prefix; any byte change in "
                       "the prefix invalidates everything after it."),
]


class ScriptedModel:
    """Returns canned responses, and records what it was asked.

    `structured` maps a schema name to the object to return, which is what the
    grading and groundedness nodes consume via with_structured_output.
    """

    def __init__(self, text: str = "Answer text [1].", structured: dict | None = None):
        self.text = text
        self.structured = structured or {}
        self.calls: list[str] = []

    class _Message:
        def __init__(self, content: str):
            self.content = content

    def invoke(self, messages):
        self.calls.append("invoke")
        return self._Message(self.text)

    async def ainvoke(self, messages):
        return self.invoke(messages)

    def with_structured_output(self, schema):
        payload = self.structured.get(schema.__name__)
        if payload is None:
            raise NotImplementedError(f"no scripted response for {schema.__name__}")

        outer = self

        class _Bound:
            def invoke(self, messages):
                outer.calls.append(schema.__name__)
                return schema.model_validate(payload)

            async def ainvoke(self, messages):
                return self.invoke(messages)

        return _Bound()


@pytest.fixture(scope="module")
def indexed():
    cfg = settings.variant(collection_base=COLLECTION)
    chunks = [
        Chunk(
            chunk_id=f"e{i}",
            text=f"Test docs > {name}\n\n{body}",
            source_url=f"https://example.test/{i}",
            title=name,
            section=name,
            doc_set="test",
            position=i,
        )
        for i, (name, body) in enumerate(CORPUS)
    ]
    store.index_chunks(cfg, chunks, recreate=True)
    return cfg


def bundle(generator: ScriptedModel, grader: ScriptedModel) -> LLMBundle:
    return LLMBundle(generator, grader, "stub", "stub-gen", "stub-grade")


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #
def test_naive_graph_runs_end_to_end(indexed):
    generator = ScriptedModel("RRF sums reciprocal ranks [1].")
    result = answer_question(
        "how does RRF fusion work",
        settings=naive_config(indexed),
        llms=bundle(generator, ScriptedModel()),
    )
    assert result.status == "answered"
    assert "[1]" in result.answer
    assert result.citations
    assert result.retrieval_mode == "dense"
    assert {e["node"] for e in result.events} == {"retrieve", "truncate", "generate"}


def test_agentic_graph_runs_end_to_end(indexed):
    grader = ScriptedModel(
        structured={
            "QueryAnalysis": {"intent": "lookup", "search_query": "RRF fusion rank formula"},
            "RelevanceGrade": {"sufficient": True, "score": 0.9, "reason": "covered"},
            "GroundednessCheck": {"grounded": True, "unsupported": [], "reason": "ok"},
        }
    )
    generator = ScriptedModel("RRF sums 1/(k + rank) with k=2 [1].")

    result = answer_question(
        "how does RRF fusion work",
        settings=indexed,
        llms=bundle(generator, grader),
    )

    assert result.status == "answered"
    assert result.grounded is True
    assert result.retrieval_mode == "hybrid"
    assert result.attempts == 1

    fired = [e["node"] for e in result.events]
    for node in ("analyze_query", "retrieve", "grade_documents",
                 "generate", "check_groundedness"):
        assert node in fired, f"{node} did not run"
    # Reranking is off by default, so a top-k cut stands in for it.
    assert "truncate" in fired or "rerank" in fired


def test_agentic_graph_runs_with_reranking_enabled(indexed):
    """The reranked path must still work, since it is one flag away."""
    grader = ScriptedModel(
        structured={
            "QueryAnalysis": {"intent": "lookup", "search_query": "RRF fusion"},
            "RelevanceGrade": {"sufficient": True, "score": 0.9, "reason": "covered"},
            "GroundednessCheck": {"grounded": True, "unsupported": [], "reason": "ok"},
        }
    )
    result = answer_question(
        "how does RRF fusion work",
        settings=indexed.variant(reranker="bge-base", use_reranker=True),
        llms=bundle(ScriptedModel("RRF sums reciprocal ranks [1]."), grader),
    )
    assert result.status == "answered"
    fired = [e["node"] for e in result.events]
    assert "rerank" in fired
    rerank_event = next(e for e in result.events if e["node"] == "rerank")
    assert rerank_event.get("scored", 0) > 0


def test_citation_numbers_line_up_with_the_source_list(indexed):
    """An answer citing [1] and [4] must produce a list where [4] exists.

    Returning only the cited passages renumbers the list, so a sparse citation
    like [1] and [4] would render against a two-item list and [4] would point
    at nothing. Every retrieved passage is returned instead, flagged with
    whether the answer used it.
    """
    grader = ScriptedModel(
        structured={
            "QueryAnalysis": {"intent": "lookup", "search_query": "RRF fusion"},
            "RelevanceGrade": {"sufficient": True, "score": 0.9, "reason": "covered"},
            "GroundednessCheck": {"grounded": True, "unsupported": [], "reason": "ok"},
        }
    )
    result = answer_question(
        "how does RRF fusion work",
        settings=indexed.variant(top_k=4),
        llms=bundle(ScriptedModel("First [1]. Also [4]."), grader),
    )

    assert len(result.citations) == 4, "all shown passages must be returned"
    assert [c["cited"] for c in result.citations] == [True, False, False, True]
    # The highest marker in the answer must be addressable in the list.
    assert result.citations[3]["chunk_id"]


def test_grouped_citations_are_all_resolved(indexed):
    """Gemini writes "[2, 3]"; both numbers must count as citations."""
    grader = ScriptedModel(
        structured={
            "QueryAnalysis": {"intent": "lookup", "search_query": "RRF fusion"},
            "RelevanceGrade": {"sufficient": True, "score": 0.9, "reason": "covered"},
            "GroundednessCheck": {"grounded": True, "unsupported": [], "reason": "ok"},
        }
    )
    result = answer_question(
        "how does RRF fusion work",
        settings=indexed.variant(top_k=3),
        llms=bundle(ScriptedModel("Supported by [2, 3]."), grader),
    )
    assert [c["cited"] for c in result.citations] == [False, True, True]


def test_citations_resolve_to_real_sources(indexed):
    grader = ScriptedModel(
        structured={
            "QueryAnalysis": {"intent": "lookup", "search_query": "modifier IDF sparse"},
            "RelevanceGrade": {"sufficient": True, "score": 0.9, "reason": "covered"},
            "GroundednessCheck": {"grounded": True, "unsupported": [], "reason": "ok"},
        }
    )
    result = answer_question(
        "what does modifier IDF do",
        settings=indexed,
        llms=bundle(ScriptedModel("Set it to idf [1]."), grader),
    )
    assert result.citations
    assert result.citations[0]["source_url"].startswith("https://example.test/")


# --------------------------------------------------------------------------- #
# Self-correction
# --------------------------------------------------------------------------- #
def test_insufficient_context_retries_then_abstains(indexed):
    """The core promise: it declines instead of answering from weak context."""
    grader = ScriptedModel(
        text="a different search query",
        structured={
            "QueryAnalysis": {"intent": "lookup", "search_query": "something obscure"},
            # Always insufficient, so the loop must terminate on its own.
            "RelevanceGrade": {"sufficient": False, "score": 0.1, "reason": "off topic"},
            "GroundednessCheck": {"grounded": True, "unsupported": [], "reason": "ok"},
        },
    )
    generator = ScriptedModel("This should never be produced.")

    cfg = indexed.variant(max_retries=2)
    result = answer_question("what is the airspeed velocity of a swallow",
                             settings=cfg, llms=bundle(generator, grader))

    assert result.status == "insufficient_context"
    assert generator.calls == [], "the generator must not run when context is insufficient"
    # One initial retrieval plus max_retries rewrites.
    assert result.attempts == 3
    assert len(result.rewrites) == 2
    assert "give_up" in [e["node"] for e in result.events]


def test_out_of_scope_abstains_without_burning_retries(indexed):
    grader = ScriptedModel(
        structured={
            "QueryAnalysis": {"intent": "out_of_scope", "search_query": "kubernetes hpa"},
            "RelevanceGrade": {"sufficient": False, "score": 0.0, "reason": "not in corpus"},
        }
    )
    result = answer_question("how do I autoscale kubernetes pods",
                             settings=indexed, llms=bundle(ScriptedModel(), grader))

    assert result.status == "insufficient_context"
    assert result.attempts == 1, "out-of-scope should not retry"
    assert result.rewrites == []


def test_ungrounded_answer_is_regenerated_exactly_once(indexed):
    grader = ScriptedModel(
        structured={
            "QueryAnalysis": {"intent": "lookup", "search_query": "RRF fusion"},
            "RelevanceGrade": {"sufficient": True, "score": 0.9, "reason": "covered"},
            # Never satisfied, so the repair loop must stop by itself.
            "GroundednessCheck": {
                "grounded": False,
                "unsupported": ["k defaults to 60"],
                "reason": "not stated",
            },
        }
    )
    generator = ScriptedModel("RRF uses k=60 [1].")

    result = answer_question("how does RRF fusion work",
                             settings=indexed, llms=bundle(generator, grader))

    assert result.grounded is False, "the answer should come back flagged, not suppressed"
    assert result.answer, "a flagged answer is still returned"
    assert generator.calls.count("invoke") == 2, "exactly one regeneration attempt"


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #
def test_a_broken_grader_fails_open_rather_than_blocking(indexed):
    """If the grader cannot be parsed, answer anyway - the user has citations."""
    grader = ScriptedModel(text="not json at all", structured={})
    generator = ScriptedModel("An answer [1].")

    result = answer_question("how does RRF fusion work",
                             settings=indexed, llms=bundle(generator, grader))

    assert result.status == "answered"
    assert generator.calls, "generation should still have run"


def test_generation_failure_surfaces_as_an_error_result(indexed):
    class Exploding(ScriptedModel):
        def invoke(self, messages):
            raise RuntimeError("provider is down")

    result = answer_question(
        "how does RRF fusion work",
        settings=naive_config(indexed),
        llms=bundle(Exploding(), ScriptedModel()),
    )
    assert result.status == "error"
    assert "provider is down" in result.answer
