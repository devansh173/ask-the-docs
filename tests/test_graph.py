"""Graph logic tests: routing, formatting, citation handling.

Everything here is a pure function or runs against a stub model, so the suite
needs no API key and no network. The routing tests in particular encode the
promises the README makes - that the pipeline retries a fixed number of times
and then abstains rather than answering from weak context.
"""

from __future__ import annotations

import pytest

from askthedocs.config import naive_config, settings
from askthedocs.graph import nodes
from askthedocs.graph.build import build_graph, config_name
from askthedocs.graph.state import GraphState, RunResult, initial_state
from askthedocs.llm import LLMBundle
from askthedocs.retrieval.schema import Hit


def make_hit(n: int, title: str = "Doc", section: str = "Section") -> Hit:
    return Hit(
        chunk_id=f"c{n}",
        text=f"{title} > {section}\n\nBody of passage {n} with enough text to matter.",
        source_url=f"https://example.test/{n}",
        title=title,
        section=section,
        doc_set="test",
        retrieval_score=1.0 / n,
    )


class StubModel:
    """Stands in for a chat model. Never called by the tests below."""

    def invoke(self, messages):  # pragma: no cover - guard, not behaviour
        raise AssertionError("no test in this module should call an LLM")


@pytest.fixture
def stub_bundle() -> LLMBundle:
    return LLMBundle(StubModel(), StubModel(), "anthropic", "gen", "grade")


# --------------------------------------------------------------------------- #
# Topology
# --------------------------------------------------------------------------- #
def test_agentic_graph_has_the_self_correction_nodes(stub_bundle):
    graph = build_graph(settings, stub_bundle)
    names = set(graph.get_graph().nodes)
    for node in (
        "analyze_query", "retrieve", "grade_documents",
        "rewrite_query", "generate", "check_groundedness", "give_up",
    ):
        assert node in names, f"agentic graph is missing {node}"


def test_rerank_node_is_wired_in_only_when_a_reranker_is_selected(stub_bundle):
    """Reranking ships off (the ablation found it not worth its latency), so the
    graph must be correct with and without the node."""
    without = set(build_graph(
        settings.variant(reranker="none", use_reranker=False), stub_bundle
    ).get_graph().nodes)
    assert "rerank" not in without
    assert "truncate" in without, "something must still cut the list to top-k"

    with_rerank = set(build_graph(
        settings.variant(reranker="bge-base", use_reranker=True), stub_bundle
    ).get_graph().nodes)
    assert "rerank" in with_rerank


def test_naive_graph_has_none_of_them(stub_bundle):
    graph = build_graph(naive_config(settings), stub_bundle)
    names = set(graph.get_graph().nodes)
    assert {"retrieve", "generate"} <= names
    for node in ("grade_documents", "rewrite_query", "check_groundedness", "analyze_query"):
        assert node not in names, f"naive baseline should not contain {node}"


def test_config_names():
    assert config_name(settings) == "agentic"
    assert config_name(naive_config(settings)) == "naive"


# --------------------------------------------------------------------------- #
# Routing - the retry and abstention promises
# --------------------------------------------------------------------------- #
def test_sufficient_context_goes_straight_to_generation():
    state = GraphState(relevant=True, attempts=1)
    assert nodes.route_after_grade(state, settings) == "generate"


def test_weak_context_triggers_a_rewrite_while_retries_remain():
    state = GraphState(relevant=False, attempts=1, intent="lookup")
    assert nodes.route_after_grade(state, settings) == "rewrite_query"


def test_pipeline_abstains_once_retries_are_exhausted():
    cfg = settings.variant(max_retries=2)
    # attempts counts retrievals, so attempt 3 is the one after two rewrites.
    assert nodes.route_after_grade(
        GraphState(relevant=False, attempts=3, intent="lookup"), cfg
    ) == "give_up"


def test_out_of_scope_abstains_immediately_without_burning_retries():
    state = GraphState(relevant=False, attempts=1, intent="out_of_scope")
    assert nodes.route_after_grade(state, settings) == "give_up"


def test_zero_retries_abstains_on_the_first_failure():
    cfg = settings.variant(max_retries=0)
    assert nodes.route_after_grade(
        GraphState(relevant=False, attempts=1, intent="lookup"), cfg
    ) == "give_up"


def test_grounded_answer_ends_the_run():
    assert nodes.route_after_groundedness(GraphState(grounded=True), settings) == "__end__"


def test_ungrounded_answer_is_regenerated_once():
    first = GraphState(grounded=False, regenerated=False)
    assert nodes.route_after_groundedness(first, settings) == "mark_regenerate"


def test_groundedness_loop_cannot_spin_forever():
    """Second failure returns the answer flagged rather than looping again."""
    second = GraphState(grounded=False, regenerated=True)
    assert nodes.route_after_groundedness(second, settings) == "__end__"


# --------------------------------------------------------------------------- #
# Context formatting and citations
# --------------------------------------------------------------------------- #
def test_passages_are_numbered_from_one():
    formatted = nodes.format_passages([make_hit(1), make_hit(2)])
    assert formatted.startswith("[1]")
    assert "[2]" in formatted


def test_formatting_drops_the_duplicated_breadcrumb():
    hit = make_hit(1, title="Hybrid Queries", section="RRF")
    formatted = nodes.format_passages([hit])
    # The breadcrumb appears once, in the header - not again in the body.
    assert formatted.count("Hybrid Queries") == 1


def test_passage_bodies_are_truncated():
    hit = make_hit(1)
    hit.text = "Title > Section\n\n" + ("x" * 5000)
    formatted = nodes.format_passages([hit], max_chars=100)
    assert len(formatted) < 400


def test_citation_indices_are_parsed_and_bounded():
    assert nodes._cited_indices("Supported by [1] and also [3].", 5) == [1, 3]
    # An index past the number of passages is a model error, not a citation.
    assert nodes._cited_indices("See [9].", 3) == []
    assert nodes._cited_indices("No citations here.", 3) == []


def test_citation_payload_shape():
    citation = make_hit(1, title="Graph API", section="Reducers").citation()
    assert set(citation) == {"chunk_id", "title", "section", "source_url", "score"}


# --------------------------------------------------------------------------- #
# Provider-shaped responses
# --------------------------------------------------------------------------- #
class _Response:
    def __init__(self, content):
        self.content = content


@pytest.mark.parametrize(
    "content,expected",
    [
        ("  a plain string  ", "a plain string"),
        # Gemini 3.x returns content blocks, not a string.
        ([{"type": "text", "text": "part one "}, {"type": "text", "text": "two"}],
         "part one two"),
        # Thinking blocks must never be concatenated into the answer.
        ([{"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": "shown"}],
         "shown"),
        (["bare", "strings"], "barestrings"),
        (None, ""),
        ([], ""),
    ],
)
def test_message_text_handles_every_provider_shape(content, expected):
    """Anthropic returns a string; Gemini returns blocks.

    Calling .strip() on the list form raises AttributeError - a
    provider-specific crash in the layer whose entire job is to be
    provider-agnostic. This is the regression test for that.
    """
    assert nodes.message_text(_Response(content)) == expected


def test_hit_score_prefers_rerank_when_present():
    hit = make_hit(2)
    assert hit.score == pytest.approx(0.5)
    hit.rerank_score = 7.5
    assert hit.score == 7.5


# --------------------------------------------------------------------------- #
# Structured-output fallback
# --------------------------------------------------------------------------- #
def test_json_is_recovered_from_a_fenced_block():
    assert nodes._extract_json('```json\n{"grounded": true}\n```') == {"grounded": True}


def test_json_is_recovered_from_surrounding_prose():
    payload = nodes._extract_json('Sure! {"sufficient": false, "score": 0.2} Hope that helps.')
    assert payload == {"sufficient": False, "score": 0.2}


def test_unparseable_output_returns_none_rather_than_raising():
    assert nodes._extract_json("I could not produce JSON.") is None


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def test_initial_state_never_mutates_the_original_question():
    state = initial_state("What is RRF?")
    assert state["question"] == state["query"] == "What is RRF?"
    assert state["attempts"] == 0


def test_run_result_serialises_for_the_api():
    result = RunResult(
        question="q", answer="a", citations=[{"title": "t"}], attempts=2, latency_ms=12.34
    )
    data = result.to_dict()
    assert data["attempts"] == 2
    assert data["latency_ms"] == 12.3
    assert "summary" not in data
