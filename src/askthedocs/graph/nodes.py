"""The graph's nodes.

Each node is a plain function of (state, runtime deps) -> partial state. They
are wrapped into LangGraph nodes in build.py, which also decides which of them
are wired in: the naive baseline uses retrieve + generate only, while the full
pipeline adds analysis, reranking, grading, rewriting and a groundedness check.
Keeping the nodes independent of the wiring is what lets the eval harness score
both topologies from one codebase.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..config import Settings
from ..llm import LLMBundle, call_with_fallback
from ..retrieval import store
from ..retrieval.encoders import get_reranker
from ..retrieval.schema import Hit
from . import prompts
from .state import GraphState

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Structured output
# --------------------------------------------------------------------------- #
class QueryAnalysis(BaseModel):
    intent: str = Field(description="lookup | howto | comparison | out_of_scope")
    search_query: str = Field(description="the question rewritten for retrieval")


class RelevanceGrade(BaseModel):
    sufficient: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class GroundednessCheck(BaseModel):
    grounded: bool
    unsupported: list[str] = Field(default_factory=list)
    reason: str


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response.

    Used only when with_structured_output is unavailable or fails, which in
    practice means a local Ollama model that has not been tuned for tool use.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start : end + 1] if start != -1 and end > start else None
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def call_structured(
    chain: list[tuple[str, Any]], schema: type[BaseModel], system: str, user: str
) -> tuple[BaseModel | None, str | None]:
    """Ask for a structured answer, degrading to JSON parsing if need be.

    Returns ``(result, model_name)``. ``result`` is None if every model in the
    chain produced nothing usable - callers must treat that as "check failed"
    and choose a safe default rather than crashing the request. ``model_name``
    is the model that actually answered, which can differ from ``chain[0]`` if
    earlier models in the chain refused on quota.
    """
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    try:
        result, used = call_with_fallback(
            chain, lambda m: m.with_structured_output(schema).invoke(messages)
        )
        return result, used
    except Exception as exc:
        log.debug("structured output failed (%s); retrying with json parsing", exc)

    schema_hint = json.dumps(schema.model_json_schema().get("properties", {}), indent=2)
    messages = [
        SystemMessage(content=system + f"\n\nRespond with JSON matching:\n{schema_hint}"),
        HumanMessage(content=user),
    ]
    try:
        raw, used = call_with_fallback(chain, lambda m: m.invoke(messages))
        payload = _extract_json(message_text(raw))
        return (schema.model_validate(payload) if payload else None), used
    except Exception as exc:
        log.warning("structured fallback failed for %s: %s", schema.__name__, exc)
        return None, None


def message_text(response: Any) -> str:
    """Flatten a chat response into plain text, whatever shape it arrives in.

    ``BaseChatModel`` responses do not have one content shape. Anthropic returns
    a plain string; Gemini 3.x returns a list of content blocks; some providers
    mix text blocks with thinking or tool-use blocks that have no "text" key at
    all. Calling .strip() on the list form raises AttributeError, which is a
    provider-specific crash in a layer whose whole purpose is to be
    provider-agnostic - so every call site goes through here.
    """
    content = getattr(response, "content", response)
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Only text blocks carry an answer; thinking and tool_use blocks
                # must not be concatenated into it.
                if block.get("type") in (None, "text") and "text" in block:
                    parts.append(str(block["text"]))
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return str(content).strip()


# --------------------------------------------------------------------------- #
# Context formatting
# --------------------------------------------------------------------------- #
def format_passages(hits: Sequence[Hit], *, max_chars: int = 1800) -> str:
    blocks = []
    for i, hit in enumerate(hits, 1):
        body = hit.text
        # Chunks carry a breadcrumb first line; the header below repeats it in a
        # cleaner form, so drop the duplicate to save tokens.
        if "\n\n" in body:
            head, rest = body.split("\n\n", 1)
            if head.count(" > ") >= 1 and len(head) < 200:
                body = rest
        body = body[:max_chars]
        label = f"{hit.title}" + (f" > {hit.section}" if hit.section else "")
        blocks.append(f"[{i}] {label}\n{body}")
    return "\n\n---\n\n".join(blocks)


# Models cite in more than one shape. Claude tends to write "[1]" per sentence;
# Gemini frequently groups them as "[2, 3]". Matching only the single form drops
# real citations on the floor, which silently shrinks the source list the user
# sees - so accept a bracket containing a comma-separated run of numbers.
_CITATION_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _cited_indices(answer: str, n: int) -> list[int]:
    found: set[int] = set()
    for group in _CITATION_GROUP.findall(answer):
        for part in group.split(","):
            part = part.strip()
            if part.isdigit():
                found.add(int(part))
    return sorted(i for i in found if 1 <= i <= n)


@dataclass
class Deps:
    """Everything the nodes need that is not in the state."""

    settings: Settings
    llms: LLMBundle


def _event(name: str, started: float, **details: Any) -> dict:
    return {"node": name, "ms": round((time.perf_counter() - started) * 1000, 1), **details}


def _fallback_kwargs(used: str | None, configured: str) -> dict:
    """``{"fallback_from": configured}`` when a fallback actually fired.

    Surfaced in the event trace (and from there, the UI) so a quota-driven
    model switch is visible rather than silently changing which model answered
    a given query.
    """
    if used and used != configured:
        return {"fallback_from": configured}
    return {}


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def analyze_query(state: GraphState, deps: Deps) -> dict:
    """Classify the question and rewrite it into a retrieval query."""
    started = time.perf_counter()
    question = state["question"]

    analysis, used = call_structured(
        deps.llms.grader_models(), QueryAnalysis, prompts.QUERY_ANALYSIS_SYSTEM, question
    )
    if analysis is None:
        # Analysis is an optimisation, not a gate - fall through on the raw text.
        return {
            "query": question,
            "intent": "unknown",
            "events": [_event("analyze_query", started, skipped="analysis_failed")],
        }

    query = (analysis.search_query or question).strip() or question
    return {
        "query": query,
        "intent": analysis.intent,
        "events": [
            _event(
                "analyze_query",
                started,
                intent=analysis.intent,
                rewritten=query != question,
                query=query,
                model=used,
                **_fallback_kwargs(used, deps.llms.grader_model),
            )
        ],
    }


def retrieve(state: GraphState, deps: Deps) -> dict:
    started = time.perf_counter()
    trace = store.search(
        deps.settings,
        state.get("query") or state["question"],
        doc_sets=state.get("doc_sets") or None,
    )
    return {
        "hits": trace.hits,
        "retrieval_mode": trace.mode,
        "attempts": state.get("attempts", 0) + 1,
        "events": [
            _event(
                "retrieve",
                started,
                mode=trace.mode,
                dense_prefetch=trace.dense_candidates,
                sparse_prefetch=trace.sparse_candidates,
                returned=len(trace.hits),
                attempt=state.get("attempts", 0) + 1,
            )
        ],
    }


def rerank(state: GraphState, deps: Deps) -> dict:
    """Cross-encoder rerank, cutting fusion_limit candidates down to top_k.

    The bi-encoder scores query and passage separately, so it never sees them
    together; the cross-encoder does one joint forward pass per pair. That is
    ~20x more compute for 20 candidates, which is why it runs on a shortlist
    rather than the whole corpus.
    """
    started = time.perf_counter()
    hits = state.get("hits") or []
    if not hits:
        return {"events": [_event("rerank", started, skipped="no_hits")]}
    if not deps.settings.reranking_enabled:
        return {
            "hits": hits[: deps.settings.top_k],
            "events": [_event("rerank", started, skipped="disabled")],
        }

    spec = deps.settings.reranker_spec
    reranker = get_reranker(deps.settings)
    scores = reranker.score(state.get("query") or state["question"],
                            [h.text for h in hits])
    for hit, score in zip(hits, scores):
        hit.rerank_score = score

    ordered = sorted(hits, key=lambda h: h.rerank_score or 0.0, reverse=True)
    kept = ordered[: deps.settings.top_k]

    # The torch rerankers are ~2.3 GB and this was built on a 7.3 GB box; hand
    # the memory back rather than holding it between queries. The ONNX ones are
    # small enough to keep resident, and reloading them per query would cost
    # more than it saves.
    if spec.backend == "torch":
        from ..retrieval.encoders import REGISTRY

        REGISTRY.release_reranker(spec)

    return {
        "hits": kept,
        "events": [
            _event(
                "rerank",
                started,
                model=spec.slug,
                scored=len(hits),
                kept=len(kept),
                top_score=round(kept[0].rerank_score or 0.0, 4) if kept else None,
                reordered=[h.chunk_id for h in kept] != [h.chunk_id for h in hits[: len(kept)]],
            )
        ],
    }


def truncate(state: GraphState, deps: Deps) -> dict:
    """Top-k cut for configurations that skip reranking."""
    started = time.perf_counter()
    hits = (state.get("hits") or [])[: deps.settings.top_k]
    return {"hits": hits, "events": [_event("truncate", started, kept=len(hits))]}


def grade_documents(state: GraphState, deps: Deps) -> dict:
    """Decide whether what we retrieved can actually answer the question.

    This is the gate that makes the pipeline corrective: without it the
    generator is handed whatever came back and will write a confident answer
    from loosely-related passages.
    """
    started = time.perf_counter()
    hits = state.get("hits") or []
    if not hits:
        return {
            "relevant": False,
            "grade_score": 0.0,
            "grade_reason": "retrieval returned nothing",
            "events": [_event("grade_documents", started, sufficient=False, score=0.0)],
        }

    user = (
        f"Question:\n{state['question']}\n\n"
        f"Retrieved passages:\n{format_passages(hits, max_chars=1200)}"
    )
    grade, used = call_structured(
        deps.llms.grader_models(), RelevanceGrade, prompts.GRADE_SYSTEM, user
    )
    if grade is None:
        # Fail open: a broken grader should not block an answer the user can
        # judge for themselves against the citations.
        return {
            "relevant": True,
            "grade_score": 0.5,
            "grade_reason": "grader unavailable; proceeding",
            "events": [_event("grade_documents", started, skipped="grader_failed")],
        }

    sufficient = grade.sufficient and grade.score >= deps.settings.relevance_threshold
    return {
        "relevant": sufficient,
        "grade_score": grade.score,
        "grade_reason": grade.reason,
        "events": [
            _event(
                "grade_documents",
                started,
                sufficient=sufficient,
                score=round(grade.score, 3),
                reason=grade.reason,
                model=used,
                **_fallback_kwargs(used, deps.llms.grader_model),
            )
        ],
    }


def rewrite_query(state: GraphState, deps: Deps) -> dict:
    """Produce a different query after a failed retrieval."""
    started = time.perf_counter()
    previous = "\n".join(f"- {q}" for q in ([state["question"]] + state.get("rewrites", [])))
    user = (
        f"Original question:\n{state['question']}\n\n"
        f"Queries already tried:\n{previous}\n\n"
        f"Why the last attempt failed: {state.get('grade_reason', 'no relevant passages')}"
    )
    messages = [SystemMessage(content=prompts.QUERY_REWRITE_SYSTEM), HumanMessage(content=user)]
    try:
        response, used = call_with_fallback(deps.llms.grader_models(), lambda m: m.invoke(messages))
        new_query = message_text(response).strip('"')
        new_query = new_query.split("\n")[0][:400] or state["question"]
    except Exception as exc:
        log.warning("rewrite_query failed on every model in the chain: %s", exc)
        new_query, used = state["question"], None

    return {
        "query": new_query,
        "rewrites": [new_query],
        "events": [
            _event(
                "rewrite_query", started, query=new_query, model=used,
                **_fallback_kwargs(used, deps.llms.grader_model),
            )
        ],
    }


def generate(state: GraphState, deps: Deps) -> dict:
    started = time.perf_counter()
    hits = state.get("hits") or []
    if not hits:
        return {
            "answer": prompts.INSUFFICIENT_ANSWER,
            "citations": [],
            "status": "insufficient_context",
            "events": [_event("generate", started, skipped="no_hits")],
        }

    user = (
        f"Question:\n{state['question']}\n\n"
        f"Documentation passages:\n{format_passages(hits)}"
    )
    system = prompts.GENERATE_SYSTEM
    if state.get("regenerated") and state.get("groundedness_reason"):
        system += prompts.REGENERATE_SUFFIX.format(
            unsupported=state.get("groundedness_reason", "")
        )

    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    response, used = call_with_fallback(deps.llms.generator_models(), lambda m: m.invoke(messages))
    answer = message_text(response)

    cited = _cited_indices(answer, len(hits))
    # Return every passage the model was shown, in the order it was shown them,
    # flagging which ones the answer actually pointed at.
    #
    # Filtering down to just the cited ones renumbers the list, so an answer
    # citing [1] and [5] would render against a four-item list where [5] refers
    # to nothing. Keeping the full list means marker N always resolves to
    # source N, and the uncited entries are still worth showing: they are what
    # the model read and chose not to use.
    cited_set = set(cited)
    citations = [
        {**hit.citation(), "cited": (i in cited_set)}
        for i, hit in enumerate(hits, 1)
    ]

    return {
        "answer": answer,
        "citations": citations,
        "status": "answered",
        "events": [
            _event(
                "generate",
                started,
                model=used,
                passages=len(hits),
                cited=len(cited),
                chars=len(answer),
                **_fallback_kwargs(used, deps.llms.generator_model),
            )
        ],
    }


def check_groundedness(state: GraphState, deps: Deps) -> dict:
    """Final check that the answer's claims appear in the cited passages."""
    started = time.perf_counter()
    hits = state.get("hits") or []
    answer = state.get("answer") or ""
    if not hits or not answer or state.get("status") == "insufficient_context":
        return {"grounded": True, "events": [_event("check_groundedness", started, skipped=True)]}

    user = (
        f"Question:\n{state['question']}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Source passages:\n{format_passages(hits, max_chars=1400)}"
    )
    check, used = call_structured(
        deps.llms.grader_models(), GroundednessCheck, prompts.GROUNDEDNESS_SYSTEM, user
    )
    if check is None:
        return {
            "grounded": True,
            "groundedness_reason": "check unavailable",
            "events": [_event("check_groundedness", started, skipped="checker_failed")],
        }

    return {
        "grounded": check.grounded,
        "groundedness_reason": "; ".join(check.unsupported) or check.reason,
        "events": [
            _event(
                "check_groundedness",
                started,
                grounded=check.grounded,
                unsupported=len(check.unsupported),
                reason=check.reason,
                model=used,
                **_fallback_kwargs(used, deps.llms.grader_model),
            )
        ],
    }


def mark_regenerate(state: GraphState, deps: Deps) -> dict:
    """Flag the run so generate() knows to repair rather than restate."""
    return {"regenerated": True}


def give_up(state: GraphState, deps: Deps) -> dict:
    """Return "not enough information" instead of answering from thin context."""
    started = time.perf_counter()
    intent = state.get("intent", "")
    answer = (
        prompts.OUT_OF_SCOPE_ANSWER
        if intent == "out_of_scope"
        else prompts.INSUFFICIENT_ANSWER
    )
    return {
        "answer": answer,
        "citations": [],
        "status": "insufficient_context",
        "grounded": True,
        "events": [
            _event("give_up", started, intent=intent,
                   attempts=state.get("attempts", 0),
                   reason=state.get("grade_reason", "")),
        ],
    }


# --------------------------------------------------------------------------- #
# Conditional edges
# --------------------------------------------------------------------------- #
def route_after_grade(state: GraphState, settings: Settings) -> str:
    if state.get("relevant"):
        return "generate"
    if state.get("intent") == "out_of_scope":
        return "give_up"
    if state.get("attempts", 0) <= settings.max_retries:
        return "rewrite_query"
    return "give_up"


def route_after_groundedness(state: GraphState, settings: Settings) -> str:
    if state.get("grounded", True):
        return "__end__"
    if state.get("regenerated"):
        # One repair attempt only. Return the answer but leave grounded=False so
        # the caller and the UI can flag it rather than silently trusting it.
        return "__end__"
    return "mark_regenerate"
