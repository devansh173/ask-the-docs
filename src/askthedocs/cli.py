"""Command line entry point.

    askthedocs scrape                 fetch documentation into data/raw
    askthedocs index --recreate       chunk + embed + upsert into Qdrant
    askthedocs inspect "<question>"   show dense / sparse / hybrid / reranked
    askthedocs ask "<question>"       run the full agentic graph once
    askthedocs serve                  run the API + frontend
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import (
    EMBEDDINGS,
    PROFILE_PRESETS,
    RERANKERS,
    naive_config,
    settings as base_settings,
)
from .logging_setup import setup as setup_logging

log = logging.getLogger("askthedocs")


def _resolve(args: argparse.Namespace):
    """Apply --profile / --embedding / --reranker to the base settings.

    --profile is a shorthand that sets both; the individual flags override it,
    so `--profile quality --reranker none` does what it looks like.
    """
    settings = base_settings
    if getattr(args, "profile", None):
        settings = settings.with_profile(args.profile)
    if getattr(args, "embedding", None):
        settings = settings.variant(embedding=args.embedding)
    if getattr(args, "reranker", None):
        settings = settings.variant(reranker=args.reranker)
    return settings


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=sorted(PROFILE_PRESETS),
                        help="preset pairing of embedding + reranker")
    parser.add_argument("--embedding", choices=sorted(EMBEDDINGS),
                        help="dense embedding model (its own collection)")
    parser.add_argument("--reranker", choices=sorted(RERANKERS),
                        help="cross-encoder, or 'none'")


def _cmd_scrape(args: argparse.Namespace) -> int:
    from .ingest.scrape import scrape_all

    counts = scrape_all(only=args.only)
    total = sum(counts.values())
    for name, n in counts.items():
        print(f"  {name:12s} {n:4d} pages")
    print(f"  {'total':12s} {total:4d} pages")
    return 0 if total else 1


def _cmd_index(args: argparse.Namespace) -> int:
    from tqdm import tqdm

    from .ingest.chunk import chunk_corpus
    from .retrieval import store

    settings = _resolve(args)
    chunks = chunk_corpus(settings, only=args.only)

    bar = tqdm(total=len(chunks), desc="embedding", unit="chunk")
    written = store.index_chunks(
        settings,
        chunks,
        recreate=args.recreate,
        progress=lambda done, total: bar.update(done - bar.n),
    )
    bar.close()

    spec = settings.embedding_spec
    print(
        f"\nindexed {written} chunks into '{settings.collection}'\n"
        f"  embedding : {spec.label}  ({spec.model}, {spec.dim}d)\n"
        f"  reranker  : {settings.reranker_spec.label}\n"
        f"  points    : {store.collection_size(settings)}"
    )
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Side-by-side retrieval arms - the manual sanity check for phase 1."""
    from .retrieval import store
    from .retrieval.encoders import get_reranker

    settings = _resolve(args)
    question = args.question

    def show(label: str, hits, score_attr="retrieval_score") -> None:
        print(f"\n  {label}")
        if not hits:
            print("    (nothing)")
        for i, h in enumerate(hits[: args.k], 1):
            score = getattr(h, score_attr)
            head = h.text.split("\n\n", 1)[-1][:90].replace("\n", " ")
            print(f"    {i}. [{score:7.4f}] {h.title} > {h.section or '-'}")
            print(f"       {head}...")

    print(f"\nQUESTION: {question}")
    show("DENSE ONLY", store.search_arm(settings, question, "dense", limit=args.k))
    show("SPARSE ONLY (BM25)", store.search_arm(settings, question, "sparse", limit=args.k))

    trace = store.search(settings, question)
    show(f"HYBRID (RRF, {trace.latency_ms:.0f}ms)", trace.hits)

    if settings.reranking_enabled and trace.hits:
        reranker = get_reranker(settings)
        scores = reranker.score(question, [h.text for h in trace.hits])
        for hit, score in zip(trace.hits, scores):
            hit.rerank_score = score
        reranked = sorted(trace.hits, key=lambda h: h.rerank_score or 0, reverse=True)
        show("AFTER CROSS-ENCODER RERANK", reranked, score_attr="rerank_score")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from .graph.build import answer_question

    settings = _resolve(args)
    if args.naive:
        settings = naive_config(settings)

    result = answer_question(args.question, settings=settings)

    print(f"\n{result.answer}\n")
    if result.citations:
        print("Sources:")
        for i, c in enumerate(result.citations, 1):
            print(f"  [{i}] {c['title']} > {c['section'] or '-'}  {c['source_url']}")
    print(f"\n({result.summary()})")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "askthedocs.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evals.run import run_comparison

    run_comparison(
        configs=args.configs,
        settings=_resolve(args),
        limit=args.limit,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="askthedocs", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-level", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scrape", help="fetch documentation into data/raw")
    p.add_argument("--only", nargs="*", help="limit to these doc sets")
    p.set_defaults(func=_cmd_scrape)

    p = sub.add_parser("index", help="chunk, embed and upsert into Qdrant")
    p.add_argument("--recreate", action="store_true", help="drop the collection first")
    p.add_argument("--only", nargs="*", help="limit to these doc sets")
    _add_model_flags(p)
    p.set_defaults(func=_cmd_index)

    p = sub.add_parser("inspect", help="compare dense / sparse / hybrid / reranked")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)
    _add_model_flags(p)
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("ask", help="run the graph once and print the answer")
    p.add_argument("question")
    p.add_argument("--naive", action="store_true", help="use the baseline config")
    _add_model_flags(p)
    p.set_defaults(func=_cmd_ask)

    p = sub.add_parser("eval", help="score configurations with RAGAS")
    p.add_argument("--configs", nargs="*", default=["naive", "agentic"])
    _add_model_flags(p)
    p.add_argument("--limit", type=int, default=None, help="only N golden questions")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("serve", help="run the API and frontend")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        log.error("%s", exc, exc_info=log.isEnabledFor(logging.DEBUG))
        return 1


if __name__ == "__main__":
    sys.exit(main())
