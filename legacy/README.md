# v1 (archived)

The original "Ask the Docs": a Streamlit app that accepted PDF/TXT uploads,
embedded them with `all-MiniLM-L6-v2` into an in-memory FAISS index, and
answered with a local `flan-t5-base` through LangChain's `RetrievalQA`.

Kept for comparison. v2 replaces every layer of it:

| | v1 | v2 |
|---|---|---|
| Retrieval | FAISS, dense only, top-k | Qdrant, dense + BM25 fused with RRF, cross-encoder reranked |
| Control flow | one `RetrievalQA` chain | LangGraph state machine with grading, query rewriting and a groundedness check |
| Generation | `flan-t5-base` (250M, local) | any of Claude / Gemini / GPT / Ollama, chosen at request time |
| Index lifetime | rebuilt per upload, in memory | persistent collection, ingested once |
| Citations | none | per-sentence, resolved to source URLs |
| Evaluation | none | offline IR metrics + RAGAS + a CI regression gate |
| Observability | none | Langfuse span per node |

Nothing here is imported by v2.
