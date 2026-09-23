# Resume bullets

Draft wording for **Ask the Docs**. Every number below comes from
`evals_out/retrieval-bge-small-*.json` and the test suite — nothing here is
estimated. Re-run `askthedocs eval` before quoting them if the corpus changes.

**One caveat before you use these:** the generation-side metrics
(faithfulness, answer relevancy, context precision/recall) have **not been
measured yet**. The pipeline itself is verified end to end against Gemini - real
answers, real citations, correct abstention on out-of-scope questions - but the
full RAGAS run needs ~352 model calls and the Gemini free tier caps `flash-lite`
at 20 requests per day, which it exhausts after about five questions.

**Do not quote RAGAS numbers until you have actually run them.** To get them:
raise the quota (any paid tier), or point `LLM_PROVIDER` at a local Ollama
model, then `askthedocs eval`. The retrieval numbers below are real, were
reproduced across three separate runs, and are safe to quote today.

---

## Recommended set (pick 3)

> **Built an agentic RAG system** over 5.2K documentation chunks using LangGraph,
> Qdrant and FastAPI, implementing hybrid dense+BM25 retrieval with reciprocal
> rank fusion — **raising top-1 retrieval accuracy from 75% to 87.5% and MRR from
> 0.860 to 0.918** against a 44-question hand-labelled evaluation set.

> **Designed a self-correcting retrieval graph** in LangGraph that grades its own
> retrieved context, rewrites the query and re-searches on weak results (max 2
> retries), and verifies every generated answer against its cited sources —
> returning an explicit "not enough information" rather than answering from
> insufficient context.

> **Built an offline evaluation harness** (hit@k, MRR, nDCG, recall) requiring no
> LLM calls, wired into a CI gate that fails builds on retrieval regression;
> **the ablation disproved my own design assumption**, showing cross-encoder
> reranking reduced MRR by 5% while adding 57× latency, so it shipped disabled
> with the measurement documented.

---

## Alternates

> **Engineered a provider-agnostic LLM layer** supporting Anthropic Claude,
> Google Gemini, OpenAI and local Ollama models, selectable per request with
> user-supplied API keys held in-request only — never persisted, logged or traced.

> **Implemented multi-model retrieval** across 7 self-hosted embedding models
> (384–1024 dim) with per-model Qdrant collections and 5 optional cross-encoder
> rerankers, plus user document upload (PDF/Markdown/HTML) into isolated
> per-workspace indexes.

> **Diagnosed and fixed a 30× indexing slowdown** by profiling the embedding
> pipeline: identified sequence length rather than parallelism as the bottleneck
> (attention is O(n²); measured 254 → 2 chunks/s from 50 to 750 characters) and
> that per-point SQLite commits into a cloud-synced directory were amplifying
> disk writes 16×.

> **Achieved 91 passing tests with zero external dependencies** — no API key, no
> network — by scripting LLM behaviour at the graph boundary, so the full
> agentic pipeline including retries and abstention is verified for free on
> every commit.

---

## Notes on wording

**Why "75% to 87.5%" and not a bigger number.** That is hit@1 over 40 answerable
questions (30 → 35 correct). It is a real, reproducible delta. Resist rounding it
up or quoting hit@5, which was already 1.000 for the baseline and would make the
improvement look like zero.

**Lead with the disproved assumption.** The reranking bullet is the strongest one
here, and counter-intuitively so: most candidates list technologies they used,
not measurements that changed their mind. It demonstrates the eval harness had
teeth, and it pre-empts "did you actually verify any of this?"

**Say "44-question hand-labelled" explicitly.** It distinguishes this from
LLM-generated eval sets, which is a real methodological difference and an easy
follow-up question to answer well.

**Numbers to have ready in an interview:**

| | |
|---|---|
| Corpus | 5,199 chunks / 75 pages / 3 documentation sets |
| Golden set | 44 questions (40 answerable + 4 deliberately out-of-scope) |
| Dense only | hit@1 0.750 · MRR 0.860 · 33 ms p50 |
| + BM25 / RRF | hit@1 0.875 · MRR 0.918 · 155 ms p50 |
| + cross-encoder | hit@1 0.750 · MRR 0.871 · 8,895 ms p50 |
| Tests | 91 passing, no API key required |

**The question you will get:** *"Why did reranking hurt?"* — Because dense
retrieval alone already reached hit@5 = 1.000 on this corpus. There was nothing
left for the reranker to recover, so reordering an already-correct list could
only move the right answer down. Cross-encoders earn their cost when the top-20
contains genuine distractors; a 75-page corpus does not produce those.
