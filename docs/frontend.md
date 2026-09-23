# The web UI

One HTML file, one stylesheet, one script, served as static assets by the same
FastAPI process that serves the API. No build step, no `node_modules`, nothing
to compile before the app runs — `askthedocs serve` is the whole thing.

## What it does

**Ask** — questions stream token by token, with the pipeline's progress shown
above the answer as it happens ("Searching the index" → "Reranking passages" →
"Grading relevance" → "Writing the answer"). Citations render as clickable
chips; clicking one scrolls to and highlights the source it refers to.

**Show its work** — every answer carries a collapsible trace: which nodes ran,
how long each took, how many candidates each stage kept, whether the grader
judged the passages sufficient, whether the groundedness check passed. This is
the part that makes the pipeline legible rather than magical.

**Upload documents** — drag PDFs, Markdown, text or HTML onto the sidebar. They
are parsed, chunked and embedded locally, then indexed into a collection of
their own. Upload progress is real (XHR `upload.onprogress`), because a large
PDF spends meaningful time on the wire before indexing even begins.

**Swap models** — pick the embedding model, the reranker, the LLM provider and
the model, per request. The embedding dropdown shows how many chunks each model
has indexed and disables the ones with none.

**Compare pipelines** — an Agentic/Naive toggle runs the same question through
the full graph or the baseline, which is the fastest way to see what the extra
machinery buys.

## Deliberate choices

**No framework.** The app has perhaps a dozen pieces of state and one streaming
endpoint. React plus a bundler would add a build step, a `node_modules`
directory and a second deployment artefact to a project whose point is the
retrieval pipeline. The whole UI is ~700 lines.

**Markdown is rendered by hand, escape-first.** Model output is HTML-escaped
before any formatting is applied, then a fixed set of safe tags is reintroduced
(code, bold, lists, citation chips). Reaching for a markdown library would mean
auditing its sanitiser; escaping first means untrusted text can never become
markup in the first place.

**Keys live in `sessionStorage`, not `localStorage`.** They survive a refresh in
the current tab and vanish when it closes. They are sent per request and the
server drops them immediately — never written to disk, never logged, never
attached to a trace.

**Progress events, not a spinner.** The graph makes several LLM calls before a
single answer token exists. A spinner for eight seconds looks broken; naming the
current stage looks like a system.

**Failure states say what to do.** Selecting an un-indexed embedding model
produces the exact command to build it. An empty upload workspace says to upload
something. A missing API key names the environment variable *and* points at the
sidebar. None of them return an empty answer and leave you guessing.

## Layout

```
frontend/
  index.html   structure and the sidebar controls
  styles.css   design tokens, light/dark, responsive at 900px
  app.js       state, SSE streaming, uploads, markdown rendering
```

Dark mode follows `prefers-color-scheme`. Below 900px the sidebar becomes a
drawer behind a scrim.

## The streaming protocol

`POST /api/ask/stream` returns Server-Sent Events. `EventSource` cannot issue a
POST, so the client reads the response body as a stream and parses frames
itself.

| Event | Payload | Used for |
|---|---|---|
| `start` | config, provider, model, embedding | header line |
| `node` | node name, ms, per-node detail | live progress, then the trace |
| `token` | text fragment | the streamed answer |
| `reset` | reason | groundedness repair: clear and rewrite |
| `done` | answer, citations, status, flags | final render |
| `error` | detail | error banner |

Only the `generate` node's tokens are streamed as answer text. The graph's other
LLM calls — query analysis, relevance grading, the groundedness check — surface
as `node` events instead, which is why the UI can narrate the pipeline without
leaking the grader's reasoning into the answer.
