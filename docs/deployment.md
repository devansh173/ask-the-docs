# Deployment

The app is one container that serves both the JSON API and the static frontend.
What it needs alongside it is a Qdrant it can reach and, optionally, an LLM key.
Users can supply their own key from the UI, so a deployment without one is still
a working demo — it just asks the visitor for a key before it answers.

## Sizing

With the shipped defaults the process settles at roughly **500–700 MB**
resident. Anything advertising 512 MB is marginal; 1 GB is comfortable.

| Model | Size on disk | Loaded at |
|---|---|---|
| `bge-small-en-v1.5` (dense) | 67 MB | startup (warmed, baked into the image) |
| `Qdrant/bm25` (sparse) | 10 MB | startup (warmed, baked into the image) |
| a cross-encoder, if enabled | 80 MB – 2.3 GB | first query that reranks |

Reranking ships disabled, so the default image stays small. **If you enable a
reranker, size the container accordingly** — `bge-base` alone is 1 GB of weights
on top of the runtime, which pushes the requirement past 2 GB.

## Option A — Google Cloud Run (recommended)

Scales to zero, bills per request, and lets you set memory explicitly, which is
what the other free tiers get wrong for this workload.

```bash
gcloud run deploy askthedocs \
  --source . \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --allow-unauthenticated \
  --set-env-vars "EMBEDDING_MODEL=bge-small,RERANKER_MODEL=none,QDRANT_URL=https://YOUR-CLUSTER.cloud.qdrant.io:6333,QDRANT_COLLECTION=askthedocs" \
  --set-secrets "QDRANT_API_KEY=qdrant-key:latest,ANTHROPIC_API_KEY=anthropic-key:latest"
```

Notes that matter:

- **1 GiB is enough with the defaults**, which settle around 500–700 MB. If you
  turn a reranker on, raise it to 2 GiB and `--cpu 2`: `bge-base` is another
  1 GB of weights, and cross-encoder inference is CPU-bound — on 1 vCPU,
  reranking 20 candidates adds seconds per query.
- **Cold starts.** Scale-to-zero means the first request after idle pays model
  load. Set `--min-instances 1` if you are sending the link to someone and want
  it to feel instant; that is the one setting that costs money while idle.
- Secrets go in Secret Manager, not `--set-env-vars`.

## Option B — Fly.io

```bash
fly launch --no-deploy
fly secrets set ANTHROPIC_API_KEY=... QDRANT_API_KEY=...
fly deploy
```

```toml
# fly.toml  — sized for the defaults; raise both if you enable a reranker
[build]
  dockerfile = "Dockerfile"

[env]
  EMBEDDING_MODEL = "bge-small"
  RERANKER_MODEL = "none"
  QDRANT_URL = "https://YOUR-CLUSTER.cloud.qdrant.io:6333"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = "suspend"
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  memory = "1gb"
  cpus = 1
```

`auto_stop_machines = "suspend"` snapshots memory rather than killing the
machine, so a resumed instance skips the model load — a better cold-start story
than Cloud Run, at the cost of paying for the suspended machine's storage.

## Option C — Hugging Face Spaces

Natural fit for an ML demo and CPU Basic gives 2 vCPU / 16 GB, comfortably more
than needed. Note that HF has moved compute-backed Spaces (Docker and Gradio)
behind a paid plan; only Static Spaces remain free. Verify current terms before
relying on it.

## Qdrant

| | Setup | Notes |
|---|---|---|
| Embedded (default) | none | Single process only. Fine locally, wrong for a deployed container whose filesystem is ephemeral. |
| Docker | `docker compose up -d` | Local development against a real server. |
| Qdrant Cloud free tier | signup, no card | 0.5 vCPU, 1 GB RAM, 4 GB disk — roughly a million 768-dim vectors, far beyond this corpus. |

**Do not deploy with embedded mode.** Local mode writes to a SQLite file that
lives in the container's ephemeral filesystem: it disappears on every restart,
and two replicas would each hold a private copy of the index.

The free Qdrant Cloud cluster **suspends after one week of inactivity and is
deleted after four**. A demo you link from a résumé needs a keep-alive — a
scheduled workflow hitting `/api/health` weekly is enough.

## Indexing the deployed instance

The container ships without data. Index from your machine against the same
Qdrant the deployment uses:

```bash
export QDRANT_URL=https://YOUR-CLUSTER.cloud.qdrant.io:6333
export QDRANT_API_KEY=...
askthedocs scrape
askthedocs index --recreate
```

Index with the **same `EMBEDDING_MODEL` the server runs**. Each model has its
own collection and its own dimensionality (384 / 768 / 1024), so a mismatch
means the server queries a collection that was never built — the API reports it
explicitly rather than returning empty results, but nothing will answer until
you index that model.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `bge-small` | Must match what was indexed; each model has its own collection. |
| `RERANKER_MODEL` | `none` | Off by default — the ablation found it not worth its latency. |
| `QDRANT_URL` | unset → embedded | Qdrant server URL. |
| `QDRANT_API_KEY` | — | Required by Qdrant Cloud. |
| `QDRANT_COLLECTION` | `askthedocs` | Collection name. |
| `LLM_PROVIDER` | `anthropic` | Server-side default; the UI can override. |
| `LLM_MODEL` | `claude-opus-5` | Generation model. |
| `GRADER_MODEL` | `claude-sonnet-5` | Grading and groundedness model. |
| `ANTHROPIC_API_KEY` | — | Optional; users may supply their own from the UI. |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | — | Tracing. Omit to disable. |
| `TOP_K` | `5` | Passages sent to the generator. |
| `MAX_RETRIES` | `2` | Query rewrites before abstaining. |

## Health check

`GET /api/health` reports the collection size, the active embedding and
reranker, which
Qdrant it is talking to, and whether tracing is on. `indexed_chunks: 0` means
the container is up but nothing was ever indexed into the collection it is
pointed at — the most common deployment mistake, and the reason the field is
exposed rather than a bare `{"status": "ok"}`.
