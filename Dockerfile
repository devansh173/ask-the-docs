# syntax=docker/dockerfile:1
#
# Runtime image for the API + frontend. Installs the core requirements only:
# the torch-backed models (qwen3 embeddings, bge-v2-m3 reranker) would add ~2 GB
# of wheels and are not part of the deployed demo, which runs ONNX through
# fastembed.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    EMBEDDING_MODEL=bge-small \
    RERANKER_MODEL=none

WORKDIR /app

# --- dependencies (cached separately from source) --------------------------
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# --- application ------------------------------------------------------------
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY frontend/ ./frontend/
RUN pip install --no-deps -e .

# Run unprivileged. The model cache lives in the home directory so the
# compose volume can persist it across restarts.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /home/app/.cache /app/data \
    && chown -R app:app /home/app /app/data
USER app
ENV HOME=/home/app \
    FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed \
    HF_HOME=/home/app/.cache/huggingface

# Bake the ONNX weights into the image so a cold container answers immediately
# instead of downloading ~1 GB on its first request.
RUN python -c "\
from fastembed import TextEmbedding, SparseTextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); \
SparseTextEmbedding(model_name='Qdrant/bm25'); \
TextCrossEncoder(model_name='BAAI/bge-reranker-base'); \
print('model cache warm')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "askthedocs.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
