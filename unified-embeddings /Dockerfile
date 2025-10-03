# ------------------------------------------------------------
# Unified Embeddings API — CPU (default) or CUDA via build args
# ------------------------------------------------------------
# CPU build:
#   docker build -t unified-embeddings:cpu .
#
# CUDA build (example: CUDA 12.1):
#   docker build \
#     --build-arg BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
#     -t unified-embeddings:cuda .
#
# Run (CPU):
#   docker run --rm -p 9005:9005 unified-embeddings:cpu
#
# Run (CUDA):
#   docker run --rm -p 9005:9005 --gpus all unified-embeddings:cuda
# ------------------------------------------------------------

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HF_HOME=/root/.cache/huggingface

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy app
COPY embedding_models.py /app/embedding_models.py
COPY requirements.txt /app/requirements.txt

# ---- Python deps ----
ARG TORCH_INDEX_URL=
RUN python -m pip install --upgrade pip \
 && if [ -n "$TORCH_INDEX_URL" ]; then \
        python -m pip install "torch==2.*" --index-url "$TORCH_INDEX_URL" ; \
    else \
        python -m pip install torch ; \
    fi \
 && python -m pip install -r requirements.txt

EXPOSE 9005

# Healthcheck hits the app once it starts
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
  CMD curl -fsS http://127.0.0.1:9005/healthz || exit 1

# Start API
CMD ["uvicorn", "embedding_models:app", "--host", "127.0.0.1", "--port", "9005"]
