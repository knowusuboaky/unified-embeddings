# Unified Embeddings API

A lightweight **FastAPI** service that unifies multiple open-source text embedding models behind a single `/embed` endpoint.
It handles model-specific quirks (prefixes, pooling, normalization) and works on both **CPU** and **GPU (CUDA)** builds.

---

## ✨ Key Features

* 🔎 **Unified interface** for multiple models (E5, MXBAI, Arctic, Nomic, BGE-M3).
* ⚡ **Auto-prefixing**: query/document handling baked in.
* 🖥️ **CPU-friendly** but with **CUDA acceleration** if available.
* 🐳 **Docker-ready** with `ghcr.io/knowusuboaky/unified-embeddings:latest`.
* 📦 **Preloading** of all registry models at startup for faster inference.

---

## 📦 Models Available

| Key         | Hugging Face ID                           | Dimensions | Notes                |
| ----------- | ----------------------------------------- | ---------- | -------------------- |
| e5-large-v2 | `intfloat/e5-large-v2`                    | 1024       | Query/doc prefixes   |
| mxbai       | `mixedbread-ai/mxbai-embed-large-v1`      | 1024       | Long context support |
| arctic      | `Snowflake/snowflake-arctic-embed-l-v2.0` | 1024       | High token limit     |
| nomic       | `nomic-ai/nomic-embed-text-v1.5`          | 768        | Fast, open-source    |
| bge-m3      | `BAAI/bge-m3`                             | 1024       | Multilingual support |

---

## 🚀 Quickstart (Local, CPU)

```bash
git clone https://github.com/knowusuboaky/unified-embeddings.git
cd unified-embeddings

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r unified-embeddings/requirements.txt

uvicorn embedding_models:app --host 0.0.0.0 --port 9005
```

Docs available at: [http://localhost:9005/**docs**](http://localhost:9005/__docs__)

---

## 🐳 Docker Usage

### Pull from GHCR

```bash
docker pull ghcr.io/knowusuboaky/unified-embeddings:latest
```

### Run (CPU)

```bash
docker run --rm -p 9005:9005 ghcr.io/knowusuboaky/unified-embeddings:latest
```

### Run (GPU, CUDA)

```bash
docker run --rm -p 9005:9005 --gpus all ghcr.io/knowusuboaky/unified-embeddings:cuda
```

---

## 🔧 docker-compose

```yaml
version: "3.9"
services:
  embeddings:
    image: ghcr.io/knowusuboaky/unified-embeddings:latest
    environment:
      - TRANSFORMERS_CACHE=/root/.cache/huggingface
    ports:
      - "9005:9005"
    volumes:
      - hf-cache:/root/.cache/huggingface
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://localhost:9005/healthz || exit 1"]
      interval: 30s
      timeout: 5s
      retries: 5

volumes:
  hf-cache:
```

Run:

```bash
docker compose up -d
```

---

## 📡 API Endpoints

* **GET `/healthz`** → Returns status and loaded models
* **GET `/models`** → Lists available models
* **POST `/embed?model=<key>`** → Generate embeddings

Example:

```bash
curl -X POST "http://localhost:9005/embed?model=e5-large-v2" \
  -H "Content-Type: application/json" \
  -d '{"texts":["hello world"], "mode":"auto"}'
```

Response:

```json
{
  "model": "e5-large-v2",
  "dims": 1024,
  "vectors": [[0.123, -0.456, ...]]
}
```

---

## 📂 Repo Structure

```plaintext
unified-embeddings/
├── unified-embeddings/
│   ├── Dockerfile           # Container build
│   ├── requirements.txt     # Dependencies
├── embedding_models.py      # FastAPI app & registry
├── .dockerignore            # Ignore files in Docker build
├── LICENSE                  # MIT license
└── README.md                # This file
```

---

## ⚡ Tips

* Mount `hf-cache` volume to avoid re-downloading models.
* Use smaller `batch_size` if you hit memory issues.
* For private Hugging Face models, set `HUGGING_FACE_HUB_TOKEN`.

---
