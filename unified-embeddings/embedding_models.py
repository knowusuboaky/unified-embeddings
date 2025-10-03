# embedding_models.py
#
# Install (CPU-only; easiest)
# python -m venv .venv && source .venv/bin/activate
# pip install --upgrade pip
# pip install fastapi uvicorn[standard] pydantic==2.* numpy torch transformers sentence-transformers
#
# (Optional) Install for NVIDIA GPU
# # Pick the CUDA build matching your driver (example: CUDA 12.1)
# pip install --upgrade pip
# pip install "torch==2.*" --index-url https://download.pytorch.org/whl/cu121
# pip install fastapi uvicorn[standard] pydantic==2.* numpy transformers sentence-transformers huggingface_hub[hf_ext]
# pip install einops safetensors
# 
# (Optional) Extra CPU speedups
# pip install onnxruntime
#
# Run (port 9005)
# uvicorn embedding_models:app --host 0.0.0.0 --port 9005

from typing import List, Dict, Literal
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import threading
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

app = FastAPI(
    title="Unified Embeddings API",
    version="0.1",
    docs_url="/__docs__",
    openapi_url="/openapi.json"
)

# --------------------------
# Model registry
# --------------------------
REGISTRY: Dict[str, Dict] = {
    "e5-large-v2": {
        "hf_id": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "normalize": True,
        "dims": 1024,
        "preferred_loader": "st",
        "max_tokens": 512,
    },
    "mxbai": {
        "hf_id": "mixedbread-ai/mxbai-embed-large-v1",
        "query_prefix": "",
        "doc_prefix": "",
        "normalize": True,
        "dims": 1024,
        "preferred_loader": "st",
        "max_tokens": 8192,
    },
    "arctic": {
        "hf_id": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "query_prefix": "",
        "doc_prefix": "",
        "normalize": True,
        "dims": 1024,
        "preferred_loader": "st",
        "max_tokens": 8192,
    },
    "nomic": {
        "hf_id": "nomic-ai/nomic-embed-text-v1.5",
        "query_prefix": "",
        "doc_prefix": "",
        "normalize": True,
        "dims": 768,
        "preferred_loader": "st",
        "max_tokens": 8192,
    },
    "bge-m3": {
        "hf_id": "BAAI/bge-m3",
        "query_prefix": "Represent this query for retrieval: ",
        "doc_prefix":   "Represent this document for retrieval: ",
        "normalize": True,
        "dims": 1024,
        "preferred_loader": "st",
        "max_tokens": 8192,
    },
}

_loaded_models: Dict[str, object] = {}
_tokenizers: Dict[str, object] = {}
_backends: Dict[str, Literal["st", "hf"]] = {}
_load_lock = threading.Lock()

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def _needs_trust_remote(key: str) -> bool:
    # Models that ship custom code requiring trust_remote_code=True
    return key in {"nomic", "bge-m3"}

def _load_model(key: str):
    if key in _loaded_models:
        return
    spec = REGISTRY[key]
    hf_id = spec["hf_id"]
    dev = _device()
    with _load_lock:
        if key in _loaded_models:
            return
        trust = _needs_trust_remote(key)

        # Prefer SentenceTransformers (handles pooling)
        if spec.get("preferred_loader") == "st":
            try:
                model = SentenceTransformer(hf_id, device=dev, trust_remote_code=trust)
                _loaded_models[key] = model
                _backends[key] = "st"
                return
            except Exception as e:
                print(f"[WARN] ST load failed for {key}: {e}")

        # Fallback to raw HF + mean pooling
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust)
        mdl = AutoModel.from_pretrained(hf_id, trust_remote_code=trust).to(dev)
        mdl.eval()
        _tokenizers[key] = tok
        _loaded_models[key] = mdl
        _backends[key] = "hf"

def _encode(key: str, texts: List[str], batch_size: int, normalize: bool) -> np.ndarray:
    _load_model(key)
    backend = _backends[key]
    dev = _device()
    if backend == "st":
        model: SentenceTransformer = _loaded_models[key]
        return model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
    # Raw HF fallback
    model_hf: AutoModel = _loaded_models[key]
    tok = _tokenizers[key]
    out_vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True, return_tensors="pt").to(dev)
            out = model_hf(**enc)
            pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out_vecs.append(pooled.cpu().numpy())
    return np.vstack(out_vecs)

def _apply_prefixes(model_key: str, mode: str, texts: List[str]) -> List[str]:
    spec = REGISTRY[model_key]
    q, d = spec["query_prefix"], spec["doc_prefix"]
    if mode == "query": return [q + t for t in texts]
    if mode == "document": return [d + t for t in texts]
    if len(texts) == 1: return [q + texts[0]]
    return [d + t for t in texts]

class EmbedRequest(BaseModel):
    texts: List[str]
    mode: Literal["auto", "query", "document"] = "auto"
    batch_size: int = Field(64, ge=1, le=1024)
    normalize: bool = True
    truncate: bool = True  # kept for compatibility; truncation is handled by tokenizer defaults

class EmbedResponse(BaseModel):
    model: str
    dims: int
    vectors: List[List[float]]

# --------------------------
# Startup: preload ALL models
# --------------------------
@app.on_event("startup")
def _startup():
    print(f"[startup] Preloading ALL models: {list(REGISTRY.keys())}")
    for name in REGISTRY.keys():
        try:
            _load_model(name)
            print(f"[startup] Loaded {name} (backend={_backends.get(name)})")
        except Exception as e:
            print(f"[startup] ERROR loading {name}: {e}")

# --------------------------
# Routes
# --------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True, "device": _device(), "loaded": list(_loaded_models.keys())}

@app.get("/models")
def models():
    return {k: {"hf_id": v["hf_id"], "dims": v["dims"], "max_tokens": v["max_tokens"]}
            for k, v in REGISTRY.items()}

@app.post("/embed", response_model=EmbedResponse)
def embed(model: str = Query(..., description=f"One of: {list(REGISTRY.keys())}"), req: EmbedRequest = None):
    if model not in REGISTRY:
        raise HTTPException(400, f"Unknown model '{model}'")
    if not req or not req.texts:
        raise HTTPException(400, "Request must include non-empty 'texts'")
    texts = _apply_prefixes(model, req.mode, req.texts)
    spec = REGISTRY[model]
    vecs = _encode(model, texts, batch_size=req.batch_size, normalize=spec["normalize"] and req.normalize)
    return EmbedResponse(model=model, dims=spec["dims"], vectors=vecs.tolist())
