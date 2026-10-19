"""API tìm Điều luật cho n8n.   uvicorn prod.app:app --port 8000   (chạy từ thư mục gốc)

Env: QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, BM25_VOCAB, EMB_MODEL, RERANK=0|1, RERANK_MODEL, DEVICE, API_TOKEN
POST /search  (header X-API-Key)  {"question": "...", "rerank_query": "...", "k": 8, "max_chars": 2500}
GET  /health
"""
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from legal_core.search import MAX_CHARS, Config
from prod.qdrant_retriever import QdrantRetriever

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
STATE: dict = {}
LOCK = threading.Lock()          # một model dùng chung -> mỗi lúc một truy vấn


@asynccontextmanager
async def lifespan(app):
    env = os.environ.get
    cfg = Config(device=env("DEVICE") or None, emb_model=env("EMB_MODEL") or Config.emb_model,
                 rerank_model=env("RERANK_MODEL") or Config.rerank_model, depth=Config.depth if env("RERANK") == "1" else 0)
    r = QdrantRetriever(cfg, collection=env("QDRANT_COLLECTION", "phapdien_v1"), vocab_path=env("BM25_VOCAB", "bm25_vocab.json.gz"),
                        url=os.environ["QDRANT_URL"], api_key=env("QDRANT_API_KEY"))
    r.search("Quyền và nghĩa vụ của người lao động", k=1)     # khởi động nóng: câu đầu tiên không bị chậm
    STATE["r"] = r
    yield
    STATE.clear()


app = FastAPI(title="Tìm Điều luật", lifespan=lifespan)


def check_key(x_api_key: str | None = Header(default=None)):
    tok = os.environ.get("API_TOKEN")
    if tok and not secrets.compare_digest((x_api_key or "").encode(), tok.encode()):
        raise HTTPException(401, "Sai hoặc thiếu X-API-Key")


class Query(BaseModel):
    question: str = Field(..., min_length=5, max_length=8000)
    rerank_query: str | None = Field(None, max_length=500)
    k: int = Field(8, ge=1, le=20)
    max_chars: int = Field(2500, ge=300, le=MAX_CHARS)


@app.get("/health")
def health():
    r = STATE.get("r")
    if r is None:
        raise HTTPException(503, "đang nạp model")
    return {"ok": True, "collection": r.collection, "rerank": r.reranker is not None}


@app.post("/search", dependencies=[Depends(check_key)])
def search(q: Query):
    with LOCK:
        return STATE["r"].search(q.question, q.rerank_query, q.k, q.max_chars)
