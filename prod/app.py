"""Backend tra cứu luật.   uvicorn prod.app:app --port 8000   (chạy từ thư mục gốc)

Env: QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, BM25_VOCAB, EMB_MODEL, RERANK=0|1, RERANK_MODEL, DEVICE, API_TOKEN,
     N8N_WEBHOOK_URL, ALLOWED_ORIGINS (domain giao diện, cách nhau dấu phẩy), ASK_PER_MINUTE
POST /ask     (giao diện web)   {"question": "...", "k": 5}  -> chuyển sang n8n (LLM viết lại + trả lời), trả nguyên kết quả
POST /search  (n8n, X-API-Key)  {"question": "...", "rerank_query": "...", "k": 8, "max_chars": 2500}
GET  /health
"""
import logging
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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
    STATE["http"] = httpx.AsyncClient(timeout=httpx.Timeout(150, connect=10))
    yield
    await STATE["http"].aclose()
    STATE.clear()


app = FastAPI(title="Tìm Điều luật", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_methods=["POST"], allow_headers=["Content-Type"],
                   allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()])


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


# ---------- /ask: giao diện web -> backend -> n8n (n8n gọi lại /search) ----------
class Ask(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)
    k: int = Field(5, ge=1, le=10)


HITS: dict = defaultdict(deque)          # IP -> thời điểm các lần hỏi trong 60 giây gần nhất


def rate_limit(request: Request):
    limit = int(os.environ.get("ASK_PER_MINUTE", "10"))
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    now, q = time.monotonic(), HITS[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Bạn hỏi quá nhanh. Đợi một phút rồi hỏi tiếp.")
    q.append(now)


@app.post("/ask", dependencies=[Depends(rate_limit)])
async def ask(q: Ask):
    url = os.environ.get("N8N_WEBHOOK_URL")
    if not url:
        raise HTTPException(503, "Chưa cấu hình N8N_WEBHOOK_URL")
    try:
        r = await STATE["http"].post(url, json=q.model_dump())
    except httpx.HTTPError:
        raise HTTPException(502, "Không kết nối được dịch vụ trả lời")
    if r.status_code != 200:
        raise HTTPException(502, f"Dịch vụ trả lời lỗi {r.status_code}")
    return r.json()
