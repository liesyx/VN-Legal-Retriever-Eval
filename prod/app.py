"""Backend tra cứu luật.   uvicorn prod.app:app --port 8000   (chạy từ thư mục gốc)

Env: QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, BM25_VOCAB, EMB_MODEL, RERANK=0|1, RERANK_MODEL, DEVICE, API_TOKEN,
     N8N_WEBHOOK_URL, ALLOWED_ORIGINS (domain giao diện, cách nhau dấu phẩy), ASK_PER_MINUTE
     + biến telemetry (OTEL_EXPORTER_OTLP_*_ENDPOINT, METRICS_PORT): xem prod/telemetry.py
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
from opentelemetry import propagate
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from legal_core.search import MAX_CHARS, Config
from prod import telemetry as tm
from prod.qdrant_retriever import QdrantRetriever

log = logging.getLogger("legal_rag")
STATE: dict = {}
LOCK = threading.Lock()          # một model dùng chung -> mỗi lúc một truy vấn


@asynccontextmanager
async def lifespan(app):
    tm.start_metrics_server()
    env = os.environ.get
    cfg = Config(device=env("DEVICE") or None, emb_model=env("EMB_MODEL") or Config.emb_model,
                 rerank_model=env("RERANK_MODEL") or Config.rerank_model, depth=Config.depth if env("RERANK") == "1" else 0)
    r = QdrantRetriever(cfg, collection=env("QDRANT_COLLECTION", "phapdien_v1"), vocab_path=env("BM25_VOCAB", "bm25_vocab.json.gz"),
                        url=os.environ["QDRANT_URL"], api_key=env("QDRANT_API_KEY"))
    r.search("Quyền và nghĩa vụ của người lao động", k=1)     # khởi động nóng: câu đầu tiên không bị chậm
    STATE["r"] = r
    tm.INFO.labels(r.collection, str(r.reranker is not None).lower(), os.environ.get("APP_VERSION", "dev")).set(1)
    STATE["http"] = httpx.AsyncClient(timeout=httpx.Timeout(150, connect=10))
    yield
    await STATE["http"].aclose()
    STATE.clear()


app = FastAPI(title="Tìm Điều luật", lifespan=lifespan)
tm.setup(app)
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
    with tm.stage("search", k=q.k, rewritten=bool(q.rerank_query)) as sp:
        with LOCK:
            out = STATE["r"].search(q.question, q.rerank_query, q.k, q.max_chars)
        sp.set_attribute("search.sources", [s["citation"] for s in out["sources"]])
    log.info("search k=%d -> %s", q.k, ", ".join(s["citation"] for s in out["sources"][:3]))
    return out


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
        tm.ASK.labels("rate_limited").inc()
        raise HTTPException(429, "Bạn hỏi quá nhanh. Đợi một phút rồi hỏi tiếp.")
    q.append(now)


@app.post("/ask", dependencies=[Depends(rate_limit)])
async def ask(q: Ask):
    url = os.environ.get("N8N_WEBHOOK_URL")
    if not url:
        tm.ASK.labels("not_configured").inc()
        raise HTTPException(503, "Chưa cấu hình N8N_WEBHOOK_URL")
    t0 = time.perf_counter()
    try:
        hdr = {}
        propagate.inject(hdr)          # traceparent -> n8n gửi lại khi gọi /search => cùng một trace trong Jaeger
        r = await STATE["http"].post(url, json=q.model_dump(), headers=hdr)
    except httpx.HTTPError as e:
        tm.ASK.labels("n8n_unreachable").inc()
        log.error("n8n không kết nối được: %r", e)
        raise HTTPException(502, "Không kết nối được dịch vụ trả lời")
    tm.N8N.observe(time.perf_counter() - t0)
    if r.status_code != 200:
        tm.ASK.labels("n8n_error").inc()
        log.error("n8n trả %d", r.status_code)
        raise HTTPException(502, f"Dịch vụ trả lời lỗi {r.status_code}")
    data = r.json()
    tm.ASK.labels("ok" if data.get("ok") else "no_answer").inc()
    log.info("ask %d ký tự -> %s, %d nguồn, n8n %.1f s", len(q.question), "ok" if data.get("ok") else "không trả lời",
             len(data.get("sources") or []), time.perf_counter() - t0)
    return data
