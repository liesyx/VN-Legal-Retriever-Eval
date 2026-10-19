"""Kiểm tra không cần GPU/Qdrant thật (model giả, Qdrant trong bộ nhớ):
    python -m tests.test_all              # index giả tự sinh (CI)
    python -m tests.test_all <INDEX_DIR>  # hoặc dữ liệu thật (chunks.parquet + embeddings.f16.npy)
"""
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from dataclasses import replace

import numpy as np
from qdrant_client import QdrantClient, models

from legal_core.search import Config, SparseQuery
from pipeline import publish_qdrant as pub
from pipeline.file_retriever import FileRetriever
from prod.qdrant_retriever import QdrantRetriever


class FakeEmbedder:
    def __init__(self, dim): self.dim = dim
    def encode(self, texts, **kw):
        v = np.stack([np.random.default_rng(sum(map(ord, t))).normal(size=self.dim) for t in texts])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class FakeReranker:
    def predict(self, pairs, **kw):
        return np.array([len(set(q.lower().split()) & set(d.lower().split())) for q, d in pairs], float)


def main(index_dir):
    dim = np.load(f"{index_dir}/embeddings.f16.npy").shape[1]
    emb, rr, cfg = FakeEmbedder(dim), FakeReranker(), Config(topn=200)   # topn > số chunk: không hoà điểm ở mốc cắt
    client, tmp = QdrantClient(":memory:"), tempfile.mkdtemp()
    v1 = f"{tmp}/bm25_vocab_v1.json.gz"

    # --- build: chia lô theo dung lượng, bị ngắt rồi đẩy tiếp
    pub.BATCH, pub.BATCH_BYTES = 16, 60_000
    sent, up = [], client.upsert
    client.upsert = lambda c, pts, **kw: (sent.append(len(pts)), up(c, pts, **kw))[1]
    name = pub.publish(client, index_dir, 1, v1, cfg)
    client.upsert = up
    assert max(sent) <= 16 and min(sent[:-1]) < 16, sent
    n = client.count(name).count
    client.delete(name, models.PointIdsList(points=list(range(40, n))))
    assert pub.completeness(client, name) == (40, n)
    pub.publish(client, index_dir, 1, v1, cfg)
    assert pub.completeness(client, name) == (n, n)
    print(f"✓ chia lô ({min(sent)}–{max(sent)} point/lô) · đẩy tiếp sau khi bị ngắt")

    # --- mã hoá câu hỏi phía prod (không sklearn) == CountVectorizer của benchmark
    fr = FileRetriever(cfg, index_dir, emb, rr)
    sq = SparseQuery.load(v1)
    qs = ["Xin hỏi về " + " ".join(fr.texts[i].split()[3:12]) + " như thế nào?" for i in (1, 5, 17, 42, 77)]
    qs += ["câu hỏi toàn từ lạ zzz qqq", fr.texts[3] + " " + fr.texts[3]]
    for q in qs:
        Q = fr.bm25.vec.transform([q]).tocsr()
        ref = dict(zip(Q.indices.tolist(), ((cfg.k3 + 1) * Q.data / (cfg.k3 + Q.data)).tolist()))
        qi, qv = sq.encode(q)
        assert len(ref) == len(qi) and all(abs(ref[int(i)] - v) < 1e-6 for i, v in zip(qi, qv)), q
    print("✓ sparse câu hỏi == sklearn của benchmark")

    # --- prod (Qdrant) == build (file), có và không rerank
    qr = QdrantRetriever(cfg, name, v1, client=client, embedder=emb, reranker=rr)
    for q in qs:
        a, b = fr.search(q, "từ12 từ40", k=8), qr.search(q, "từ12 từ40", k=8)
        assert [s["key"] for s in a["sources"]] == [s["key"] for s in b["sources"]], q
        assert [s["text"] for s in a["sources"]] == [s["text"] for s in b["sources"]], q
        ia, sa = fr._hybrid(q); ib, sb = qr._hybrid(q)
        assert len(set(ia[:20]) & set(ib[:20])) >= 18 and np.allclose(sa[:20], sb[:20], atol=1e-4), q
    c0 = replace(cfg, depth=0)
    fr0, qr0 = FileRetriever(c0, index_dir, emb, None), QdrantRetriever(c0, name, v1, client=client, embedder=emb)
    assert fr0.reranker is None and qr0.reranker is None
    for q in qs:
        assert [s["key"] for s in fr0.search(q)["sources"]] == [s["key"] for s in qr0.search(q)["sources"]], q
    print(f"✓ Qdrant == file: {len(qs)} câu hỏi (rerank bật/tắt)")

    # --- bảng từ lệch phiên bản bị chặn
    name2 = pub.publish(client, index_dir, 2, f"{tmp}/bm25_vocab_v2.json.gz", cfg)
    try:
        QdrantRetriever(cfg, name2, v1, client=client, embedder=emb); raise AssertionError
    except RuntimeError as e:
        print("✓ chặn lệch phiên bản:", e)

    # --- Điều quá dài: chỉ lưu tiêu đề + độ dài, kết quả vẫn y hệt file
    old, pub.MAX_CHARS = pub.MAX_CHARS, 400
    name3 = pub.publish(client, index_dir, 3, f"{tmp}/v3.json.gz", cfg)
    pub.MAX_CHARS = old
    assert any(p.payload["article_len"] > 400 for p in client.scroll(name3, limit=10_000, with_payload=["article_len"])[0])
    qr3 = QdrantRetriever(cfg, name3, f"{tmp}/v3.json.gz", client=client, embedder=emb, reranker=rr)
    for q in qs:
        assert [s["text"] for s in fr.search(q, "từ12", 8, 400)["sources"]] == [s["text"] for s in qr3.search(q, "từ12", 8, 400)["sources"]]
    print("✓ Điều dài: không lưu toàn văn, kết quả y hệt")

    # --- API
    from fastapi.testclient import TestClient
    import prod.app as api

    @asynccontextmanager
    async def no_load(app):
        yield
    api.app.router.lifespan_context = no_load
    os.environ["API_TOKEN"] = "k3y"
    h = {"X-API-Key": "k3y"}
    with TestClient(api.app) as c:
        assert c.get("/health").status_code == 503
        api.STATE["r"] = qr
        assert c.post("/search", json={"question": qs[0]}).status_code == 401
        js = c.post("/search", headers=h, json={"question": qs[0], "rerank_query": "câu ngắn", "k": 3, "max_chars": 300}).json()
        assert js["rerank_query"] == "câu ngắn" and [s["n"] for s in js["sources"]] == [1, 2, 3]
        assert all(s["text"] and len(s["text"]) <= 320 for s in js["sources"])
        assert c.post("/search", headers=h, json={"question": "hi"}).status_code == 422
        assert c.get("/health").json() == {"ok": True, "collection": name, "rerank": True}

        # /ask: chuyển nguyên câu hỏi sang n8n, trả nguyên kết quả; giới hạn số lần hỏi theo IP
        import httpx
        seen = []
        def n8n(req):
            seen.append(req)
            return httpx.Response(200, json={"ok": True, "answer": "Trả lời [1]", "sources": []})
        api.STATE["http"] = httpx.AsyncClient(transport=httpx.MockTransport(n8n))
        os.environ.update(N8N_WEBHOOK_URL="https://n8n.test/webhook/legal-ask", ASK_PER_MINUTE="2")
        api.HITS.clear()
        ip = {"X-Forwarded-For": "1.2.3.4, 10.0.0.1"}
        r = c.post("/ask", headers=ip, json={"question": "Ly hôn thì ai nuôi con?", "k": 5})
        assert r.status_code == 200 and r.json()["answer"] == "Trả lời [1]"
        assert str(seen[0].url) == "https://n8n.test/webhook/legal-ask" and b"Ly h" in seen[0].content
        assert c.post("/ask", headers=ip, json={"question": "Câu hỏi thứ hai", "k": 5}).status_code == 200
        assert c.post("/ask", headers=ip, json={"question": "Câu hỏi thứ ba", "k": 5}).status_code == 429
        assert c.post("/ask", headers={"X-Forwarded-For": "5.6.7.8"}, json={"question": "IP khác vẫn hỏi được"}).status_code == 200
        api.STATE["http"] = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(500)))
        assert c.post("/ask", headers={"X-Forwarded-For": "9.9.9.9"}, json={"question": "n8n đang lỗi thì sao"}).status_code == 502
    print("✓ API (/search, /ask, giới hạn theo IP)")
    check_telemetry(api, qr, qs[0])
    print("ALL OK")


def check_telemetry(api, qr, question):
    """Trace /ask -> n8n -> /search nối chung một trace_id; log mang trace_id; metrics có số liệu."""
    import httpx
    from opentelemetry import trace
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from prometheus_client import generate_latest
    import logging

    spans, logs = InMemorySpanExporter(), InMemoryLogRecordExporter()
    tp = TracerProvider(); tp.add_span_processor(SimpleSpanProcessor(spans)); trace.set_tracer_provider(tp)
    lp = LoggerProvider(); lp.add_log_record_processor(SimpleLogRecordProcessor(logs)); set_logger_provider(lp)
    logging.getLogger().addHandler(LoggingHandler(logger_provider=lp))

    from fastapi.testclient import TestClient
    api.STATE["r"] = qr
    api.HITS.clear()
    os.environ.update(ASK_PER_MINUTE="100")
    with TestClient(api.app) as c:
        # n8n giả: nhận /ask từ backend rồi gọi lại /search kèm traceparent nhận được (như node HTTP trong n8n)
        async def n8n(req):
            assert req.headers.get("traceparent"), "backend phải gửi traceparent sang n8n"
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://api") as inner:
                r = await inner.post("/search", headers={"X-API-Key": "k3y", "traceparent": req.headers["traceparent"]},
                                     json={"question": question, "k": 3})
            assert r.status_code == 200, r.text
            return httpx.Response(200, json={"ok": True, "answer": "Trả lời [1]", "sources": r.json()["sources"]})
        api.STATE["http"] = httpx.AsyncClient(transport=httpx.MockTransport(n8n))
        assert c.post("/ask", headers={"X-Forwarded-For": "7.7.7.7"}, json={"question": question}).status_code == 200

    got = spans.get_finished_spans()
    names = {s.name for s in got}
    for need in ("POST /ask", "POST /search", "search", "embed", "qdrant_query", "fuse", "qdrant_payload", "rerank"):
        assert need in names, (need, names)
    ask_span = next(s for s in got if s.name == "POST /ask")
    search_span = next(s for s in got if s.name == "POST /search")
    assert len({s.context.trace_id for s in got}) == 1, "phải cùng một trace"
    assert search_span.context.trace_id == ask_span.context.trace_id
    ids = {s.context.span_id for s in got}
    assert search_span.parent and search_span.parent.span_id in ids   # /search nằm dưới /ask, không phải trace mới
    recs = [r.log_record for r in logs.get_finished_logs()]
    assert any("search k=3" in str(r.body) and r.trace_id == ask_span.context.trace_id for r in recs), [str(r.body) for r in recs]
    m = generate_latest().decode()
    assert 'legal_search_stage_seconds_count{stage="embed"}' in m and 'legal_ask_total{result="ok"}' in m
    assert 'legal_http_request_seconds_count{method="POST",route="/ask",status="200"}' in m
    print(f"✓ telemetry: {len(got)} span cùng 1 trace (/ask → n8n → /search → embed/qdrant/rerank), log có trace_id, metrics có số liệu")


if __name__ == "__main__":
    from tests.fixture import make_index
    main(sys.argv[1] if len(sys.argv) > 1 else make_index(tempfile.mkdtemp()))
