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
    check_mlops(fr, qr, name, tmp)
    check_build_tools(fr)
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
    # 👍/👎: /ask trả trace_id -> giao diện gửi lại cùng phiếu; phiếu thành metric + log có vote/ask_trace_id
    with TestClient(api.app) as c:
        api.STATE["http"] = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"ok": True, "answer": "x [1]", "sources": []})))
        tid = c.post("/ask", headers={"X-Forwarded-For": "8.8.8.8"}, json={"question": question}).json()["trace_id"]
        assert len(tid) == 32
        fb = {"vote": "down", "trace_id": tid, "question": question, "sources": ["k1", "k2"]}
        assert c.post("/feedback", headers={"X-Forwarded-For": "8.8.8.8"}, json=fb).status_code == 204
        assert c.post("/feedback", json={**fb, "vote": "meh"}).status_code == 422
        assert c.post("/feedback", json={**fb, "trace_id": "không-phải-hex"}).status_code == 422
    rec = next(r.log_record for r in logs.get_finished_logs() if str(r.log_record.body).startswith("feedback down"))
    assert rec.attributes["vote"] == "down" and rec.attributes["ask_trace_id"] == tid and rec.attributes["sources"] == "k1,k2"
    m = generate_latest().decode()
    assert 'legal_feedback_total{vote="down"} 1.0' in m and "legal_dense_top1_score_count" in m
    print(f"✓ telemetry: {len(got)} span cùng 1 trace (/ask → n8n → /search → embed/qdrant/rerank), log có trace_id, metrics có số liệu")


def check_mlops(fr, qr, collection, tmp):
    """Đánh giá trên bộ câu hỏi chuẩn + cổng chất lượng + MLflow (giả) — đúng luồng Job deploy: check -> run -> promote."""
    import gzip, json
    from mlops import evaluate as ev, golden
    from mlops.metrics import per_query_metrics
    from tests.fake_mlflow import FakeMlflow
    import prod.qdrant_retriever as pq

    # chỉ số khớp công thức benchmark (tính tay)
    m = per_query_metrics(["a", "x", "b"], ["a", "b"])
    assert abs(m["ndcg@10"] - (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))) < 1e-12 and m["mrr@10"] == 1 and m["recall@3"] == 1
    assert per_query_metrics(["x", "a"], ["a"])["mrr@10"] == 0.5 and per_query_metrics(["x"], ["a"])["first_rank"] == 0

    # bộ câu hỏi chuẩn giả: câu hỏi lấy từ một chunk, đáp án = Điều chứa chunk đó
    rows = [{"item_id": f"q{i}", "split": "test" if i % 4 else "dev", "lv_name": ["Lao động", "Dân sự"][i % 2],
             "query": "Cho hỏi " + " ".join(fr.texts[i].split()[2:14]), "gold": [fr.keys[i]]} for i in range(0, 120, 3)]
    gpath = f"{tmp}/golden_v1.jsonl.gz"
    with gzip.open(gpath, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    items = golden.load(gpath)
    assert len(items) == 30 and all(r["split"] == "test" for r in items)
    assert [r["item_id"] for r in golden.load(gpath, sample=10)] == [r["item_id"] for r in golden.load(gpath, sample=10)]

    rq, sq = ev.evaluate(qr, items, progress=0)
    rf, sf = ev.evaluate(fr, items, progress=0)
    assert [r["ndcg@10"] for r in rq] == [r["ndcg@10"] for r in rf], "Qdrant và file phải cho cùng điểm"
    assert sq["n_queries"] == 30 and 0 < sq[ev.MAIN] <= 1 and "ndcg_at_10/lao_dong" in sq

    # luồng Job: check -> run -> promote, rồi lần deploy sau (chỉ đổi code) được bỏ qua
    fake = FakeMlflow()
    ev.Mlflow = lambda: ev.__dict__["_RealMlflow"](uri="http://mlflow.test", client=fake.client())
    os.environ.update(QDRANT_COLLECTION=collection, GOLDEN=gpath, GOLDEN_REF="eval/golden_v1.jsonl.gz", APP_VERSION="7-abc",
                      MLFLOW_TRACKING_URI="http://mlflow.test", RERANK="1", EMB_MODEL_REF="models/emb/main")
    pq.from_env = lambda cfg=None: qr
    skip = f"{tmp}/skip"
    run = lambda *a: ev.main(["run", "--skip", skip, "--out", f"{tmp}/eval", *a])
    assert ev.main(["check", "--skip", skip]) == 0 and not os.path.exists(skip)
    assert run("--min-ndcg", "0.05") == 0
    r1 = next(iter(fake.runs.values()))
    t1 = {t["key"]: t["value"] for t in r1["data"]["tags"]}
    assert t1["gate"] == "passed" and t1["model_version"] == "1" and fake.models["legal-rag-retriever"]["aliases"] == {"candidate": "1"}
    assert any(a.endswith("per_query.csv") for a in fake.artifacts) and json.load(open(f"{tmp}/eval/summary.json"))["gate"] == "passed"
    assert ev.main(["promote"]) == 0 and fake.models["legal-rag-retriever"]["aliases"]["champion"] == "1"
    assert ev.main(["check", "--skip", skip]) == 0 and open(skip).read() == r1["info"]["run_id"]
    assert run() == 0 and len(fake.runs) == 1, "đã đạt -> không chạy lại"
    os.remove(skip)

    # cấu hình mới (vd. đổi model) kém hơn champion quá 0,01 -> KHÔNG ĐẠT -> Job exit 1 -> Helm dừng deploy
    os.environ["EMB_MODEL_REF"] = "models/emb/v2"
    ch = fake.runs[r1["info"]["run_id"]]["data"]["metrics"]
    for x in ch:
        if x["key"] == ev.MAIN:
            x["value"] = sq[ev.MAIN] + 0.05
    assert run("--min-ndcg", "0.05") == 1
    r2 = list(fake.runs.values())[-1]
    assert {t["key"]: t["value"] for t in r2["data"]["tags"]}["gate"] == "failed" and r2["info"]["status"] == "FAILED"
    assert fake.models["legal-rag-retriever"]["aliases"]["champion"] == "1"
    assert ev.main(["promote"]) == 0                         # không có run đạt -> không đổi champion, không báo lỗi
    # bộ câu hỏi chuẩn mới: điểm không so được với champion (chấm trên bộ cũ) -> chỉ xét ngưỡng tối thiểu
    os.environ["GOLDEN_REF"] = "eval/golden_v2.jsonl.gz"
    assert run("--min-ndcg", "0.05") == 0
    # thí nghiệm: không xét cổng, luôn exit 0
    assert run("--no-gate", "--alpha", "0.5", "--name", "thử alpha") == 0
    assert {t["key"]: t["value"] for t in list(fake.runs.values())[-1]["data"]["tags"]}["purpose"] == "experiment"
    # MLflow sập: vẫn đánh giá và quyết định theo ngưỡng tối thiểu
    ev.Mlflow = lambda: (_ for _ in ()).throw(ev.MlflowError(0, "DOWN", "down"))
    assert run("--min-ndcg", "0.05") == 0 and run("--min-ndcg", "0.99") == 1
    print(f"✓ MLOps: nDCG@10 {sq[ev.MAIN]:.3f} (Qdrant == file) · check → run → promote · đã đạt thì bỏ qua · tụt so với champion thì chặn")


def check_build_tools(fr):
    """Chunker + tạo nhãn bộ câu hỏi chuẩn (code dùng trong notebook Kaggle) — chép self-test của 2 notebook cũ."""
    import pandas as pd
    from pipeline import chunker as lc
    from mlops import labels as lb
    c = lc.WhitespaceCounter()
    assert lc.split_khoan("Thực hiện theo Điều 8. Nghị định số 38/2006/NĐ-CP. 1.1. Hàng tháng.") is None
    pre, items = lc.split_khoan("Gồm: 1. Khoản một; 2. Khoản hai; 3. Khoản ba.")
    assert pre == "Gồm:" and [l for l, _ in items] == ["1", "2", "3"]
    long = "Trong Luật này: " + " ".join(f"{i}. Thuật ngữ {i} " + "giải thích " * 30 + "." for i in range(1, 6))
    ch = lc.chunk_article(long, "Điều 3. Giải thích từ ngữ", "Đề mục", "", counter=c, max_tokens=80)
    assert all(x.tokens <= 80 for x in ch) and ch[0].khoan_from == "1"
    m = lc.parse_source_note("(Điều 26 Bộ luật số 45/2019/QH14 Lao động ngày 20/11/2019 của Quốc hội, có hiệu lực thi hành kể từ ngày 01/01/2021)")
    assert (m["origin_doc_type"], m["origin_so_hieu"], m["origin_article"], m["origin_effective_date"]) == ("Bộ luật", "45/2019/QH14", "Điều 26", "2021-01-01")

    docs, by = lb.build_catalog(["(Điều 1 Bộ luật số 91/2015/QH13 Dân sự ngày 24/11/2015 của Quốc hội, có hiệu lực thi hành kể từ ngày 01/01/2017)"],
                                ["91/2015/QH13"], ["Bộ luật"], ["2017-01-01"])
    r = lb.CitationResolver(docs, by)
    assert r.gold_keys("Theo khoản 1 Điều 241 và Ðiều 242 Bộ luật dân sự năm 2015", None,
                       {"91/2015/qh13|241", "91/2015/qh13|242"})[0] == ["91/2015/qh13|241", "91/2015/qh13|242"]
    assert r.gold_keys("Điều 129 Bộ luật dân sự", "2012-01-01", {"91/2015/qh13|129"})[0] == []
    V = lb.ContentValidator({"91/2015/qh13|129": "Điều 9.1.LQ.129. Hiệu lực của giao dịch dân sự vô hiệu"},
                            {"91/2015/qh13|129": "Giao dịch dân sự vô hiệu không làm phát sinh, thay đổi, chấm dứt quyền"})
    assert r.gold_keys("Theo Điều 129 Bộ luật dân sự năm 2015 thì người trộm cắp tài sản bị xử lý", None, {"91/2015/qh13|129"}, validator=V)[0] == []
    assert r.gold_keys("Theo Điều 129 Bộ luật dân sự năm 2015 về hiệu lực của giao dịch dân sự vô hiệu", None,
                       {"91/2015/qh13|129"}, validator=V)[0] == ["91/2015/qh13|129"]
    assert lb.to_iso_date("05/03/2020") == "2020-03-05" and lb.to_iso_date(None) is None

    # cả quy trình trên index giả: câu trả lời chép nội dung Điều + ghi số hiệu -> có đáp án; câu lạc đề -> không
    ch = fr.chunks
    rows = []
    for i in range(0, len(ch), 2):
        c0 = ch.iloc[i]
        rows.append(dict(item_id=f"a{i}", lv_name="Lao động", title="t", question_text="Xin hỏi luật sư về trường hợp của tôi như sau ạ?",
                         date_sent="2022-05-01", answer_text=f"Căn cứ {c0.origin_article} {c0.origin_doc_type} số {c0.origin_so_hieu}: {c0.body}"))
    rows.append(dict(item_id="x", lv_name="", title="t", question_text="Câu hỏi không có căn cứ nào cả, đúng không ạ?",
                     date_sent="2022-05-01", answer_text="Bạn nên liên hệ cơ quan có thẩm quyền."))
    Q, st = lb.build_golden(ch, pd.DataFrame(rows), log=lambda *a: None)
    assert "x" not in set(Q.item_id) and len(Q) >= 0.9 * (len(rows) - 1), st
    key_of = dict(zip(ch.index, fr.keys))
    assert all(g == [key_of[int(it[1:])]] for it, g in zip(Q.item_id, Q.gold)), "đáp án phải đúng Điều đã chép"
    print(f"✓ chunker v3 + tạo nhãn bộ câu hỏi chuẩn ({len(Q)}/{len(rows)} câu có đáp án đúng Điều)")


if __name__ == "__main__":
    from tests.fixture import make_index
    main(sys.argv[1] if len(sys.argv) > 1 else make_index(tempfile.mkdtemp()))
