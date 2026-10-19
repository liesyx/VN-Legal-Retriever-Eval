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
    print("✓ API")
    print("ALL OK")


if __name__ == "__main__":
    from tests.fixture import make_index
    main(sys.argv[1] if len(sys.argv) > 1 else make_index(tempfile.mkdtemp()))
