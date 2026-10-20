"""Production: dense + BM25 sparse trên Qdrant (chỉ đọc)."""
from __future__ import annotations

import logging
import os
import time

from legal_core.search import Config, Searcher, SparseQuery, encode_query, fuse, load_embedder, load_reranker, qdrant_client
from prod.telemetry import DENSE_TOP1, stage

log = logging.getLogger("legal_rag")
PAYLOAD = ["key", "embed_text", "body", "citation", "title", "effective_date", "article_text", "article_len"]


class QdrantRetriever(Searcher):
    def __init__(self, cfg: Config, collection: str, vocab_path: str, url: str | None = None, api_key: str | None = None,
                 client=None, embedder=None, reranker=None):
        from qdrant_client import models
        t0 = time.time()
        self.cfg, self.m, self.collection = cfg, models, collection
        self.client = client or qdrant_client(url, api_key, timeout=30)
        if not self.client.collection_exists(collection):
            raise RuntimeError(f"Không có collection '{collection}' trên Qdrant")
        self.sparse = SparseQuery.load(vocab_path)
        if self.sparse.collection != collection:      # bảng từ phải cùng phiên bản dữ liệu
            raise RuntimeError(f"{vocab_path} thuộc '{self.sparse.collection}', không phải '{collection}'")
        self.embedder = embedder or load_embedder(cfg.emb_model, cfg.device)
        self.reranker = reranker or (load_reranker(cfg.rerank_model, cfg.device) if cfg.depth else None)
        if self.reranker is not None:
            self.reranker = _TracedReranker(self.reranker)
        log.info("Sẵn sàng: %s, rerank=%s (%.0fs)", collection, self.reranker is not None, time.time() - t0)

    def _hybrid(self, question):
        m, n = self.m, self.cfg.topn
        with stage("embed"):
            dense = encode_query(self.embedder, question).tolist()
            qi, qv = self.sparse.encode(question)
        reqs = [m.QueryRequest(query=dense, using="dense", limit=n, params=m.SearchParams(exact=True))]
        if len(qi):                                   # câu hỏi toàn từ lạ -> bỏ nhánh BM25
            reqs.append(m.QueryRequest(query=m.SparseVector(indices=qi.tolist(), values=qv.tolist()), using="bm25", limit=n))
        with stage("qdrant_query", collection=self.collection, bm25_terms=len(qi)):
            res = self.client.query_batch_points(self.collection, requests=reqs)
        run = [([p.id for p in r.points], [p.score for p in r.points]) for r in res] + [([], [])]
        if run[0][1]:
            DENSE_TOP1.observe(run[0][1][0])          # cosine của chunk gần nhất: thấp = câu hỏi lạ so với dữ liệu
        with stage("fuse"):
            return fuse(*run[0], *run[1], self.cfg)

    def _payload(self, ids):
        out = {}
        if not ids:
            return out
        with stage("qdrant_payload", n=len(ids)):
            recs = self.client.retrieve(self.collection, ids, with_payload=PAYLOAD)
        for r in recs:
            p = r.payload
            out[int(r.id)] = dict(embed_text=p["embed_text"], body=p["body"], key=p["key"],
                                  article=dict(citation=p["citation"], title=p["title"], effective_date=p.get("effective_date"),
                                               text=p["article_text"], len=p.get("article_len")))
        return out


def config_from_env() -> Config:
    env = os.environ.get
    return Config(device=env("DEVICE") or None, emb_model=env("EMB_MODEL") or Config.emb_model,
                  rerank_model=env("RERANK_MODEL") or Config.rerank_model, depth=Config.depth if env("RERANK") == "1" else 0)


def from_env(cfg: Config | None = None) -> "QdrantRetriever":
    """Cùng cách dựng cho API và Job đánh giá -> đánh giá đúng cấu hình đang chạy production."""
    env = os.environ.get
    return QdrantRetriever(cfg or config_from_env(), collection=env("QDRANT_COLLECTION", "phapdien_v1"),
                           vocab_path=env("BM25_VOCAB", "bm25_vocab.json.gz"), url=os.environ["QDRANT_URL"],
                           api_key=env("QDRANT_API_KEY"))


class _TracedReranker:
    """Bọc reranker để mỗi lần chấm điểm có span "rerank" + thời gian trong Prometheus."""
    def __init__(self, inner):
        self.inner = inner

    def predict(self, pairs, **kw):
        with stage("rerank", pairs=len(pairs)):
            return self.inner.predict(pairs, **kw)
