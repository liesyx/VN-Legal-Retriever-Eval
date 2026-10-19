"""Production: dense + BM25 sparse trên Qdrant (chỉ đọc)."""
from __future__ import annotations

import logging
import time

from legal_core.search import Config, Searcher, SparseQuery, encode_query, fuse, load_embedder, load_reranker, qdrant_client

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
        log.info("Sẵn sàng: %s, rerank=%s (%.0fs)", collection, self.reranker is not None, time.time() - t0)

    def _hybrid(self, question):
        m, n = self.m, self.cfg.topn
        qi, qv = self.sparse.encode(question)
        reqs = [m.QueryRequest(query=encode_query(self.embedder, question).tolist(), using="dense", limit=n,
                               params=m.SearchParams(exact=True))]
        if len(qi):                                   # câu hỏi toàn từ lạ -> bỏ nhánh BM25
            reqs.append(m.QueryRequest(query=m.SparseVector(indices=qi.tolist(), values=qv.tolist()), using="bm25", limit=n))
        run = [([p.id for p in r.points], [p.score for p in r.points]) for r in self.client.query_batch_points(self.collection, requests=reqs)]
        run += [([], [])]
        return fuse(*run[0], *run[1], self.cfg)

    def _payload(self, ids):
        out = {}
        for r in (self.client.retrieve(self.collection, ids, with_payload=PAYLOAD) if ids else []):
            p = r.payload
            out[int(r.id)] = dict(embed_text=p["embed_text"], body=p["body"], key=p["key"],
                                  article=dict(citation=p["citation"], title=p["title"], effective_date=p.get("effective_date"),
                                               text=p["article_text"], len=p.get("article_len")))
        return out
