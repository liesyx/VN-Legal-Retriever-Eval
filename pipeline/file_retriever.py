"""Tìm trực tiếp trên file output notebook (luồng build): thử nghiệm, đánh giá, và đối chiếu với Qdrant trước khi đẩy."""
from __future__ import annotations

import time

from legal_core.search import Config, Searcher, encode_query, fuse, load_embedder, load_reranker
from pipeline.bm25 import topk_rows
from pipeline.index_data import article_keys, build_articles, build_bm25, load_index


class FileRetriever(Searcher):
    def __init__(self, cfg: Config, index_dir: str, embedder=None, reranker=None):
        self.cfg = cfg
        t0 = time.time()
        self.chunks, self.E = load_index(index_dir)
        self.texts = self.chunks["embed_text"].tolist()
        self.keys = article_keys(self.chunks)
        self.articles = build_articles(self.chunks, self.keys)
        self.bm25 = build_bm25(self.texts, cfg)
        self.embedder = embedder or load_embedder(cfg.emb_model, cfg.device)
        self.reranker = reranker or (load_reranker(cfg.rerank_model, cfg.device) if cfg.depth else None)
        print(f"FileRetriever: {len(self.chunks):,} chunk, {len(self.articles):,} Điều, {time.time()-t0:.0f}s")

    def _hybrid(self, question):
        q = encode_query(self.embedder, question)
        di, ds = topk_rows(q[None, :] @ self.E.T, self.cfg.topn)
        li, ls = self.bm25.search([question], k=self.cfg.topn)
        return fuse(di[0], ds[0], li[0], ls[0], self.cfg)

    def _payload(self, ids):
        return {i: dict(embed_text=self.texts[i], body=self.chunks.body.iat[i], key=self.keys[i],
                        article=self.articles[self.keys[i]]) for i in ids}
