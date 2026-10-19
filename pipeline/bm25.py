"""BM25 trên ma trận thưa — chép nguyên từ legal_eval.py của benchmark (chỉ dùng ở luồng build)."""
from __future__ import annotations

from typing import Callable

import numpy as np

class BM25:
    def __init__(self, analyzer: Callable[[str], list], k1: float = 1.2, b: float = 0.75,
                 max_df: float = 0.5, min_df: int = 1, qtf: str = "binary", k3: float = 8.0):
        from sklearn.feature_extraction.text import CountVectorizer
        self.vec = CountVectorizer(analyzer=analyzer, max_df=max_df, min_df=min_df, dtype=np.float32)
        self.k1, self.b = k1, b
        self.qtf, self.k3 = qtf, k3   # "binary": mỗi từ của câu hỏi tính 1 lần · "k3": tần suất bão hoà (k3+1)·qtf/(k3+qtf)

    def fit(self, texts: list[str]):
        self.X = self.vec.fit_transform(texts).tocsr().astype(np.float32)
        self.N = self.X.shape[0]
        self.df = np.bincount(self.X.indices, minlength=self.X.shape[1]).astype(np.float32)
        self.idf = np.log1p((self.N - self.df + 0.5) / (self.df + 0.5)).astype(np.float32)
        self.dl = np.asarray(self.X.sum(axis=1)).ravel()
        self.avgdl = float(self.dl.mean()) if self.N else 1.0
        self.set_params(self.k1, self.b)
        return self

    def set_params(self, k1: float, b: float):
        """Đổi k1, b mà không phải tách token lại."""
        self.k1, self.b = k1, b
        X = self.X
        rows = np.repeat(np.arange(self.N), np.diff(X.indptr))
        tf = X.data
        denom = tf + k1 * (1 - b + b * self.dl[rows] / self.avgdl)
        W = X.copy()
        W.data = (tf * (k1 + 1) / denom * self.idf[X.indices]).astype(np.float32)
        self.WT = W.T.tocsr()
        return self

    def search(self, queries: list[str], k: int = 200, batch: int = 128):
        idx_all, sc_all = [], []
        for s in range(0, len(queries), batch):
            Q = self.vec.transform(queries[s:s + batch]).tocsr()
            if self.qtf == "k3":
                Q.data = ((self.k3 + 1) * Q.data / (self.k3 + Q.data)).astype(np.float32)
            else:
                Q.data[:] = 1.0
            S = (Q @ self.WT).toarray()
            ix, sc = topk_rows(S, k)
            idx_all += list(ix); sc_all += list(sc)
        return idx_all, sc_all


def topk_rows(S: np.ndarray, k: int):
    k = min(k, S.shape[1])
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    rows = np.arange(S.shape[0])[:, None]
    order = np.argsort(-S[rows, part], axis=1)
    idx = part[rows, order]
    return idx, S[rows, idx]
