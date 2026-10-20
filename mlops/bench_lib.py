"""Thư viện cho notebook benchmark (kaggle/legal_rag_benchmark.ipynb) = legal_eval.py (legal-eval-v5) của benchmark cũ.

Phần dùng chung với production/build (chuẩn hoá chữ, BM25, trộn điểm, tạo nhãn) KHÔNG chép lại mà lấy từ
legal_core / pipeline / mlops.labels — đã đối chiếu từng hàm, giống hệt bản trong notebook cũ. Ở đây chỉ còn phần
riêng của benchmark: TF-IDF, sparse bge-m3, RRF, gộp chunk -> Điều, đủ bộ chỉ số, kiểm định thống kê.
Notebook dùng: `from mlops import bench_lib as le` (giữ tên `le` như notebook cũ).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Callable, Iterable, Optional

import numpy as np

from legal_core.fusion import _minmax, weighted_fusion                       # noqa: F401
from legal_core.text import article_key, dieu_number, norm, syllable_bigrams, syllables  # noqa: F401
from mlops.labels import (EVAL_VERSION, Citation, CitationResolver, ContentValidator, DocInfo,  # noqa: F401
                          build_catalog, calibrate_validator, to_iso_date)
from pipeline.bm25 import BM25, topk_rows                                    # noqa: F401

_WORD = re.compile(r"\w+")


def make_pyvi_analyzer():
    from pyvi import ViTokenizer   # import lỗi -> người gọi tự bỏ qua biến thể này

    def words(text: str) -> list[str]:
        # Tách từ trên văn bản CÒN chữ hoa (pyvi dùng đặc trưng viết hoa), rồi mới chuẩn hoá từng từ
        t = unicodedata.normalize("NFC", text or "").replace("Ð", "Đ").replace("ð", "đ")
        return [norm(w) for w in ViTokenizer.tokenize(t).split() if _WORD.search(w)]
    return words


class TFIDF:
    """TF-IDF (sublinear tf) + cosine; cùng giao diện search() với BM25."""
    def __init__(self, analyzer: Callable[[str], list], max_df: float = 0.5, min_df: int = 1):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(analyzer=analyzer, max_df=max_df, min_df=min_df, sublinear_tf=True,
                                   norm="l2", dtype=np.float32)

    def fit(self, texts: list[str]):
        self.WT = self.vec.fit_transform(texts).T.tocsr()
        return self

    def search(self, queries: list[str], k: int = 200, batch: int = 128):
        idx_all, sc_all = [], []
        for s in range(0, len(queries), batch):
            S = (self.vec.transform(queries[s:s + batch]) @ self.WT).toarray()
            ix, sc = topk_rows(S, k)
            idx_all += list(ix); sc_all += list(sc)
        return idx_all, sc_all


class SparseWeightsIndex:
    """Chỉ mục cho learned sparse (vd. bge-m3 lexical weights): điểm = tổng q_w * d_w trên token chung."""
    def __init__(self, doc_csr):
        self.WT = doc_csr.T.tocsr()

    def search(self, query_csr, k: int = 200, batch: int = 256):
        idx_all, sc_all = [], []
        for s in range(0, query_csr.shape[0], batch):
            S = (query_csr[s:s + batch] @ self.WT).toarray()
            ix, sc = topk_rows(S, k)
            idx_all += list(ix); sc_all += list(sc)
        return idx_all, sc_all


def weights_to_csr(rows_ids: list, rows_w: list, vocab_size: int):
    """rows_ids/rows_w: list các mảng (token id, trọng số) -> CSR, gộp trùng id bằng max."""
    from scipy.sparse import csr_matrix
    indptr, indices, data = [0], [], []
    for ids, w in zip(rows_ids, rows_w):
        ids = np.asarray(ids, dtype=np.int64); w = np.asarray(w, dtype=np.float32)
        keep = w > 0
        ids, w = ids[keep], w[keep]
        if ids.size:
            order = np.lexsort((-w, ids))
            ids, w = ids[order], w[order]
            first = np.r_[True, ids[1:] != ids[:-1]]
            ids, w = ids[first], w[first]
        indices.append(ids); data.append(w); indptr.append(indptr[-1] + ids.size)
    return csr_matrix((np.concatenate(data) if data else np.zeros(0, np.float32),
                       np.concatenate(indices) if indices else np.zeros(0, np.int64), np.array(indptr)),
                      shape=(len(rows_ids), vocab_size))


def rrf(runs: list, k: int = 60, topn: int = 200):
    """runs: list các (idx_list, score_list); mỗi list dài bằng số câu hỏi."""
    out_i, out_s = [], []
    n_q = len(runs[0][0])
    for q in range(n_q):
        sc = defaultdict(float)
        for idx_list, _ in runs:
            for r, i in enumerate(idx_list[q], start=1):
                sc[int(i)] += 1.0 / (k + r)
        best = sorted(sc.items(), key=lambda x: -x[1])[:topn]
        out_i.append(np.array([b[0] for b in best])); out_s.append(np.array([b[1] for b in best]))
    return out_i, out_s


def chunks_to_articles(run, chunk_keys: np.ndarray, topn: int = 100) -> list[list]:
    """Chuyển xếp hạng chunk -> xếp hạng Điều (giữ lần xuất hiện đầu = điểm cao nhất)."""
    out = []
    for idx, _ in zip(*run):
        seen, lst = set(), []
        for i in idx:
            key = chunk_keys[int(i)]
            if key not in seen:
                seen.add(key); lst.append(key)
                if len(lst) == topn:
                    break
        out.append(lst)
    return out


KS = (1, 3, 5, 10, 20, 50, 100)


def per_query_metrics(ranked: list, gold: Iterable) -> dict:
    gold = set(gold)
    m = {}
    rel = np.array([1.0 if x in gold else 0.0 for x in ranked[:100]])
    for k in KS:
        hits = rel[:k].sum()
        m[f"hit@{k}"] = float(hits > 0)
        m[f"recall@{k}"] = float(hits / len(gold)) if gold else 0.0
    first = np.flatnonzero(rel[:10])
    m["mrr@10"] = float(1.0 / (first[0] + 1)) if first.size else 0.0
    disc = 1.0 / np.log2(np.arange(2, 12))
    dcg = float((rel[:10] * disc[:len(rel[:10])]).sum())
    idcg = float(disc[:min(len(gold), 10)].sum())
    m["ndcg@10"] = dcg / idcg if idcg else 0.0
    if rel.sum():
        prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
        m["ap@100"] = float((prec * rel).sum() / len(gold))
    else:
        m["ap@100"] = 0.0
    for k in (5, 10):
        m[f"p@{k}"] = float(rel[:k].sum() / k)
        p_k, r_k = m[f"p@{k}"], m[f"recall@{k}"]
        m[f"f2@{k}"] = float(5 * p_k * r_k / (4 * p_k + r_k)) if (p_k + r_k) else 0.0   # F2 (ALQAC dùng F2)
    R = len(gold)
    m["r_prec"] = float(rel[:R].sum() / R) if R else 0.0
    first_any = np.flatnonzero(rel)
    m["first_rank"] = int(first_any[0] + 1) if first_any.size else 0   # 0 = không thấy trong top-100
    return m


def holm(pvals: dict) -> dict:
    """Hiệu chỉnh Holm–Bonferroni cho nhiều phép so sánh."""
    items = sorted(pvals.items(), key=lambda x: x[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def paired_randomization(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> float:
    """p-value hai phía cho H0: hai phương pháp như nhau (đảo dấu ngẫu nhiên từng cặp)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if not d.size or np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (n + 1))


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n, x.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
