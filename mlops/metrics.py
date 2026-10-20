"""Chỉ số truy xuất — chép nguyên per_query_metrics của legal_eval.py (benchmark) để số đo ở đây khớp số trong benchmark."""
from __future__ import annotations

from typing import Iterable

import numpy as np

KS = (1, 3, 5, 10)


def per_query_metrics(ranked: list, gold: Iterable) -> dict:
    """ranked: danh sách khoá Điều theo thứ tự; gold: các khoá Điều đúng (nhãn bạc của benchmark)."""
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
    first_any = np.flatnonzero(rel)
    m["first_rank"] = int(first_any[0] + 1) if first_any.size else 0   # 0 = không thấy trong danh sách
    return m


def bootstrap_ci(x, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, float)
    if not x.size:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n, x.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
