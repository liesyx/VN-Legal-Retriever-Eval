"""Chuẩn hoá min-max và trộn có trọng số — chép nguyên từ legal_eval.py của benchmark."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

def _minmax(s, floor: float = 0.0):
    """Min-max về [floor, 1]. floor > 0: phần tử cuối danh sách vẫn hơn phần tử KHÔNG có trong danh sách (= 0)."""
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0:
        return s
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return np.ones_like(s)
    return floor + (1.0 - floor) * (s - lo) / (hi - lo)


def weighted_fusion(run_a, run_b, alpha: float, topn: int = 200, floor: float = 0.0):
    """alpha * dense + (1-alpha) * lexical, sau khi min-max mỗi danh sách theo từng câu (về [floor, 1])."""
    out_i, out_s = [], []
    for (ia, sa), (ib, sb) in zip(zip(*run_a), zip(*run_b)):
        sc = defaultdict(float)
        for i, v in zip(ia, _minmax(sa, floor)):
            sc[int(i)] += alpha * v
        for i, v in zip(ib, _minmax(sb, floor)):
            sc[int(i)] += (1 - alpha) * v
        best = sorted(sc.items(), key=lambda x: -x[1])[:topn]
        out_i.append(np.array([b[0] for b in best])); out_s.append(np.array([b[1] for b in best]))
    return out_i, out_s
