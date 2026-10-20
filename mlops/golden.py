"""Bộ câu hỏi chuẩn (golden set) = qrels.parquet do notebook benchmark xuất ra (494 câu test + câu dev).

Mỗi dòng: item_id, split (dev|test), lv_name (lĩnh vực), query (câu hỏi người dân), gold (khoá Điều đúng).
Image production không có pandas -> đổi sang JSON Lines nén một lần, rồi đẩy lên MinIO:

    python -m mlops.golden benchmark/qrels.parquet golden_v1.jsonl.gz
    mc cp golden_v1.jsonl.gz <minio>/legal-rag/eval/golden_v1.jsonl.gz

Thêm câu mới (vd. từ phản hồi 👎 đã gán nhãn, xem mlops/feedback.py) -> lưu thành golden_v2 (không sửa đè v1:
kết quả cũ trong MLflow phải so được với nhau) rồi đổi quality.golden trong values.yaml.
"""
from __future__ import annotations

import gzip
import json
import sys

import numpy as np

FIELDS = ("item_id", "split", "lv_name", "query", "gold")


def load(path: str, split: str | None = "test", sample: int = 0, seed: int = 0) -> list[dict]:
    if path.endswith(".parquet"):
        import pandas as pd
        rows = pd.read_parquet(path).to_dict("records")
    else:
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    items = [{"item_id": str(r["item_id"]), "split": r.get("split", "test"), "lv_name": r.get("lv_name") or "",
              "query": str(r["query"]), "gold": [str(g) for g in r["gold"]]} for r in rows]
    items = [r for r in items if r["gold"] and (split is None or r["split"] == split)]
    if sample and sample < len(items):            # lấy mẫu cố định (seed) để các lần chạy so được với nhau
        idx = np.sort(np.random.default_rng(seed).choice(len(items), sample, replace=False))
        items = [items[i] for i in idx]
    return items


def export(src: str, dst: str):
    items = load(src, split=None)
    with gzip.open(dst, "wt", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps({k: r[k] for k in FIELDS}, ensure_ascii=False) + "\n")
    n = {s: sum(r["split"] == s for r in items) for s in ("dev", "test")}
    print(f"{dst}: {len(items)} câu ({n['test']} test, {n['dev']} dev)")


if __name__ == "__main__":
    export(*sys.argv[1:3])
