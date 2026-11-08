"""Đo cách tạo câu truy vấn ngay trên API /search đang chạy (không cần GPU, không cần index cục bộ).

    API_TOKEN=... python -m mlops.eval_api golden_v1.jsonl[.gz] [--url https://legal-api.../search] [--split test]

So sánh 3 cách gửi `question` cho /search trên cùng bộ câu hỏi chuẩn:
  full        câu hỏi đầy đủ của người dân                 (cách benchmark/gate đang đo)
  title       chỉ tiêu đề ngắn                              (giả lập câu LLM viết lại)
  title+full  "tiêu đề. câu hỏi đầy đủ"                     (cách n8n đang gửi từ 2026-11-08)
Tiêu đề lấy từ cột `title` của bộ câu hỏi (do người viết) -> là cận trên của tiêu đề LLM viết.
Kết quả 08/11/2026, 483 câu test golden_v1, API không rerank: 0,4296 / 0,4374 / 0,4665 (title+full hơn full p = 0,0001).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import os

import httpx
import numpy as np

from mlops.bench_lib import bootstrap_ci, paired_randomization, per_query_metrics

VARIANTS = {
    "full": lambda g: g["query"],
    "title": lambda g: g["title"] or g["query"],
    "title+full": lambda g: (g["title"] + ". " if g.get("title") else "") + g["query"],
}


def _read(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("--url", default=os.environ.get("SEARCH_URL", "https://legal-api.oci.lexsy.pro.vn/search"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    items = [g for g in _read(a.golden) if g.get("split") == a.split and g.get("gold")]
    cli = httpx.Client(timeout=120, headers={"X-API-Key": os.environ.get("API_TOKEN", "")})

    def one(make, g):
        body = {"question": make(g)[:8000], "rerank_query": "", "k": 10, "max_chars": 300}
        for i in range(3):
            try:
                r = cli.post(a.url, json=body)
                r.raise_for_status()
                return [s["key"] for s in r.json()["sources"]]
            except httpx.HTTPError:
                if i == 2:
                    raise

    nd = {}
    for name, make in VARIANTS.items():
        with cf.ThreadPoolExecutor(a.workers) as ex:
            ranked = list(ex.map(lambda g: one(make, g), items))
        m = [per_query_metrics(r, g["gold"]) for r, g in zip(ranked, items)]
        nd[name] = np.array([x["ndcg@10"] for x in m])
        lo, hi = bootstrap_ci(nd[name])
        print(f"{name:11s} nDCG@10 {nd[name].mean():.4f} [{lo:.3f}, {hi:.3f}]  "
              f"Hit@1 {np.mean([x['hit@1'] for x in m]):.3f}  Hit@10 {np.mean([x['hit@10'] for x in m]):.3f}", flush=True)
    for x, y in [("title", "full"), ("title+full", "full"), ("title+full", "title")]:
        print(f"{x} − {y}: {(nd[x] - nd[y]).mean():+.4f}  p = {paired_randomization(nd[x], nd[y]):.4f}")
    print(f"n = {len(items)} câu ({a.split})")


if __name__ == "__main__":
    main()
