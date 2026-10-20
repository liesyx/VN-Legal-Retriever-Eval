"""Gom phản hồi 👎 từ Loki thành file để gán nhãn rồi bổ sung bộ câu hỏi chuẩn (vòng phản hồi của MLOps).

    kubectl -n observability port-forward svc/loki 3100:3100 &
    python -m mlops.feedback --days 7 -o feedback_down.jsonl
    # mở file, điền "gold": ["<khoá Điều đúng>", ...] cho từng câu (khoá Điều lấy từ trường key của /search),
    # câu nào không gán được thì xoá; rồi ghép vào bộ câu hỏi chuẩn mới:
    python -m mlops.feedback merge golden_v1.jsonl.gz feedback_down.jsonl -o golden_v2.jsonl.gz

Mỗi dòng xuất ra: item_id, split="feedback", query, gold=[], shown (các Điều đã hiện cho người dùng), ask_trace_id
(mở trong Grafana/Jaeger để xem lại đúng lần hỏi đó), time.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time

import httpx


def pull(loki: str, days: float, vote: str = "down", limit: int = 5000) -> list[dict]:
    end = time.time_ns()
    q = f'{{service_name="legal-rag-api"}} | vote="{vote}"'
    r = httpx.get(f"{loki.rstrip('/')}/loki/api/v1/query_range", timeout=60,
                  params={"query": q, "start": end - int(days * 86400e9), "end": end, "limit": limit, "direction": "backward"})
    r.raise_for_status()
    out, seen = [], set()
    for stream in r.json()["data"]["result"]:
        md = stream["stream"]                      # structured metadata (vote, question, sources, ask_trace_id) nằm cùng label
        for ts, _line in stream["values"]:
            q = (md.get("question") or "").strip()
            if not q or q in seen:
                continue
            seen.add(q)
            tid = md.get("ask_trace_id") or ""
            out.append({"item_id": f"fb-{tid[:12] or ts}", "split": "feedback", "lv_name": "", "query": q, "gold": [],
                        "shown": [s for s in (md.get("sources") or "").split(",") if s], "comment": md.get("comment", ""),
                        "ask_trace_id": tid, "time": time.strftime("%Y-%m-%d %H:%M", time.localtime(int(ts) / 1e9))})
    return sorted(out, key=lambda x: x["time"])


def _read(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def merge(base: str, extra: str, dst: str):
    rows = _read(base)
    have = {r["query"] for r in rows}
    add = [{"item_id": r["item_id"], "split": "test", "lv_name": r.get("lv_name", ""), "query": r["query"], "gold": r["gold"]}
           for r in _read(extra) if r.get("gold") and r["query"] not in have]
    op = gzip.open if dst.endswith(".gz") else open
    with op(dst, "wt", encoding="utf-8") as f:
        for r in rows + add:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{dst}: {len(rows)} câu cũ + {len(add)} câu mới từ phản hồi (đã gán nhãn)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["merge"]:
        m = argparse.ArgumentParser()
        m.add_argument("base"); m.add_argument("extra"); m.add_argument("-o", required=True)
        a = m.parse_args(argv[1:])
        return merge(a.base, a.extra, a.o)
    p.add_argument("--loki", default="http://localhost:3100")
    p.add_argument("--days", type=float, default=7)
    p.add_argument("--vote", default="down", choices=["down", "up"])
    p.add_argument("-o", default="feedback_down.jsonl")
    a = p.parse_args(argv)
    rows = pull(a.loki, a.days, a.vote)
    with open(a.o, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{a.o}: {len(rows)} câu {a.vote} trong {a.days:g} ngày")


if __name__ == "__main__":
    main()
