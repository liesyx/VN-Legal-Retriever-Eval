"""Ghi kết quả notebook benchmark (kaggle/legal_rag_benchmark.ipynb) vào MLflow, experiment "legal-rag-benchmark".

  run cha  "benchmark <index> (Kaggle)": dashboard.html + các file kết quả, nDCG@10 từng phương pháp,
           đường "nDCG trên dev theo α" và "theo β" (MLflow vẽ thành biểu đồ đường, step = α×10 / β×10)
  run con  mỗi phương pháp một run: đủ chỉ số trên tập test, khoảng tin cậy, ms/câu, p (Holm), đường Recall@k (step = k)
So sánh: mở experiment -> tích các run con -> biểu tượng biểu đồ -> chọn ndcg_at_10 / hit_at_1 / recall_at_k.
Tách riêng khỏi experiment cổng chất lượng (legal-rag-retrieval) nên chạy bao nhiêu lần cũng không ảnh hưởng deploy.
"""
from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path

from mlops.evaluate import _ml

log = logging.getLogger("legal_rag.bench")
key = lambda m: m.replace("@", "_at_")          # MLflow không nhận "@" trong tên chỉ số
safe = lambda n: re.sub(r"[^\w./\- :]", "_", n.replace("+", "__"))   # tên phương pháp có "+": không hợp lệ trong tên chỉ số


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def log_run(*, res, per_q, metrics, ks, ci, alpha_curve, rr_mix_rows, best, production, name, params, tags,
            out_dir, files=("dashboard.html", "results_test.csv", "results_by_domain.csv", "summary.json")) -> str | None:
    """res: DataFrame kết quả (cột method, family, metrics…, dev_ndcg@10, ms/query, p_holm, sig, desc)."""
    ml = _ml()
    if not ml:
        return None
    try:
        exp = ml.experiment_id(os.environ.get("MLFLOW_BENCH_EXPERIMENT", "legal-rag-benchmark"))
        parent = ml.create_run(exp, name, {"purpose": "benchmark", "best_method": best,
                                           "production_method": production or "-", **tags})
        pid = parent["run_id"]
        top = res.iloc[0]
        ml.log_batch(pid, params=params,
                     metrics={"best_ndcg_at_10": top[metrics[0]], "n_methods": len(res),
                              **{f"ndcg_at_10/{safe(r.method)}": r[metrics[0]] for _, r in res.iterrows()}})
        if alpha_curve:
            ml.log_series(pid, "dev_ndcg_vs_alpha", [(round(d["alpha"] * 10), d["score"]) for d in alpha_curve])
        for r in rr_mix_rows or []:
            ml.log_series(pid, f"dev_ndcg_vs_beta/{r['variant']}_top{r['depth']}", [(round(b * 10), v) for b, v in r["curve"]])
        for f in files:
            p = Path(out_dir) / f
            if p.exists():
                ml.log_text(parent, f, p.read_text(encoding="utf-8"))
        dev_col = next((c for c in res.columns if c.startswith("dev_")), None)
        for _, r in res.iterrows():
            child = ml.create_run(exp, r.method, {"mlflow.parentRunId": pid, "family": r.family, "sig": str(r.sig),
                                                  "production": str(r.method == production).lower(),
                                                  "best": str(r.method == best).lower(), **tags})
            m = {key(k): r[k] for k in metrics if _num(r[k]) is not None}
            m.update({k2: v for k2, v in {"ms_per_query": _num(r["ms/query"]), "p_holm": _num(r.p_holm),
                                          "dev_ndcg_at_10": _num(r[dev_col]) if dev_col else None}.items() if v is not None})
            for mk, (lo, hi) in (ci.get(r.method) or {}).items():
                m[f"{key(mk)}_ci_low"], m[f"{key(mk)}_ci_high"] = lo, hi
            ml.log_batch(child["run_id"], params={"method": r.method, "family": r.family, "desc": str(r.desc)[:500]}, metrics=m)
            if r.method in per_q:
                ml.log_series(child["run_id"], "recall_at_k", [(k, per_q[r.method][f"recall@{k}"].mean()) for k in ks])
            ml.finish(child["run_id"])
        ml.finish(pid)
        log.info("MLflow: run cha %s + %d run con (experiment legal-rag-benchmark)", pid, len(res))
        return pid
    except Exception as e:
        log.warning("ghi MLflow lỗi: %r", e)
        return None
