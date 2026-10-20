"""Đánh giá tìm Điều luật trên bộ câu hỏi chuẩn + ghi MLflow + cổng chất lượng.

  check    (Job deploy, bước 1) đã có lần đánh giá ĐẠT cho đúng cấu hình này chưa? có -> ghi file --skip, khỏi chạy lại
  run      (Job deploy, bước 2) chạy 494 câu test trên Qdrant với đúng cấu hình production, ghi MLflow, so với champion;
           không đạt -> exit 1 -> Helm dừng deploy, production giữ bản cũ
  promote  (sau khi deploy xong) gắn alias "champion" trong MLflow Model Registry cho phiên bản vừa lên production

Thí nghiệm (local / Colab, có dữ liệu file của notebook):
  python -m mlops.evaluate run --index-dir <thư mục chunks.parquet + embeddings> --golden benchmark/qrels.parquet \\
         --alpha 0.6 --no-gate --name "thử alpha 0.6"
  -> mỗi lần chạy là một run trong experiment MLflow, so sánh trên giao diện MLflow.

"Cấu hình" = vân tay (fingerprint) của: collection Qdrant (phapdien_vN — không bao giờ sửa đè), model embedding/rerank
(đường dẫn trên MinIO — đổi model thì đổi đường dẫn), tham số Config, bộ câu hỏi chuẩn. Đổi code mà không đổi những thứ
này thì không cần chạy lại (tests/test_all.py đã khoá: tìm trên Qdrant == tìm trên file == benchmark).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import unicodedata
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from legal_core.search import Config
from mlops import golden, report
from mlops.metrics import bootstrap_ci, per_query_metrics
from mlops.mlflow_rest import Mlflow, MlflowError, metrics_of, tags_of

log = logging.getLogger("legal_rag.eval")
env = os.environ.get
K = 10
MAIN = "ndcg_at_10"


# ---------- cấu hình & vân tay ----------
def describe(cfg: Config, collection: str) -> dict:
    c = {k: v for k, v in asdict(cfg).items() if k not in ("device", "emb_model", "rerank_model")}
    if not cfg.depth:
        c.pop("beta")
    return {"collection": collection,
            "emb_model": env("EMB_MODEL_REF") or cfg.emb_model,
            "rerank_model": (env("RERANK_MODEL_REF") or cfg.rerank_model) if cfg.depth else "-",
            "rerank": bool(cfg.depth), "golden": env("GOLDEN_REF") or env("GOLDEN", ""),
            "sample": int(env("EVAL_SAMPLE") or 0), **{f"cfg.{k}": v for k, v in c.items()}}


def fingerprint(desc: dict) -> str:
    return hashlib.sha256(json.dumps(desc, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return re.sub(r"[^a-z0-9]+", "_", "".join(ch for ch in s if not unicodedata.combining(ch)).lower()).strip("_") or "khac"


# ---------- đánh giá ----------
def evaluate(retriever, items: list[dict], k: int = K, progress: int = 50, rerank_field: str | None = None) -> tuple[list[dict], dict]:
    """rerank_field: lấy câu ngắn cho reranker từ trường này của bộ câu hỏi (vd. "title"); None = tự rút từ câu hỏi."""
    rows, t_all = [], time.perf_counter()
    for i, it in enumerate(items, 1):
        t0 = time.perf_counter()
        out = retriever.search(it["query"], it.get(rerank_field) if rerank_field else None, k, 300)
        dt = time.perf_counter() - t0
        ranked = [s["key"] for s in out["sources"]]
        m = per_query_metrics(ranked, it["gold"])
        rows.append({"item_id": it["item_id"], "lv_name": it["lv_name"], "seconds": round(dt, 3), **m,
                     "gold": "|".join(it["gold"]), "top3": "|".join(ranked[:3]),
                     "top1_title": (out["sources"][0]["citation"] + " — " + out["sources"][0]["title"])[:160] if out["sources"] else "",
                     "query": it["query"][:300]})
        if progress and i % progress == 0:
            nd = np.mean([r["ndcg@10"] for r in rows])
            log.info("%d/%d câu · nDCG@10 tạm %.4f · %.2f s/câu", i, len(items), nd, (time.perf_counter() - t_all) / i)
    s = summarize(rows)
    s["eval_minutes"] = round((time.perf_counter() - t_all) / 60, 2)
    return rows, s


def summarize(rows: list[dict]) -> dict:
    col = lambda k: np.array([r[k] for r in rows], float)
    nd = col("ndcg@10")
    lo, hi = bootstrap_ci(nd)
    s = {MAIN: nd.mean(), "ndcg_at_10_ci_low": lo, "ndcg_at_10_ci_high": hi, "mrr_at_10": col("mrr@10").mean(),
         **{f"hit_at_{k}": col(f"hit@{k}").mean() for k in (1, 3, 5, 10)},
         **{f"recall_at_{k}": col(f"recall@{k}").mean() for k in (5, 10)},
         "latency_p50_s": float(np.percentile(col("seconds"), 50)), "latency_p95_s": float(np.percentile(col("seconds"), 95)),
         "n_queries": len(rows)}
    by = {}
    for r in rows:
        by.setdefault(r["lv_name"], []).append(r["ndcg@10"])
    for lv, v in by.items():
        if len(v) >= 10:                                   # lĩnh vực quá ít câu thì số đo không có nghĩa
            s[f"ndcg_at_10/{_slug(lv)}"] = float(np.mean(v))
    return {k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v) for k, v in s.items()}


def decide(s: dict, champion: dict | None, min_ndcg: float, max_drop: float, golden: str | None = None) -> tuple[bool, str]:
    nd = s[MAIN]
    if nd < min_ndcg:
        return False, f"nDCG@10 {nd:.4f} < ngưỡng tối thiểu {min_ndcg:.4f}"
    if champion and golden and champion.get("params", {}).get("golden") not in (None, golden):
        # khác bộ câu hỏi chuẩn -> điểm không so được với nhau, chỉ xét ngưỡng tối thiểu
        return True, f"nDCG@10 {nd:.4f} ≥ {min_ndcg:.4f} (bộ câu hỏi mới {golden}, champion chấm trên {champion['params']['golden']})"
    if champion and MAIN in champion["metrics"]:
        ref = champion["metrics"][MAIN]
        if nd < ref - max_drop:
            return False, f"nDCG@10 {nd:.4f} tụt hơn {max_drop:.3f} so với champion v{champion['version']} ({ref:.4f})"
        return True, f"nDCG@10 {nd:.4f} (champion v{champion['version']}: {ref:.4f}, cho phép tụt {max_drop:.3f})"
    return True, f"nDCG@10 {nd:.4f} ≥ {min_ndcg:.4f} (chưa có champion)"


# ---------- MLflow ----------
def _ml() -> Mlflow | None:
    try:
        return Mlflow()
    except MlflowError as e:
        log.warning("không ghi MLflow: %s", e)
        return None


_RealMlflow = Mlflow
EXPERIMENT = lambda: env("MLFLOW_EXPERIMENT", "legal-rag-retrieval")
MODEL = lambda: env("MLFLOW_MODEL", "legal-rag-retriever")


def find_passed(ml: Mlflow, fp: str) -> dict | None:
    runs = ml.search_runs([ml.experiment_id(EXPERIMENT())], f"tags.fingerprint = '{fp}' and tags.gate = 'passed'")
    return runs[0] if runs else None


def champion(ml: Mlflow) -> dict | None:
    mv = ml.get_alias(MODEL(), "champion")
    if not mv:
        return None
    run = ml.get_run(mv["run_id"]) if mv.get("run_id") else {}
    return {"version": mv["version"], "run_id": mv.get("run_id"), "metrics": metrics_of(run),
            "params": {p["key"]: p["value"] for p in run.get("data", {}).get("params", [])}}


def cmd_check(a) -> int:
    from prod.qdrant_retriever import config_from_env
    desc = describe(config_from_env(), env("QDRANT_COLLECTION", "phapdien_v1"))
    fp = fingerprint(desc)
    ml = _ml()
    try:
        hit = find_passed(ml, fp) if ml else None
    except Exception as e:                                 # MLflow lỗi -> cứ đánh giá lại, không chặn vì MLflow
        log.warning("không tra được MLflow (%r) -> sẽ đánh giá lại", e)
        hit = None
    if hit:
        m = metrics_of(hit)
        log.info("✓ cấu hình %s đã ĐẠT ở run %s (nDCG@10 %.4f) -> bỏ qua đánh giá", fp, hit["info"]["run_id"], m.get(MAIN, 0))
        if a.skip:
            Path(a.skip).write_text(hit["info"]["run_id"])
    else:
        log.info("cấu hình %s chưa được đánh giá -> chạy bộ câu hỏi chuẩn: %s", fp, json.dumps(desc, ensure_ascii=False))
    return 0


def cmd_run(a) -> int:
    if a.skip and Path(a.skip).exists():
        log.info("bỏ qua: đã đạt ở run %s", Path(a.skip).read_text())
        return 0
    if a.index_dir:                                        # thí nghiệm trên file output notebook
        from pipeline.file_retriever import FileRetriever
        cfg = Config(device=env("DEVICE") or None, depth=a.depth if a.depth is not None else 0)
        cfg = replace(cfg, **({"alpha": a.alpha} if a.alpha is not None else {}))
        r, collection = FileRetriever(cfg, a.index_dir), f"file:{Path(a.index_dir).name}"
    else:
        from prod.qdrant_retriever import config_from_env, from_env
        cfg = config_from_env()
        cfg = replace(cfg, **({"alpha": a.alpha} if a.alpha is not None else {}), **({"depth": a.depth} if a.depth is not None else {}))
        r, collection = from_env(cfg), env("QDRANT_COLLECTION", "phapdien_v1")

    path = a.golden or env("GOLDEN")
    items = golden.load(path, a.split, int(env("EVAL_SAMPLE") or a.sample or 0))
    desc = describe(cfg, collection)
    fp = fingerprint(desc)
    log.info("đánh giá %d câu (%s) · %s · vân tay %s", len(items), a.split, collection, fp)
    rows, s = evaluate(r, items)

    gate_on = not a.no_gate
    ok, why, champ = gate(s, desc, a.min_ndcg, a.max_drop) if gate_on else (True, "thí nghiệm (không xét cổng)", None)
    record(desc, s, rows, "passed" if ok and gate_on else ("failed" if gate_on else "experiment"), why,
           name=a.name or f"{collection} · {'rerank' if cfg.depth else 'hybrid'} · {env('APP_VERSION', 'local')}",
           out=a.out, champion=champ, min_ndcg=a.min_ndcg,
           tags={"purpose": "gate" if gate_on else "experiment", "split": a.split,
                 **({"champion_version": champ["version"]} if champ else {})})
    return 0 if (ok or not gate_on) else 1


def gate(s: dict, desc: dict, min_ndcg: float = 0.40, max_drop: float = 0.01):
    """So với ngưỡng tối thiểu + champion trong MLflow (đọc được thì so, không thì chỉ xét ngưỡng)."""
    champ, ml = None, _ml()
    if ml:
        try:
            champ = champion(ml)
        except Exception as e:
            log.warning("không đọc được champion (%r) -> chỉ xét ngưỡng tối thiểu", e)
    ok, why = decide(s, champ, float(env("GATE_MIN_NDCG") or min_ndcg), float(env("GATE_MAX_DROP") or max_drop), str(desc["golden"]))
    log.info("%s %s", "✓ ĐẠT" if ok else "✗ KHÔNG ĐẠT", why)
    return ok, why, champ


def record(desc: dict, s: dict, rows: list, status: str, why: str, *, name: str, out: str | None = None,
           tags: dict | None = None, champion: dict | None = None, min_ndcg: float = 0.40) -> str | None:
    """Lưu summary.json + per_query.csv và ghi một run MLflow; status passed -> đăng ký phiên bản model (alias candidate).
    Lỗi MLflow chỉ cảnh báo. -> run_id hoặc None."""
    fp = fingerprint(desc)
    log.info("kết quả: %s", json.dumps({k: v for k, v in s.items() if "/" not in k}, ensure_ascii=False))
    o = Path(out or "eval_out")
    o.mkdir(parents=True, exist_ok=True)
    summary = json.dumps({"fingerprint": fp, "config": desc, "gate": status, "reason": why, "metrics": s}, ensure_ascii=False, indent=2)
    csv = to_csv(rows)
    (o / "summary.json").write_text(summary)
    (o / "per_query.csv").write_text(csv, encoding="utf-8")
    page = report.build(s, rows, desc, status, why, min_ndcg=float(env("GATE_MIN_NDCG") or min_ndcg),
                        champion=(champion or {}).get("metrics", {}).get(MAIN), title=name)
    (o / "report.html").write_text(page, encoding="utf-8")
    ml = _ml()
    if not ml:
        return None
    try:
        info = ml.create_run(ml.experiment_id(EXPERIMENT()), name,
                             {"fingerprint": fp, "gate": status, "gate_reason": why, "collection": desc["collection"],
                              "app_version": env("APP_VERSION", "local"), **(tags or {})})
        rid = info["run_id"]
        ml.log_batch(rid, params=desc, metrics=s)
        ml.log_text(info, "report.html", page)          # mở trong tab Artifacts: biểu đồ + giải thích
        ml.log_text(info, "summary.json", summary)
        ml.log_text(info, "per_query.csv", csv)
        ml.log_text(info, "failures.csv", to_csv([x for x in rows if x["hit@10"] == 0]))
        if status == "passed":
            v = ml.register(MODEL(), f"{info['artifact_uri']}/summary.json", rid,
                            {"fingerprint": fp, "collection": desc["collection"], MAIN: s[MAIN]}, why)
            ml.set_tag(rid, "model_version", v)
            ml.set_alias(MODEL(), "candidate", v)
            log.info("MLflow: run %s · model %s v%s (alias candidate)", rid, MODEL(), v)
        ml.finish(rid, status != "failed")
        return rid
    except Exception as e:                                 # MLflow lỗi không được làm hỏng kết quả đánh giá
        log.warning("ghi MLflow lỗi: %r", e)
        return None


def cmd_promote(a) -> int:
    """Chạy sau khi Helm deploy thành công. Không bao giờ trả lỗi (lỗi ở đây mà làm hỏng deploy thì vô lý)."""
    try:
        from prod.qdrant_retriever import config_from_env
        fp = fingerprint(describe(config_from_env(), env("QDRANT_COLLECTION", "phapdien_v1")))
        ml = Mlflow()
        run = find_passed(ml, fp)
        if not run:
            log.warning("không có run ĐẠT cho cấu hình %s (cổng đang tắt?) -> không đổi champion", fp)
            return 0
        t = tags_of(run)
        v = t.get("model_version")
        if v:
            ml.set_alias(MODEL(), "champion", v)
        ml.set_tag(run["info"]["run_id"], "deployed", time.strftime("%Y-%m-%d %H:%M:%S"))
        ml.set_tag(run["info"]["run_id"], "deployed_app_version", env("APP_VERSION", "?"))
        log.info("✓ champion = %s v%s (run %s, nDCG@10 %.4f)", MODEL(), v, run["info"]["run_id"], metrics_of(run).get(MAIN, 0))
    except Exception as e:
        log.warning("promote lỗi (bỏ qua): %r", e)
    return 0


def to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    q = lambda v: '"' + str(v).replace('"', '""') + '"' if any(c in str(v) for c in ',"\n') else str(v)
    return "\n".join([",".join(cols)] + [",".join(q(r[c]) for c in cols) for r in rows]) + "\n"


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--skip")
    r = sub.add_parser("run")
    r.add_argument("--skip", help="file do bước check ghi: có thì bỏ qua")
    r.add_argument("--golden", help="qrels.parquet hoặc golden_vN.jsonl.gz (mặc định env GOLDEN)")
    r.add_argument("--split", default="test")
    r.add_argument("--sample", type=int, default=0, help="chỉ lấy N câu (0 = tất cả)")
    r.add_argument("--index-dir", help="thí nghiệm trên file thay vì Qdrant")
    r.add_argument("--alpha", type=float); r.add_argument("--depth", type=int)
    r.add_argument("--min-ndcg", type=float, default=0.40); r.add_argument("--max-drop", type=float, default=0.01)
    r.add_argument("--no-gate", action="store_true", help="chỉ ghi thí nghiệm, không xét đạt/không đạt")
    r.add_argument("--name"); r.add_argument("--out")
    sub.add_parser("promote")
    a = p.parse_args(argv)
    return {"check": cmd_check, "run": cmd_run, "promote": cmd_promote}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
