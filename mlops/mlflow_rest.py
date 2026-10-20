"""Client MLflow tối thiểu qua REST API (không cài gói mlflow vào image production — nặng và hay đổi API).

Env chuẩn của MLflow: MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD (nếu server bật basic auth)
hoặc MLFLOW_TRACKING_TOKEN. Artifact chỉ ghi được khi server chạy `--serve-artifacts` (artifact_uri dạng mlflow-artifacts:/);
không ghi được thì bỏ qua — chỉ số, tham số, tag và model registry vẫn đủ.
"""
from __future__ import annotations

import logging
import os
import time

import httpx

log = logging.getLogger("legal_rag.mlflow")


class MlflowError(RuntimeError):
    def __init__(self, status, code, message):
        super().__init__(f"MLflow {status} {code}: {message}")
        self.status, self.code = status, code


def _kv(d: dict) -> list:
    return [{"key": str(k), "value": str(v)[:5000]} for k, v in d.items()]


class Mlflow:
    def __init__(self, uri: str | None = None, client: httpx.Client | None = None):
        env = os.environ.get
        self.uri = (uri or env("MLFLOW_TRACKING_URI") or "").strip().rstrip("/")
        if not self.uri:
            raise MlflowError(0, "NOT_CONFIGURED", "thiếu MLFLOW_TRACKING_URI")
        if "://" not in self.uri:                  # "mlflow.example.com" -> "https://mlflow.example.com"
            self.uri = "https://" + self.uri
        auth = (env("MLFLOW_TRACKING_USERNAME"), env("MLFLOW_TRACKING_PASSWORD"))
        headers = {"Authorization": f"Bearer {env('MLFLOW_TRACKING_TOKEN')}"} if env("MLFLOW_TRACKING_TOKEN") else {}
        self.http = client or httpx.Client(timeout=30, headers=headers, auth=auth if all(auth) else None)

    def _call(self, method: str, path: str, **kw) -> dict:
        r = self.http.request(method, f"{self.uri}/api/2.0/mlflow/{path}", **kw)
        if r.status_code >= 400:
            try:
                e = r.json()
            except ValueError:
                e = {"error_code": "HTTP", "message": r.text[:200]}
            raise MlflowError(r.status_code, e.get("error_code"), e.get("message"))
        return r.json() if r.content else {}

    # ---------- tracking ----------
    def experiment_id(self, name: str) -> str:
        try:
            return self._call("GET", "experiments/get-by-name", params={"experiment_name": name})["experiment"]["experiment_id"]
        except MlflowError as e:
            if e.code != "RESOURCE_DOES_NOT_EXIST":
                raise
        return self._call("POST", "experiments/create", json={"name": name})["experiment_id"]

    def create_run(self, experiment_id: str, name: str, tags: dict) -> dict:
        tags = {"mlflow.runName": name, "mlflow.source.type": "JOB", **tags}
        return self._call("POST", "runs/create", json={"experiment_id": experiment_id, "run_name": name,
                                                      "start_time": int(time.time() * 1000), "tags": _kv(tags)})["run"]["info"]

    def log_batch(self, run_id: str, params: dict | None = None, metrics: dict | None = None, tags: dict | None = None):
        ts = int(time.time() * 1000)
        m = [{"key": k, "value": float(v), "timestamp": ts, "step": 0} for k, v in (metrics or {}).items()]
        p, t = _kv(params or {}), _kv(tags or {})
        while m or p or t:                       # giới hạn của MLflow: 1000 metric / 100 param / 100 tag mỗi lần
            self._call("POST", "runs/log-batch", json={"run_id": run_id, "metrics": m[:1000], "params": p[:100], "tags": t[:100]})
            m, p, t = m[1000:], p[100:], t[100:]

    def set_tag(self, run_id: str, key: str, value):
        self._call("POST", "runs/set-tag", json={"run_id": run_id, "key": key, "value": str(value)})

    def finish(self, run_id: str, ok: bool = True):
        self._call("POST", "runs/update", json={"run_id": run_id, "status": "FINISHED" if ok else "FAILED",
                                               "end_time": int(time.time() * 1000)})

    def search_runs(self, experiment_ids: list, filter: str, max_results: int = 1) -> list:
        return self._call("POST", "runs/search", json={"experiment_ids": experiment_ids, "filter": filter,
                                                      "max_results": max_results,
                                                      "order_by": ["attributes.start_time DESC"]}).get("runs", [])

    def get_run(self, run_id: str) -> dict:
        return self._call("GET", "runs/get", params={"run_id": run_id})["run"]

    def log_text(self, info: dict, path: str, text: str) -> bool:
        """Ghi một file artifact (qua proxy --serve-artifacts). Lỗi thì bỏ qua."""
        uri = info.get("artifact_uri", "")
        if not uri.startswith("mlflow-artifacts:"):
            log.warning("bỏ qua artifact %s: artifact_uri=%s không đi qua server", path, uri)
            return False
        rel = uri.split(":", 1)[1].lstrip("/")
        try:
            r = self.http.put(f"{self.uri}/api/2.0/mlflow-artifacts/artifacts/{rel}/{path}", content=text.encode("utf-8"))
            r.raise_for_status()
            return True
        except httpx.HTTPError as e:
            log.warning("không ghi được artifact %s: %r", path, e)
            return False

    # ---------- model registry ----------
    def register(self, name: str, source: str, run_id: str, tags: dict, description: str = "") -> str:
        try:
            self._call("POST", "registered-models/create", json={"name": name, "description":
                       "Cấu hình tìm Điều luật (collection Qdrant + model + tham số) đã qua cổng chất lượng"})
        except MlflowError as e:
            if e.code != "RESOURCE_ALREADY_EXISTS":
                raise
        mv = self._call("POST", "model-versions/create", json={"name": name, "source": source, "run_id": run_id,
                                                              "tags": _kv(tags), "description": description})
        return str(mv["model_version"]["version"])

    def set_alias(self, name: str, alias: str, version: str):
        self._call("POST", "registered-models/alias", json={"name": name, "alias": alias, "version": str(version)})

    def get_alias(self, name: str, alias: str) -> dict | None:
        try:
            return self._call("GET", "registered-models/alias", params={"name": name, "alias": alias})["model_version"]
        except MlflowError as e:
            if e.code == "RESOURCE_DOES_NOT_EXIST" or e.status == 404:
                return None
            raise


def tags_of(run: dict) -> dict:
    return {t["key"]: t["value"] for t in run.get("data", {}).get("tags", [])}


def metrics_of(run: dict) -> dict:
    return {m["key"]: m["value"] for m in run.get("data", {}).get("metrics", [])}
