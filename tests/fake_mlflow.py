"""MLflow giả trong bộ nhớ (httpx.MockTransport) — đủ các API REST mà mlops/mlflow_rest.py dùng."""
import json
import re

import httpx


class FakeMlflow:
    def __init__(self):
        self.exps, self.runs, self.models, self.artifacts = {}, {}, {}, {}

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self.handle))

    def _err(self, code, status=404):
        return httpx.Response(status, json={"error_code": code, "message": code})

    def handle(self, req: httpx.Request):
        path = req.url.path
        if path.startswith("/api/2.0/mlflow-artifacts/artifacts/"):
            self.artifacts[path.split("/artifacts/", 1)[1]] = req.content.decode()
            return httpx.Response(200, json={})
        ep = path.split("/api/2.0/mlflow/", 1)[1]
        q = dict(req.url.params)
        b = json.loads(req.content) if req.content else {}
        if ep == "experiments/get-by-name":
            e = self.exps.get(q["experiment_name"])
            return httpx.Response(200, json={"experiment": {"experiment_id": e}}) if e else self._err("RESOURCE_DOES_NOT_EXIST")
        if ep == "experiments/create":
            self.exps[b["name"]] = str(len(self.exps) + 1)
            return httpx.Response(200, json={"experiment_id": self.exps[b["name"]]})
        if ep == "runs/create":
            rid = f"run{len(self.runs) + 1}"
            info = {"run_id": rid, "experiment_id": b["experiment_id"], "start_time": b["start_time"],
                    "artifact_uri": f"mlflow-artifacts:/{b['experiment_id']}/{rid}/artifacts", "status": "RUNNING"}
            self.runs[rid] = {"info": info, "data": {"tags": list(b["tags"]), "params": [], "metrics": []}}
            return httpx.Response(200, json={"run": {"info": info}})
        if ep == "runs/log-batch":
            d = self.runs[b["run_id"]]["data"]
            d["params"] += b["params"]; d["metrics"] += b["metrics"]; d["tags"] += b["tags"]
            assert all(re.fullmatch(r"[\w./\- :]+", m["key"]) for m in b["metrics"]), "tên metric MLflow không hợp lệ"
            return httpx.Response(200, json={})
        if ep == "runs/set-tag":
            tags = self.runs[b["run_id"]]["data"]["tags"]
            tags[:] = [t for t in tags if t["key"] != b["key"]] + [{"key": b["key"], "value": b["value"]}]
            return httpx.Response(200, json={})
        if ep == "runs/update":
            self.runs[b["run_id"]]["info"]["status"] = b["status"]
            return httpx.Response(200, json={})
        if ep == "runs/get":
            return httpx.Response(200, json={"run": self.runs[q["run_id"]]})
        if ep == "runs/search":
            conds = re.findall(r"tags\.(\w+) = '([^']*)'", b["filter"])
            def ok(r):
                t = {x["key"]: x["value"] for x in r["data"]["tags"]}
                return r["info"]["experiment_id"] in b["experiment_ids"] and all(t.get(k) == v for k, v in conds)
            hits = sorted((r for r in self.runs.values() if ok(r)), key=lambda r: -r["info"]["start_time"])
            return httpx.Response(200, json={"runs": hits[: b["max_results"]]})
        if ep == "registered-models/create":
            if b["name"] in self.models:
                return self._err("RESOURCE_ALREADY_EXISTS", 400)
            self.models[b["name"]] = {"versions": [], "aliases": {}}
            return httpx.Response(200, json={})
        if ep == "model-versions/create":
            m = self.models[b["name"]]
            v = {"version": str(len(m["versions"]) + 1), "run_id": b["run_id"], "source": b["source"]}
            m["versions"].append(v)
            return httpx.Response(200, json={"model_version": v})
        if ep == "registered-models/alias" and req.method == "POST":
            self.models[b["name"]]["aliases"][b["alias"]] = b["version"]
            return httpx.Response(200, json={})
        if ep == "registered-models/alias":
            m = self.models.get(q["name"])
            v = m and m["aliases"].get(q["alias"])
            if not v:
                return self._err("RESOURCE_DOES_NOT_EXIST")
            return httpx.Response(200, json={"model_version": m["versions"][int(v) - 1]})
        return self._err("ENDPOINT_NOT_FOUND")
