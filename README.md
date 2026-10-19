# legal_rag — API tìm Điều luật cho n8n

Hybrid 0,7·dense (AITeamVN/Vietnamese_Embedding) + 0,3·BM25 trên Qdrant, gộp theo Điều. Rerank tuỳ chọn (`RERANK=1`).
n8n lo phần LLM (viết lại câu hỏi, trả lời); API chỉ trả các Điều liên quan.

```
legal_core/   thuật toán dùng chung (tách từ, trộn điểm, Config)
pipeline/     build dữ liệu: dựng BM25, đẩy collection phapdien_vN lên Qdrant (chạy local/Colab)
prod/         API FastAPI chạy trong pod (chỉ đọc Qdrant)
tests/        test với dữ liệu giả, Qdrant trong bộ nhớ (Jenkins chạy)
web/          giao diện người dùng (HTML/CSS/JS thuần, busybox httpd), gọi backend /ask
helm/legal-rag  Dockerfile Jenkinsfile   CI/CD lên k3s
helm/observability   Jaeger · Prometheus · Loki · Grafana (cài tay 1 lần)
```

Luồng người dùng: trình duyệt (`legal.oci.lexsy.pro.vn`, pod web NodePort 30082) → backend `legal-api…/ask` (NodePort 30081) → n8n (LLM viết lại câu hỏi)
→ API `/search` (NodePort 30081) → Qdrant → n8n (LLM trả lời có trích dẫn) → trình duyệt.

## API
```
POST /ask      (giao diện web, CORS + giới hạn 10 câu/phút/IP) {"question", "k"} -> chuyển sang n8n, trả nguyên kết quả
POST /search   header X-API-Key   {"question": "...", "rerank_query": "...", "k": 8, "max_chars": 2500}
  -> {"rerank_query", "sources": [{"n", "key", "citation", "title", "effective_date", "text"}], "timings_ms"}
GET  /health   -> {"ok": true, "collection": "phapdien_v1", "rerank": false}
```

## Dữ liệu mới (local / Colab)
```bash
pip install -r pipeline/requirements.txt
export QDRANT_URL=https://qdrant.oci.lexsy.pro.vn QDRANT_API_KEY=...
python -m pipeline.publish_qdrant --index-dir <thư mục chunks.parquet + embeddings.f16.npy> --version 2
# -> collection phapdien_v2 + bm25_vocab_v2.json.gz ; bị ngắt thì chạy lại đúng lệnh
mc cp bm25_vocab_v2.json.gz <minio>/legal-rag/data/phapdien_v2/bm25_vocab.json.gz
# sửa helm/legal-rag/values.yaml: data.version: v2 -> push main -> Jenkins deploy
python -m pipeline.publish_qdrant --list            # xem các bản ; --delete phapdien_v1 để xoá bản cũ
```

## Triển khai
```
push main -> Jenkins: gitleaks · hadolint · helm lint · trivy config -> test -> build -> trivy image -> Harbor -> helm upgrade
pod: initContainer (rclone) tải model + bảng từ từ MinIO -> uvicorn
https://legal-api.oci.lexsy.pro.vn -> Coolify proxy -> http://10.0.0.47:30081 (NodePort) -> pod
```

| Ở đâu | Có gì |
|---|---|
| Qdrant | collection `phapdien_vN` (dense + sparse `bm25` + payload) |
| MinIO `legal-rag/` | `models/vietnamese-embedding/main/`, `data/phapdien_vN/bm25_vocab.json.gz` |
| Harbor `legal-rag/api` | image code + torch CPU |
| k8s ns `legal-rag` | Secret `harbor-pull`, `legal-rag-secrets` (QDRANT_API_KEY, API_TOKEN, S3_ACCESS_KEY, S3_SECRET_KEY) |
| Jenkins | credential `harbor-legal-rag` (robot jenkins), kubeconfig của `demo:jenkins-deployer` |
| `values.yaml` | `data.version`, `rerank.enabled`, URL Qdrant/MinIO, `service.nodePort`, `resources` |
| `legal_core/search.py` → `Config` | tham số thuật toán đã chốt bằng benchmark |

Cài một lần: namespace `legal-rag` + RoleBinding cho `demo:jenkins-deployer`; 2 Secret; project + 2 robot Harbor;
Multibranch Pipeline + webhook GitHub; route Coolify `legal-api.oci.lexsy.pro.vn -> http://10.0.0.47:30081`.

Vận hành:
```bash
kubectl -n legal-rag get pods
kubectl -n legal-rag logs deploy/legal-rag -c fetch-artifacts   # tải model
kubectl -n legal-rag logs deploy/legal-rag -c api
helm -n legal-rag rollback legal-rag                            # quay lại bản trước (image + phiên bản dữ liệu)
```

## Test
```bash
python -m tests.test_all              # dữ liệu giả
python -m tests.test_all <INDEX_DIR>  # dữ liệu thật
```
Đẩy/đẩy tiếp/chia lô · mã hoá câu hỏi == sklearn · Qdrant == file (rerank bật/tắt) · chặn bảng từ lệch bản · Điều dài · API.

## Ghi chú
- BM25 lưu sẵn trọng số (idf × tf bão hoà × chuẩn hoá độ dài) thành sparse vector → Qdrant chỉ nhân với vector câu hỏi,
  điểm đúng bằng BM25 của benchmark. Bảng từ (từ → số) cần cho phía câu hỏi.
- Điều dài hơn 20.000 ký tự: Qdrant chỉ lưu tiêu đề + độ dài; API trả tiêu đề + chunk khớp nhất.
- nDCG@10 benchmark: hybrid 0,428; hybrid + rerank tiêu đề 0,475 (tiêu đề do biên tập viên viết — cận trên).

## Quan sát (trace · log · metrics)
```
API ──OTLP trace──> Jaeger  (jaeger.observability:4318, lưu RAM)
    ──OTLP log────> Loki    (loki.observability:3100/otlp, kèm trace_id, giữ 7 ngày)
    <──scrape :9100─ Prometheus (+ CPU/RAM mọi container qua cAdvisor, giữ 7 ngày)
Grafana: dashboard "legal-rag API" · log -> bấm TraceID mở trace · trace -> "Logs for this span"
Một câu hỏi = 1 trace: POST /ask -> n8n -> POST /search -> search > embed · qdrant_query · fuse · qdrant_payload (· rerank)
```
Jenkins deploy chart này (stage "Deploy observability", release `obs`) ngay trước API. Admin cụm làm 1 lần trước
(ClusterRole cho Prometheus là quyền cấp cụm, SA Jenkins chỉ có quyền trong namespace):
```bash
kubectl apply -f helm/observability/cluster-rbac.yaml   # ns observability + RoleBinding admin cho demo:jenkins-deployer + ClusterRole Prometheus
kubectl -n observability create secret generic grafana-admin \
  --from-literal=admin-user=admin --from-literal=admin-password='<mật khẩu>'
kubectl auth can-i create deployments -n observability --as=system:serviceaccount:demo:jenkins-deployer   # yes
```
Route Coolify: `grafana.oci.lexsy.pro.vn -> http://10.0.0.47:30083`, `jaeger.oci.lexsy.pro.vn -> http://10.0.0.47:30084`
(Jaeger UI không có đăng nhập — có thể bỏ route này, xem trace ngay trong Grafana › Explore › Jaeger).

API bật/tắt ở `values.yaml › observability.enabled` (Jenkins deploy như cũ). Chưa cài chart observability thì API vẫn chạy, chỉ ghi cảnh báo không gửi được.
n8n: node "Tìm Điều luật" gửi lại header `traceparent` nhận từ webhook để /search nằm chung trace với /ask.

Metrics của API: `legal_http_request_seconds{route,status}`, `legal_search_stage_seconds{stage}`, `legal_ask_total{result}`,
`legal_n8n_seconds`, `legal_info{collection,rerank,version}`.
Tài nguyên: requests tổng ~150m CPU / ~700Mi RAM (máy 2 CPU); limits cho phép dùng thêm khi rảnh. Chỉ log của API vào Loki
(log pod khác: `kubectl logs`). Trace lưu RAM nên mất khi Jaeger restart.
