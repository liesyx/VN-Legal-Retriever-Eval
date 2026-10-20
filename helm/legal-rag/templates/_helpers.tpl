{{- define "legal-rag.labels" -}}
app.kubernetes.io/name: legal-rag
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/* initContainer tải model + bảng từ (+ bộ câu hỏi chuẩn cho Job đánh giá) từ MinIO.
     Gọi: include "legal-rag.fetch" (dict "root" . "golden" true|false) */}}
{{- define "legal-rag.fetch" -}}
{{- $v := .root.Values -}}
- name: fetch-artifacts           # tải model + bảng từ từ MinIO vào emptyDir (image app không chứa model)
  image: {{ $v.fetchImage }}
  command: ["sh", "-ec"]
  args:
    - |
      {{- if .golden }}
      if [ -f /work/skip ]; then echo "đã đánh giá cấu hình này -> không tải model"; exit 0; fi
      rclone copyto "s3:$S3_BUCKET/{{ $v.quality.golden }}" /models/golden.jsonl.gz
      {{- end }}
      rclone copy --stats-one-line "s3:$S3_BUCKET/{{ $v.model.embedding }}" /models/embedding
      {{- if $v.rerank.enabled }}
      rclone copy --stats-one-line "s3:$S3_BUCKET/{{ $v.model.reranker }}" /models/reranker
      {{- end }}
      rclone copyto "s3:$S3_BUCKET/data/phapdien_{{ $v.data.version }}/bm25_vocab.json.gz" /models/bm25_vocab.json.gz
      test -s /models/bm25_vocab.json.gz && test -n "$(ls /models/embedding)"
  env:
    - { name: S3_BUCKET, value: {{ $v.s3.bucket | quote }} }
    - { name: HOME, value: /tmp }
    - { name: RCLONE_CONFIG_S3_TYPE, value: s3 }
    - { name: RCLONE_CONFIG_S3_PROVIDER, value: Minio }
    - { name: RCLONE_CONFIG_S3_ENDPOINT, value: {{ $v.s3.endpoint | quote }} }
    - name: RCLONE_CONFIG_S3_ACCESS_KEY_ID
      valueFrom: { secretKeyRef: { name: {{ $v.existingSecret }}, key: S3_ACCESS_KEY } }
    - name: RCLONE_CONFIG_S3_SECRET_ACCESS_KEY
      valueFrom: { secretKeyRef: { name: {{ $v.existingSecret }}, key: S3_SECRET_KEY } }
  securityContext:
    allowPrivilegeEscalation: false
    readOnlyRootFilesystem: true
    capabilities: { drop: ["ALL"] }
  resources:
    requests: { cpu: 100m, memory: 128Mi }
    limits: { cpu: "1", memory: 512Mi }
  volumeMounts:
    - { name: models, mountPath: /models }
    - { name: tmp, mountPath: /tmp }
    {{- if .golden }}
    - { name: work, mountPath: /work }
    {{- end }}
{{- end }}

{{/* Biến môi trường dựng bộ tìm kiếm — API và Job đánh giá dùng chung => đánh giá đúng cấu hình production */}}
{{- define "legal-rag.retrieverEnv" -}}
- { name: QDRANT_URL, value: {{ .Values.qdrant.url | quote }} }
- { name: QDRANT_COLLECTION, value: "phapdien_{{ .Values.data.version }}" }
- { name: BM25_VOCAB, value: /models/bm25_vocab.json.gz }
- { name: EMB_MODEL, value: /models/embedding }
- { name: RERANK_MODEL, value: /models/reranker }
- { name: RERANK, value: {{ ternary "1" "0" .Values.rerank.enabled | quote }} }
- { name: DEVICE, value: cpu }
- { name: APP_VERSION, value: {{ .Values.image.tag | quote }} }
- name: QDRANT_API_KEY
  valueFrom: { secretKeyRef: { name: {{ .Values.existingSecret }}, key: QDRANT_API_KEY } }
{{- end }}

{{/* Biến cho mlops.evaluate (check / run / promote) */}}
{{- define "legal-rag.qualityEnv" -}}
{{- $q := .Values.quality -}}
- { name: EMB_MODEL_REF, value: {{ .Values.model.embedding | quote }} }
- { name: RERANK_MODEL_REF, value: {{ .Values.model.reranker | quote }} }
- { name: GOLDEN, value: /models/golden.jsonl.gz }
- { name: GOLDEN_REF, value: {{ $q.golden | quote }} }
- { name: EVAL_SAMPLE, value: {{ $q.sample | quote }} }
- { name: GATE_MIN_NDCG, value: {{ $q.minNdcg10 | quote }} }
- { name: GATE_MAX_DROP, value: {{ $q.maxDrop | quote }} }
- { name: MLFLOW_TRACKING_URI, value: {{ $q.mlflow.uri | quote }} }
- { name: MLFLOW_EXPERIMENT, value: {{ $q.mlflow.experiment | quote }} }
- { name: MLFLOW_MODEL, value: {{ $q.mlflow.model | quote }} }
- name: MLFLOW_TRACKING_USERNAME          # chỉ cần khi MLflow bật basic auth: thêm 2 key này vào Secret
  valueFrom: { secretKeyRef: { name: {{ .Values.existingSecret }}, key: MLFLOW_TRACKING_USERNAME, optional: true } }
- name: MLFLOW_TRACKING_PASSWORD
  valueFrom: { secretKeyRef: { name: {{ .Values.existingSecret }}, key: MLFLOW_TRACKING_PASSWORD, optional: true } }
{{- end }}
