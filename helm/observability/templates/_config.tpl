{{- define "obs.prometheusConfig" -}}
global:
  scrape_interval: {{ .Values.prometheus.scrapeInterval }}
  evaluation_interval: {{ .Values.prometheus.scrapeInterval }}
scrape_configs:
  - job_name: prometheus
    static_configs: [{ targets: ["localhost:9090"] }]

  # Pod nào có annotation prometheus.io/scrape: "true" (+ port, path) thì được đọc — legal-rag API ở cổng 9100
  - job_name: pods
    kubernetes_sd_configs: [{ role: pod }]
    relabel_configs:
      - { source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape], action: keep, regex: "true" }
      - { source_labels: [__meta_kubernetes_pod_phase], action: keep, regex: Running }
      - { source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path], target_label: __metrics_path__, regex: (.+) }
      - source_labels: [__meta_kubernetes_pod_ip, __meta_kubernetes_pod_annotation_prometheus_io_port]
        regex: (.+);(\d+)
        replacement: $1:$2
        target_label: __address__
      - { source_labels: [__meta_kubernetes_namespace], target_label: namespace }
      - { source_labels: [__meta_kubernetes_pod_name], target_label: pod }
      - { source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_name], target_label: app }

  # CPU/RAM/mạng của mọi container (cAdvisor trong kubelet), đọc qua API server
  - job_name: cadvisor
    scheme: https
    tls_config: { ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt }
    authorization: { credentials_file: /var/run/secrets/kubernetes.io/serviceaccount/token }
    kubernetes_sd_configs: [{ role: node }]
    relabel_configs:
      - { target_label: __address__, replacement: kubernetes.default.svc:443 }
      - { source_labels: [__meta_kubernetes_node_name], regex: (.+), target_label: __metrics_path__, replacement: /api/v1/nodes/$1/proxy/metrics/cadvisor }
    metric_relabel_configs:
      - { source_labels: [__name__], action: keep, regex: "container_(cpu_usage_seconds_total|memory_working_set_bytes|network_receive_bytes_total|network_transmit_bytes_total|spec_memory_limit_bytes|cpu_cfs_throttled_periods_total|cpu_cfs_periods_total)|machine_(cpu_cores|memory_bytes)" }
{{- end }}

{{- define "obs.lokiConfig" -}}
# Loki 1 tiến trình, lưu trên đĩa. Nhận log OTLP từ API ở /otlp/v1/logs:
# service.name, k8s.pod.name... thành label; trace_id/span_id thành structured metadata (lọc: | trace_id="...")
auth_enabled: false
server:
  http_listen_port: 3100
  grpc_listen_port: 9095
  log_level: warn
common:
  instance_addr: 127.0.0.1
  path_prefix: /loki
  replication_factor: 1
  ring: { kvstore: { store: inmemory } }
  storage:
    filesystem: { chunks_directory: /loki/chunks, rules_directory: /loki/rules }
schema_config:
  configs:
    - { from: "2024-01-01", store: tsdb, object_store: filesystem, schema: v13, index: { prefix: index_, period: 24h } }
limits_config:
  retention_period: {{ .Values.loki.retention }}
  allow_structured_metadata: true
  volume_enabled: true
compactor:
  working_directory: /loki/compactor
  retention_enabled: true
  delete_request_store: filesystem
analytics:
  reporting_enabled: false
{{- end }}

{{- define "obs.grafanaDatasources" -}}
# "$$" = ký tự $ (Grafana thay biến môi trường trong file provisioning)
apiVersion: 1
datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    url: http://prometheus:9090
    isDefault: true
    jsonData: { timeInterval: {{ .Values.prometheus.scrapeInterval }} }
  - name: Loki
    uid: loki
    type: loki
    url: http://loki:3100
    jsonData:
      derivedFields:                    # log có trace_id -> nút mở trace trong Jaeger
        - name: TraceID
          matcherType: label
          matcherRegex: trace_id
          datasourceUid: jaeger
          url: "$${__value.raw}"
  - name: Jaeger
    uid: jaeger
    type: jaeger
    url: http://jaeger:16686
    jsonData:
      tracesToLogsV2:                   # span -> log cùng trace trong Loki
        datasourceUid: loki
        spanStartTimeShift: "-5m"
        spanEndTimeShift: "5m"
        customQuery: true
        query: '{service_name="$${__span.tags["service.name"]}"} | trace_id="$${__trace.traceId}"'
      tracesToMetrics:
        datasourceUid: prometheus
      nodeGraph: { enabled: true }
{{- end }}
