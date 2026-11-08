# Chạy trong thư mục chứa ảnh (PowerShell): đổi tên ảnh cho khớp README
$map = [ordered]@{
  "1_architecture"       = "01_architecture"
  "minio"                = "02_minio"
  "qdrant"               = "03_qdrant_collections"
  "qdrant_1"             = "04_qdrant_collection"
  "qdrant_2"             = "05_qdrant_visualize"
  "10_n8n_workflow"      = "06_n8n_workflow"
  "n8n_1"                = "07_n8n_execution"
  "3_mlflow_stack"       = "08_mlflow_stack"
  "5_kaggle_report"      = "09_kaggle_report"
  "6_mlflow_run"         = "10_mlflow_run"
  "8_mlflow_compare"     = "11_mlflow_compare"
  "9_web_ui"             = "12_web_ui"
  "12_jaeger_trace"      = "13_jaeger_trace"
  "13_grafana_dashboard" = "14_grafana_dashboard"
  "grafana_1"            = "15_grafana_logs"
  "grafana_2"            = "16_grafana_quality"
  "jenkin"               = "17_jenkins_stages"
}
foreach ($old in $map.Keys) {
  $f = Get-ChildItem -File | Where-Object { $_.BaseName -eq $old } | Select-Object -First 1
  if ($f) { Rename-Item $f.FullName ($map[$old] + ".png") -Force; Write-Host "$($f.Name) -> $($map[$old]).png" }
  else    { Write-Host "Không thấy: $old" -ForegroundColor Yellow }
}
