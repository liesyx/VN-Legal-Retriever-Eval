{{- define "legal-rag.labels" -}}
app.kubernetes.io/name: legal-rag
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
