"""Quan sát hệ thống cho API: trace (OTLP → Jaeger), log (OTLP → Loki, kèm trace_id), metrics (Prometheus).

Bật bằng biến môi trường, để trống = tắt (chạy local / test không gửi gì ra ngoài):
  OTEL_EXPORTER_OTLP_TRACES_ENDPOINT   vd http://jaeger.observability:4318/v1/traces
  OTEL_EXPORTER_OTLP_LOGS_ENDPOINT     vd http://loki.observability:3100/otlp/v1/logs
  METRICS_PORT                         vd 9100 — Prometheus đọc pod:9100/metrics (cổng riêng, không qua NodePort)
  OTEL_SERVICE_NAME (mặc định legal-rag-api), OTEL_RESOURCE_ATTRIBUTES (k8s.pod.name=..., do Helm đặt)
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

from opentelemetry import trace
from prometheus_client import Counter, Gauge, Histogram, start_http_server

SERVICE = os.environ.get("OTEL_SERVICE_NAME", "legal-rag-api")
tracer = trace.get_tracer("legal_rag")
log = logging.getLogger("legal_rag")

# ---------- metrics ----------
_SLOW = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 45, 60, 90, 120)
HTTP = Histogram("legal_http_request_seconds", "Thời gian xử lý request", ["method", "route", "status"], buckets=_SLOW)
STAGE = Histogram("legal_search_stage_seconds", "Thời gian từng bước tìm kiếm", ["stage"],
                  buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10))
ASK = Counter("legal_ask_total", "Câu hỏi từ giao diện web theo kết quả", ["result"])
N8N = Histogram("legal_n8n_seconds", "Thời gian chờ n8n trả lời (2 lần gọi LLM + /search)", buckets=_SLOW)
INFO = Gauge("legal_info", "Cấu hình đang chạy", ["collection", "rerank", "version"])


@contextmanager
def stage(name: str, **attrs):
    """Một bước xử lý: tạo span con (Jaeger) + ghi vào histogram legal_search_stage_seconds (Prometheus)."""
    t0 = time.perf_counter()
    with tracer.start_as_current_span(name, attributes=attrs) as sp:
        yield sp
    STAGE.labels(name).observe(time.perf_counter() - t0)


class _TraceIds(logging.Filter):
    """Thêm trace_id vào log in ra stdout (kubectl logs) để tra ngược sang Jaeger."""
    def filter(self, record):
        ctx = trace.get_current_span().get_span_context()
        record.trace_id = format(ctx.trace_id, "032x") if ctx.is_valid else "-"
        return True


def setup_logging():
    h = logging.StreamHandler()
    h.addFilter(_TraceIds())
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s trace=%(trace_id)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def setup(app, span_exporter=None, log_exporter=None):
    """Gọi một lần lúc import app. Exporter truyền vào dùng cho test; bình thường lấy từ biến môi trường."""
    setup_logging()
    traces_on = span_exporter is not None or bool(os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
    logs_on = log_exporter is not None or bool(os.environ.get("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"))
    if traces_on or logs_on:
        from opentelemetry.sdk.resources import Resource
        resource = Resource.create({"service.name": SERVICE, "service.version": os.environ.get("APP_VERSION", "dev")})
    if traces_on:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
        if span_exporter is None:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            span_exporter, proc = OTLPSpanExporter(), BatchSpanProcessor
        else:
            proc = SimpleSpanProcessor
        tp = TracerProvider(resource=resource)
        tp.add_span_processor(proc(span_exporter))
        trace.set_tracer_provider(tp)
    if logs_on:
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, SimpleLogRecordProcessor
        if log_exporter is None:
            from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
            log_exporter, proc = OTLPLogExporter(), BatchLogRecordProcessor
        else:
            proc = SimpleLogRecordProcessor
        lp = LoggerProvider(resource=resource)
        lp.add_log_record_processor(proc(log_exporter))
        set_logger_provider(lp)
        logging.getLogger().addHandler(LoggingHandler(level=logging.INFO, logger_provider=lp))

    # Span cho mọi request vào FastAPI + mọi request httpx đi ra (n8n, Qdrant), kèm header traceparent
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health")
    HTTPXClientInstrumentor().instrument()

    @app.middleware("http")
    async def access_log(request, call_next):
        t0 = time.perf_counter()
        status = 500
        try:
            resp = await call_next(request)
            status = resp.status_code
            return resp
        finally:
            route = getattr(request.scope.get("route"), "path", request.url.path)
            dt = time.perf_counter() - t0
            HTTP.labels(request.method, route, str(status)).observe(dt)
            if route != "/health":
                log.info("%s %s -> %d (%.0f ms)", request.method, route, status, dt * 1000)

    log.info("telemetry: traces=%s logs=%s metrics_port=%s", traces_on, logs_on, os.environ.get("METRICS_PORT") or "off")


def start_metrics_server():
    port = os.environ.get("METRICS_PORT")
    if port:
        start_http_server(int(port))
