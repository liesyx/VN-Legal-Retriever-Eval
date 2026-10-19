# syntax=docker/dockerfile:1
# Stage: base (deps + code prod) -> test (CI: docker build --target test) -> runtime (image chạy trên k8s)
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
# Layer 1 (nặng, ít đổi): torch bản CPU thay cho bản CUDA mặc định (~2,5 GB) + sentence-transformers.
# Phiên bản torch lấy từ requirements-ml.txt (đã ghim ==) nên bỏ qua DL3013.
COPY prod/requirements-ml.txt prod/requirements-ml.txt
# hadolint ignore=DL3013
RUN TORCH="$(grep '^torch==' prod/requirements-ml.txt)" \
 && pip install --index-url https://download.pytorch.org/whl/cpu "$TORCH" \
 && pip install -r prod/requirements-ml.txt
# Layer 2 (nhẹ, hay đổi): FastAPI, Qdrant, OpenTelemetry... — đổi ở đây chỉ push vài MB
COPY prod/requirements.txt prod/requirements.txt
RUN pip install -r prod/requirements.txt
COPY legal_core/ legal_core/
COPY prod/ prod/

FROM base AS test
COPY pipeline/ pipeline/
COPY tests/ tests/
RUN pip install -r pipeline/requirements.txt \
 && python -m tests.test_all \
 && python -c "import torch, sentence_transformers, prod.app; print('torch', torch.__version__)"

FROM base AS runtime
RUN useradd --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin app
USER 10001:10001
# Model + bảng từ do initContainer tải vào /models; không gọi HuggingFace lúc chạy
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf \
    OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
EXPOSE 8000
CMD ["uvicorn", "prod.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
