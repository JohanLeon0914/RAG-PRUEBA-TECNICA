FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN pip install --no-cache-dir uv==0.9.18

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev \
      --no-install-package torch \
      --no-install-package triton \
      --no-install-package cuda-bindings \
      --no-install-package cuda-pathfinder \
      --no-install-package cuda-toolkit \
      --no-install-package nvidia-cublas \
      --no-install-package nvidia-cuda-cupti \
      --no-install-package nvidia-cuda-nvrtc \
      --no-install-package nvidia-cuda-runtime \
      --no-install-package nvidia-cudnn-cu13 \
      --no-install-package nvidia-cufft \
      --no-install-package nvidia-cufile \
      --no-install-package nvidia-curand \
      --no-install-package nvidia-cusolver \
      --no-install-package nvidia-cusparse \
      --no-install-package nvidia-cusparselt-cu13 \
      --no-install-package nvidia-nccl-cu13 \
      --no-install-package nvidia-nvjitlink \
      --no-install-package nvidia-nvshmem-cu13 \
      --no-install-package nvidia-nvtx \
    && uv pip install --python .venv/bin/python --no-cache \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.6.0+cpu

COPY app ./app
COPY scripts ./scripts
COPY data/corpus ./data/corpus

EXPOSE 8000

CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
