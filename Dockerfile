FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # CPU-only torch: eliminates ~1.5 GB of CUDA/nvidia packages
    UV_TORCH_BACKEND=cpu \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Install deps first (separate layer → cached until lockfile changes)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project \
    && uv cache clean

# Copy source and install project
COPY . .
RUN uv sync --frozen --no-dev \
    && uv cache clean

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/.cache \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "RAG.app:app", "--host", "0.0.0.0", "--port", "8000"]
