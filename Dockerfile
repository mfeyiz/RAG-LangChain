FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# curl/ca-certificates for tooling; the libpango/cairo/gdk-pixbuf/ffi stack and
# fonts are required by WeasyPrint to render the regenerated workspace PDFs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl ca-certificates \
        libpango-1.0-0 libpangocairo-1.0-0 libcairo2 \
        libgdk-pixbuf-2.0-0 libffi-dev libharfbuzz0b \
        fonts-dejavu \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Install deps first — cached until lockfile changes
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy source and install project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/.cache \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "RAG.app:app", "--host", "0.0.0.0", "--port", "8000"]
