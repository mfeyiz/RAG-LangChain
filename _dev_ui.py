"""Local dev launcher: serve the static frontend AND the API on one origin.

In production nginx serves index.html/script.js/style.css and the FastAPI app
only serves the API. For local testing this mounts the static files on the same
origin as the API so the UI works without nginx.

Run:  uv run uvicorn _dev_ui:app --port 8080
"""
import os
from pathlib import Path

# Allow /upload and /admin/* without a JWT for local testing.
os.environ.setdefault("ALLOW_INSECURE_DEV", "1")

from fastapi.staticfiles import StaticFiles  # noqa: E402

from RAG.app import app  # noqa: E402

ROOT = Path(__file__).resolve().parent

# Mounted last, so all explicit API routes take precedence; everything else
# (/, /script.js, /style.css, …) is served from disk. html=True serves index.html at /.
app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="static")
