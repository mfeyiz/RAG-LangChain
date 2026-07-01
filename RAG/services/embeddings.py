"""Gemini Embedding 2 — multimodal, via direct :embedContent REST call.

The google-genai SDK routes embed_content() through the legacy :predict
endpoint, which Google has cut for gemini-embedding-2-preview. We bypass
the SDK and POST to :embedContent directly so the model is accessible.

Both text and image inputs are supported in the same vector space, so a
text query can retrieve a relevant figure directly (no separate pipeline).

EMBEDDING_LOCATION must be us-central1 for gemini-embedding-2-preview,
even when the project's primary region is "eu".
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-2-preview")
_LOCATION = os.getenv("EMBEDDING_LOCATION", "us-central1")
_BATCH = int(os.getenv("EMBEDDING_BATCH", "32"))


class GoogleMultimodalEmbeddings:
    """Gemini Embedding 2 via Vertex AI :embedContent REST endpoint.

    Implements embed_documents / embed_query (LangChain interface) plus
    embed_image for figure indexing.
    """

    def __init__(self, model: str | None = None, dim: int | None = None):
        self.model = model or _MODEL
        self.dim = dim or EMBEDDING_DIM
        self._session = None  # requests.Session, lazy
        self._creds = None    # google.oauth2 credentials, lazy

    # ── auth ────────────────────────────────────────────────────────────────

    def _get_headers(self) -> dict[str, str]:
        """Return Bearer-auth headers, refreshing the token when needed."""
        import google.auth.transport.requests

        if self._creds is None:
            self._creds = self._build_creds()
        req = google.auth.transport.requests.Request(session=self._get_session())
        if not self._creds.valid:
            self._creds.refresh(req)
        return {
            "Authorization": f"Bearer {self._creds.token}",
            "Content-Type": "application/json",
        }

    def _build_creds(self):
        key_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if key_file and Path(key_file).exists():
            from google.oauth2 import service_account
            return service_account.Credentials.from_service_account_file(
                key_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        # Fall back to Application Default Credentials (gcloud ADC)
        import google.auth
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return creds

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    # ── REST call ────────────────────────────────────────────────────────────

    def _url(self) -> str:
        project = os.environ["GOOGLE_CLOUD_PROJECT"]
        return (
            f"https://{_LOCATION}-aiplatform.googleapis.com/v1beta1"
            f"/projects/{project}/locations/{_LOCATION}"
            f"/publishers/google/models/{self.model}:embedContent"
        )

    def _embed_single(self, content_part: dict[str, Any]) -> list[float]:
        body = {
            "content": {"parts": [content_part]},
            "outputDimensionality": self.dim,
        }
        resp = self._get_session().post(
            self._url(), json=body, headers=self._get_headers(), timeout=30
        )
        if not resp.ok:
            raise RuntimeError(
                f"Embedding API error {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()["embedding"]["values"]

    def _embed_texts_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, one call per text (batchEmbedContents not supported)."""
        return [self._embed_single({"text": t}) for t in texts]

    # ── public interface (LangChain + embed_image) ───────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            out.extend(self._embed_texts_batch(texts[start: start + _BATCH]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed_single({"text": text})

    def embed_image(self, path: str | Path) -> list[float]:
        """Embed a figure image into the same space as text."""
        data = Path(path).read_bytes()
        b64 = base64.b64encode(data).decode()
        mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
        return self._embed_single({"inlineData": {"mimeType": mime, "data": b64}})


def create_embeddings() -> GoogleMultimodalEmbeddings:
    return GoogleMultimodalEmbeddings()
