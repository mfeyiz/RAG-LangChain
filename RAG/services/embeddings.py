"""Embedding backends.

``EMBEDDING_PROVIDER`` selects the implementation:
  - ``google`` (default): Gemini Embedding 2 (``gemini-embedding-2-preview``) —
    natively multimodal, so text chunks AND figure images land in ONE shared
    vector space. A text query can therefore retrieve a relevant figure directly.
  - ``hf``: local sentence-transformers bi-encoder (offline / no GCP).

Both expose ``embed_documents`` / ``embed_query``; the Google backend adds
``embed_image`` so ingestion can index figures alongside text.
"""
import os

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google").strip().lower()
# Google's recommended production sweet spot is 768; 1536 trades storage for a
# little more quality. Must match the dimension the vector store was created with.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


class GoogleMultimodalEmbeddings:
    """Gemini Embedding 2 via Vertex AI (text + image in one space).

    Implements the LangChain embeddings interface (``embed_documents`` /
    ``embed_query``) plus ``embed_image`` for figures. ``dim`` reports the output
    dimensionality the vector store must be provisioned with.
    """

    def __init__(self, model: str | None = None, dim: int | None = None):
        self.model = model or os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-2-preview")
        self.dim = dim or EMBEDDING_DIM
        self._client = None
        self._types = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                vertexai=True,
                project=os.environ["GOOGLE_CLOUD_PROJECT"],
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "us"),
            )
            self._types = types
        return self._client, self._types

    def _embed(self, contents: list) -> list[list[float]]:
        client, types = self._ensure_client()
        config = types.EmbedContentConfig(output_dimensionality=self.dim)
        result = client.models.embed_content(model=self.model, contents=contents, config=config)
        return [list(e.values) for e in result.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Batch in modest groups to stay within request limits.
        out: list[list[float]] = []
        batch = int(os.getenv("EMBEDDING_BATCH", "32"))
        for start in range(0, len(texts), batch):
            out.extend(self._embed(list(texts[start: start + batch])))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def embed_image(self, path) -> list[float]:
        """Embed a figure image into the same space as text (for figure points)."""
        from pathlib import Path

        _, types = self._ensure_client()
        data = Path(path).read_bytes()
        mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
        part = types.Part.from_bytes(data=data, mime_type=mime)
        return self._embed([part])[0]


def create_embeddings():
    if EMBEDDING_PROVIDER == "google":
        return GoogleMultimodalEmbeddings()
    # Local fallback. HF_EMBEDDING_MODEL_NAME keeps the legacy model name distinct
    # from EMBEDDING_MODEL_NAME (which now holds the Google model id).
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = os.getenv(
        "HF_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    return HuggingFaceEmbeddings(model_name=model_name)
