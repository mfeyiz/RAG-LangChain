"""Vision-LLM image captioning for text-anchored image search.

At ingest each extracted figure is captioned with a short factual description /
OCR via a vision-capable OpenRouter model. The caption is indexed as text
(alongside the figure's heading context), which is how "show me the diagram
about X" works without a separate image-vector index.

Captioning is gated by IMAGE_CAPTIONS (default on) and is best-effort: any
failure returns an empty caption so ingestion never hard-fails.
"""
import base64
import mimetypes
import os
from functools import lru_cache
from pathlib import Path

_CAPTION_PROMPT = (
    "Describe this figure from a document in one or two factual sentences so it "
    "can be found by search. Include any visible title, axis labels, key numbers, "
    "or text (OCR). Do not speculate. Respond in the document's language."
)


def _encode(path: Path) -> str | None:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    try:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:{mime};base64,{data}"


@lru_cache(maxsize=1)
def _caption_llm():
    # Built lazily and cached; reuse the shared vision-model configuration.
    from RAG.agents.supervisor import get_vision_llm

    return get_vision_llm(streaming=False)


def caption_image(path: Path) -> str:
    """Return a short caption for an image, or "" if captioning is unavailable."""
    if os.getenv("IMAGE_CAPTIONS", "1") != "1":
        return ""
    path = Path(path)
    data_url = _encode(path)
    if not data_url:
        return ""
    try:
        from langchain_core.messages import HumanMessage

        message = HumanMessage(
            content=[
                {"type": "text", "text": _CAPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )
        response = _caption_llm().invoke([message])
        return (response.content or "").strip().replace("\n", " ")
    except Exception as exc:  # pragma: no cover - network/model failures
        print(f"[Captioning] Failed for {path.name}: {exc}")
        return ""
