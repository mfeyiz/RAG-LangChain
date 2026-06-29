"""Pluggable PDF/DOCX → Markdown conversion.

Backend is selected via the DOC_CONVERTER env var:
  - "markitdown" (default): Microsoft markitdown — lightweight, fast, handles
    both PDF and DOCX. Good general-purpose markdown.
  - "docling": IBM docling — far better table/heading/layout fidelity, but heavy
    (large model downloads, slow on CPU). Imported lazily so the dependency is
    only required when actually selected.

When multimodal ingestion is enabled, `convert_to_markdown_with_images` always
uses docling regardless of DOC_CONVERTER, because docling preserves document
structure and writes image references inline under the correct headings — which
is exactly what heading-aware, text-anchored chunking needs.
"""
import os
import re
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".docx"}

# Matches Markdown image links so we can rewrite docling's relative artifact
# paths down to bare filenames (images are re-homed into per-channel dirs).
_IMAGE_LINK_RE = re.compile(r"(!\[[^\]]*\]\()\s*<?([^)\s>]+)>?(\s*(?:[\"'][^\"']*[\"'])?\s*\))")


class ConversionError(RuntimeError):
    """Raised when a document cannot be converted to non-empty Markdown."""


def convert_to_markdown(file_path: Path) -> str:
    """Convert a PDF or DOCX file to a Markdown string.

    Raises ConversionError on unsupported types, backend failures, or empty output.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ConversionError(
            f"Unsupported file type: {suffix}. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    backend = os.getenv("DOC_CONVERTER", "markitdown").strip().lower()
    try:
        if backend == "docling":
            markdown = _convert_with_docling(file_path)
        else:
            markdown = _convert_with_markitdown(file_path)
    except ConversionError:
        raise
    except Exception as exc:  # pragma: no cover - backend-specific failures
        raise ConversionError(f"{backend} failed to convert {file_path.name}: {exc}") from exc

    markdown = (markdown or "").strip()
    if not markdown:
        raise ConversionError(f"{backend} produced empty Markdown for {file_path.name}")
    return markdown


def _convert_with_markitdown(file_path: Path) -> str:
    from markitdown import MarkItDown

    result = MarkItDown().convert(str(file_path))
    return getattr(result, "text_content", "") or ""


def _convert_with_docling(file_path: Path) -> str:
    # Lazy import — docling pulls in large dependencies and models.
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(file_path))
    return result.document.export_to_markdown()


def convert_to_markdown_with_images(file_path: Path, images_dir: Path) -> tuple[str, list[str]]:
    """Convert a document to Markdown while extracting its figures.

    Images are written into ``images_dir`` and the returned Markdown references
    them by bare filename (e.g. ``![](figure1.png)``) so the chunker can anchor
    each figure to its section. Returns ``(markdown, [image_filenames])``.

    Falls back to the text-only path (no images) if docling or its image export
    is unavailable, so ingestion never hard-fails on the multimodal path.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ConversionError(
            f"Unsupported file type: {suffix}. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    try:
        markdown, image_names = _docling_export_with_images(file_path, images_dir)
    except Exception as exc:  # pragma: no cover - backend-specific failures
        print(f"[Converter] Image extraction unavailable, falling back to text-only: {exc}")
        return convert_to_markdown(file_path), []

    markdown = (markdown or "").strip()
    if not markdown:
        raise ConversionError(f"docling produced empty Markdown for {file_path.name}")
    return markdown, image_names


def _docling_export_with_images(file_path: Path, images_dir: Path) -> tuple[str, list[str]]:
    import tempfile

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import ImageRefMode

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True
    # OCR is slow and noisy here; figure text is recovered via captioning instead.
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(str(file_path))

    # save_as_markdown (REFERENCED mode) writes each picture into artifacts_dir
    # and links it from the Markdown by relative path; we then flatten those
    # links down to bare basenames so they survive the per-channel image copy.
    with tempfile.TemporaryDirectory() as tmp:
        md_path = Path(tmp) / "doc.md"
        result.document.save_as_markdown(
            md_path,
            artifacts_dir=images_dir,
            image_mode=ImageRefMode.REFERENCED,
        )
        raw_md = md_path.read_text(encoding="utf-8")

    image_names: list[str] = []

    def _flatten(match: re.Match) -> str:
        name = os.path.basename(match.group(2).strip())
        if name and name not in image_names:
            image_names.append(name)
        return f"{match.group(1)}{name}{match.group(3)}"

    markdown = _IMAGE_LINK_RE.sub(_flatten, raw_md)

    # docling may nest images one level deep (an artifacts subfolder); hoist any
    # extracted files up to images_dir so basenames resolve directly.
    for path in list(images_dir.rglob("*")):
        if path.is_file() and path.parent != images_dir:
            target = images_dir / path.name
            if not target.exists():
                path.replace(target)

    # Drop trivial figures (logos/icons/bullets) so retrieval surfaces real
    # diagrams. Tunable via MIN_IMAGE_DIM (min width AND height, px).
    markdown, image_names = _filter_small_images(markdown, images_dir, image_names)
    return markdown, image_names


def _filter_small_images(markdown: str, images_dir: Path, image_names: list[str]) -> tuple[str, list[str]]:
    min_dim = int(os.getenv("MIN_IMAGE_DIM", "96"))
    if min_dim <= 0:
        return markdown, image_names
    try:
        from PIL import Image
    except Exception:
        return markdown, image_names

    dropped: set[str] = set()
    kept: list[str] = []
    for name in image_names:
        path = images_dir / name
        try:
            with Image.open(path) as img:
                width, height = img.size
        except Exception:
            kept.append(name)
            continue
        if width < min_dim or height < min_dim:
            dropped.add(name)
            path.unlink(missing_ok=True)
        else:
            kept.append(name)

    if not dropped:
        return markdown, image_names

    def _drop(match: re.Match) -> str:
        return "" if os.path.basename(match.group(2).strip()) in dropped else match.group(0)

    return _IMAGE_LINK_RE.sub(_drop, markdown), kept
