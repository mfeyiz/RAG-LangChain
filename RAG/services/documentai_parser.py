"""Google Document AI Layout Parser → heading-aware Markdown + cropped figures.

Selected via ``DOC_PARSER=documentai``. The Layout Parser processor returns a
structured document layout (headings, paragraphs, lists, tables) which we render
to Markdown so the existing heading-aware, parent-aware chunker can consume it.

Figures are recovered by cropping the page raster at each figure block's bounding
box: we prefer Document AI's own page image, and fall back to rendering the PDF
page locally (PyMuPDF) at the same normalized box. Each figure is written into
``images_dir`` and linked inline (``![](name.png)``) right after the page it
belongs to, so it is anchored to that section.

Requires env: GOOGLE_CLOUD_PROJECT, DOCAI_LOCATION (or GOOGLE_CLOUD_LOCATION),
DOCAI_PROCESSOR_ID, and GOOGLE_APPLICATION_CREDENTIALS (service account).
"""
import io
import os
from pathlib import Path

_MIME_BY_SUFFIX = {".pdf": "application/pdf", ".html": "text/html", ".htm": "text/html"}

# Layout Parser text-block types map to Markdown heading levels.
_HEADING_LEVEL = {
    "heading-1": 1, "heading-2": 2, "heading-3": 3,
    "heading-4": 3, "heading-5": 3, "heading-6": 3,  # clamp to the chunker's 3 levels
    "title": 1,
}


def _processor_name(client) -> str:
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.getenv("DOCAI_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION", "us")
    processor_id = os.environ["DOCAI_PROCESSOR_ID"]
    return client.processor_path(project, location, processor_id)


def _client():
    from google.api_core.client_options import ClientOptions
    from google.cloud import documentai_v1 as documentai

    location = os.getenv("DOCAI_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION", "us")
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    return documentai.DocumentProcessorServiceClient(client_options=opts), documentai


def documentai_export_with_images(file_path: Path, images_dir: Path) -> tuple[str, list[str]]:
    """Process ``file_path`` with the Layout Parser; return (markdown, image_names)."""
    file_path = Path(file_path)
    mime = _MIME_BY_SUFFIX.get(file_path.suffix.lower())
    if not mime:
        # DOCX etc. aren't supported by Layout Parser — let the caller fall back.
        raise ValueError(f"Document AI Layout Parser does not support {file_path.suffix}")

    client, documentai = _client()
    raw = documentai.RawDocument(content=file_path.read_bytes(), mime_type=mime)
    process_options = documentai.ProcessOptions(
        layout_config=documentai.ProcessOptions.LayoutConfig(
            chunking_config=documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
                chunk_size=900,
                include_ancestor_headings=True,
            )
        )
    )
    request = documentai.ProcessRequest(
        name=_processor_name(client),
        raw_document=raw,
        process_options=process_options,
    )
    document = client.process_document(request=request).document

    markdown, figure_boxes = _layout_to_markdown(document)
    image_names = _extract_figures(file_path, document, figure_boxes, images_dir)
    markdown = _inject_figure_links(markdown, image_names)
    return markdown, image_names


def _layout_to_markdown(document) -> tuple[str, list[dict]]:
    """Render Document AI ``document_layout`` blocks to Markdown.

    Returns (markdown, figure_boxes) where each figure_box is
    {"page": int, "vertices": [(x,y)...] normalized} for later cropping.
    """
    layout = getattr(document, "document_layout", None)
    figure_boxes: list[dict] = []
    if layout is None or not getattr(layout, "blocks", None):
        # No structured layout — fall back to the flat OCR text.
        return (getattr(document, "text", "") or "").strip(), figure_boxes

    parts: list[str] = []

    def render_block(block, depth: int = 0) -> None:
        text_block = getattr(block, "text_block", None)
        table_block = getattr(block, "table_block", None)
        list_block = getattr(block, "list_block", None)
        _record_figure(block, figure_boxes)

        if text_block and getattr(text_block, "text", ""):
            btype = (getattr(text_block, "type_", "") or getattr(text_block, "type", "")).lower()
            text = text_block.text.strip()
            level = _HEADING_LEVEL.get(btype)
            if level:
                parts.append(f"\n{'#' * level} {text}\n")
            elif "list" in btype:
                parts.append(f"{'  ' * depth}- {text}")
            else:
                parts.append(f"\n{text}\n")
            for child in getattr(text_block, "blocks", []) or []:
                render_block(child, depth + 1)
        elif list_block:
            for child in getattr(list_block, "list_entries", []) or getattr(list_block, "blocks", []) or []:
                render_block(child, depth + 1)
        elif table_block:
            parts.append(_render_table(table_block))

    for block in layout.blocks:
        render_block(block)

    return "\n".join(p for p in parts).strip(), figure_boxes


def _render_table(table_block) -> str:
    """Render a Document AI table block to a GitHub-style Markdown table."""
    def cell_text(cell) -> str:
        for block in getattr(cell, "blocks", []) or []:
            text_block = getattr(block, "text_block", None)
            if text_block and getattr(text_block, "text", ""):
                return text_block.text.strip().replace("\n", " ")
        return ""

    def row_cells(row):
        return [cell_text(cell) for cell in getattr(row, "cells", [])]

    header_rows = getattr(table_block, "header_rows", []) or []
    body_rows = getattr(table_block, "body_rows", []) or []
    if not header_rows and not body_rows:
        return ""
    lines: list[str] = []
    headers = row_cells(header_rows[0]) if header_rows else row_cells(body_rows[0])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in (body_rows if header_rows else body_rows[1:]):
        lines.append("| " + " | ".join(row_cells(row)) + " |")
    return "\n" + "\n".join(lines) + "\n"


def _record_figure(block, figure_boxes: list[dict]) -> None:
    """If a block looks like a figure/image, record its normalized bounding box."""
    btype = ""
    text_block = getattr(block, "text_block", None)
    if text_block:
        btype = (getattr(text_block, "type_", "") or getattr(text_block, "type", "")).lower()
    if btype not in ("figure", "image", "picture"):
        return
    page_span = getattr(block, "page_span", None)
    page = getattr(page_span, "page_start", 1) if page_span else 1
    bbox = getattr(block, "bounding_box", None) or getattr(block, "bounding_poly", None)
    vertices = []
    if bbox is not None:
        for v in getattr(bbox, "normalized_vertices", []) or []:
            vertices.append((getattr(v, "x", 0.0), getattr(v, "y", 0.0)))
    figure_boxes.append({"page": int(page), "vertices": vertices})


def _extract_figures(file_path: Path, document, figure_boxes: list[dict], images_dir: Path) -> list[str]:
    """Crop each recorded figure box from its page raster into images_dir.

    Prefers Document AI's page image; falls back to rendering the PDF page with
    PyMuPDF. Returns the list of written filenames (filtered for trivial sizes).
    """
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    if not figure_boxes:
        return []

    try:
        from PIL import Image
    except Exception:
        return []

    pages = {int(getattr(p, "page_number", i + 1)): p for i, p in enumerate(getattr(document, "pages", []) or [])}
    min_dim = int(os.getenv("MIN_IMAGE_DIM", "96"))
    names: list[str] = []

    for idx, fig in enumerate(figure_boxes):
        page_no = fig["page"]
        page_img = _page_raster(file_path, pages.get(page_no), page_no, Image)
        if page_img is None:
            continue
        crop = _crop_normalized(page_img, fig["vertices"])
        if crop is None or crop.width < min_dim or crop.height < min_dim:
            continue
        name = f"p{page_no}_fig{idx + 1}.png"
        crop.save(images_dir / name)
        names.append(name)

    return names


def _page_raster(file_path: Path, docai_page, page_no: int, Image):
    """Return a PIL image of the page — from Document AI if present, else PyMuPDF."""
    content = getattr(getattr(docai_page, "image", None), "content", None) if docai_page else None
    if content:
        try:
            return Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            pass
    try:
        import fitz  # PyMuPDF

        with fitz.open(str(file_path)) as doc:
            page = doc[page_no - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    except Exception:
        return None


def _crop_normalized(page_img, vertices):
    """Crop ``page_img`` to the normalized polygon's bounding rect."""
    if not vertices:
        return None
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    w, h = page_img.size
    left, right = int(min(xs) * w), int(max(xs) * w)
    top, bottom = int(min(ys) * h), int(max(ys) * h)
    if right <= left or bottom <= top:
        return None
    return page_img.crop((left, top, right, bottom))


def _inject_figure_links(markdown: str, image_names: list[str]) -> str:
    """Append figure links so they are part of the markdown (and thus chunked).

    We can't always map a crop back to its exact section, so figures are appended
    under their own subheading; the heading-aware chunker keeps each figure
    retrievable and the multimodal embedder indexes the image itself.
    """
    if not image_names:
        return markdown
    links = "\n".join(f"![]({name})" for name in image_names)
    return f"{markdown}\n\n## Figürler\n\n{links}\n"
