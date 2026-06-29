"""Regenerate a styled PDF from edited workspace Markdown (WeasyPrint).

Best-effort visual mimicry of the original upload: page size/orientation, page
margins, and the dominant body font/size are extracted from the original PDF
when available and injected as CSS variables ahead of the corporate base
stylesheet. When extraction is not possible (DOCX original, missing file, or a
parser error), sensible A4 defaults are used.

WeasyPrint and pdfminer are imported lazily so this module can be imported (and
the rest of the app can run) on hosts without the native libraries installed.
"""
from collections import Counter
from pathlib import Path

from RAG.services import paths

_TEMPLATE_CSS = Path(__file__).resolve().parent / "templates" / "pdf_style.css"

_DEFAULTS = {
    "page_size": "A4",          # CSS @page size value
    "margin": "20mm",
    "font": '"DejaVu Serif", "Times New Roman", serif',
    "font_size": "11pt",
}


def render(source: str) -> Path:
    """Render the workspace markdown for `source` to a PDF and return its path."""
    import markdown as md_lib
    from weasyprint import CSS, HTML

    md_path = paths.workspace_md_path(source)
    if not md_path.exists():
        raise FileNotFoundError(f"Workspace markdown not found: {md_path}")

    paths.ensure_dirs()
    text = md_path.read_text(encoding="utf-8")
    html_body = md_lib.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "nl2br"],
    )
    full_html = (
        "<html><head><meta charset='utf-8'></head>"
        f"<body>{html_body}</body></html>"
    )

    style = _extract_style(paths.original_doc_path(source))
    css = _build_css(style)

    out_path = paths.workspace_pdf_path(source)
    HTML(string=full_html).write_pdf(str(out_path), stylesheets=[CSS(string=css)])
    return out_path


def _build_css(style: dict) -> str:
    base = _TEMPLATE_CSS.read_text(encoding="utf-8") if _TEMPLATE_CSS.exists() else ""
    dynamic = f"""
    @page {{
        size: {style['page_size']};
        margin: {style['margin']};
        @bottom-center {{
            content: counter(page) " / " counter(pages);
            font-family: "DejaVu Sans", sans-serif;
            font-size: 9pt;
            color: #666;
        }}
    }}
    :root {{
        --doc-font: {style['font']};
        --doc-font-size: {style['font_size']};
    }}
    """
    return dynamic + "\n" + base


def _extract_style(original: Path | None) -> dict:
    style = dict(_DEFAULTS)
    if original is None or original.suffix.lower() != ".pdf":
        return style

    try:
        _apply_page_geometry(original, style)
        _apply_dominant_font(original, style)
    except Exception as exc:  # pragma: no cover - parser/format edge cases
        print(f"[PDFRenderer] Style extraction failed for {original.name}, using defaults: {exc}")
    return style


def _apply_page_geometry(original: Path, style: dict) -> None:
    from pypdf import PdfReader

    page = PdfReader(str(original)).pages[0]
    box = page.mediabox
    width_pt = float(box.width)
    height_pt = float(box.height)
    if width_pt > 0 and height_pt > 0:
        style["page_size"] = f"{width_pt:.0f}pt {height_pt:.0f}pt"


def _apply_dominant_font(original: Path, style: dict) -> None:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTChar

    def chars(obj):
        if isinstance(obj, LTChar):
            yield obj
        elif hasattr(obj, "__iter__"):
            for child in obj:
                yield from chars(child)

    sizes: Counter = Counter()
    for page_count, page_layout in enumerate(extract_pages(str(original)), start=1):
        for char in chars(page_layout):
            sizes[round(char.size)] += 1
        if page_count >= 2:  # sampling two pages is enough for a dominant size
            break

    if sizes:
        dominant = sizes.most_common(1)[0][0]
        if 7 <= dominant <= 16:  # ignore headers/footnotes outliers
            style["font_size"] = f"{dominant}pt"
