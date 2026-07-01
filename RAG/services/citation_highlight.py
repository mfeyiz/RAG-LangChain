"""Citation → PDF page lookup + highlighted page rendering.

Citations in an answer look like ``[1]``, ``[2]`` and map back to a retrieved
chunk (source + text snippet) via the search-metadata that ships with each
answer. When the user clicks a citation, the backend searches the ORIGINAL pdf
for the snippet's page, renders that page to a PNG, and draws yellow highlight
rectangles over the matching text spans (fosforlu kalem effect). The frontend
just shows the returned <img> in a preview panel.

Uses PyMuPDF (fitz), which is already a dependency (pymupdf). DOCX originals
are converted to PDF on the fly so citations resolve for them too.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

from RAG.services import paths

# Rendered highlighted pages are cached on disk under data/workspace/cite/
CITE_DIR = paths.WORKSPACE_MD_DIR.parent / "cite"

_HIGHLIGHT_COLOR = (1.0, 0.92, 0.23)  # yellow highlighter (RGB 0-1)


def ensure_dir() -> None:
    CITE_DIR.mkdir(parents=True, exist_ok=True)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def locate_page(pdf_path: Path, snippet: str) -> int | None:
    """Return the 0-based index of the first page whose text contains the
    snippet (whitespace-normalised). None when not found / no snippet."""
    if not snippet or len(snippet.strip()) < 3:
        return None
    import fitz  # type: ignore

    target = _normalize(snippet)
    # Use the first ~40 words of the snippet as the needle — full chunk text is
    # usually longer than what survives pdf text extraction verbatim.
    needle = " ".join(target.split()[:40])
    if not needle:
        return None

    doc = fitz.open(str(pdf_path))
    try:
        for i, page in enumerate(doc):
            page_text = _normalize(page.get_text("text"))
            if not page_text:
                continue
            if needle in page_text:
                return i
            # Try the last few words too (chunks often start mid-sentence).
            tail = " ".join(target.split()[-12:])
            if len(tail) > 8 and tail in page_text:
                return i
        return None
    finally:
        doc.close()


def render_highlighted_page(
    source: str,
    snippet: str,
    page_index: int | None = None,
    max_page: int | None = None,
) -> dict:
    """Locate + render. Returns {page, total_pages, image_url} or {error}.

    When page_index is None the snippet is searched across all pages first.
    """
    import fitz  # type: ignore

    pdf_path = paths.original_doc_path(source)
    if pdf_path is None or pdf_path.suffix.lower() != ".pdf":
        # DOCX original: we don't page-render it. Fallback: return nothing.
        return {"error": "Highlights yalnızca PDF orijinalleri için destekleniyor."}
    if not pdf_path.exists():
        return {"error": "Orijinal PDF bulunamadı."}

    target_page = page_index if page_index is not None else locate_page(pdf_path, snippet)
    if target_page is None:
        return {"error": "Atıf metni orijinal PDF içinde bulunamadı.", "page": None}

    ensure_dir()
    doc = fitz.open(str(pdf_path))
    try:
        total = doc.page_count
        if max_page is not None and target_page > max_page:
            target_page = max(0, max_page)
        target_page = max(0, min(target_page, total - 1))
        page = doc[target_page]

        # Highlight every occurrence of the snippet's distinctive phrase.
        if snippet and len(snippet.strip()) >= 3:
            for phrase in _search_phrases(snippet):
                rects = page.search_for(phrase)
                for r in rects:
                    page.add_highlight_annot(r)
            for annot in list(page.annots() or []):
                annot.set_colors(stroke=_HIGHLIGHT_COLOR)
                annot.update()

        # Render at ~2x for crisp display.
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        stem = paths.stem_of(source)
        ts = int(time.time())
        out_name = f"{stem}_p{target_page + 1}_{ts}.png"
        out_path = CITE_DIR / out_name
        pix.save(str(out_path))

        return {
            "page": target_page + 1,        # 1-based for display
            "page_index": target_page,       # 0-based for nav
            "total_pages": total,
            "image_url": f"/cite/image/{out_name}",
        }
    finally:
        doc.close()


def _snippets_hash(snippets: list[str]) -> str:
    """Stable short hash of the snippet set — used to cache rendered docs so
    re-clicking the same answer's citations reuses the highlighted pages."""
    h = hashlib.sha1()
    for s in snippets:
        h.update(_normalize(s).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def render_highlighted_document(
    source: str,
    snippets: list[str],
    focus_snippet: str | None = None,
) -> dict:
    """Render EVERY page of the original PDF with ALL retrieved chunks of that
    source highlighted, so the viewer can scroll the whole document with every
    cited passage marked (not just the clicked one).

    Returns {total_pages, pages:[{page, image_url}], focus_page_index} or {error}.
    Pages are cached on disk by (source, mtime, snippet-set) so repeat clicks are
    cheap.
    """
    import fitz  # type: ignore

    pdf_path = paths.original_doc_path(source)
    if pdf_path is None or pdf_path.suffix.lower() != ".pdf":
        return {"error": "Highlights yalnızca PDF orijinalleri için destekleniyor."}
    if not pdf_path.exists():
        return {"error": "Orijinal PDF bulunamadı."}

    ensure_dir()
    stem = paths.stem_of(source)
    shash = _snippets_hash(snippets)
    try:
        mtime = int(pdf_path.stat().st_mtime)
    except OSError:
        mtime = 0

    # Highlight phrases for every chunk, computed once and reused per page.
    phrase_sets = [_search_phrases(s) for s in snippets if s and len(s.strip()) >= 3]

    try:
        focus_page_index = locate_page(pdf_path, focus_snippet) if focus_snippet else None
    except Exception as exc:
        print(f"[Citation] focus page lookup failed for {source}: {exc}")
        focus_page_index = None

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        print(f"[Citation] could not open PDF {pdf_path}: {exc}")
        return {"error": "Orijinal PDF açılamadı."}

    try:
        total = doc.page_count
        pages: list[dict] = []
        for i in range(total):
            out_name = f"{stem}_{mtime}_{shash}_p{i + 1}.png"
            out_path = CITE_DIR / out_name
            # Render each page independently — a bad annotation or an unusual
            # page must not abort the whole document (that would 500 and leave
            # the viewer showing an empty grey panel).
            try:
                if not out_path.exists():
                    page = doc[i]
                    seen: set[str] = set()
                    for phrases in phrase_sets:
                        for phrase in phrases:
                            if phrase in seen:
                                continue
                            seen.add(phrase)
                            for r in page.search_for(phrase):
                                page.add_highlight_annot(r)
                    for annot in list(page.annots() or []):
                        annot.set_colors(stroke=_HIGHLIGHT_COLOR)
                        annot.update()
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    pix.save(str(out_path))
                pages.append({"page": i + 1, "image_url": f"/cite/image/{out_name}"})
            except Exception as exc:
                print(f"[Citation] page {i + 1} render failed for {source}: {exc}")
                continue

        if not pages:
            return {"error": "Orijinal PDF işlenemedi."}

        return {
            "total_pages": total,
            "pages": pages,
            "focus_page_index": focus_page_index if focus_page_index is not None else 0,
        }
    finally:
        doc.close()


def _search_phrases(snippet: str) -> list[str]:
    """Distinctive substrings to highlight, longest first."""
    words = _normalize(snippet).split()
    phrases = []
    for n in (12, 8, 6):
        for i in range(0, max(0, len(words) - n + 1)):
            phrases.append(" ".join(words[i:i + n]))
    # Also the whole thing trimmed.
    if snippet.strip():
        phrases.append(" ".join(words[:40]))
    # De-dup, keep order, drop empties.
    seen: set[str] = set()
    out: list[str] = []
    for p in phrases:
        if p and p not in seen and len(p) >= 5:
            seen.add(p)
            out.append(p)
    return out[:6]