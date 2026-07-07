"""Shared on-disk storage layout for the dual-channel (originals/workspace) RAG.

    data/
      originals/  pdf/  docx/  markdown/   # read-only ground truth — never edited
      workspace/  pdf/  markdown/          # edited markdown + regenerated PDFs

A document is keyed by its sanitized stem; its logical `source` (stored in chunk
metadata) is "<stem>.md" and is identical across both channels, so a single
source maps deterministically to every on-disk artifact.
"""
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SUPPORTED_DOC_SUFFIXES = {".pdf", ".docx"}

ORIGINALS_PDF_DIR = DATA_DIR / "originals" / "pdf"
ORIGINALS_DOCX_DIR = DATA_DIR / "originals" / "docx"
ORIGINALS_MD_DIR = DATA_DIR / "originals" / "markdown"
ORIGINALS_IMG_DIR = DATA_DIR / "originals" / "images"
WORKSPACE_MD_DIR = DATA_DIR / "workspace" / "markdown"
WORKSPACE_PDF_DIR = DATA_DIR / "workspace" / "pdf"
WORKSPACE_IMG_DIR = DATA_DIR / "workspace" / "images"
WORKSPACE_DOCX_DIR = DATA_DIR / "workspace" / "docx"

# Extracted document figures live under a per-document subdir keyed by stem, so a
# source maps deterministically to its image folder in either channel.
_CHANNEL_IMG_DIR = {
    "originals": ORIGINALS_IMG_DIR,
    "workspace": WORKSPACE_IMG_DIR,
}

_ALL_DIRS = (
    ORIGINALS_PDF_DIR,
    ORIGINALS_DOCX_DIR,
    ORIGINALS_MD_DIR,
    ORIGINALS_IMG_DIR,
    WORKSPACE_MD_DIR,
    WORKSPACE_PDF_DIR,
    WORKSPACE_IMG_DIR,
    WORKSPACE_DOCX_DIR,
)


def ensure_dirs() -> None:
    for directory in _ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def sanitize_stem(name: str) -> str:
    """Filesystem- and URL-friendly stem derived from an upload filename."""
    stem = re.sub(r"[^\w.-]+", "_", Path(name).stem).strip("._")
    return stem or "document"


def source_for(stem: str) -> str:
    return f"{stem}.md"


def stem_of(source: str) -> str:
    return Path(source).stem


def workspace_md_path(source: str) -> Path:
    return WORKSPACE_MD_DIR / Path(source).name


def originals_md_path(source: str) -> Path:
    return ORIGINALS_MD_DIR / Path(source).name


def workspace_pdf_path(source: str) -> Path:
    return WORKSPACE_PDF_DIR / f"{stem_of(source)}.pdf"


def workspace_docx_path(source: str) -> Path:
    return WORKSPACE_DOCX_DIR / f"{stem_of(source)}.docx"


def originals_images_dir(source: str) -> Path:
    """Per-document folder holding figures extracted from the original upload."""
    return ORIGINALS_IMG_DIR / stem_of(source)


def workspace_images_dir(source: str) -> Path:
    """Per-document folder holding the editable copy of extracted figures."""
    return WORKSPACE_IMG_DIR / stem_of(source)


def image_path(channel: str, stem: str, name: str) -> Path | None:
    """Resolve a single figure file, guarding against path traversal."""
    base = (_CHANNEL_IMG_DIR.get(channel) or WORKSPACE_IMG_DIR) / sanitize_stem(stem)
    base_resolved = base.resolve()
    candidate = (base / Path(name).name).resolve()
    if not candidate.is_relative_to(base_resolved):
        return None
    return candidate if candidate.exists() else None


def original_doc_path(source: str) -> Path | None:
    """Return the original uploaded PDF/DOCX for a source, or None if missing."""
    stem = stem_of(source)
    for directory, ext in ((ORIGINALS_PDF_DIR, ".pdf"), (ORIGINALS_DOCX_DIR, ".docx")):
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None
