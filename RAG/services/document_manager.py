"""Dual-channel incremental indexing.

Ingestion writes every uploaded document into BOTH channels (read-only
`originals` and editable `workspace`); the editor write-back only touches
`workspace`. Add/delete keep pgvector and the per-channel corpus.jsonl in sync
without a full re-index.
"""
import json
import os
import re
import shutil
import uuid
from pathlib import Path

from langchain_core.documents import Document

from RAG.services.converter import convert_to_markdown, convert_to_markdown_with_images
from RAG.services import paths, vector_store
from RAG.services.table_store import extract_tables, TABLES_DIR_BASE as _TABLES_DIR_BASE
from RAG.services.retrieval import (
    CHANNELS,
    DEFAULT_CHANNEL,
    channel_collection,
    channel_corpus_path,
    get_retriever,
    invalidate_channel_cache,
    _get_embeddings,
)
from RAG.services.rag_service import semantic_chunk_documents

# Multimodal ingestion: extract figures from documents and anchor them to their
# section. Disable (MULTIMODAL=0) to fall back to the legacy text-only path.
_MULTIMODAL = os.getenv("MULTIMODAL", "1") == "1"
_IMAGE_CAPTIONS = os.getenv("IMAGE_CAPTIONS", "1") == "1"

_IMAGE_LINE_RE = re.compile(r"^!\[[^\]]*\]\(\s*<?([^)\s>]+)>?[^)]*\)\s*$")


def add_document(file_bytes: bytes, filename: str) -> dict:
    """Save the original, convert to Markdown, and index into both channels.

    Returns {filename, source, chunks_added, originals_md, workspace_md}.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in paths.SUPPORTED_DOC_SUFFIXES:
        raise ValueError(
            f"Unsupported file type: {suffix}. Only {sorted(paths.SUPPORTED_DOC_SUFFIXES)} are accepted."
        )

    paths.ensure_dirs()
    stem = paths.sanitize_stem(filename)
    source = paths.source_for(stem)

    # 1. Persist the original upload (read-only ground truth).
    original_dir = paths.ORIGINALS_PDF_DIR if suffix == ".pdf" else paths.ORIGINALS_DOCX_DIR
    original_path = original_dir / f"{stem}{suffix}"
    original_path.write_bytes(file_bytes)

    # 2. Convert to Markdown and write to both originals (read-only) and workspace.
    #    On the multimodal path also extract figures into per-channel image dirs.
    if _MULTIMODAL:
        originals_img = paths.originals_images_dir(source)
        workspace_img = paths.workspace_images_dir(source)
        # Re-upload of the same file: clear prior figures so stale images from an
        # earlier version don't linger and get served alongside the new ones.
        for stale in (originals_img, workspace_img):
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=True)
        markdown, image_names = convert_to_markdown_with_images(original_path, originals_img)
        if _IMAGE_CAPTIONS and image_names:
            markdown = _inject_image_captions(markdown, originals_img)
        # Mirror extracted figures into the editable workspace image dir.
        _copy_tree(originals_img, workspace_img)
    else:
        markdown = convert_to_markdown(original_path)

    originals_md = paths.originals_md_path(source)
    workspace_md = paths.workspace_md_path(source)
    originals_md.write_text(markdown, encoding="utf-8")
    workspace_md.write_text(markdown, encoding="utf-8")

    # Persist structured table sidecars (CSV+JSON) so the code-interpreter node
    # can do real arithmetic over financial/table-heavy documents instead of
    # asking the LLM to approximate it.
    try:
        extract_tables(markdown, source)
    except Exception as exc:
        print(f"[DocumentManager] table extraction failed for {source}: {exc}")

    # 3. Chunk once; index the same chunks into both channels.
    chunks = _chunk_markdown(markdown, source, stem)
    if not chunks:
        return {
            "filename": filename,
            "source": source,
            "chunks_added": 0,
            "originals_md": str(originals_md),
            "workspace_md": str(workspace_md),
        }

    vectors = _get_embeddings().embed_documents([c.page_content for c in chunks])
    # Index figures as their own points in the shared multimodal space.
    figure_docs, figure_vectors = _build_figure_index(chunks, source, stem)
    for channel in CHANNELS:
        # Replace any prior copy of this source (re-upload == update).
        _delete_from_vector_store_by_source(source, channel)
        _remove_from_corpus(source, channel)
        _upsert_to_vector_store(chunks, vectors, channel)
        _append_to_corpus(chunks, channel)
        if figure_docs:
            _upsert_to_vector_store(figure_docs, figure_vectors, channel)
            _append_to_corpus(figure_docs, channel)
        get_retriever(channel).reload_corpus()
        invalidate_channel_cache(channel)

    return {
        "filename": filename,
        "source": source,
        "chunks_added": len(chunks),
        "originals_md": str(originals_md),
        "workspace_md": str(workspace_md),
    }


def reindex_workspace_source(source: str) -> dict:
    """Re-chunk and re-index a workspace markdown file after an edit.

    Touches the workspace channel only — the originals index is never modified.
    Returns {source, chunks_added}.
    """
    md_path = paths.workspace_md_path(source)
    if not md_path.exists():
        raise FileNotFoundError(f"Workspace markdown not found: {md_path}")

    markdown = md_path.read_text(encoding="utf-8")
    stem = paths.stem_of(source)
    chunks = _chunk_markdown(markdown, source, stem)

    _delete_from_vector_store_by_source(source, "workspace")
    _remove_from_corpus(source, "workspace")

    if chunks:
        vectors = _get_embeddings().embed_documents([c.page_content for c in chunks])
        _upsert_to_vector_store(chunks, vectors, "workspace")
        _append_to_corpus(chunks, "workspace")
        figure_docs, figure_vectors = _build_figure_index(chunks, source, stem)
        if figure_docs:
            _upsert_to_vector_store(figure_docs, figure_vectors, "workspace")
            _append_to_corpus(figure_docs, "workspace")

    get_retriever("workspace").reload_corpus()
    invalidate_channel_cache("workspace")
    return {"source": source, "chunks_added": len(chunks)}


def delete_document_by_source(source: str, channel: str | None = None) -> dict:
    """Remove a document's chunks. By default removes from BOTH channels and deletes
    the on-disk artifacts; pass a channel to scope the vector delete only."""
    channels = [channel] if channel else list(CHANNELS)
    deleted = 0
    for ch in channels:
        deleted += max(_delete_from_vector_store_by_source(source, ch), _remove_from_corpus(source, ch))
        get_retriever(ch).reload_corpus()
        invalidate_channel_cache(ch)

    if channel is None:
        _delete_artifacts(source)

    return {"source": source, "chunks_deleted": deleted}


def list_documents(channel: str = DEFAULT_CHANNEL) -> list[dict]:
    """Return unique sources with chunk counts from a channel's corpus.jsonl."""
    corpus_path = channel_corpus_path(channel)
    if not corpus_path.exists():
        return []

    counts: dict[str, int] = {}
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            src = rec.get("metadata", {}).get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1

    return [{"source": src, "chunks": cnt} for src, cnt in sorted(counts.items())]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _copy_tree(src: Path, dst: Path) -> None:
    """Mirror src into dst (used to seed the workspace image copy from originals)."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)


def _inject_image_captions(markdown: str, images_dir: Path) -> str:
    """Append a `Figure: <caption>` line under each image link so figures are
    findable through the text retrieval pipeline (text-anchored image search)."""
    from RAG.services.image_captioning import caption_image

    out_lines: list[str] = []
    for line in markdown.splitlines():
        out_lines.append(line)
        match = _IMAGE_LINE_RE.match(line.strip())
        if not match:
            continue
        name = os.path.basename(match.group(1).strip())
        img_file = images_dir / name
        if not img_file.exists():
            continue
        caption = caption_image(img_file)
        if caption:
            out_lines.append(f"Figure: {caption}")
    return "\n".join(out_lines)


def _chunk_markdown(markdown: str, source: str, stem: str) -> list:
    doc = Document(
        page_content=markdown,
        metadata={"source": source, "kind": "markdown", "title": stem},
    )
    return semantic_chunk_documents([doc])


def _build_figure_index(chunks, source: str, stem: str):
    """Embed each figure image as its OWN point in the shared (multimodal) space.

    With Gemini Embedding 2 text and images live in one vector space, so a text
    query can hit a figure's image vector directly. Returns (figure_docs,
    figure_vectors); empty when the embedder can't embed images (e.g. the local
    HF fallback) or no figures are present.
    """
    embeddings = _get_embeddings()
    if not hasattr(embeddings, "embed_image"):
        return [], []

    # Unique figures with the section they were anchored to (first occurrence).
    seen: dict[str, dict] = {}
    for chunk in chunks:
        meta = chunk.metadata
        for name in meta.get("images", []) or []:
            if name not in seen:
                seen[name] = {
                    "heading_path": meta.get("heading_path", ""),
                    "parent_id": meta.get("parent_id", ""),
                }
    if not seen:
        return [], []

    figure_docs, figure_vectors = [], []
    for name, anchor in seen.items():
        img_path = paths.image_path("workspace", stem, name) or paths.image_path("originals", stem, name)
        if img_path is None:
            continue
        try:
            vector = embeddings.embed_image(img_path)
        except Exception as exc:
            print(f"[DocumentManager] Figure embed failed for {name}: {exc}")
            continue
        heading_path = anchor["heading_path"]
        content = f"{heading_path} (figure: {name})" if heading_path else f"figure: {name}"
        doc_id = "fig-" + uuid.uuid5(uuid.NAMESPACE_URL, f"{source}|{name}").hex
        figure_docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": source,
                    "title": stem,
                    "kind": "figure",
                    "name": name,
                    "images": [name],
                    "heading_path": heading_path,
                    "parent_id": anchor["parent_id"],
                    "doc_id": doc_id,
                },
            )
        )
        figure_vectors.append(vector)
    return figure_docs, figure_vectors


def _upsert_to_vector_store(chunks, vectors, channel: str):
    if not chunks or not vectors:
        return
    vector_store.ensure_schema(len(vectors[0]))
    rows = [
        {
            "doc_id": chunk.metadata["doc_id"],
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": vec,
        }
        for chunk, vec in zip(chunks, vectors)
    ]
    vector_store.upsert(channel_collection(channel), rows)


def _append_to_corpus(chunks, channel: str):
    corpus_path = channel_corpus_path(channel)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("a", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(
                json.dumps(
                    {"doc_id": chunk.metadata["doc_id"], "content": chunk.page_content, "metadata": chunk.metadata},
                    ensure_ascii=False,
                ) + "\n"
            )


def _delete_from_vector_store_by_source(source: str, channel: str) -> int:
    return vector_store.delete_by_source(channel_collection(channel), source)


def _remove_from_corpus(source: str, channel: str) -> int:
    corpus_path = channel_corpus_path(channel)
    if not corpus_path.exists():
        return 0

    kept, removed = [], 0
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("metadata", {}).get("source") == source:
                removed += 1
            else:
                kept.append(line)

    tmp = corpus_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.writelines(kept)
    tmp.replace(corpus_path)
    return removed


def _delete_artifacts(source: str) -> None:
    """Remove on-disk markdown/pdf/original artifacts for a source."""
    candidates = [
        paths.originals_md_path(source),
        paths.workspace_md_path(source),
        paths.workspace_pdf_path(source),
        paths.original_doc_path(source),
    ]
    for path in candidates:
        if path and path.exists():
            path.unlink(missing_ok=True)

    for img_dir in (paths.originals_images_dir(source), paths.workspace_images_dir(source)):
        if img_dir.exists():
            shutil.rmtree(img_dir, ignore_errors=True)

    # Remove structured table sidecars for this source.
    tables_dir = _TABLES_DIR_BASE / paths.stem_of(source)
    if tables_dir.exists():
        shutil.rmtree(tables_dir, ignore_errors=True)
