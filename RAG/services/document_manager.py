"""Dual-channel incremental indexing.

Ingestion writes every uploaded document into BOTH channels (read-only
`originals` and editable `workspace`); the editor write-back only touches
`workspace`. Add/delete keep Qdrant and the per-channel corpus.jsonl in sync
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
from RAG.services import paths
from RAG.services.retrieval import (
    CHANNELS,
    DEFAULT_CHANNEL,
    channel_collection,
    channel_corpus_path,
    get_qdrant_client,
    get_retriever,
    _collection_exists,
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
    for channel in CHANNELS:
        # Replace any prior copy of this source (re-upload == update).
        _delete_from_qdrant_by_source(source, channel)
        _remove_from_corpus(source, channel)
        _upsert_to_qdrant(chunks, vectors, channel)
        _append_to_corpus(chunks, channel)
        get_retriever(channel).reload_corpus()

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

    _delete_from_qdrant_by_source(source, "workspace")
    _remove_from_corpus(source, "workspace")

    if chunks:
        vectors = _get_embeddings().embed_documents([c.page_content for c in chunks])
        _upsert_to_qdrant(chunks, vectors, "workspace")
        _append_to_corpus(chunks, "workspace")

    get_retriever("workspace").reload_corpus()
    return {"source": source, "chunks_added": len(chunks)}


def delete_document_by_source(source: str, channel: str | None = None) -> dict:
    """Remove a document's chunks. By default removes from BOTH channels and deletes
    the on-disk artifacts; pass a channel to scope the vector delete only."""
    channels = [channel] if channel else list(CHANNELS)
    deleted = 0
    for ch in channels:
        deleted += max(_delete_from_qdrant_by_source(source, ch), _remove_from_corpus(source, ch))
        get_retriever(ch).reload_corpus()

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


def _upsert_to_qdrant(chunks, vectors, channel: str):
    try:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        client = get_qdrant_client(create=True)
        if client is None:
            return
        collection = channel_collection(channel)
        if not _collection_exists(client, collection):
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
            )

        points = [
            PointStruct(
                id=uuid.uuid4().hex,
                vector=vec,
                payload={
                    "doc_id": chunk.metadata["doc_id"],
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vec in zip(chunks, vectors)
        ]
        for start in range(0, len(points), 64):
            client.upsert(collection_name=collection, points=points[start: start + 64])
    except Exception as exc:
        print(f"[DocumentManager] Qdrant upsert failed ({channel}): {exc}")


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


def _delete_from_qdrant_by_source(source: str, channel: str) -> int:
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = get_qdrant_client()
        collection = channel_collection(channel)
        if client is None or not _collection_exists(client, collection):
            return 0

        result = client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]
            ),
        )
        return getattr(result, "deleted_count", 0) or 0
    except Exception as exc:
        print(f"[DocumentManager] Qdrant delete failed ({channel}): {exc}")
        return 0


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
