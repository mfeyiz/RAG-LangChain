"""Incremental document indexing: add or delete documents without full re-index."""
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from RAG.services.retrieval import (
    COLLECTION_NAME,
    CORPUS_PATH,
    QDRANT_PATH,
    get_retriever,
    _get_embeddings,
)
from RAG.services.rag_service import semantic_chunk_documents, write_corpus, _stable_doc_id

UPLOADS_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"


def add_document(file_bytes: bytes, filename: str) -> dict:
    """
    Save file, chunk it, embed and upsert into Qdrant, append to corpus, reload BM25.
    Returns {filename, chunks_added, doc_ids}.
    """
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_core.documents import Document

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Write to a temp file first so loaders can read it from disk.
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOADS_DIR) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        relative = tmp_path.relative_to(tmp_path.parent.parent.parent / "data")
        rel_str = str(relative).replace("\\", "/")

        if suffix == ".pdf":
            raw_docs = PyPDFLoader(str(tmp_path)).load()
            for doc in raw_docs:
                doc.metadata.update({"source": rel_str, "kind": "pdf", "title": tmp_path.stem})
        elif suffix == ".txt":
            raw_docs = TextLoader(str(tmp_path), encoding="utf-8").load()
            for doc in raw_docs:
                doc.metadata.update({"source": rel_str, "kind": "text", "title": tmp_path.stem})
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Only .pdf and .txt are accepted.")

        chunks = semantic_chunk_documents(raw_docs)

        # Persist a named copy (stem + timestamp) so it survives server restarts.
        final_path = UPLOADS_DIR / f"{tmp_path.stem}_{int(time.time())}{suffix}"
        shutil.move(str(tmp_path), str(final_path))

        if not chunks:
            return {"filename": filename, "chunks_added": 0, "doc_ids": []}

        for chunk in chunks:
            chunk.metadata["source"] = str(final_path.relative_to(final_path.parent.parent.parent / "data")).replace("\\", "/")

        _upsert_to_qdrant(chunks)
        _append_to_corpus(chunks)

        retriever = get_retriever()
        retriever.reload_corpus()

        doc_ids = [c.metadata["doc_id"] for c in chunks]
        return {"filename": filename, "chunks_added": len(chunks), "doc_ids": doc_ids}

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def delete_document_by_source(source: str) -> dict:
    """
    Remove all chunks matching `source` from Qdrant and corpus.jsonl.
    Returns {source, chunks_deleted}.
    """
    deleted_ids = _delete_from_qdrant_by_source(source)
    removed = _remove_from_corpus(source)
    retriever = get_retriever()
    retriever.reload_corpus()
    return {"source": source, "chunks_deleted": max(deleted_ids, removed)}


def list_documents() -> list[dict]:
    """Return unique sources with chunk counts from corpus.jsonl."""
    if not CORPUS_PATH.exists():
        return []

    counts: dict[str, int] = {}
    with CORPUS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            src = rec.get("metadata", {}).get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1

    return [{"source": src, "chunks": cnt} for src, cnt in sorted(counts.items())]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _upsert_to_qdrant(chunks):
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct

        retriever = get_retriever()
        client = retriever.qdrant

        if client is None or not client.collection_exists(COLLECTION_NAME):
            return

        embeddings = _get_embeddings()
        texts = [c.page_content for c in chunks]
        vectors = embeddings.embed_documents(texts)

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
            client.upsert(collection_name=COLLECTION_NAME, points=points[start: start + 64])
    except Exception as exc:
        print(f"[DocumentManager] Qdrant upsert failed: {exc}")


def _append_to_corpus(chunks):
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_PATH.open("a", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(
                json.dumps(
                    {"doc_id": chunk.metadata["doc_id"], "content": chunk.page_content, "metadata": chunk.metadata},
                    ensure_ascii=False,
                ) + "\n"
            )


def _delete_from_qdrant_by_source(source: str) -> int:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        retriever = get_retriever()
        client = retriever.qdrant

        if client is None or not client.collection_exists(COLLECTION_NAME):
            return 0

        result = client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]
            ),
        )
        return getattr(result, "deleted_count", 0) or 0
    except Exception as exc:
        print(f"[DocumentManager] Qdrant delete failed: {exc}")
        return 0


def _remove_from_corpus(source: str) -> int:
    if not CORPUS_PATH.exists():
        return 0

    kept, removed = [], 0
    with CORPUS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("metadata", {}).get("source") == source:
                removed += 1
            else:
                kept.append(line)

    tmp = CORPUS_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.writelines(kept)
    tmp.replace(CORPUS_PATH)
    return removed
