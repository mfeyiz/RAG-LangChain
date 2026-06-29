import hashlib
import json
import os
import sys
import time
from pathlib import Path

import re

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from RAG.services.embeddings import create_embeddings
from RAG.services import paths
from RAG.services.retrieval import (
    CHANNELS,
    QDRANT_PATH,
    channel_collection,
    channel_corpus_path,
    get_qdrant_client,
)


# Markdown source directory per channel. Workspace is the editable copy; both are
# rebuilt from disk if the vector store is missing (e.g. fresh PVC after restart).
_CHANNEL_MD_DIR = {
    "originals": paths.ORIGINALS_MD_DIR,
    "workspace": paths.WORKSPACE_MD_DIR,
}

_LOCK_KEY = "rag:index:lock"
_LOCK_TTL = 900  # 15 min — upper bound for full indexing run


def _collection_ready(client, collection_name: str) -> bool:
    try:
        if client and client.collection_exists(collection_name):
            count = client.get_collection(collection_name).points_count
            return bool(count and count > 0)
    except Exception:
        pass
    return False


def _stored_vector_dim(client, collection_name: str) -> int | None:
    """Return the vector dimension stored in a Qdrant collection, or None if unknown."""
    try:
        info = client.get_collection(collection_name)
        vc = info.config.params.vectors
        if hasattr(vc, "size"):          # unnamed default vector
            return vc.size
        if isinstance(vc, dict) and vc:  # named vectors
            return next(iter(vc.values())).size
    except Exception:
        pass
    return None


def ensure_index() -> None:
    """Build the vector index for any channel whose collection is missing, empty, or stale.

    Detects stale indexes (e.g. indexed with a wrong embedding model) by comparing
    the stored vector dimension against the current embedding model output.

    Uses a Redis distributed lock so only one pod runs indexing when multiple
    replicas start simultaneously. The lock is released whether indexing
    succeeds or fails so other pods are never blocked permanently.
    """
    client = get_qdrant_client(create=True)

    from RAG.services.retrieval import _get_embeddings
    expected_dim = len(_get_embeddings().embed_query("probe"))

    channels_to_build = []
    for channel in CHANNELS:
        collection = channel_collection(channel)
        if _collection_ready(client, collection):
            stored_dim = _stored_vector_dim(client, collection)
            if stored_dim is not None and stored_dim != expected_dim:
                print(f"[Index] {collection}: vector dim mismatch (stored={stored_dim}, expected={expected_dim}) — rebuilding.")
                try:
                    client.delete_collection(collection)
                except Exception as exc:
                    print(f"[Index] Could not delete stale collection {collection}: {exc}")
                    continue
                channels_to_build.append(channel)
            else:
                count = client.get_collection(collection).points_count
                print(f"[Index] {collection} already has {count} points — skipping.")
        else:
            channels_to_build.append(channel)

    if not channels_to_build:
        return

    import redis as redis_lib

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    r = None
    acquired = False
    try:
        r = redis_lib.from_url(redis_url)
        acquired = bool(r.set(_LOCK_KEY, "1", nx=True, ex=_LOCK_TTL))
    except Exception as exc:
        print(f"[Index] Redis unavailable, proceeding without lock: {exc}")
        acquired = True

    if not acquired:
        print("[Index] Another pod is indexing, waiting…")
        for _ in range(180):
            time.sleep(5)
            if all(_collection_ready(client, channel_collection(ch)) for ch in channels_to_build):
                print("[Index] Index is ready.")
                return
        print("[Index] Timed out waiting for index — proceeding anyway.")
        return

    try:
        for channel in channels_to_build:
            if _collection_ready(client, channel_collection(channel)):
                continue
            print(f"[Index] Building vector index for channel '{channel}'…")
            create_channel_index(channel)
        print("[Index] Indexing complete.")
    except Exception as exc:
        import traceback
        print(f"[Index] Indexing failed: {exc}")
        traceback.print_exc()
    finally:
        if r:
            r.delete(_LOCK_KEY)


def create_vector_db():
    """Rebuild both channels from their on-disk markdown."""
    for channel in CHANNELS:
        create_channel_index(channel)


def create_channel_index(channel: str):
    # Reuse the already-loaded embedding model from retrieval.
    from RAG.services.retrieval import _get_embeddings
    embeddings = _get_embeddings()
    documents = load_markdown_documents(_CHANNEL_MD_DIR[channel])
    chunks = semantic_chunk_documents(documents)
    write_corpus(chunks, channel)
    write_qdrant_index(chunks, channel, embeddings=embeddings)
    print(f"Indexed {len(chunks)} chunks into channel '{channel}'.")


def load_markdown_documents(md_dir: Path) -> list[Document]:
    documents = []
    if not md_dir.exists():
        return documents
    for filepath in sorted(md_dir.glob("*.md")):
        text = filepath.read_text(encoding="utf-8")
        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": filepath.name,  # "<stem>.md" — uniform across channels
                    "kind": "markdown",
                    "title": filepath.stem,
                },
            )
        )
        print(f"Loaded: {filepath.name}")
    return documents


# Markdown heading levels we split on. Order matters: the splitter assigns the
# nearest enclosing heading at each level to every section it produces.
_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# Matches Markdown image links: ![alt](path "title"). We only keep the basename
# of the path so the reference survives the dual-channel copy into per-doc image
# dirs (the on-disk path differs per channel; the filename does not).
_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)")


def _heading_path(metadata: dict) -> str:
    """Human-readable section path, e.g. 'Security > Threats > Malware'."""
    parts = [metadata.get(level) for _, level in _HEADERS_TO_SPLIT_ON]
    return " > ".join(p for p in parts if p)


def _section_images(text: str) -> list[str]:
    """Basenames of every image referenced in a section's Markdown."""
    seen: list[str] = []
    for match in _IMAGE_LINK_RE.finditer(text):
        name = os.path.basename(match.group(1).strip())
        if name and name not in seen:
            seen.append(name)
    return seen


def semantic_chunk_documents(documents: list[Document]) -> list[Document]:
    """Heading-aware Markdown chunking.

    Stage 1 splits on the ``#``/``##``/``###`` heading tree so each chunk knows
    which section it belongs to (captured in ``h1``/``h2``/``h3`` metadata).
    Stage 2 bounds oversized sections with the character splitter. The heading
    path is prepended to every chunk's content so the embedding and BM25 index
    carry section context, and any image references in the section are recorded
    in ``metadata['images']`` so retrieval can surface figures (text-anchored).
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in documents:
        base_metadata = dict(doc.metadata)
        sections = header_splitter.split_text(doc.page_content)
        # Fallback: a document with no Markdown headings yields a single section.
        if not sections:
            sections = [Document(page_content=doc.page_content, metadata={})]

        index = 0
        for section in sections:
            heading_meta = {
                level: section.metadata[level]
                for _, level in _HEADERS_TO_SPLIT_ON
                if section.metadata.get(level)
            }
            heading_path = _heading_path(heading_meta)
            images = _section_images(section.page_content)

            for piece in size_splitter.split_text(section.page_content):
                content = f"{heading_path}\n\n{piece}" if heading_path else piece
                metadata = {**base_metadata, **heading_meta}
                if heading_path:
                    metadata["heading_path"] = heading_path
                metadata["images"] = images
                metadata["chunk_index"] = index
                metadata["chunking"] = "heading-aware-markdown"
                metadata["doc_id"] = _stable_doc_id(content, metadata)
                chunks.append(Document(page_content=content, metadata=metadata))
                index += 1
    return chunks


def write_corpus(chunks: list[Document], channel: str):
    corpus_path = channel_corpus_path(channel)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = corpus_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    {
                        "doc_id": chunk.metadata["doc_id"],
                        "content": chunk.page_content,
                        "metadata": chunk.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    tmp_path.replace(corpus_path)  # atomic — readers see old or new, never partial
    print(f"Corpus written: {corpus_path}")


def write_qdrant_index(chunks: list[Document], channel: str, embeddings=None):
    try:
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except Exception as exc:
        print(f"Qdrant dependency unavailable; corpus-only index created. {exc}")
        return

    collection = channel_collection(channel)
    client = get_qdrant_client(create=True)
    if client is None:
        print(f"Qdrant client unavailable; skipping {collection}.")
        return

    if not chunks:
        # Drop any stale collection so an empty channel is genuinely empty.
        if client.collection_exists(collection):
            client.delete_collection(collection)
        return

    if embeddings is None:
        embeddings = create_embeddings()

    texts = [chunk.page_content for chunk in chunks]
    print(f"[Index] Embedding {len(texts)} chunks ({collection}) with {embeddings.model_name}…")
    vectors = embeddings.embed_documents(texts)
    print(f"[Index] Embedding complete — {len(vectors)} vectors generated.")
    if not vectors:
        print("No vectors created.")
        return

    vector_size = len(vectors[0])
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=index,
            vector=vector,
            payload={
                "doc_id": chunk.metadata["doc_id"],
                "content": chunk.page_content,
                "metadata": chunk.metadata,
            },
        )
        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    for start in range(0, len(points), 64):
        client.upsert(collection_name=collection, points=points[start: start + 64])

    print(f"Qdrant collection written: {QDRANT_PATH} / {collection}")


def _stable_doc_id(content: str, metadata: dict) -> str:
    key = "|".join(
        [
            metadata.get("source", ""),
            metadata.get("title", ""),
            str(metadata.get("chunk_index", "")),
            content,
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    create_vector_db()
