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
from RAG.services import paths, vector_store
from RAG.services.retrieval import (
    CHANNELS,
    channel_collection,
    channel_corpus_path,
)


# Markdown source directory per channel. Workspace is the editable copy; both are
# rebuilt from disk if the vector store is missing (e.g. fresh PVC after restart).
_CHANNEL_MD_DIR = {
    "originals": paths.ORIGINALS_MD_DIR,
    "workspace": paths.WORKSPACE_MD_DIR,
}

_LOCK_KEY = "rag:index:lock"
_LOCK_TTL = 900  # 15 min — upper bound for full indexing run


def _collection_ready(collection_name: str) -> bool:
    return vector_store.channel_ready(collection_name)


def ensure_index() -> None:
    """Build the vector index for any channel whose data is missing, empty, or stale.

    Stale indexes (e.g. indexed with a wrong embedding model) are detected inside
    ``vector_store.ensure_schema``: it compares the table's embedding dimension
    against the current model and drops/recreates the table on mismatch, so a
    dimension change clears every channel and they get rebuilt here.

    Uses a Redis distributed lock so only one pod runs indexing when multiple
    replicas start simultaneously. The lock is released whether indexing
    succeeds or fails so other pods are never blocked permanently.
    """
    from RAG.services.retrieval import _get_embeddings
    expected_dim = len(_get_embeddings().embed_query("probe"))

    # Provisions the table (dropping it first if the stored dimension is stale).
    if not vector_store.ensure_schema(expected_dim):
        print("[Index] Postgres unavailable — skipping vector indexing.")
        return

    channels_to_build = []
    for channel in CHANNELS:
        collection = channel_collection(channel)
        if _collection_ready(collection):
            count = vector_store.channel_count(collection)
            print(f"[Index] {collection} already has {count} rows — skipping.")
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
            if all(_collection_ready(channel_collection(ch)) for ch in channels_to_build):
                print("[Index] Index is ready.")
                return
        print("[Index] Timed out waiting for index — proceeding anyway.")
        return

    try:
        for channel in channels_to_build:
            if _collection_ready(channel_collection(channel)):
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
    write_vector_index(chunks, channel, embeddings=embeddings)
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


def _parent_id(source: str, heading_path: str, ordinal: int) -> str:
    """Stable id for a heading section (the 'parent' of its size-split pieces)."""
    key = f"{source}|{heading_path}|{ordinal}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def semantic_chunk_documents(documents: list[Document]) -> list[Document]:
    """Heading-aware, parent-aware Markdown chunking.

    Stage 1 splits on the ``#``/``##``/``###`` heading tree so each chunk knows
    which section it belongs to (captured in ``h1``/``h2``/``h3`` metadata). Each
    such section is the *parent*: its full verbatim text is recorded on every
    child piece as ``metadata['parent_content']`` (with a shared ``parent_id``)
    so retrieval can expand a matched child back to its whole section, and the
    editor can rewrite that section in place.

    Stage 2 bounds oversized sections with the character splitter. The heading
    path is prepended to every chunk's content so the embedding and BM25 index
    carry section context, and any image references in the section are recorded
    in ``metadata['images']`` so retrieval can surface figures.
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
        source = base_metadata.get("source", "")
        sections = header_splitter.split_text(doc.page_content)
        # Fallback: a document with no Markdown headings yields a single section.
        if not sections:
            sections = [Document(page_content=doc.page_content, metadata={})]

        index = 0
        for ordinal, section in enumerate(sections):
            heading_meta = {
                level: section.metadata[level]
                for _, level in _HEADERS_TO_SPLIT_ON
                if section.metadata.get(level)
            }
            heading_path = _heading_path(heading_meta)
            images = _section_images(section.page_content)
            # The parent is the section's full, verbatim markdown — a byte-for-byte
            # slice of the source file, so the editor can locate and replace it.
            parent_content = section.page_content
            parent_id = _parent_id(source, heading_path, ordinal)

            for piece in size_splitter.split_text(section.page_content):
                content = f"{heading_path}\n\n{piece}" if heading_path else piece
                metadata = {**base_metadata, **heading_meta}
                if heading_path:
                    metadata["heading_path"] = heading_path
                metadata["images"] = images
                metadata["parent_id"] = parent_id
                metadata["parent_content"] = parent_content
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


def write_vector_index(chunks: list[Document], channel: str, embeddings=None):
    collection = channel_collection(channel)
    if vector_store.get_pool() is None:
        print(f"Postgres unavailable; corpus-only index created ({collection}).")
        return

    if not chunks:
        # Clear the channel so an empty channel is genuinely empty.
        vector_store.replace_channel(collection, [])
        return

    if embeddings is None:
        embeddings = create_embeddings()

    texts = [chunk.page_content for chunk in chunks]
    model_label = getattr(embeddings, "model_name", None) or getattr(embeddings, "model", "?")
    print(f"[Index] Embedding {len(texts)} chunks ({collection}) with {model_label}…")
    vectors = embeddings.embed_documents(texts)
    print(f"[Index] Embedding complete — {len(vectors)} vectors generated.")
    if not vectors:
        print("No vectors created.")
        return

    vector_store.ensure_schema(len(vectors[0]))
    rows = [
        {
            "doc_id": chunk.metadata["doc_id"],
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    vector_store.replace_channel(collection, rows)
    print(f"pgvector channel written: {collection} ({len(rows)} rows)")


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
