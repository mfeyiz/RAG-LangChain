import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from RAG.services.embeddings import create_embeddings
from RAG.services.retrieval import COLLECTION_NAME, CORPUS_PATH, QDRANT_PATH


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


_LOCK_KEY = "rag:index:lock"
_LOCK_TTL = 900  # 15 min — upper bound for full indexing run


def _qdrant_client():
    from qdrant_client import QdrantClient

    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    if qdrant_url:
        return QdrantClient(url=qdrant_url)
    if QDRANT_PATH.exists():
        return QdrantClient(path=str(QDRANT_PATH))
    return None


def _collection_ready(client) -> bool:
    try:
        if client and client.collection_exists(COLLECTION_NAME):
            count = client.get_collection(COLLECTION_NAME).points_count
            return bool(count and count > 0)
    except Exception:
        pass
    return False


def _stored_vector_dim(client) -> int | None:
    """Return the vector dimension stored in the Qdrant collection, or None if unknown."""
    try:
        info = client.get_collection(COLLECTION_NAME)
        vc = info.config.params.vectors
        if hasattr(vc, "size"):          # unnamed default vector
            return vc.size
        if isinstance(vc, dict) and vc:  # named vectors
            return next(iter(vc.values())).size
    except Exception:
        pass
    return None


def ensure_index() -> None:
    """Build the vector index only if the Qdrant collection is missing, empty, or stale.

    Detects stale indexes (e.g. indexed with a degraded/wrong model) by comparing
    the stored vector dimension against the current embedding model output.

    Uses a Redis distributed lock so only one pod runs indexing when multiple
    replicas start simultaneously. The lock is released whether indexing
    succeeds or fails so other pods are never blocked permanently.
    """
    try:
        client = _qdrant_client()
    except Exception as exc:
        print(f"[Index] Could not connect to Qdrant: {exc}")
        client = None

    # Fast path — already indexed. Validate vector dim before skipping.
    if _collection_ready(client):
        stored_dim = _stored_vector_dim(client)
        if stored_dim is not None:
            expected_dim = len(create_embeddings().embed_query("probe"))
            if stored_dim != expected_dim:
                print(f"[Index] Vector dim mismatch (stored={stored_dim}, expected={expected_dim}) — rebuilding.")
                try:
                    client.delete_collection(COLLECTION_NAME)
                except Exception as exc:
                    print(f"[Index] Could not delete stale collection: {exc}")
                    return
                # Fall through to lock + rebuild.
            else:
                count = client.get_collection(COLLECTION_NAME).points_count
                print(f"[Index] Collection already has {count} points — skipping.")
                return
        else:
            count = client.get_collection(COLLECTION_NAME).points_count
            print(f"[Index] Collection already has {count} points — skipping.")
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
        # Another pod holds the lock — wait for it to finish.
        print("[Index] Another pod is indexing, waiting…")
        for _ in range(180):
            time.sleep(5)
            if _collection_ready(client):
                print("[Index] Index is ready.")
                return
        print("[Index] Timed out waiting for index — proceeding anyway.")
        return

    try:
        # Re-check after acquiring: another pod may have finished between our
        # first check and acquiring the lock.
        if _collection_ready(client):
            print("[Index] Collection indexed by another pod — skipping.")
            return

        print("[Index] Collection missing or empty — building vector index…")
        create_vector_db()
        print("[Index] Indexing complete.")
    finally:
        if r:
            r.delete(_LOCK_KEY)


def create_vector_db():
    documents = load_documents(DATA_DIR)
    chunks = semantic_chunk_documents(documents)
    write_corpus(chunks)
    write_qdrant_index(chunks)
    print(f"Indexed {len(chunks)} chunks.")


def load_documents(data_dir: Path) -> list[Document]:
    documents = []
    has_hotpotqa_json = (data_dir / "hotpotqa").is_dir()

    for root, _, files in os.walk(data_dir):
        for filename in sorted(files):
            filepath = Path(root) / filename
            relative_path = filepath.relative_to(data_dir).as_posix()

            if filename == "hotpotqa_knowledge_base.txt" and has_hotpotqa_json:
                continue

            if filename.endswith(".pdf"):
                for page in PyPDFLoader(str(filepath)).load():
                    page.metadata.update(
                        {
                            "source": relative_path,
                            "kind": "pdf",
                            "title": filepath.stem,
                        }
                    )
                    documents.append(page)
                print(f"Loaded: {relative_path}")

            elif filename.endswith(".txt"):
                for doc in TextLoader(str(filepath), encoding="utf-8").load():
                    doc.metadata.update(
                        {
                            "source": relative_path,
                            "kind": "text",
                            "title": filepath.stem,
                        }
                    )
                    documents.append(doc)
                print(f"Loaded: {relative_path}")

            elif filename.endswith(".json"):
                documents.extend(_load_hotpotqa_json(filepath, relative_path))
                print(f"Loaded: {relative_path}")

    return documents


def semantic_chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n## ", "\n# ", "\n\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        for index, chunk in enumerate(splitter.split_documents([doc])):
            chunk.metadata = dict(chunk.metadata)
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunking"] = "semantic-heading-recursive"
            chunk.metadata["doc_id"] = _stable_doc_id(chunk.page_content, chunk.metadata)
            chunks.append(chunk)
    return chunks


def write_corpus(chunks: list[Document]):
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CORPUS_PATH.with_suffix(".tmp")
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
    tmp_path.replace(CORPUS_PATH)  # atomic — readers see old or new, never partial
    print(f"Corpus written: {CORPUS_PATH}")


def write_qdrant_index(chunks: list[Document]):
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except Exception as exc:
        print(f"Qdrant dependency unavailable; corpus-only index created. {exc}")
        return

    embeddings = create_embeddings()
    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)
    if not vectors:
        print("No vectors created.")
        return

    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    if qdrant_url:
        client = QdrantClient(url=qdrant_url)
        print(f"Connecting to Qdrant server: {qdrant_url}")
    else:
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(QDRANT_PATH))
    vector_size = len(vectors[0])

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points = []
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            PointStruct(
                id=index,
                vector=vector,
                payload={
                    "doc_id": chunk.metadata["doc_id"],
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                },
            )
        )

    for start in range(0, len(points), 64):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start : start + 64],
        )

    print(f"Qdrant collection written: {QDRANT_PATH} / {COLLECTION_NAME}")


def _load_hotpotqa_json(filepath: Path, relative_path: str) -> list[Document]:
    with filepath.open(encoding="utf-8") as file:
        data = json.load(file)

    question = data.get("question", "")
    answer = data.get("answer", "")
    documents = []
    base_metadata = {
        "source": relative_path,
        "dataset": "hotpotqa",
        "question": question,
        "answer": answer,
        "question_type": data.get("type", ""),
        "level": data.get("level", ""),
    }

    if question and answer:
        documents.append(
            Document(
                page_content=f"Question: {question}\nAnswer: {answer}",
                metadata={**base_metadata, "kind": "qa", "title": "Gold QA"},
            )
        )

    for title, body in _extract_sections(data.get("context", "")):
        documents.append(
            Document(
                page_content=f"{title}\n{body}".strip(),
                metadata={**base_metadata, "kind": "context", "title": title},
            )
        )

    return documents


def _extract_sections(context: str) -> list[tuple[str, str]]:
    sections = []
    for raw_section in re.split(r"(?=\n?## )", context.strip()):
        section = raw_section.strip()
        if not section:
            continue
        match = re.match(r"##\s*(.+?)\n(.+)", section, flags=re.DOTALL)
        if match:
            sections.append((match.group(1).strip(), match.group(2).strip()))
        else:
            sections.append(("Untitled", section))
    return sections


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
