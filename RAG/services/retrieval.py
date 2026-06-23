import asyncio
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from RAG.services.embeddings import create_embeddings


COLLECTION_NAME = "rag_documents"
DENSE_SEARCH_K = 30
BM25_SEARCH_K = 30
FINAL_CONTEXT_K = 5
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
# Candidates below this score are dropped after reranking to avoid irrelevant results.
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.05"))

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "vector_db"
QDRANT_PATH = VECTOR_DIR / "qdrant"
CORPUS_PATH = VECTOR_DIR / "corpus.jsonl"


@dataclass
class RetrievalCandidate:
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        if self.rerank_score is not None:
            return self.rerank_score
        return self.dense_score + self.bm25_score

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "dense_score": self.dense_score,
            "bm25_score": self.bm25_score,
            "rerank_score": self.rerank_score,
        }

    @classmethod
    def from_dict(cls, data: dict):
        rerank_score = data.get("rerank_score")
        return cls(
            doc_id=str(data.get("doc_id", "")),
            content=str(data.get("content", "")),
            metadata=data.get("metadata", {}),
            dense_score=float(data.get("dense_score", 0.0) or 0.0),
            bm25_score=float(data.get("bm25_score", 0.0) or 0.0),
            rerank_score=float(rerank_score) if rerank_score is not None else None,
        )


def retrieve_context(query: str, top_k: int = FINAL_CONTEXT_K) -> tuple[str, list[dict]]:
    retriever = get_retriever()
    candidates = retriever.retrieve(query, top_k=top_k)
    return format_docs(candidates), search_metadata(candidates)


async def retrieve_context_async(query: str, top_k: int = FINAL_CONTEXT_K) -> tuple[str, list[dict]]:
    return await asyncio.to_thread(retrieve_context, query, top_k)


# Module-level embeddings singleton — loaded once at startup, never tied to a retriever instance.
_embeddings_model = None


def _get_embeddings():
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = create_embeddings()
    return _embeddings_model


def get_retriever():
    # Never recreate the instance: if qdrant was None at startup, _dense_search retries inline.
    if not hasattr(get_retriever, "_instance"):
        get_retriever._instance = HybridRetriever()
    return get_retriever._instance


_models_ready = False


def warmup_models() -> None:
    """Eagerly load embedding and reranker models at startup to avoid OOM spikes on first query."""
    global _models_ready
    _get_embeddings()   # load once; result is cached in _embeddings_model
    _load_reranker()
    get_retriever()     # init corpus + BM25 + attempt Qdrant connection
    _models_ready = True


def models_ready() -> bool:
    return _models_ready


class HybridRetriever:
    def __init__(self):
        self.corpus = _load_corpus()
        self.bm25 = BM25Index(self.corpus)
        self.qdrant = _load_qdrant_client()
        # embeddings live in _embeddings_model, not here

    def reload_corpus(self) -> None:
        """Reload corpus.jsonl and rebuild BM25 index after incremental updates."""
        self.corpus = _load_corpus()
        self.bm25 = BM25Index(self.corpus)
        print(f"[Retrieval] Corpus reloaded — {len(self.corpus)} chunks.")

    def retrieve(self, query: str, top_k: int = FINAL_CONTEXT_K) -> list[RetrievalCandidate]:
        rewritten_query = query.strip()
        cached = _get_cached_results(rewritten_query, top_k)
        if cached is not None:
            print(f"[Retrieval] Cache hit for query: {rewritten_query}")
            return cached

        dense_candidates = self._dense_search(rewritten_query)
        bm25_candidates = self.bm25.search(rewritten_query, k=BM25_SEARCH_K)
        merged = _merge_candidates(dense_candidates, bm25_candidates)

        if not merged:
            return []

        reranked = rerank_candidates(rewritten_query, merged)
        above = [c for c in reranked if c.final_score >= RERANK_SCORE_THRESHOLD]
        # Keep at least 1 result even if everything is below threshold
        selected = (above if above else reranked[:1])[:top_k]
        _cache_results(rewritten_query, top_k, selected)
        return selected

    def _dense_search(self, query: str) -> list[RetrievalCandidate]:
        # Retry Qdrant connection on every call if previously unavailable.
        if self.qdrant is None:
            self.qdrant = _load_qdrant_client()
        if self.qdrant is None:
            print("[Retrieval] Dense search skipped: Qdrant unavailable")
            return []

        try:
            query_vector = _get_embeddings().embed_query(query)
            points = _qdrant_query(self.qdrant, query_vector)
        except Exception as exc:
            print(f"[Retrieval] Dense search skipped: {exc}")
            self.qdrant = None  # force reconnect next time
            return []

        candidates = []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            content = payload.get("content", "")
            metadata = payload.get("metadata", {})
            if not content:
                continue
            score = getattr(point, "score", None)
            candidates.append(
                RetrievalCandidate(
                    doc_id=str(payload.get("doc_id") or getattr(point, "id", "")),
                    content=content,
                    metadata=metadata,
                    dense_score=float(score) if score is not None else 0.0,
                )
            )
        return candidates


class BM25Index:
    def __init__(self, corpus: list[RetrievalCandidate]):
        self.corpus = corpus
        self.tokenized = [_tokenize(doc.content) for doc in corpus]
        self.avg_doc_len = (
            sum(len(tokens) for tokens in self.tokenized) / len(self.tokenized)
            if self.tokenized
            else 0
        )
        self.doc_freqs = self._document_frequencies(self.tokenized)

    def search(self, query: str, k: int) -> list[RetrievalCandidate]:
        query_terms = _tokenize(query)
        if not query_terms or not self.corpus:
            return []

        scored = []
        for index, doc in enumerate(self.corpus):
            score = self._score(query_terms, self.tokenized[index])
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        max_score = scored[0][0] if scored else 1.0

        results = []
        for score, doc in scored[:k]:
            results.append(
                RetrievalCandidate(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    metadata=doc.metadata,
                    bm25_score=score / max_score,
                )
            )
        return results

    def _score(self, query_terms: list[str], doc_terms: list[str]) -> float:
        k1 = 1.5
        b = 0.75
        doc_len = len(doc_terms)
        if doc_len == 0:
            return 0.0

        term_counts = {}
        for term in doc_terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        score = 0.0
        total_docs = len(self.corpus)
        for term in query_terms:
            if term not in term_counts:
                continue
            doc_freq = self.doc_freqs.get(term, 0)
            idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            frequency = term_counts[term]
            denominator = frequency + k1 * (1 - b + b * doc_len / self.avg_doc_len)
            score += idf * (frequency * (k1 + 1)) / denominator
        return score

    @staticmethod
    def _document_frequencies(tokenized_docs: list[list[str]]) -> dict[str, int]:
        frequencies = {}
        for tokens in tokenized_docs:
            for term in set(tokens):
                frequencies[term] = frequencies.get(term, 0) + 1
        return frequencies


def rerank_candidates(query: str, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    reranker = _load_reranker()
    if reranker is None:
        return sorted(candidates, key=lambda item: item.final_score, reverse=True)

    try:
        pairs = [(query, candidate.content) for candidate in candidates]
        scores = reranker.predict(pairs)
    except Exception as exc:
        print(f"[Retrieval] Reranker skipped: {exc}")
        return sorted(candidates, key=lambda item: item.final_score, reverse=True)

    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = float(score)

    return sorted(candidates, key=lambda item: item.final_score, reverse=True)


def format_docs(candidates: Iterable[RetrievalCandidate]) -> str:
    blocks = []
    for index, candidate in enumerate(candidates, start=1):
        source = candidate.metadata.get("source", "unknown")
        title = candidate.metadata.get("title", "untitled")
        kind = candidate.metadata.get("kind", "document")
        score = candidate.final_score
        blocks.append(
            "\n".join(
                [
                    f"[{index}] Source: {source} | Title: {title} | Kind: {kind} | Score: {score:.4f}",
                    candidate.content,
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "No relevant documents found."


def search_metadata(candidates: Iterable[RetrievalCandidate], content_limit: int = 500) -> list[dict]:
    return [
        {
            "content": _snippet(candidate.content, content_limit),
            "source": candidate.metadata.get("source", "unknown"),
            "title": candidate.metadata.get("title", "untitled"),
            "kind": candidate.metadata.get("kind", "document"),
            "dense_score": candidate.dense_score,
            "bm25_score": candidate.bm25_score,
            "rerank_score": candidate.rerank_score,
            "score": candidate.final_score,
            "relevant": True,
        }
        for candidate in candidates
    ]


def _merge_candidates(*candidate_groups: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    merged = {}
    for group in candidate_groups:
        for candidate in group:
            key = candidate.doc_id or candidate.content[:120]
            if key not in merged:
                merged[key] = candidate
                continue
            merged[key].dense_score = max(merged[key].dense_score, candidate.dense_score)
            merged[key].bm25_score = max(merged[key].bm25_score, candidate.bm25_score)
    return list(merged.values())


def _load_corpus() -> list[RetrievalCandidate]:
    if not CORPUS_PATH.exists():
        print(f"[Retrieval] Corpus not found at {CORPUS_PATH}. Run RAG/services/rag_service.py.")
        return []

    corpus = []
    with CORPUS_PATH.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            corpus.append(
                RetrievalCandidate(
                    doc_id=str(record["doc_id"]),
                    content=record["content"],
                    metadata=record.get("metadata", {}),
                )
            )
    return corpus


def _load_qdrant_client():
    try:
        from qdrant_client import QdrantClient

        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        if qdrant_url:
            client = QdrantClient(url=qdrant_url)
        else:
            if not QDRANT_PATH.exists():
                return None
            client = QdrantClient(path=str(QDRANT_PATH))

        if not client.collection_exists(COLLECTION_NAME):
            return None
        return client
    except Exception as exc:
        print(f"[Retrieval] Qdrant unavailable: {exc}")
        return None


def _qdrant_query(client, query_vector: list[float]):
    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=DENSE_SEARCH_K,
            with_payload=True,
        )
        return response.points
    except AttributeError:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=DENSE_SEARCH_K,
            with_payload=True,
        )


@lru_cache(maxsize=1)
def _load_reranker():
    if os.getenv("RAG_ENABLE_RERANKER", "1") == "0":
        return None

    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(RERANKER_MODEL_NAME)
    except Exception as exc:
        print(f"[Retrieval] Reranker unavailable: {exc}")
        return None


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[\w'-]+", text, flags=re.UNICODE)
        if len(token) > 1
    ]


def _snippet(content: str, content_limit: int) -> str:
    content = content.strip()
    if len(content) <= content_limit:
        return content
    return f"{content[:content_limit].rstrip()}..."


_RETRIEVAL_MEMORY_CACHE: dict[str, tuple[float, list[dict]]] = {}


def _get_cached_results(query: str, top_k: int) -> list[RetrievalCandidate] | None:
    ttl = _cache_ttl_seconds()
    if ttl <= 0:
        return None

    key = _cache_key(query, top_k)
    cached = _redis_get(key) or _memory_get(key, ttl)
    if cached is None:
        return None

    return [RetrievalCandidate.from_dict(item) for item in cached]


def _cache_results(query: str, top_k: int, candidates: list[RetrievalCandidate]):
    ttl = _cache_ttl_seconds()
    if ttl <= 0:
        return

    key = _cache_key(query, top_k)
    payload = [candidate.to_dict() for candidate in candidates]
    if not _redis_set(key, payload, ttl):
        _RETRIEVAL_MEMORY_CACHE[key] = (time.time(), payload)


def _memory_get(key: str, ttl: int) -> list[dict] | None:
    cached = _RETRIEVAL_MEMORY_CACHE.get(key)
    if cached is None:
        return None

    created_at, payload = cached
    if time.time() - created_at > ttl:
        _RETRIEVAL_MEMORY_CACHE.pop(key, None)
        return None
    return payload


def _redis_get(key: str) -> list[dict] | None:
    client = _load_redis_client()
    if client is None:
        return None

    raw = client.get(key)
    return json.loads(raw) if raw else None


def _redis_set(key: str, payload: list[dict], ttl: int) -> bool:
    client = _load_redis_client()
    if client is None:
        return False

    try:
        client.setex(key, ttl, json.dumps(payload, ensure_ascii=False))
        return True
    except Exception as exc:
        print(f"[Retrieval] Redis write failed, falling back to memory cache: {exc}")
        return False


_redis_client = None
_redis_retry_after: float = 0.0
_REDIS_RETRY_INTERVAL = 30.0


def _load_redis_client():
    global _redis_client, _redis_retry_after
    if _redis_client is not None:
        return _redis_client

    now = time.time()
    if now < _redis_retry_after:
        return None

    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return None

    try:
        from redis import Redis

        client = Redis.from_url(redis_url, decode_responses=True)
        client.ping()
        _redis_client = client
        return client
    except Exception as exc:
        print(f"[Retrieval] Redis unavailable, retrying in {_REDIS_RETRY_INTERVAL}s: {exc}")
        _redis_retry_after = now + _REDIS_RETRY_INTERVAL
        return None


def _cache_key(query: str, top_k: int) -> str:
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    return f"rag:retrieval:{top_k}:{digest}"


def _cache_ttl_seconds() -> int:
    return int(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS", "900"))
