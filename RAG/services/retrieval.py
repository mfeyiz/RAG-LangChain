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

from RAG.services import vector_store
from RAG.services.embeddings import create_embeddings


DENSE_SEARCH_K = int(os.getenv("DENSE_SEARCH_K", "15"))
BM25_SEARCH_K = int(os.getenv("BM25_SEARCH_K", "15"))
FINAL_CONTEXT_K = 5
# Cap how many merged candidates get reranked. The reranker (cross-encoder on
# CPU) is the dominant latency cost — fewer pairs = much faster retrieval.
RERANK_INPUT_K = int(os.getenv("RERANK_INPUT_K", "12"))
# bge-reranker-base is ~2x faster than -large on CPU with similar ranking quality.
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
# Reranker provider: "vertex" = Vertex AI Ranking API (Discovery Engine);
# "local" = the bge cross-encoder above (offline / no GCP).
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "local").strip().lower()
VERTEX_RANKER_MODEL = os.getenv("VERTEX_RANKER_MODEL", "semantic-ranker-default-004")
# Candidates below this score are dropped after reranking to avoid irrelevant results.
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.05"))

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "vector_db"

# ── Dual-channel configuration ────────────────────────────────────────────────
# Two isolated retrieval channels share a single pgvector table (one ``rag_chunks``
# table, distinguished by the ``channel`` column) but keep separate BM25 corpora
# on disk:
#   - "originals":  read-only ground-truth index (never edited)
#   - "workspace":  active RAG index the chat queries and the editor mutates
# The "collection" value is used as the pgvector ``channel`` column value.
CHANNELS: dict[str, dict] = {
    "originals": {"collection": "rag_originals", "corpus": VECTOR_DIR / "corpus_originals.jsonl"},
    "workspace": {"collection": "rag_workspace", "corpus": VECTOR_DIR / "corpus_workspace.jsonl"},
}
DEFAULT_CHANNEL = "workspace"


def channel_collection(channel: str = DEFAULT_CHANNEL) -> str:
    return CHANNELS[channel]["collection"]


def channel_corpus_path(channel: str = DEFAULT_CHANNEL) -> Path:
    return CHANNELS[channel]["corpus"]


# Backward-compatible aliases — default to the workspace channel.
COLLECTION_NAME = CHANNELS[DEFAULT_CHANNEL]["collection"]
CORPUS_PATH = CHANNELS[DEFAULT_CHANNEL]["corpus"]


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


def retrieve_context(
    query: str, top_k: int = FINAL_CONTEXT_K, channel: str = DEFAULT_CHANNEL
) -> tuple[str, list[dict]]:
    retriever = get_retriever(channel)
    candidates = retriever.retrieve(query, top_k=top_k)
    # Feed the LLM the full parent SECTION of each matched child (deduped by
    # parent_id) for richer context; keep child-level metadata for the evidence
    # panel and for the editor's anchor.
    return format_docs_with_parents(candidates), search_metadata(candidates)


async def retrieve_context_async(
    query: str, top_k: int = FINAL_CONTEXT_K, channel: str = DEFAULT_CHANNEL
) -> tuple[str, list[dict]]:
    return await asyncio.to_thread(retrieve_context, query, top_k, channel)


# Module-level embeddings singleton — loaded once at startup, never tied to a retriever instance.
_embeddings_model = None


def _get_embeddings():
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = create_embeddings()
    return _embeddings_model


# Per-channel retriever singletons.
_RETRIEVERS: dict[str, "HybridRetriever"] = {}


def get_retriever(channel: str = DEFAULT_CHANNEL):
    # Never recreate the instance: if Postgres was None at startup, _dense_search retries inline.
    if channel not in _RETRIEVERS:
        _RETRIEVERS[channel] = HybridRetriever(channel)
    return _RETRIEVERS[channel]


_models_ready = False


def warmup_models() -> None:
    """Eagerly load embedding and reranker models at startup to avoid OOM spikes on first query."""
    global _models_ready
    _get_embeddings()   # load once; result is cached in _embeddings_model
    _load_reranker()
    for channel in CHANNELS:
        get_retriever(channel)   # init corpus + BM25 per channel
    vector_store.get_pool()      # attempt shared Postgres connection
    _models_ready = True


def models_ready() -> bool:
    return _models_ready


class HybridRetriever:
    def __init__(self, channel: str = DEFAULT_CHANNEL):
        self.channel = channel
        self.collection_name = channel_collection(channel)
        self.corpus_path = channel_corpus_path(channel)
        self.corpus = _load_corpus(self.corpus_path)
        self.bm25 = BM25Index(self.corpus)
        # embeddings live in _embeddings_model; the pgvector pool is shared via vector_store.get_pool()

    def reload_corpus(self) -> None:
        """Reload this channel's corpus and rebuild BM25 index after incremental updates."""
        self.corpus = _load_corpus(self.corpus_path)
        self.bm25 = BM25Index(self.corpus)
        print(f"[Retrieval] Corpus reloaded ({self.channel}) — {len(self.corpus)} chunks.")

    def retrieve(self, query: str, top_k: int = FINAL_CONTEXT_K) -> list[RetrievalCandidate]:
        rewritten_query = query.strip()
        cached = _get_cached_results(rewritten_query, top_k, self.channel)
        if cached is not None:
            print(f"[Retrieval] Cache hit for query: {rewritten_query}")
            return cached

        t0 = time.time()
        dense_candidates = self._dense_search(rewritten_query)
        t1 = time.time()
        bm25_candidates = self.bm25.search(rewritten_query, k=BM25_SEARCH_K)
        merged = _merge_candidates(dense_candidates, bm25_candidates)

        if not merged:
            return []

        # Only rerank the most promising candidates by fused dense+bm25 score —
        # the cross-encoder is the latency bottleneck, so fewer pairs is faster.
        merged.sort(key=lambda c: c.dense_score + c.bm25_score, reverse=True)
        rerank_input = merged[:RERANK_INPUT_K]

        t2 = time.time()
        reranked = rerank_candidates(rewritten_query, rerank_input)
        t3 = time.time()
        print(
            f"[Retrieval] timings — dense {t1 - t0:.2f}s | "
            f"rerank {t3 - t2:.2f}s ({len(rerank_input)} pairs) | total {t3 - t0:.2f}s"
        )

        above = [c for c in reranked if c.final_score >= RERANK_SCORE_THRESHOLD]
        # Keep at least 1 result even if everything is below threshold
        selected = (above if above else reranked[:1])[:top_k]
        _cache_results(rewritten_query, top_k, selected, self.channel)
        return selected

    def _dense_search(self, query: str) -> list[RetrievalCandidate]:
        if vector_store.get_pool() is None:
            print(f"[Retrieval] Dense search skipped: Postgres/{self.collection_name} unavailable")
            return []

        try:
            query_vector = _get_embeddings().embed_query(query)
            rows = vector_store.query(self.collection_name, query_vector, DENSE_SEARCH_K)
        except Exception as exc:
            print(f"[Retrieval] Dense search skipped: {exc}")
            return []

        candidates = []
        for row in rows:
            content = row.get("content", "")
            if not content:
                continue
            candidates.append(
                RetrievalCandidate(
                    doc_id=str(row.get("doc_id", "")),
                    content=content,
                    metadata=row.get("metadata", {}),
                    dense_score=float(row.get("score", 0.0) or 0.0),
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
    if not candidates:
        return candidates

    if RERANKER_PROVIDER == "vertex":
        scored = _vertex_rerank(query, candidates)
        if scored is not None:
            return scored
        # fall through to the local reranker if the Vertex call failed

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


def _vertex_rerank(
    query: str, candidates: list[RetrievalCandidate]
) -> list[RetrievalCandidate] | None:
    """Rerank via the Vertex AI Ranking API (Discovery Engine rank service).

    Returns the re-scored, re-sorted candidates, or None on failure so the caller
    can fall back to the local cross-encoder.
    """
    try:
        from google.cloud import discoveryengine_v1 as discoveryengine

        project = os.environ["GOOGLE_CLOUD_PROJECT"]
        client = discoveryengine.RankServiceClient()
        ranking_config = client.ranking_config_path(
            project=project, location="global", ranking_config="default_ranking_config"
        )
        records = [
            discoveryengine.RankingRecord(
                id=str(i),
                # Figures have no real text — fall back to their heading/caption.
                content=(candidate.content or candidate.metadata.get("heading_path", ""))[:8000],
            )
            for i, candidate in enumerate(candidates)
        ]
        response = client.rank(
            request=discoveryengine.RankRequest(
                ranking_config=ranking_config,
                model=VERTEX_RANKER_MODEL,
                query=query,
                records=records,
            )
        )
        for record in response.records:
            candidates[int(record.id)].rerank_score = float(record.score)
        return sorted(candidates, key=lambda item: item.final_score, reverse=True)
    except Exception as exc:
        print(f"[Retrieval] Vertex rerank skipped: {exc}")
        return None


def format_docs_with_parents(candidates: Iterable[RetrievalCandidate]) -> str:
    """Format retrieved candidates for the LLM, expanding each child to its full
    parent SECTION.

    Sections are emitted in the children's rank order and deduplicated by
    ``parent_id`` so the LLM sees each relevant section once, whole — giving it
    the surrounding context the matched snippet lives in. Falls back to the
    child's own content when a parent section isn't recorded (older index).
    """
    blocks: list[str] = []
    seen: set[str] = set()
    index = 0
    for candidate in candidates:
        parent_id = candidate.metadata.get("parent_id")
        parent_content = candidate.metadata.get("parent_content")
        if parent_id and parent_id in seen:
            continue
        if parent_id:
            seen.add(parent_id)
        index += 1
        source = candidate.metadata.get("source", "unknown")
        title = candidate.metadata.get("title", "untitled")
        kind = candidate.metadata.get("kind", "document")
        body = parent_content or candidate.content
        blocks.append(
            "\n".join(
                [
                    f"[{index}] Source: {source} | Title: {title} | Kind: {kind} | Score: {candidate.final_score:.4f}",
                    body,
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
            "images": candidate.metadata.get("images", []),
            "parent_id": candidate.metadata.get("parent_id", ""),
            "parent_content": candidate.metadata.get("parent_content", ""),
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


def _load_corpus(corpus_path: Path = CORPUS_PATH) -> list[RetrievalCandidate]:
    if not corpus_path.exists():
        print(f"[Retrieval] Corpus not found at {corpus_path}. Run RAG/services/rag_service.py.")
        return []

    corpus = []
    with corpus_path.open(encoding="utf-8") as file:
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


def _get_cached_results(
    query: str, top_k: int, channel: str = DEFAULT_CHANNEL
) -> list[RetrievalCandidate] | None:
    ttl = _cache_ttl_seconds()
    if ttl <= 0:
        return None

    key = _cache_key(query, top_k, channel)
    cached = _redis_get(key) or _memory_get(key, ttl)
    if cached is None:
        return None

    return [RetrievalCandidate.from_dict(item) for item in cached]


def _cache_results(
    query: str, top_k: int, candidates: list[RetrievalCandidate], channel: str = DEFAULT_CHANNEL
):
    ttl = _cache_ttl_seconds()
    if ttl <= 0:
        return

    key = _cache_key(query, top_k, channel)
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


def _cache_key(query: str, top_k: int, channel: str = DEFAULT_CHANNEL) -> str:
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    return f"rag:retrieval:{channel}:{top_k}:{digest}"


def invalidate_channel_cache(channel: str = DEFAULT_CHANNEL) -> None:
    """Drop every cached retrieval result for a channel.

    Called after a channel's index changes (add / delete / @update reindex) so a
    fresh chat with a previously-asked question retrieves the UPDATED corpus
    instead of stale pre-update candidates that would otherwise live for the
    ``RETRIEVAL_CACHE_TTL_SECONDS`` window.
    """
    prefix = f"rag:retrieval:{channel}:"

    # In-process memory cache (used when Redis is unavailable).
    for key in [k for k in _RETRIEVAL_MEMORY_CACHE if k.startswith(prefix)]:
        _RETRIEVAL_MEMORY_CACHE.pop(key, None)

    client = _load_redis_client()
    if client is None:
        return
    try:
        stale = list(client.scan_iter(match=f"{prefix}*"))
        if stale:
            client.delete(*stale)
    except Exception as exc:
        print(f"[Retrieval] cache invalidation failed for {channel}: {exc}")


def _cache_ttl_seconds() -> int:
    return int(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS", "900"))
