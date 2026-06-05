import json
import math
import os
import re
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


def retrieve_context(query: str, top_k: int = FINAL_CONTEXT_K) -> tuple[str, list[dict]]:
    retriever = get_retriever()
    candidates = retriever.retrieve(query, top_k=top_k)
    return format_docs(candidates), search_metadata(candidates)


@lru_cache(maxsize=1)
def get_retriever():
    return HybridRetriever()


class HybridRetriever:
    def __init__(self):
        self.corpus = _load_corpus()
        self.bm25 = BM25Index(self.corpus)
        self.qdrant = _load_qdrant_client()
        self.embeddings = None

    def retrieve(self, query: str, top_k: int = FINAL_CONTEXT_K) -> list[RetrievalCandidate]:
        rewritten_query = query.strip()
        dense_candidates = self._dense_search(rewritten_query)
        bm25_candidates = self.bm25.search(rewritten_query, k=BM25_SEARCH_K)
        merged = _merge_candidates(dense_candidates, bm25_candidates)

        if not merged:
            return []

        reranked = rerank_candidates(rewritten_query, merged)
        return reranked[:top_k]

    def _dense_search(self, query: str) -> list[RetrievalCandidate]:
        if self.qdrant is None:
            return []

        try:
            if self.embeddings is None:
                self.embeddings = create_embeddings()
            query_vector = self.embeddings.embed_query(query)
            points = _qdrant_query(self.qdrant, query_vector)
        except Exception as exc:
            print(f"[Retrieval] Dense search skipped: {exc}")
            return []

        candidates = []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            content = payload.get("content", "")
            metadata = payload.get("metadata", {})
            if not content:
                continue
            candidates.append(
                RetrievalCandidate(
                    doc_id=str(payload.get("doc_id") or getattr(point, "id", "")),
                    content=content,
                    metadata=metadata,
                    dense_score=float(getattr(point, "score", 0.0) or 0.0),
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
    if not QDRANT_PATH.exists():
        return None

    try:
        from qdrant_client import QdrantClient

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
