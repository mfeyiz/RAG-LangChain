# RAG Multi-Agent System

FastAPI, LangGraph, BGE-M3, Qdrant, hybrid retrieval, BGE reranking, and optional RAGAS/DeepEval evaluation.

## Pipeline

```text
Documents
  -> Semantic Chunking
  -> Metadata Extraction
  -> BGE-M3 Embeddings
  -> Qdrant + corpus.jsonl

User Query
  -> Query Rewrite
  -> BM25 + Dense Retrieval
  -> BGE Reranker
  -> Top 5 Context
  -> GPT / Gemini / Claude compatible LLM
  -> Answer with citations

Evaluation
  -> RAGAS + DeepEval
```

## Setup

```bash
uv sync --dev
```

Create `.env`:

```bash
OPENROUTER_API_KEY=your_key
```

Build the retrieval index:

```bash
uv run python RAG/services/rag_service.py
```

Run the app:

```bash
uv run python RAG/app.py
```

Open `http://localhost:8000`.

## Notes

- `RAG/services/rag_service.py` builds `RAG/vector_db/corpus.jsonl` and a local Qdrant collection under `RAG/vector_db/qdrant`.
- `RAG/services/retrieval.py` merges BM25 and Qdrant dense candidates, then reranks with `BAAI/bge-reranker-large`.
- If Qdrant or the reranker is unavailable, the app falls back gracefully to corpus/BM25 retrieval.
- The old FAISS and unused tool-calling service path have been removed.
