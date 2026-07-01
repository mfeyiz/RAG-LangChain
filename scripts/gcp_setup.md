# Google Cloud retrieval stack — setup

The retrieval stack is **provider-agnostic**: each component is chosen by an env
flag and imported lazily, so you can mix Google and local, and swap providers
later without touching code. Only the components whose flag points at Google need
the steps below.

| Component | Env flag | Google value | Local fallback |
|-----------|----------|--------------|----------------|
| Doc parser | `DOC_PARSER` | `documentai` | `docling`, `markitdown` |
| Embeddings | `EMBEDDING_PROVIDER` | `google` (Gemini Embedding 2) | `hf` (sentence-transformers) |
| Reranker | `RERANKER_PROVIDER` | `vertex` (Ranking API) | `local` (bge cross-encoder) |
| Vector DB | — | **self-hosted Postgres + pgvector** (cluster pod) | local Postgres (docker) |

> The vector DB is your own Postgres with the pgvector extension. Set
> `DATABASE_URL` to the in-cluster service (e.g.
> `postgresql://rag:PASSWORD@postgres:5432/rag`). For local dev:
> `docker run -e POSTGRES_PASSWORD=pg -p 5432:5432 pgvector/pgvector:pg16`.

## 1. One-time GCP project setup

```bash
gcloud auth login
gcloud config set project "$GOOGLE_CLOUD_PROJECT"

# Enable the APIs for whichever Google components you turn on:
gcloud services enable documentai.googleapis.com        # DOC_PARSER=documentai
gcloud services enable aiplatform.googleapis.com         # EMBEDDING_PROVIDER=google
gcloud services enable discoveryengine.googleapis.com    # RERANKER_PROVIDER=vertex

# Service account + key (fills GOOGLE_APPLICATION_CREDENTIALS):
gcloud iam service-accounts create rag-stack --display-name "RAG stack"
SA="rag-stack@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com"
for role in roles/documentai.apiUser roles/aiplatform.user roles/discoveryengine.user; do
  gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" --member "serviceAccount:$SA" --role "$role"
done
gcloud iam service-accounts keys create ./service-account.json --iam-account "$SA"
```

Point `GOOGLE_APPLICATION_CREDENTIALS` at the absolute path of `service-account.json`.

## 2. Document AI Layout Parser processor

Create a **Layout Parser** processor (Console → Document AI → Create processor →
Layout Parser), then copy its ID into `DOCAI_PROCESSOR_ID` and its region into
`DOCAI_LOCATION` (`us` or `eu`). It parses PDF/HTML into layout-aware Markdown;
figures are cropped from the page raster (Document AI page image, else PyMuPDF).

## 3. Gemini Embedding 2

No provisioning — just `EMBEDDING_PROVIDER=google`,
`EMBEDDING_MODEL_NAME=gemini-embedding-2-preview`, and `EMBEDDING_DIM`
(768 / 1536 / 3072). It is natively multimodal, so text chunks **and** figure
images share one space and a text query can retrieve a figure directly.

> Changing the embedding model or dim **requires a one-time reindex** (the vector
> dimension changes): the indexer drops/recreates the `rag_chunks` table on a
> dimension mismatch automatically — just delete `RAG/vector_db/corpus_*.jsonl`
> and re-upload the documents (or run the ingest path).

## 4. Vertex AI Ranking API

No provisioning — `RERANKER_PROVIDER=vertex` and
`VERTEX_RANKER_MODEL=semantic-ranker-default-004` (or `semantic-ranker-fast-004`).

## 5. Install deps

```bash
uv sync
```

The Google client libraries (`google-cloud-documentai`,
`google-cloud-discoveryengine`, `google-genai`, `pymupdf`) are only imported when
the corresponding flag is set, so the app still runs on the local providers
without GCP credentials.

## Swapping providers later

Add a new branch to the relevant factory and select it by flag — no other code
changes:
- Embeddings: `create_embeddings()` in `RAG/services/embeddings.py`
- Parser: `convert_to_markdown_with_images()` in `RAG/services/converter.py`
- Reranker: `rerank_candidates()` in `RAG/services/retrieval.py`
