import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from RAG.agents.graph import create_graph
from RAG.services.auth import authenticate_request
from RAG.services.document_manager import add_document, delete_document_by_source, list_documents
from RAG.services.feedback import save_feedback, get_feedback_stats, list_feedback, update_feedback_comment
from RAG.services.guardrails import guardrails
from RAG.services.rag_service import ensure_index
from RAG.services.retrieval import (
    COLLECTION_NAME,
    get_retriever,
    models_ready,
    warmup_models,
    _get_embeddings,
)
from RAG.services.session_store import session_store
from RAG.services.tracing import new_trace_id, start_request_trace, trace_event

_START_TIME = asyncio.get_event_loop().time() if False else __import__("time").time()


_index_ready = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _index_ready
    # Load models first — ensures _embeddings_model is set before ensure_index
    # touches it. Starting both concurrently caused a race condition where two
    # threads both tried to load BGE-M3 simultaneously.
    await asyncio.to_thread(warmup_models)
    _app.state.graph, _app.state.checkpointer = await create_graph()

    async def _run_index():
        global _index_ready
        await asyncio.to_thread(ensure_index)
        _index_ready = True

    asyncio.create_task(_run_index())
    yield
    try:
        aclose = getattr(_app.state.checkpointer, "aclose", None)
        if aclose:
            await aclose()
    except Exception:
        pass
    await session_store.close()


app = FastAPI(title="RAG Multi-Agent System", lifespan=lifespan)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/healthz")
async def healthz():
    if not models_ready():
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.get("/status")
async def status():
    """Lightweight status for the UI: are models loaded and is the index ready?"""
    models = models_ready()
    if not models:
        phase, message = "loading_models", "Modeller yükleniyor…"
    elif not _index_ready:
        phase, message = "indexing", "Belgeler indeksleniyor, lütfen bekleyin…"
    else:
        phase, message = "ready", "Hazır"
    return {
        "models_ready": models,
        "index_ready": _index_ready,
        "phase": phase,
        "message": message,
    }


# ── Ask (SSE with token streaming) ───────────────────────────────────────────

@app.post("/ask")
async def handle_query(request: Request):
    auth = authenticate_request(request)
    if not auth.allowed:
        return JSONResponse({"error": auth.error}, status_code=401)

    data = await request.json()
    user_query = data.get("query", "")

    guard = guardrails.check_all(user_query, auth.user_id)
    if not guard.allowed:
        return JSONResponse({"error": guard.error}, status_code=guard.status_code)

    session_id = session_store.get_or_create(data.get("session_id") or auth.session_id)
    previous_session = await session_store.load(session_id)
    conversation_history = previous_session.get("history", [])

    trace_id = new_trace_id()
    await trace_event(
        trace_id,
        "request.received",
        {"user_id": auth.user_id, "session_id": session_id, "query": user_query},
    )

    graph = request.app.state.graph

    async def event_generator():
        initial_state = {
            "messages": [],
            "next_agent": "supervisor",
            "query": user_query,
            "research_results": "",
            "draft_response": "",
            "final_response": "",
            "review_feedback": "",
            "revision_count": 0,
            "search_metadata": [],
            "user_id": auth.user_id,
            "session_id": session_id,
            "trace_id": trace_id,
            "rewritten_query": "",
            "conversation_history": conversation_history,
            "hop_steps": [],
            "hop_context": "",
            "source_type": "rag",
            "web_sources": [],
        }

        try:
            yield {
                "event": "session_info",
                "data": json.dumps({"session_id": session_id, "trace_id": trace_id}, ensure_ascii=False),
            }

            with start_request_trace(
                trace_name="rag-ask",
                trace_id=trace_id,
                user_id=auth.user_id,
                session_id=session_id,
                input_payload={"query": user_query},
            ) as root_span:
                final_response = ""
                active_node = ""

                async for event in graph.astream_events(
                    initial_state,
                    config={
                        "configurable": {
                            "thread_id": session_id,
                            "user_id": auth.user_id,
                        }
                    },
                    version="v2",
                ):
                    event_type = event["event"]
                    node = event.get("metadata", {}).get("langgraph_node", "")

                    # Agent start notification
                    if event_type == "on_chain_start" and node and node != active_node:
                        active_node = node
                        await trace_event(trace_id, f"agent.{node}.start", {})
                        yield {
                            "event": "agent_update",
                            "data": json.dumps({"agent": node, "status": "working"}, ensure_ascii=False),
                        }

                    # Token-level streaming from writer
                    elif event_type == "on_chat_model_stream" and node == "writer":
                        chunk = event["data"].get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            yield {"event": "token", "data": chunk.content}

                    # Search results from researcher
                    elif event_type == "on_chain_end" and node == "researcher":
                        output = event["data"].get("output", {})
                        metadata = output.get("search_metadata", [])
                        if metadata:
                            yield {
                                "event": "search_results",
                                "data": json.dumps(metadata, ensure_ascii=False),
                            }

                    # Final response
                    elif event_type == "on_chain_end" and node in ("reviewer", "writer"):
                        output = event["data"].get("output", {})
                        fr = output.get("final_response", "")
                        if fr and not final_response:
                            final_response = fr
                            yield {"event": "message", "data": fr}
                            root_span.update(output={"answer": fr})

                            next_state = session_store.build_next_state(
                                previous_session,
                                user_id=auth.user_id,
                                query=user_query,
                                response=fr,
                                trace_id=trace_id,
                            )
                            await session_store.save(session_id, next_state)

            await trace_event(trace_id, "request.completed", {"session_id": session_id})
            yield {"event": "done", "data": "[DONE]"}

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            await trace_event(trace_id, "request.error", {"error": str(e)})
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


# ── Query Suggestions (embedding-based) ──────────────────────────────────────

@app.get("/suggestions")
async def get_suggestions(
    q: str = Query(default="", min_length=1),
    limit: int = Query(default=5, ge=1, le=10),
):
    q = q.strip()
    if len(q) < 2:
        return {"suggestions": []}

    retriever = get_retriever()
    if retriever.qdrant is None:
        return {"suggestions": _bm25_suggestions(q, limit)}

    try:
        query_vector = await asyncio.to_thread(_get_embeddings().embed_query, q)
        response = retriever.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit * 4,
            with_payload=True,
        )
        points = response.points
    except Exception as exc:
        print(f"[Suggestions] Qdrant search failed: {exc}")
        return {"suggestions": _bm25_suggestions(q, limit)}

    seen: set[str] = set()
    suggestions: list[dict] = []

    for point in points:
        payload = point.payload or {}
        metadata = payload.get("metadata", {})
        kind = metadata.get("kind", "")
        source = metadata.get("source", "unknown")

        # Prefer real questions from QA chunks
        if kind == "qa":
            text = metadata.get("question", "").strip()
        else:
            title = metadata.get("title", "").strip()
            text = title if title and title.lower() != "untitled" else ""

        if text and text not in seen:
            seen.add(text)
            suggestions.append({"text": text, "source": source, "kind": kind})
            if len(suggestions) >= limit:
                break

    return {"suggestions": suggestions}


def _bm25_suggestions(q: str, limit: int) -> list[dict]:
    """Fallback: BM25 prefix match on corpus titles when Qdrant is unavailable."""
    retriever = get_retriever()
    q_lower = q.lower()
    seen: set[str] = set()
    results: list[dict] = []
    for candidate in retriever.corpus:
        title = candidate.metadata.get("title", "").strip()
        if title and title.lower() != "untitled" and q_lower in title.lower() and title not in seen:
            seen.add(title)
            results.append({"text": title, "source": candidate.metadata.get("source", ""), "kind": candidate.metadata.get("kind", "")})
            if len(results) >= limit:
                break
    return results


# ── User Feedback ─────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    session_id: str
    trace_id: str = ""
    rating: int           # 1 = thumbs up, -1 = thumbs down
    query: str = ""
    comment: str = ""


@app.post("/feedback")
async def post_feedback(request: Request, body: FeedbackRequest):
    auth = authenticate_request(request)
    if not auth.allowed:
        return JSONResponse({"error": auth.error}, status_code=401)

    if body.rating not in (1, -1):
        return JSONResponse({"error": "rating must be 1 or -1"}, status_code=400)

    save_feedback(
        session_id=body.session_id,
        trace_id=body.trace_id,
        rating=body.rating,
        query=body.query,
        comment=body.comment,
        user_id=auth.user_id,
    )
    return {"ok": True}


class FeedbackCommentRequest(BaseModel):
    session_id: str
    trace_id: str = ""
    comment: str


@app.post("/feedback/comment")
async def post_feedback_comment(request: Request, body: FeedbackCommentRequest):
    auth = authenticate_request(request)
    if not auth.allowed:
        return JSONResponse({"error": auth.error}, status_code=401)

    updated = await asyncio.to_thread(
        update_feedback_comment, body.trace_id, body.session_id, body.comment
    )
    return {"ok": updated}


# ── Document Upload ───────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    auth = authenticate_request(request)
    if not auth.allowed:
        return JSONResponse({"error": auth.error}, status_code=401)
    if not auth.is_admin:
        return JSONResponse({"error": "Admin access required."}, status_code=403)

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".pdf", ".txt"):
        return JSONResponse({"error": "Only .pdf and .txt files are supported."}, status_code=400)

    file_bytes = await file.read()
    if len(file_bytes) > 50 * 1024 * 1024:
        return JSONResponse({"error": "File too large (max 50 MB)."}, status_code=413)

    result = await asyncio.to_thread(add_document, file_bytes, file.filename)
    return result


# ── Admin API ─────────────────────────────────────────────────────────────────

def _require_admin(request: Request):
    auth = authenticate_request(request)
    if not auth.allowed:
        return None, JSONResponse({"error": auth.error}, status_code=401)
    if not auth.is_admin:
        return None, JSONResponse({"error": "Admin access required."}, status_code=403)
    return auth, None


@app.get("/admin/stats")
async def admin_stats(request: Request):
    _, err = _require_admin(request)
    if err:
        return err

    import time as _time
    retriever = get_retriever()
    feedback = get_feedback_stats()

    return {
        "uptime_seconds": round(_time.time() - _START_TIME),
        "corpus_chunks": len(retriever.corpus),
        "qdrant_available": retriever.qdrant is not None,
        "models_ready": models_ready(),
        "feedback": feedback,
    }


@app.get("/admin/documents")
async def admin_list_documents(request: Request):
    _, err = _require_admin(request)
    if err:
        return err
    return {"documents": list_documents()}


@app.delete("/admin/documents/{source:path}")
async def admin_delete_document(source: str, request: Request):
    _, err = _require_admin(request)
    if err:
        return err
    result = await asyncio.to_thread(delete_document_by_source, source)
    return result


@app.get("/admin/feedback")
async def admin_feedback(request: Request, limit: int = Query(default=50, le=200)):
    _, err = _require_admin(request)
    if err:
        return err
    return {"feedback": list_feedback(limit=limit)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
