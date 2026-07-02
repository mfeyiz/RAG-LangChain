import asyncio
import json
import mimetypes
import re
import secrets
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from RAG.agents.graph import create_graph
from RAG.services import users, vector_store
from RAG.services.auth import (
    authenticate_request,
    auth_configured,
    create_access_token,
)
from RAG.services.document_manager import add_document, delete_document_by_source, list_documents
from RAG.services.feedback import save_feedback, get_feedback_stats, list_feedback, update_feedback_comment
from RAG.services.guardrails import guardrails
from RAG.services.rag_service import ensure_index
from RAG.services import paths
from RAG.services.retrieval import (
    COLLECTION_NAME,
    get_retriever,
    models_ready,
    warmup_models,
    _get_embeddings,
)
from RAG.services.session_store import session_store
from RAG.services.tracing import get_langfuse_handler, new_trace_id, start_request_trace, trace_event
from RAG.services.pending_edits import pop as pop_pending_edit, reject as reject_pending_edit
from RAG.services import version_control
from RAG.services.citation_highlight import (
    render_highlighted_page,
    render_highlighted_document,
    CITE_DIR,
)

_START_TIME = time.time()


_index_ready = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Set up graph and checkpointer first so FastAPI is fully configured
    # to handle incoming requests.
    _app.state.graph, _app.state.checkpointer = await create_graph()

    async def startup_sequence():
        global _index_ready
        try:
            # 1. Warm up models in background to avoid blocking ASGI startup
            await asyncio.to_thread(warmup_models)
            # 2. Provision the auth user table and seed the bootstrap admin.
            await asyncio.to_thread(users.ensure_users_schema)
            await asyncio.to_thread(users.ensure_admin_seed)
            # 2b. Initialise the workspace Git repo (best-effort; no-op if
            # GitPython / git is unavailable).
            await asyncio.to_thread(version_control.ensure_repo)
            # 3. Run indexing after models are ready to avoid model
            # loading race condition
            await asyncio.to_thread(ensure_index)
            _index_ready = True
            print("[Startup] System is fully ready.")
        except Exception as e:
            print(f"[Lifespan] Error in background startup sequence: {e}")
            import traceback
            traceback.print_exc()

    asyncio.create_task(startup_sequence())
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
        phase, message = "loading_models", "Loading models…"
    elif not _index_ready:
        phase, message = "indexing", "Indexing documents, please wait…"
    else:
        phase, message = "ready", "Ready"
    return {
        "models_ready": models,
        "index_ready": _index_ready,
        "phase": phase,
        "message": message,
        "auth_required": auth_configured(),
    }


# ── Authentication ───────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


@app.post("/auth/login")
async def login(body: LoginRequest):
    """Exchange username/password for a signed JWT used by @update and admin APIs."""
    if not auth_configured():
        return JSONResponse(
            {"error": "Authentication not configured (JWT_SECRET not set)."},
            status_code=503,
        )

    role = await asyncio.to_thread(users.verify_credentials, body.username, body.password)
    if role is None:
        return JSONResponse({"error": "Invalid username or password."}, status_code=401)

    token = create_access_token(sub=body.username, role=role)
    return {"token": token, "role": role, "username": body.username}


@app.post("/auth/register")
async def register(body: RegisterRequest, request: Request):
    """Create a new user — admin only."""
    auth = authenticate_request(request)
    if not auth.is_admin:
        return JSONResponse({"error": "Admin access required."}, status_code=403)

    role = body.role if body.role in ("user", "admin") else "user"
    ok = await asyncio.to_thread(users.create_user, body.username, body.password, role)
    if not ok:
        return JSONResponse({"error": "Could not create user."}, status_code=400)
    return {"ok": True, "username": body.username, "role": role}


# ── Ask (SSE with token streaming) ───────────────────────────────────────────

@app.post("/ask")
async def handle_query(request: Request):
    auth = authenticate_request(request)
    if not auth.allowed:
        return JSONResponse({"error": auth.error}, status_code=401)

    data = await request.json()
    user_query = data.get("query", "")

    # @update triggers the editor write-back path (it mutates the workspace index),
    # so it is restricted to authenticated callers. The regex mirrors the one in
    # RAG/agents/supervisor.py that routes @update to the editor.
    if re.search(r"@update\b", user_query, flags=re.IGNORECASE) and not auth.authenticated:
        return JSONResponse(
            {"error": "Login required to use @update."},
            status_code=403,
        )

    allow_web = bool(data.get("allow_web", False))
    # User-attached images (multimodal): list of base64 data URLs. Cap to bound
    # the vision prompt size.
    query_images = [img for img in (data.get("images") or []) if isinstance(img, str)][:4]

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
            "allow_web": allow_web,
            "needs_web_approval": False,
            "query_images": query_images,
            "context_images": [],
            "edit_instruction": "",
            "edit_target_file": "",
            "edit_summary": "",
            "regenerated_pdf": "",
            "fast_track": False,
            "edit_preview": {},
            "edit_pending": False,
            "needs_edit_approval": False,
            "edit_token": "",
            "needs_calculation": False,
            "calc_request": "",
            "calc_result": "",
            "table_data": [],
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

                # Attach the Langfuse handler at the graph level so it propagates
                # to every node via the run tree — this lets astream_events still
                # capture on_chat_model_stream token events (per-call callback
                # overrides would detach that tap and break streaming).
                handler = get_langfuse_handler()
                run_config = {
                    "configurable": {
                        "thread_id": session_id,
                        "user_id": auth.user_id,
                    }
                }
                if handler is not None:
                    run_config["callbacks"] = [handler]

                async for event in graph.astream_events(
                    initial_state,
                    config=run_config,
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
                            # parent_content (the full section) is only needed by
                            # the editor's in-process retrieval; the evidence panel
                            # never renders it, so keep it out of the SSE payload.
                            slim = [
                                {k: v for k, v in item.items() if k != "parent_content"}
                                for item in metadata
                            ]
                            yield {
                                "event": "search_results",
                                "data": json.dumps(slim, ensure_ascii=False),
                            }
                        # Figures anchored to the retrieved context, as servable URLs.
                        context_images = output.get("context_images", [])
                        if context_images:
                            image_payload = [
                                {
                                    "source": img.get("source", ""),
                                    "name": img.get("name", ""),
                                    "url": f"/images/{img.get('channel', 'workspace')}"
                                    f"/{paths.stem_of(img.get('source', ''))}/{img.get('name', '')}",
                                }
                                for img in context_images
                            ]
                            yield {
                                "event": "context_images",
                                "data": json.dumps(image_payload, ensure_ascii=False),
                            }
                        # Weak RAG match: ask the user before searching the web.
                        if output.get("needs_web_approval"):
                            yield {
                                "event": "web_search_prompt",
                                "data": json.dumps({"query": user_query}, ensure_ascii=False),
                            }

                    # Final response (editor also surfaces an edit_preview / edit_result event)
                    elif event_type == "on_chain_end" and node in ("reviewer", "writer", "editor"):
                        output = event["data"].get("output", {})
                        if node == "editor":
                            # Human-in-the-loop: editor produced a diff preview
                            # awaiting approval; surface it instead of a result.
                            if output.get("needs_edit_approval"):
                                preview = output.get("edit_preview", {}) or {}
                                yield {
                                    "event": "edit_preview",
                                    "data": json.dumps(
                                        {
                                            "token": output.get("edit_token", ""),
                                            "file": output.get("edit_target_file", ""),
                                            "instruction": output.get("edit_instruction", ""),
                                            "change_kind": preview.get("change_kind", ""),
                                            "diff": preview.get("diff", []),
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            else:
                                pdf_source = output.get("regenerated_pdf", "")
                                yield {
                                    "event": "edit_result",
                                    "data": json.dumps(
                                        {
                                            "file": output.get("edit_target_file", ""),
                                            "summary": output.get("edit_summary", ""),
                                            "pdf_url": f"/workspace/pdf/{pdf_source}" if pdf_source else "",
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
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

def _humanize_title(text: str) -> str:
    """Turn a filename-style title into a readable topic label:
    'Lec3_Scanning' → 'Lec3 Scanning', 'a-b_c.md' → 'a b c'."""
    text = re.sub(r"\.(md|pdf|docx?|txt)$", "", (text or "").strip(), flags=re.IGNORECASE)
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def _title_suggestions(q: str, limit: int) -> list[dict]:
    """Readable topic suggestions whose (humanized) title contains every word the
    user typed — predictable and relevant for short prefixes like 'scan'."""
    retriever = get_retriever()
    words = [w for w in q.lower().split() if w]
    seen: set[str] = set()
    results: list[dict] = []
    for candidate in retriever.corpus:
        raw = candidate.metadata.get("title", "").strip()
        if not raw or raw.lower() == "untitled":
            continue
        label = _humanize_title(raw)
        hay = label.lower()
        if label.lower() in seen or not all(w in hay for w in words):
            continue
        seen.add(label.lower())
        results.append({
            "text": label,
            "source": candidate.metadata.get("source", ""),
            "kind": candidate.metadata.get("kind", ""),
        })
        if len(results) >= limit:
            break
    return results


@app.get("/suggestions")
async def get_suggestions(
    q: str = Query(default="", min_length=1),
    limit: int = Query(default=5, ge=1, le=10),
):
    q = q.strip()
    if len(q) < 2:
        return {"suggestions": []}

    seen: set[str] = set()
    suggestions: list[dict] = []

    # 1. Direct title/topic matches first — readable and reliably relevant.
    for item in _title_suggestions(q, limit):
        key = item["text"].lower()
        if key not in seen:
            seen.add(key)
            suggestions.append(item)

    # 2. Fill remaining slots with semantic matches (real questions from QA
    #    chunks, otherwise the section heading or the humanized document title).
    if len(suggestions) < limit and vector_store.get_pool() is not None:
        try:
            query_vector = await asyncio.to_thread(_get_embeddings().embed_query, q)
            rows = await asyncio.to_thread(
                vector_store.query, COLLECTION_NAME, query_vector, limit * 4
            )
        except Exception as exc:
            print(f"[Suggestions] pgvector search failed: {exc}")
            rows = []

        for row in rows:
            metadata = row.get("metadata", {})
            kind = metadata.get("kind", "")
            if kind == "qa":
                text = metadata.get("question", "").strip()
            else:
                text = (metadata.get("heading_path", "").strip()
                        or _humanize_title(metadata.get("title", "")))
            if not text or text.lower() == "untitled":
                continue
            key = text.lower()
            if key not in seen:
                seen.add(key)
                suggestions.append({
                    "text": text,
                    "source": metadata.get("source", "unknown"),
                    "kind": kind,
                })
            if len(suggestions) >= limit:
                break

    return {"suggestions": suggestions[:limit]}


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
    if suffix not in paths.SUPPORTED_DOC_SUFFIXES:
        return JSONResponse({"error": "Only .pdf and .docx files are supported."}, status_code=400)

    file_bytes = await file.read()
    if len(file_bytes) > 50 * 1024 * 1024:
        return JSONResponse({"error": "File too large (max 50 MB)."}, status_code=413)

    try:
        result = await asyncio.to_thread(add_document, file_bytes, file.filename)
    except Exception as exc:
        return JSONResponse({"error": f"Conversion/indexing failed: {exc}"}, status_code=500)
    return result


# ── Workspace PDF download ────────────────────────────────────────────────────

@app.get("/workspace/pdf/{source:path}")
async def download_workspace_pdf(source: str):
    """Serve a regenerated workspace PDF for a given source ("<stem>.md")."""
    pdf_path = paths.workspace_pdf_path(source)
    # Path-traversal guard: the resolved path must stay inside WORKSPACE_PDF_DIR.
    try:
        pdf_path.resolve().relative_to(paths.WORKSPACE_PDF_DIR.resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path."}, status_code=400)
    if not pdf_path.exists():
        return JSONResponse({"error": "PDF not found."}, status_code=404)
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)


@app.get("/workspace/docx/{source:path}")
async def download_workspace_docx(source: str):
    """Serve a regenerated workspace DOCX for a given source ("<stem>.md")."""
    docx_path = paths.workspace_docx_path(source)
    # Path-traversal guard: the resolved path must stay inside WORKSPACE_DOCX_DIR.
    try:
        docx_path.resolve().relative_to(paths.WORKSPACE_DOCX_DIR.resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path."}, status_code=400)
    if not docx_path.exists():
        return JSONResponse({"error": "Word file not found. Please save the document first."}, status_code=404)
    return FileResponse(
        str(docx_path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=docx_path.name
    )


@app.get("/workspace/markdown/{source:path}")
async def download_workspace_markdown(source: str):
    """Serve the workspace Markdown file directly."""
    md_path = paths.workspace_md_path(source)
    try:
        md_path.resolve().relative_to(paths.WORKSPACE_MD_DIR.resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path."}, status_code=400)
    if not md_path.exists():
        return JSONResponse({"error": "Markdown file not found."}, status_code=404)
    return FileResponse(
        str(md_path),
        media_type="text/markdown",
        filename=md_path.name
    )


@app.get("/workspace/html/{source:path}")
async def download_workspace_html(source: str):
    """Serve a compiled, styled HTML copy of the workspace document."""
    md_path = paths.workspace_md_path(source)
    try:
        md_path.resolve().relative_to(paths.WORKSPACE_MD_DIR.resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path."}, status_code=400)
    if not md_path.exists():
        return JSONResponse({"error": "Document not found."}, status_code=404)

    text = md_path.read_text(encoding="utf-8")
    import markdown as md_lib
    html_body = md_lib.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "nl2br"],
    )

    css_path = Path(__file__).resolve().parent / "services" / "templates" / "pdf_style.css"
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{source}</title>
    <style>
        body {{ max-width: 800px; margin: 40px auto; padding: 0 20px; }}
        {css}
    </style>
</head>
<body>
    {html_body}
</body>
</html>"""

    from fastapi.responses import HTMLResponse
    return HTMLResponse(
        content=full_html,
        headers={
            "Content-Disposition": f"attachment; filename={Path(source).stem}.html"
        }
    )


@app.get("/originals/file/{source:path}")
async def download_original(source: str):
    """Serve the original uploaded PDF/DOCX for a source ("<stem>.md")."""
    orig = paths.original_doc_path(source)
    if orig is None:
        return JSONResponse({"error": "Original not found."}, status_code=404)
    # Path-traversal guard: must resolve inside an originals directory.
    inside = False
    for base in (paths.ORIGINALS_PDF_DIR, paths.ORIGINALS_DOCX_DIR):
        try:
            orig.resolve().relative_to(base.resolve())
            inside = True
            break
        except ValueError:
            continue
    if not inside:
        return JSONResponse({"error": "Invalid path."}, status_code=400)
    media = "application/pdf" if orig.suffix.lower() == ".pdf" else (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    return FileResponse(str(orig), media_type=media, filename=orig.name)


@app.get("/images/{channel}/{stem}/{name}")
async def serve_image(channel: str, stem: str, name: str):
    """Serve an extracted document figure for inline display in answers."""
    if channel not in ("originals", "workspace"):
        return JSONResponse({"error": "Invalid channel."}, status_code=400)
    img = paths.image_path(channel, stem, name)  # resolves with traversal guard
    if img is None:
        return JSONResponse({"error": "Image not found."}, status_code=404)
    media, _ = mimetypes.guess_type(str(img))
    return FileResponse(str(img), media_type=media or "image/png", filename=img.name)


@app.get("/documents/{source:path}/content")
async def document_content(source: str, channel: str = Query(default="workspace")):
    """Return a document's Markdown for a channel, plus download URLs for viewing/compare."""
    if channel not in ("originals", "workspace"):
        return JSONResponse({"error": "channel must be 'originals' or 'workspace'."}, status_code=400)

    md_path = paths.originals_md_path(source) if channel == "originals" else paths.workspace_md_path(source)
    base = paths.ORIGINALS_MD_DIR if channel == "originals" else paths.WORKSPACE_MD_DIR
    try:
        md_path.resolve().relative_to(base.resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path."}, status_code=400)
    if not md_path.exists():
        return JSONResponse({"error": "Document not found."}, status_code=404)

    return {
        "source": source,
        "channel": channel,
        "markdown": md_path.read_text(encoding="utf-8"),
        "original_url": f"/originals/file/{source}" if paths.original_doc_path(source) else "",
        "workspace_pdf_url": f"/workspace/pdf/{source}" if paths.workspace_pdf_path(source).exists() else "",
        "workspace_docx_url": f"/workspace/docx/{source}" if paths.workspace_docx_path(source).exists() else "",
    }


class SaveDocumentRequest(BaseModel):
    markdown: str


@app.post("/documents/{source:path}/save")
async def save_document(source: str, body: SaveDocumentRequest, request: Request):
    """Save manual markdown edits, perform re-indexing, regenerate PDF/DOCX, and commit."""
    auth = authenticate_request(request)
    if not auth.authenticated:
        return JSONResponse({"error": "Login required to save changes."}, status_code=401)

    md_path = paths.workspace_md_path(source)
    try:
        md_path.resolve().relative_to(paths.WORKSPACE_MD_DIR.resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path."}, status_code=400)

    paths.ensure_dirs()

    def _save_and_reindex():
        md_path.write_text(body.markdown, encoding="utf-8")

        # Re-index workspace source chunks
        from RAG.services.document_manager import reindex_workspace_source
        reindex = reindex_workspace_source(source)

        # PDF Regeneration
        from RAG.agents.editor import _render_pdf_safe
        pdf_ok = _render_pdf_safe(source)

        # DOCX Regeneration
        from RAG.services.docx_exporter import render as render_docx
        try:
            render_docx(source)
            docx_ok = True
        except Exception as exc:
            print(f"[app] DOCX regeneration failed for {source} during save: {exc}")
            docx_ok = False

        # Git Commit
        from RAG.services.version_control import commit_change
        commit_sha = commit_change(source, "manual edit")

        return reindex.get("chunks_added", 0), pdf_ok, docx_ok, commit_sha

    chunks_added, pdf_ok, docx_ok, commit_sha = await asyncio.to_thread(_save_and_reindex)

    return {
        "ok": True,
        "source": source,
        "pdf_ok": pdf_ok,
        "docx_ok": docx_ok,
        "git_sha": commit_sha,
        "chunks_added": chunks_added,
    }


@app.post("/documents/{source:path}/save-draft")
async def save_document_draft(source: str, body: SaveDocumentRequest, request: Request):
    """Fast, cheap autosave — write the workspace markdown only.

    No re-index / PDF / DOCX / git commit. Used by the studio's per-keystroke
    debounce so active editing stays responsive; the heavier ``/save`` runs on
    idle/blur and before export. Returns quickly.
    """
    auth = authenticate_request(request)
    if not auth.authenticated:
        return JSONResponse({"error": "Login required to save changes."}, status_code=401)

    md_path = paths.workspace_md_path(source)
    try:
        md_path.resolve().relative_to(paths.WORKSPACE_MD_DIR.resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path."}, status_code=400)

    paths.ensure_dirs()
    await asyncio.to_thread(md_path.write_text, body.markdown, "utf-8")
    return {"ok": True, "source": source, "draft": True}


class EditChatRequest(BaseModel):
    query: str
    current_markdown: str


@app.post("/documents/{source:path}/edit-chat")
async def edit_document_chat(source: str, body: EditChatRequest, request: Request):
    """Directly edit a document using instructions without supervisor routing."""
    auth = authenticate_request(request)
    if not auth.authenticated:
        return JSONResponse({"error": "Authentication required."}, status_code=401)

    from RAG.agents.editor import direct_edit_markdown
    try:
        updated_md = await direct_edit_markdown(body.current_markdown, body.query)
    except Exception as exc:
        return JSONResponse({"error": f"AI-assisted edit failed: {exc}"}, status_code=500)

    return {
        "ok": True,
        "source": source,
        "before": body.current_markdown,
        "after": updated_md
    }


# ── Report studio: create + asset endpoints ────────────────────────────────────

class CreateReportRequest(BaseModel):
    title: str
    template: str = "blank"


@app.post("/documents/create")
async def create_report(body: CreateReportRequest, request: Request):
    """Create a new blank/template report as a workspace Markdown source.

    Seeds the source with a template skeleton, indexes it, renders PDF/DOCX, and
    commits — so it behaves exactly like an uploaded document from then on.
    """
    auth = authenticate_request(request)
    if not auth.allowed:
        return JSONResponse({"error": auth.error}, status_code=401)
    if not auth.is_admin:
        return JSONResponse({"error": "Admin access required."}, status_code=403)

    from RAG.services import report_templates

    title = (body.title or "").strip()
    if not title:
        return JSONResponse({"error": "A report title is required."}, status_code=400)

    paths.ensure_dirs()

    # Derive a unique <stem>.md that doesn't clash with an existing workspace doc.
    base_stem = paths.sanitize_stem(title)
    stem = base_stem
    n = 2
    while paths.workspace_md_path(paths.source_for(stem)).exists():
        stem = f"{base_stem}-{n}"
        n += 1
    source = paths.source_for(stem)
    markdown = report_templates.render(body.template, title)

    def _create_and_index():
        paths.workspace_md_path(source).write_text(markdown, encoding="utf-8")

        from RAG.services.document_manager import reindex_workspace_source
        reindex_workspace_source(source)

        from RAG.agents.editor import _render_pdf_safe
        _render_pdf_safe(source)
        try:
            from RAG.services.docx_exporter import render as render_docx
            render_docx(source)
        except Exception as exc:
            print(f"[app] DOCX generation failed for new report {source}: {exc}")

        from RAG.services.version_control import commit_change
        return commit_change(source, "create report")

    try:
        git_sha = await asyncio.to_thread(_create_and_index)
    except Exception as exc:
        return JSONResponse({"error": f"Report creation failed: {exc}"}, status_code=500)

    from RAG.services import report_registry
    report_registry.record(source, title, body.template)

    return {"ok": True, "source": source, "markdown": markdown, "git_sha": git_sha}


class GenerateReportRequest(BaseModel):
    title: str
    topic: str
    template: str = "business-report"


@app.post("/documents/generate")
async def generate_report(body: GenerateReportRequest, request: Request):
    """Generate a new report section-by-section using the document store and stream SSE progress."""
    auth = authenticate_request(request)
    if not auth.allowed:
        return JSONResponse({"error": auth.error}, status_code=401)
    if not auth.is_admin:
        return JSONResponse({"error": "Admin access required."}, status_code=403)

    title = (body.title or "").strip()
    topic = (body.topic or "").strip()
    template = (body.template or "business-report").strip()

    if not title:
        return JSONResponse({"error": "A report title is required."}, status_code=400)
    if not topic:
        return JSONResponse({"error": "A report topic/instruction is required."}, status_code=400)

    async def event_generator():
        from RAG.agents.supervisor import get_llm
        from RAG.services.retrieval import retrieve_context_async
        from langchain_core.messages import SystemMessage

        llm = get_llm()
        trace_id = new_trace_id()

        try:
            # 1. Outline Planning
            yield {
                "event": "status",
                "data": json.dumps({"message": "Planning report outline..."}, ensure_ascii=False),
            }

            outline_prompt = f"""You are a professional report planner. Plan the outline for a report titled "{title}" on the topic: "{topic}".
The report template chosen is "{template}".
Based on this template, generate a list of section objects in JSON format.
Each object must have "title" (the section heading) and "description" (a brief guide of what to write about, referencing specific aspects of the topic).

Standard templates and their required sections:
- business-report: Executive Summary, Background, Analysis, Key Metrics, Recommendations, Conclusion
- research-summary: Abstract, Question, Findings, Discussion, References
- project-status: Status Overview, Progress This Period, Upcoming, Risks & Blockers, Metrics
- blank: Create 4-6 logical, well-structured sections appropriate for the topic.

Return ONLY a valid JSON array of objects, containing no markdown code fences, no introductory or concluding text.
Example:
[
  {{"title": "Section Title 1", "description": "What to write..."}},
  {{"title": "Section Title 2", "description": "What to write..."}}
]
"""
            response = await llm.ainvoke([SystemMessage(content=outline_prompt)])
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n", "", content)
                content = re.sub(r"\n```$", "", content)
            
            try:
                sections = json.loads(content.strip())
            except Exception:
                if template == "research-summary":
                    sections = [
                        {"title": "Abstract", "description": "Summary of research question and findings"},
                        {"title": "Question", "description": "Core research question"},
                        {"title": "Findings", "description": "Key findings and cited evidence"},
                        {"title": "Discussion", "description": "Discussion and limitations"},
                        {"title": "References", "description": "Sources used"}
                    ]
                elif template == "project-status":
                    sections = [
                        {"title": "Status Overview", "description": "Overall status summary"},
                        {"title": "Progress This Period", "description": "Achievements in this period"},
                        {"title": "Upcoming", "description": "Next steps planned"},
                        {"title": "Risks & Blockers", "description": "Risks, blockers and mitigations"},
                        {"title": "Metrics", "description": "Performance metrics"}
                    ]
                else:
                    sections = [
                        {"title": "Executive Summary", "description": "High-level summary of findings"},
                        {"title": "Background", "description": "Context and goals"},
                        {"title": "Analysis", "description": "Detailed analysis of topic"},
                        {"title": "Key Metrics", "description": "Key metrics table"},
                        {"title": "Recommendations", "description": "Actionable recommendations"},
                        {"title": "Conclusion", "description": "Summary and close"}
                    ]

            yield {
                "event": "status",
                "data": json.dumps({"message": f"Outline created with {len(sections)} sections."}, ensure_ascii=False),
            }

            section_contents = []
            for idx, sec in enumerate(sections):
                sec_title = sec.get("title", f"Section {idx+1}")
                sec_desc = sec.get("description", "")

                yield {
                    "event": "status",
                    "data": json.dumps({"message": f"Writing section: {sec_title}..."}, ensure_ascii=False),
                }
                yield {
                    "event": "section_start",
                    "data": json.dumps({"title": sec_title}, ensure_ascii=False),
                }

                query = f"{title} {sec_title} {sec_desc}".strip()
                context, metadata = await retrieve_context_async(query, top_k=5)

                writer_prompt = f"""You are a professional technical writer generating a specific section of a comprehensive report.
Prepare an accurate, detailed, and professional markdown text for the section based ONLY on the provided research context and topic guidelines.

Topic Guidelines: {topic}
Section Title: {sec_title}
Section Description: {sec_desc}

Research Context:
{context}

Hard Rules:
1. Respond in the same language as the topic guidelines (Turkish or English).
2. Base your section writing only on the provided research context. If the context does not contain enough info, write the section to the best of your ability using topic instructions, but clearly state what is verified.
3. Cite supporting context by converting the citation index to a markdown link with the document's filename and the exact short matching phrase. Format: `[N](cite://filename.pdf?snippet=Exact+Cited+Phrase+Here)`. Replace all spaces in the snippet query with '+' signs. Do not use any other link scheme.
Example citation: If document "annual_report.pdf" states "revenue grew by 15% in 2023" and you write that, cite it as `[1](cite://annual_report.pdf?snippet=revenue+grew+by+15%+in+2023)`.
4. Write only the section content (do NOT output the main document title, only start with the section heading e.g. `## Section Title`).
5. Output raw markdown. No markdown code blocks wrapping the entire response.
"""
                section_markdown = ""
                async for chunk in llm.astream([SystemMessage(content=writer_prompt)]):
                    if chunk.content:
                        section_markdown += chunk.content
                        yield {
                            "event": "token",
                            "data": chunk.content,
                        }

                section_contents.append(section_markdown.strip())
                yield {
                    "event": "section_complete",
                    "data": json.dumps({"title": sec_title, "markdown": section_markdown}, ensure_ascii=False),
                }

            yield {
                "event": "status",
                "data": json.dumps({"message": "Compiling final document..."}, ensure_ascii=False),
            }

            full_markdown = f"# {title}\n\n" + "\n\n".join(section_contents)
            source = _unique_workspace_source(title)
            
            def _save_and_index():
                paths.workspace_md_path(source).write_text(full_markdown, encoding="utf-8")
                return _index_and_export_workspace(source, "generate report")

            git_sha = await asyncio.to_thread(_save_and_index)
            
            from RAG.services import report_registry
            report_registry.record(source, title, template, generated=True)

            yield {
                "event": "complete",
                "data": json.dumps({"source": source, "markdown": full_markdown, "git_sha": git_sha}, ensure_ascii=False),
            }
            yield {"event": "done", "data": "[DONE]"}

        except Exception as e:
            print(f"Report Generation Error: {e}")
            import traceback
            traceback.print_exc()
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


def _unique_workspace_source(title: str) -> str:
    """Slugify `title` into a `<stem>.md` that doesn't collide with an existing
    workspace document (suffixing -2, -3, … on conflict)."""
    base_stem = paths.sanitize_stem(title)
    stem = base_stem
    n = 2
    while paths.workspace_md_path(paths.source_for(stem)).exists():
        stem = f"{base_stem}-{n}"
        n += 1
    return paths.source_for(stem)


def _index_and_export_workspace(source: str, commit_msg: str) -> str:
    """Reindex a workspace source, regenerate PDF/DOCX (best-effort), commit."""
    from RAG.services.document_manager import reindex_workspace_source
    from RAG.agents.editor import _render_pdf_safe
    from RAG.services.version_control import commit_change

    reindex_workspace_source(source)
    _render_pdf_safe(source)
    try:
        from RAG.services.docx_exporter import render as render_docx
        render_docx(source)
    except Exception as exc:
        print(f"[app] DOCX generation failed for {source}: {exc}")
    return commit_change(source, commit_msg)


class RenameReportRequest(BaseModel):
    title: str


@app.post("/documents/{source:path}/duplicate")
async def duplicate_report(source: str, request: Request):
    """Copy a workspace report (markdown + figures) to a new titled source."""
    auth = authenticate_request(request)
    if not auth.is_admin:
        return JSONResponse({"error": "Admin access required."}, status_code=403)

    import shutil
    from RAG.services import report_registry

    src_md = paths.workspace_md_path(source)
    if not src_md.exists():
        return JSONResponse({"error": "Document not found."}, status_code=404)

    meta = report_registry.get(source) or {}
    old_title = meta.get("title") or paths.stem_of(source)
    new_title = f"{old_title} copy"
    new_source = _unique_workspace_source(new_title)

    def _do():
        paths.ensure_dirs()
        paths.workspace_md_path(new_source).write_text(src_md.read_text(encoding="utf-8"), encoding="utf-8")
        old_imgs = paths.workspace_images_dir(source)
        if old_imgs.exists():
            shutil.copytree(old_imgs, paths.workspace_images_dir(new_source), dirs_exist_ok=True)
        return _index_and_export_workspace(new_source, "duplicate report")

    try:
        git_sha = await asyncio.to_thread(_do)
    except Exception as exc:
        return JSONResponse({"error": f"Duplicate failed: {exc}"}, status_code=500)

    report_registry.record(new_source, new_title, meta.get("template", "blank"),
                           generated=meta.get("generated", False))
    return {"ok": True, "source": new_source, "git_sha": git_sha}


@app.post("/documents/{source:path}/rename")
async def rename_report(source: str, body: RenameReportRequest, request: Request):
    """Rename a workspace report: copy to a new titled source, delete the old."""
    auth = authenticate_request(request)
    if not auth.is_admin:
        return JSONResponse({"error": "Admin access required."}, status_code=403)

    import shutil
    from RAG.services import report_registry
    from RAG.services.document_manager import delete_document_by_source

    title = (body.title or "").strip()
    if not title:
        return JSONResponse({"error": "A title is required."}, status_code=400)

    src_md = paths.workspace_md_path(source)
    if not src_md.exists():
        return JSONResponse({"error": "Document not found."}, status_code=404)

    meta = report_registry.get(source) or {}
    new_source = _unique_workspace_source(title)

    def _do():
        paths.ensure_dirs()
        paths.workspace_md_path(new_source).write_text(src_md.read_text(encoding="utf-8"), encoding="utf-8")
        old_imgs = paths.workspace_images_dir(source)
        if old_imgs.exists():
            shutil.copytree(old_imgs, paths.workspace_images_dir(new_source), dirs_exist_ok=True)
        git_sha = _index_and_export_workspace(new_source, f"rename report → {title}")
        # Remove the old source everywhere (index + on-disk artifacts).
        delete_document_by_source(source)
        return git_sha

    try:
        git_sha = await asyncio.to_thread(_do)
    except Exception as exc:
        return JSONResponse({"error": f"Rename failed: {exc}"}, status_code=500)

    report_registry.rename(source, new_source, title=title)
    return {"ok": True, "source": new_source, "git_sha": git_sha}


class ReportImageRequest(BaseModel):
    data_url: str
    name: str | None = None


@app.post("/documents/{source:path}/images")
async def upload_report_image(source: str, body: ReportImageRequest, request: Request):
    """Persist a PNG (chart canvas or a dragged figure) into a report's workspace
    image folder and return its same-origin URL for inline Markdown embedding."""
    auth = authenticate_request(request)
    if not auth.authenticated:
        return JSONResponse({"error": "Authentication required."}, status_code=401)

    import base64

    data_url = body.data_url or ""
    if "," in data_url and data_url.strip().lower().startswith("data:"):
        payload = data_url.split(",", 1)[1]
    else:
        payload = data_url
    try:
        raw = base64.b64decode(payload)
    except Exception:
        return JSONResponse({"error": "Invalid image data."}, status_code=400)

    stem = paths.stem_of(source)
    img_dir = paths.workspace_images_dir(source)
    img_dir.mkdir(parents=True, exist_ok=True)

    name = paths.sanitize_stem(body.name or f"chart_{secrets.token_hex(4)}")
    if not name.lower().endswith(".png"):
        name = f"{name}.png"

    # Path-traversal guard: the resolved file must stay inside the image dir.
    out_path = (img_dir / Path(name).name).resolve()
    try:
        out_path.relative_to(img_dir.resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path."}, status_code=400)

    out_path.write_bytes(raw)
    return {"ok": True, "url": f"/images/workspace/{stem}/{out_path.name}"}


@app.get("/library/assets")
async def library_assets(request: Request):
    """List reusable figures and parsed tables across the workspace for the
    report studio's right-hand tools panel."""
    auth = authenticate_request(request)
    if not auth.authenticated:
        return JSONResponse({"error": "Authentication required."}, status_code=401)

    figures: list[dict] = []
    base = paths.WORKSPACE_IMG_DIR
    if base.exists():
        for stem_dir in sorted(base.iterdir()):
            if not stem_dir.is_dir():
                continue
            for img in sorted(stem_dir.iterdir()):
                if img.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                    figures.append({
                        "source": paths.source_for(stem_dir.name),
                        "name": img.name,
                        "url": f"/images/workspace/{stem_dir.name}/{img.name}",
                    })

    tables: list[dict] = []
    try:
        from RAG.services.table_store import list_all_tables

        for t in list_all_tables():
            rows = t.get("rows") or []
            tables.append({
                "source": t.get("source", ""),
                "name": t.get("name", ""),
                "headers": t.get("headers") or [],
                "rows_preview": rows[:5],
                "row_count": len(rows),
            })
    except Exception as exc:
        print(f"[app] library_assets: table listing failed: {exc}")

    return {"figures": figures, "tables": tables}


# ── Citation → original PDF page + highlight ───────────────────────────────────

class CiteRequest(BaseModel):
    source: str
    snippet: str
    page: int | None = None


@app.post("/cite")
async def cite_lookup(body: CiteRequest):
    """Resolve a cited snippet to its page in the original PDF and return a
    rendered, highlighted page image for the chat preview panel."""
    result = await asyncio.to_thread(render_highlighted_page, body.source, body.snippet, body.page)
    if "error" in result:
        return JSONResponse(result, status_code=404)
    return result


class CiteDocRequest(BaseModel):
    source: str
    snippets: list[str] = []
    focus_snippet: str | None = None


@app.post("/cite/doc")
async def cite_doc(body: CiteDocRequest):
    """Render the whole original PDF with every retrieved chunk of `source`
    highlighted, for the scrollable citation viewer."""
    try:
        result = await asyncio.to_thread(
            render_highlighted_document, body.source, body.snippets, body.focus_snippet
        )
    except Exception as exc:
        # Never surface a bare 500 — the viewer would just show an empty grey
        # panel. Return a structured error the frontend can display.
        print(f"[Citation] /cite/doc failed for {body.source}: {exc}")
        return JSONResponse({"error": "Could not render citation image."}, status_code=404)
    if "error" in result:
        return JSONResponse(result, status_code=404)
    return result


@app.get("/cite/image/{name}")
async def cite_image(name: str):
    img = (CITE_DIR / Path(name).name).resolve()
    try:
        img.relative_to(CITE_DIR.resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path."}, status_code=400)
    if not img.exists():
        return JSONResponse({"error": "Image not found."}, status_code=404)
    return FileResponse(str(img), media_type="image/png", filename=img.name)


# ── @update approval (human-in-the-loop) + Git version control ────────────────

class UpdateApplyRequest(BaseModel):
    token: str


@app.post("/update/apply")
async def update_apply(request: Request, body: UpdateApplyRequest):
    """Persist a previously-stashed @update edit the user approved in the diff
    viewer. Requires authentication (same as @update itself)."""
    auth = authenticate_request(request)
    if not auth.authenticated:
        return JSONResponse({"error": "Login required to apply @update."}, status_code=403)

    edit = pop_pending_edit(body.token)
    if not edit:
        return JSONResponse({"error": "Pending change not found or expired."}, status_code=404)

    from RAG.agents.editor import apply_pending_edit

    try:
        result = await asyncio.to_thread(apply_pending_edit, edit)
    except Exception as exc:
        return JSONResponse({"error": f"Could not apply change: {exc}"}, status_code=500)

    pdf_url = f"/workspace/pdf/{result['source']}" if result.get("pdf_ok") else ""
    return {
        "ok": True,
        "file": result["source"],
        "summary": result["summary"],
        "reply": result["reply"],
        "pdf_url": pdf_url,
        "git_sha": result.get("git_sha"),
        "chunks_added": result.get("chunks_added", 0),
    }


@app.post("/update/reject")
async def update_reject(request: Request, body: UpdateApplyRequest):
    """Discard a stashed @update edit (user clicked Reddet)."""
    auth = authenticate_request(request)
    if not auth.allowed:
        return JSONResponse({"error": auth.error}, status_code=401)
    ok = reject_pending_edit(body.token)
    return {"ok": ok}


@app.get("/admin/history")
async def admin_history(request: Request, source: str = Query(default=""), limit: int = Query(default=50, le=200)):
    _, err = _require_admin(request)
    if err:
        return err
    return {"source": source or None, "commits": version_control.history(source or None, limit)}


@app.post("/admin/restore")
async def admin_restore(request: Request, body: dict):
    _, err = _require_admin(request)
    if err:
        return err
    source = (body or {}).get("source", "")
    ref = (body or {}).get("ref", "")
    if not source or not ref:
        return JSONResponse({"error": "source and ref are required."}, status_code=400)

    try:
        result = await asyncio.to_thread(version_control.restore, source, ref)
    except Exception as exc:
        return JSONResponse({"error": f"Restore failed: {exc}"}, status_code=500)

    # Re-index the restored workspace markdown + regenerate its PDF.
    from RAG.services.document_manager import reindex_workspace_source
    from RAG.agents.editor import _render_pdf_safe
    try:
        await asyncio.to_thread(reindex_workspace_source, source)
        await asyncio.to_thread(_render_pdf_safe, source)
    except Exception as exc:
        print(f"[restore] reindex/pdf failed: {exc}")
    return {"ok": True, **result}


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

    retriever = get_retriever()
    feedback = get_feedback_stats()

    return {
        "uptime_seconds": round(time.time() - _START_TIME),
        "corpus_chunks": len(retriever.corpus),
        "vector_store_available": vector_store.get_pool() is not None,
        "models_ready": models_ready(),
        "feedback": feedback,
    }


@app.get("/admin/documents")
async def admin_list_documents(request: Request, channel: str = Query(default="workspace")):
    auth = authenticate_request(request)
    if not auth.authenticated:
        return JSONResponse({"error": "Authentication required."}, status_code=401)
    if channel not in ("originals", "workspace"):
        return JSONResponse({"error": "channel must be 'originals' or 'workspace'."}, status_code=400)

    docs = list_documents(channel)
    # Annotate workspace docs with report metadata so the studio can group
    # studio-created "Reports" apart from "Uploaded" documents.
    if channel == "workspace":
        from RAG.services import report_registry
        registry = report_registry.all_reports()
        for d in docs:
            meta = registry.get(d["source"])
            d["kind"] = "report" if meta else "upload"
            if meta:
                d["title"] = meta.get("title")
                d["generated"] = meta.get("generated", False)
                d["created_at"] = meta.get("created_at")
    return {"channel": channel, "documents": docs}


@app.delete("/admin/documents/{source:path}")
async def admin_delete_document(source: str, request: Request):
    _, err = _require_admin(request)
    if err:
        return err
    result = await asyncio.to_thread(delete_document_by_source, source)
    from RAG.services import report_registry
    report_registry.remove(source)
    return result


@app.get("/admin/feedback")
async def admin_feedback(request: Request, limit: int = Query(default=50, le=200)):
    _, err = _require_admin(request)
    if err:
        return err
    return {"feedback": list_feedback(limit=limit)}


from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
