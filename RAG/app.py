import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from RAG.agents.graph import create_graph
from RAG.services.auth import authenticate_request
from RAG.services.guardrails import guardrails
from RAG.services.session_store import session_store
from RAG.services.tracing import new_trace_id, start_request_trace, trace_event

STATIC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _app.state.graph, _app.state.checkpointer = await create_graph()
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
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/{path:path}")
async def serve_static(path: str):
    file_path = os.path.join(STATIC_DIR, path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return JSONResponse({"error": "Not found"}, status_code=404)


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
        }

        try:
            with start_request_trace(
                trace_name="rag-ask",
                trace_id=trace_id,
                user_id=auth.user_id,
                session_id=session_id,
                input_payload={"query": user_query},
            ) as root_span:
                async for event in graph.astream(
                    initial_state,
                    config={
                        "configurable": {
                            "thread_id": session_id,
                            "user_id": auth.user_id,
                        }
                    },
                    stream_mode="updates",
                ):
                    for node_name, node_output in event.items():
                        await trace_event(trace_id, f"agent.{node_name}", _safe_trace_payload(node_output))
                        yield {
                            "event": "agent_update",
                            "data": json.dumps(
                                {"agent": node_name, "status": "working"},
                                ensure_ascii=False,
                            ),
                        }

                        if node_name == "researcher" and "search_metadata" in node_output:
                            yield {
                                "event": "search_results",
                                "data": json.dumps(
                                    node_output.get("search_metadata", []),
                                    ensure_ascii=False,
                                ),
                            }

                        # Both reviewer (normal flow) and writer (no-docs shortcut) can set final_response.
                        if node_name in ("reviewer", "writer") and node_output.get("final_response"):
                            final = node_output["final_response"]
                            yield {"event": "message", "data": final}

                            await session_store.save(
                                session_id,
                                {
                                    "user_id": auth.user_id,
                                    "last_query": user_query,
                                    "last_response": final,
                                    "last_trace_id": trace_id,
                                    "request_count": previous_session.get("request_count", 0) + 1,
                                },
                            )
                            root_span.update(output={"answer": final})

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


def _safe_trace_payload(node_output: dict) -> dict:
    payload = {}
    for key, value in node_output.items():
        if key == "messages":
            payload[key] = [message.content for message in value]
        else:
            payload[key] = value
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
