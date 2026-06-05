import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from RAG.agents.graph import create_graph

STATIC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="RAG Multi-Agent System")

graph = create_graph()


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
    data = await request.json()
    user_query = data.get("query")

    if not user_query:
        return JSONResponse({"error": "Query field cannot be empty"}, status_code=400)

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
        }

        try:
            async for event in graph.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in event.items():
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

                    if node_name == "reviewer" and node_output.get("final_response"):
                        yield {
                            "event": "message",
                            "data": node_output["final_response"],
                        }

            yield {"event": "done", "data": "[DONE]"}

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
