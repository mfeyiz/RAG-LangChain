import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from RAG.agents.state import AgentState
from RAG.services.tracing import invoke_with_langfuse, trace_event, traced_observation


env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek/deepseek-v4-flash"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
        streaming=True,
        # Route to the lowest-latency OpenRouter provider — cut writer time
        # from ~10s to ~3s in local benchmarks.
        extra_body={"provider": {"sort": "latency"}},
    )


def get_vision_llm(streaming: bool = True):
    """Vision-capable OpenRouter model used by the writer (and ingest captioning)
    when images are in play. Kept separate from `get_llm()` so routing, query
    rewriting, and review stay on the cheaper text model."""
    return ChatOpenAI(
        model=os.getenv("VISION_LLM_MODEL", "google/gemini-2.5-flash"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
        streaming=streaming,
        extra_body={"provider": {"sort": "latency"}},
    )


async def supervisor_node(state: AgentState) -> dict:
    with traced_observation(
        "supervisor",
        input_payload={
            "query": state["query"],
            "has_research": bool(state.get("research_results")),
            "has_draft": bool(state.get("draft_response")),
            "has_feedback": bool(state.get("review_feedback")),
            "revision_count": state.get("revision_count", 0),
        },
    ) as span:
        if state.get("final_response"):
            print("[Supervisor] Decision: finish")
            await trace_event(state["trace_id"], "supervisor.decision", {"next_agent": "finish"})
            span.update(output={"next_agent": "finish"})
            return {"next_agent": "finish"}

        # Researcher found weak RAG results and is asking the user before going to
        # the web — stop the graph and let the UI surface the approval prompt.
        if state.get("needs_web_approval"):
            print("[Supervisor] Decision: finish (awaiting web search approval)")
            await trace_event(state["trace_id"], "supervisor.decision", {"next_agent": "finish"})
            span.update(output={"next_agent": "finish"})
            return {"next_agent": "finish"}

        if state.get("draft_response"):
            print("[Supervisor] Decision: reviewer")
            await trace_event(state["trace_id"], "supervisor.decision", {"next_agent": "reviewer"})
            span.update(output={"next_agent": "reviewer"})
            return {"next_agent": "reviewer"}

        if state.get("review_feedback"):
            print("[Supervisor] Decision: writer")
            await trace_event(state["trace_id"], "supervisor.decision", {"next_agent": "writer"})
            span.update(output={"next_agent": "writer"})
            return {"next_agent": "writer"}

        if state.get("research_results"):
            print("[Supervisor] Decision: writer")
            await trace_event(state["trace_id"], "supervisor.decision", {"next_agent": "writer"})
            span.update(output={"next_agent": "writer"})
            return {"next_agent": "writer"}

        # Initial state: nothing set yet. An @update message is a write-back
        # request and goes to the editor instead of the read-only RAG path.
        if re.search(r"@update\b", state["query"], flags=re.IGNORECASE):
            print("[Supervisor] Decision: editor")
            await trace_event(state["trace_id"], "supervisor.decision", {"next_agent": "editor"})
            span.update(output={"next_agent": "editor"})
            return {"next_agent": "editor"}

        # Route to researcher unless it is a short social phrase that needs no
        # document lookup.
        _SOCIAL = frozenset([
            "merhaba", "selam", "hello", "hi", "hey", "teşekkür", "teşekkürler",
            "thanks", "thank", "günaydın", "iyi", "nasılsın", "görüşürüz", "bye",
        ])
        words = state["query"].lower().split()
        if len(words) <= 6 and _SOCIAL.intersection(words):
            decision = "writer"
        else:
            decision = "researcher"

        print(f"[Supervisor] Decision: {decision}")
        await trace_event(state["trace_id"], "supervisor.decision", {"next_agent": decision})
        span.update(output={"next_agent": decision})

        return {"next_agent": decision}
