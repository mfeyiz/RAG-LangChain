import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from RAG.agents.state import AgentState, AgentRole
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
    )


SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor agent. Your task is to analyze the user's question and decide which agent should run next.

Available agents:
- researcher: Searches the document database. Use for technical questions, product information, or definitions.
- writer: Prepares a response for the user using research results.
- reviewer: Checks the quality of the prepared response.

Decision rules:
1. If the user asks a technical question or requests information -> "researcher"
2. Social interactions like greetings, thanks -> "writer" (no research needed)
3. If research is done -> "writer"
4. If a response is drafted -> "reviewer"
5. If reviewer approved or revision count reached 2 -> "FINISH"

Return ONLY one of these values: researcher, writer, reviewer, FINISH"""


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

        # Initial state: nothing set yet. Route to researcher unless it is a
        # short social phrase that needs no document lookup.
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
