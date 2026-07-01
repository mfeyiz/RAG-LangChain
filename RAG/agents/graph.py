import os

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import StateGraph, END
from redis.exceptions import ResponseError

from RAG.agents.state import AgentState
from RAG.agents.supervisor import supervisor_node
from RAG.agents.researcher import researcher_node
from RAG.agents.writer import writer_node
from RAG.agents.reviewer import reviewer_node
from RAG.agents.editor import editor_node
from RAG.agents.code_interpreter import code_interpreter_node


def route_supervisor(state: AgentState) -> str:
    next_agent = state.get("next_agent", "finish")

    if next_agent == "finish":
        return "finish"

    return next_agent


async def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("code_interpreter", code_interpreter_node)

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "editor": "editor",
            "code_interpreter": "code_interpreter",
            "finish": END,
        },
    )

    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("writer", "supervisor")
    workflow.add_edge("reviewer", "supervisor")
    workflow.add_edge("editor", "supervisor")
    # Code interpreter computes/returns research_results → back to supervisor
    # which then routes to writer to fold the result into the user-facing answer.
    workflow.add_edge("code_interpreter", "supervisor")

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")

    checkpointer = AsyncRedisSaver(redis_url=redis_url)
    try:
        await checkpointer.asetup()
    except ResponseError as exc:
        if "FT." in str(exc) or "unknown command" in str(exc):
            raise RuntimeError(
                "AsyncRedisSaver requires Redis Stack or a Redis server with the RediSearch "
                "module enabled. The current Redis instance does not support the FT.* "
                "commands needed to create checkpoint indexes."
            ) from exc
        raise

    return workflow.compile(checkpointer=checkpointer), checkpointer
