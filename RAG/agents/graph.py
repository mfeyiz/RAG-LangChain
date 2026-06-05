from langgraph.graph import StateGraph, END

from RAG.agents.state import AgentState
from RAG.agents.supervisor import supervisor_node
from RAG.agents.researcher import researcher_node
from RAG.agents.writer import writer_node
from RAG.agents.reviewer import reviewer_node


def route_supervisor(state: AgentState) -> str:
    next_agent = state.get("next_agent", "finish")
    if next_agent == "finish":
        return "finish"
    return next_agent


def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "finish": END,
        },
    )

    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("writer", "supervisor")
    workflow.add_edge("reviewer", "supervisor")

    return workflow.compile()
