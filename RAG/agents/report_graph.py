"""Report generation graph — a StateGraph independent of the /ask graph.

    entry → supervisor ─┬─► researcher ─┐
                        ├─► writer ──────┤ (all workers loop back to supervisor)
                        ├─► reviewer ────┘
                        └─► finalize → END

Deliberately has NO editor / code_interpreter node and does NOT touch
`RAG.agents.graph` or `AgentState`, keeping the Bidirectional-RAG Editor path
fully isolated. Compiled without a checkpointer — the report flow runs to
completion in one shot and needs no human-in-the-loop resume.
"""

from langgraph.graph import StateGraph, END

from RAG.agents.report_state import ReportState
from RAG.agents.report_supervisor import report_supervisor_node
from RAG.agents.report_researcher import report_researcher_node
from RAG.agents.report_writer import report_writer_node
from RAG.agents.report_reviewer import report_reviewer_node


def _route(state: ReportState) -> str:
    return state.get("next_agent", "finalize")


async def _finalize_node(state: ReportState) -> dict:
    """Assemble the final body markdown (sections in outline order)."""
    outline = state.get("outline", [])
    sections = state.get("sections") or {}
    body = "\n\n".join(
        sections[i] for i in range(len(outline)) if sections.get(i)
    )
    return {"final_markdown": body}


def create_report_graph():
    workflow = StateGraph(ReportState)

    workflow.add_node("supervisor", report_supervisor_node)
    workflow.add_node("researcher", report_researcher_node)
    workflow.add_node("writer", report_writer_node)
    workflow.add_node("reviewer", report_reviewer_node)
    workflow.add_node("finalize", _finalize_node)

    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        _route,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("writer", "supervisor")
    workflow.add_edge("reviewer", "supervisor")
    workflow.add_edge("finalize", END)

    # No checkpointer: single-shot run, no resume needed (keeps the report path
    # off the Redis Stack / RediSearch dependency the /ask graph requires).
    return workflow.compile()
