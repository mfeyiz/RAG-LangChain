from typing import TypedDict, Annotated, Literal, Sequence
from langchain_core.messages import BaseMessage
import operator


AgentRole = Literal["supervisor", "researcher", "writer", "reviewer", "finish"]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: AgentRole
    query: str
    research_results: str
    draft_response: str
    final_response: str
    review_feedback: str
    revision_count: int
    search_metadata: list
    user_id: str
    session_id: str
    trace_id: str
    rewritten_query: str
    conversation_history: list  # [{query, response}, ...] last N turns
    hop_steps: list             # multi-hop decomposed steps
    hop_context: str            # aggregated context from all hops
