from typing import TypedDict, Annotated, Literal, Sequence
from langchain_core.messages import BaseMessage
import operator


AgentRole = Literal["supervisor", "researcher", "writer", "reviewer", "editor", "finish"]


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
    source_type: str            # "rag" | "web" — where research_results came from
    # ── Multimodal ──
    query_images: list          # user-attached images as data URLs (base64)
    context_images: list        # [{source, name, channel}, ...] figures from retrieved chunks
    web_sources: list           # [{title, url, content}, ...] when source_type == "web"
    allow_web: bool             # user approved falling back to web search for this query
    needs_web_approval: bool    # researcher found weak RAG results and is awaiting web approval
    # ── Editor (@update write-back) ──
    edit_instruction: str       # user's update/add instruction (without the @update tag)
    edit_target_file: str       # workspace markdown source the editor modified ("<stem>.md")
    edit_summary: str           # short human-readable summary of what changed
    regenerated_pdf: str        # source for the regenerated workspace PDF (download key)
