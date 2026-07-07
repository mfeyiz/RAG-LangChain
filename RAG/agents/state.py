from typing import TypedDict, Literal


AgentRole = Literal["supervisor", "researcher", "writer", "reviewer", "editor", "finish"]


class AgentState(TypedDict):
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
    # ── Fast-track routing ──
    fast_track: bool            # supervisor classified as simple: bypass researcher & reviewer
    # ── Diff Viewer / Human-in-the-loop @update ──
    edit_preview: dict          # pending {source, before, after, diff, instruction} awaiting approval
    edit_pending: bool          # editor produced a preview waiting for the user to approve/reject
    needs_edit_approval: bool   # surface an approve/reject prompt to the UI
    edit_token: str             # opaque key that /update/apply uses to look up a stashed edit
    # ── Tabular / code interpreter ──
    needs_calculation: bool     # question requires arithmetic over table data
    calc_request: str           # the natural-language calculation request
    calc_result: str            # the computed numeric/string result from code interpreter
    table_data: list            # structured [{name, headers, rows}] tables relevant to the query
