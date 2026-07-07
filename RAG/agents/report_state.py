"""State contract for the multi-agent report generation pipeline.

This is intentionally SEPARATE from `RAG.agents.state.AgentState` (the /ask graph
state). The report pipeline runs on its own StateGraph and must stay fully
decoupled from the Bidirectional-RAG Editor path — do not merge these.
"""

from typing import TypedDict, Literal


ReportAgent = Literal["researcher", "writer", "reviewer", "finalize"]


class ReportState(TypedDict, total=False):
    # ── Inputs ──
    topic: str                 # user's report topic / instruction
    title: str                 # report title
    template: str              # "business-report" | "research-summary" | ...
    trace_id: str

    # ── Supervisor: plan ──
    outline: list              # [{"title": str, "description": str}, ...]

    # ── Researcher: evidence ──
    # section_evidence[title] = {"context": <rag>, "web_context": <web>}
    section_evidence: dict
    sources: list              # [{"title": str, "url": str}, ...] collected web refs
    research_gaps: list        # section titles the reviewer wants re-researched

    # ── Writer: draft ──
    sections: dict             # {idx: markdown} per outline section
    draft: str                 # assembled body markdown (no H1 title, no sources)

    # ── Reviewer: critique ──
    review_feedback: str       # short human-readable critique
    section_feedback: dict     # {section_title: "what to fix"}
    revision_count: int
    reviewed: bool             # True once the reviewer has judged the current draft

    # ── Routing / output ──
    next_agent: ReportAgent
    final_markdown: str        # assembled body handed back to the endpoint driver
