"""Report Supervisor — orchestrator for the report generation pipeline.

Plans the outline (splits the topic into subtasks), manages the flow/state, and
routes between researcher → writer → reviewer, looping on reviewer feedback until
approval (or the revision cap), then hands off to `finalize`.

Fully independent of the /ask supervisor and the Bidirectional-RAG Editor.
"""

import json
import os
import re

from langchain_core.messages import SystemMessage
from langgraph.config import get_stream_writer

from RAG.agents.report_state import ReportState
from RAG.agents.supervisor import get_llm

# Reviewer loop cap. 0 → single pass (write once, no revision).
MAX_REVISIONS = int(os.getenv("REPORT_MAX_REVISIONS", "2"))
MAX_SECTIONS = int(os.getenv("REPORT_MAX_SECTIONS", "6"))


_TEMPLATE_FALLBACKS = {
    "research-summary": [
        {"title": "Abstract", "description": "Summary of research question and findings"},
        {"title": "Question", "description": "Core research question"},
        {"title": "Findings", "description": "Key findings and cited evidence"},
        {"title": "Discussion", "description": "Discussion and limitations"},
        {"title": "References", "description": "Sources used"},
    ],
    "project-status": [
        {"title": "Status Overview", "description": "Overall status summary"},
        {"title": "Progress This Period", "description": "Achievements in this period"},
        {"title": "Upcoming", "description": "Next steps planned"},
        {"title": "Risks & Blockers", "description": "Risks, blockers and mitigations"},
        {"title": "Metrics", "description": "Performance metrics"},
    ],
    "business-report": [
        {"title": "Executive Summary", "description": "High-level summary of findings"},
        {"title": "Background", "description": "Context and goals"},
        {"title": "Analysis", "description": "Detailed analysis of topic"},
        {"title": "Key Metrics", "description": "Key metrics table"},
        {"title": "Recommendations", "description": "Actionable recommendations"},
        {"title": "Conclusion", "description": "Summary and close"},
    ],
}


def _outline_prompt(title: str, topic: str, template: str) -> str:
    return f"""You are a professional report planner. Plan the outline for a report titled "{title}" on the topic: "{topic}".
The report template chosen is "{template}".
Based on this template, generate a list of section objects in JSON format.
Each object must have "title" (the section heading) and "description" (a brief guide of what to write about, referencing specific aspects of the topic).

Standard templates and their required sections:
- business-report: Executive Summary, Background, Analysis, Key Metrics, Recommendations, Conclusion
- research-summary: Abstract, Question, Findings, Discussion, References
- project-status: Status Overview, Progress This Period, Upcoming, Risks & Blockers, Metrics
- blank: Create 4-6 logical, well-structured sections appropriate for the topic.

Return ONLY a valid JSON array of objects, containing no markdown code fences, no introductory or concluding text.
Example:
[
  {{"title": "Section Title 1", "description": "What to write..."}},
  {{"title": "Section Title 2", "description": "What to write..."}}
]
"""


async def _plan_outline(state: ReportState) -> list:
    """LLM-plan the section outline, falling back to the template skeleton."""
    llm = get_llm()
    title = state["title"]
    topic = state["topic"]
    template = state.get("template", "business-report")

    try:
        resp = await llm.ainvoke([SystemMessage(content=_outline_prompt(title, topic, template))])
        content = (resp.content or "").strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n", "", content)
            content = re.sub(r"\n```$", "", content)
        sections = json.loads(content.strip())
        if not isinstance(sections, list) or not sections:
            raise ValueError("empty outline")
    except Exception as exc:
        print(f"[ReportSupervisor] outline parse failed ({exc}); using template fallback.")
        sections = _TEMPLATE_FALLBACKS.get(template, _TEMPLATE_FALLBACKS["business-report"])

    # Keep only well-formed section objects, cap the count.
    clean = [
        {"title": str(s.get("title", f"Section {i+1}")), "description": str(s.get("description", ""))}
        for i, s in enumerate(sections)
        if isinstance(s, dict)
    ]
    return clean[:MAX_SECTIONS]


def route_report_supervisor(state: ReportState) -> str:
    """Decide the next hop from the current state. Returns a node name."""
    if not state.get("outline"):
        return "researcher"  # freshly planned in the node → go research
    if state.get("research_gaps"):
        return "researcher"
    if not state.get("section_evidence"):
        return "researcher"
    if not state.get("draft"):
        return "writer"
    if not state.get("reviewed"):
        return "reviewer"
    # Draft has been reviewed:
    if state.get("section_feedback") and state.get("revision_count", 0) < MAX_REVISIONS:
        return "writer"
    return "finalize"


async def report_supervisor_node(state: ReportState) -> dict:
    writer = get_stream_writer()

    # First entry: no outline yet → plan the report into subtasks.
    if not state.get("outline"):
        writer({"event": "status", "data": {"message": "Planning report outline..."}})
        outline = await _plan_outline(state)
        writer({"event": "status", "data": {"message": f"Outline created with {len(outline)} sections."}})
        for sec in outline:
            writer({"event": "section_start", "data": {"title": sec["title"]}})
        print(f"[ReportSupervisor] Planned {len(outline)} sections.")
        return {"outline": outline, "next_agent": "researcher"}

    route = route_report_supervisor(state)
    print(f"[ReportSupervisor] route → {route} (rev={state.get('revision_count', 0)})")
    return {"next_agent": route}
