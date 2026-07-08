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

# length → (max sections, per-section token budget). Lets the caller trade off
# depth vs. speed/cost from the Report Studio modal.
_LENGTH_PROFILES = {
    "brief":    (4, 450),
    "standard": (MAX_SECTIONS, 800),
    "detailed": (max(MAX_SECTIONS, 8), 1200),
}


def length_profile(length: str) -> tuple[int, int]:
    return _LENGTH_PROFILES.get((length or "standard").lower(), _LENGTH_PROFILES["standard"])


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
    "threat-intelligence": [
        {"title": "Executive Summary", "description": "Key takeaways for decision makers"},
        {"title": "Threat Overview", "description": "The threat actor / campaign / vulnerability at a glance"},
        {"title": "Technical Analysis", "description": "TTPs, IOCs, attack chain and affected systems"},
        {"title": "Impact Assessment", "description": "Business and operational impact, affected assets"},
        {"title": "Mitigation & Recommendations", "description": "Detection, response and hardening actions"},
        {"title": "Conclusion", "description": "Summary and outlook"},
    ],
    "blank": [
        {"title": "Introduction", "description": "Set the context and goals"},
        {"title": "Overview", "description": "Main overview of the topic"},
        {"title": "Analysis", "description": "Detailed analysis"},
        {"title": "Conclusion", "description": "Summary and next steps"},
    ],
}


def _outline_prompt(title: str, topic: str, template: str, max_sections: int,
                    tone: str = "", audience: str = "", language: str = "") -> str:
    hints = []
    if tone:
        hints.append(f"Tone: {tone}.")
    if audience:
        hints.append(f"Target audience: {audience}.")
    if language and language.lower() not in ("auto", ""):
        hints.append(f"Write the section titles/descriptions in {language}.")
    hint_line = (" ".join(hints) + "\n") if hints else ""
    return f"""You are a professional report planner. Plan the outline for a report titled "{title}" on the topic: "{topic}".
The report template chosen is "{template}".
{hint_line}Based on this template, generate a list of at most {max_sections} section objects in JSON format.
Each object must have "title" (the section heading) and "description" (a brief guide of what to write about, referencing specific aspects of the topic).

Standard templates and their required sections:
- business-report: Executive Summary, Background, Analysis, Key Metrics, Recommendations, Conclusion
- research-summary: Abstract, Question, Findings, Discussion, References
- project-status: Status Overview, Progress This Period, Upcoming, Risks & Blockers, Metrics
- threat-intelligence: Executive Summary, Threat Overview, Technical Analysis, Impact Assessment, Mitigation & Recommendations, Conclusion
- blank: Create logical, well-structured sections appropriate for the topic.

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
    max_sections, _ = length_profile(state.get("length", "standard"))
    tone = state.get("tone", "")
    audience = state.get("audience", "")
    language = state.get("language", "")

    try:
        resp = await llm.ainvoke([SystemMessage(
            content=_outline_prompt(title, topic, template, max_sections, tone, audience, language)
        )])
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
    return clean[:max_sections]


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
