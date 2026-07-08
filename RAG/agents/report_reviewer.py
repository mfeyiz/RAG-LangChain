"""Report Reviewer — critic / QA for the assembled draft.

Checks the draft for accuracy, missing information, flow and SOURCE CONSISTENCY.
On problems it returns per-section feedback (→ back to the Writer) and, when a
section needs fresh evidence, `research_gaps` (→ back to the Researcher). Loops
are bounded by REPORT_MAX_REVISIONS (enforced by the supervisor router).
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer

from RAG.agents.report_state import ReportState
from RAG.agents.supervisor import get_llm

REVIEWER_SYSTEM_PROMPT = """You are a meticulous report reviewer. Evaluate the DRAFT against the evidence.

Criteria:
1. Accuracy: claims consistent with the provided evidence; no invented statistics.
2. Completeness: each planned section fully covers its focus.
3. Flow & structure: coherent, professional, well-ordered.
4. Source consistency: factual claims are supported by the listed sources; the draft does not contradict them.

Return ONLY valid JSON (no code fences):
{
  "approved": true,
  "feedback": "one short sentence, empty if approved",
  "section_feedback": {"Section Title": "what to fix"},
  "research_gaps": ["Section Title needing fresh evidence"]
}

Rules:
- Only list a section in section_feedback if it genuinely needs rewriting.
- Only list a section in research_gaps if the CURRENT evidence is insufficient (the researcher will fetch more).
- If the draft is solid, set approved=true and leave the maps/arrays empty.
- Keep feedback under 2 sentences."""


def _parse_review(content: str) -> dict:
    try:
        text = (content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        parsed = json.loads(text.strip())
        section_feedback = parsed.get("section_feedback") or {}
        return {
            "approved": bool(parsed.get("approved", False)),
            "feedback": str(parsed.get("feedback", "")).strip(),
            "section_feedback": section_feedback if isinstance(section_feedback, dict) else {},
            "research_gaps": [str(x) for x in (parsed.get("research_gaps") or []) if x],
        }
    except Exception:
        normalized = (content or "").strip()
        return {
            "approved": normalized.upper().startswith("APPROVED"),
            "feedback": normalized[:200],
            "section_feedback": {},
            "research_gaps": [],
        }


_EVIDENCE_SNIPPET_CHARS = 600
_MAX_DIGEST_SOURCES = 12


def _evidence_digest(state: ReportState) -> str:
    outline = state.get("outline", [])
    references = state.get("references") or []
    evidence_map = state.get("section_evidence") or {}
    lines = ["Planned sections: " + ", ".join(s["title"] for s in outline)]

    if references:
        lines.append("\nAvailable sources:")
        for r in references[:_MAX_DIGEST_SOURCES]:
            lines.append(f"  [{r.get('n')}] {r.get('label', '')} {('— ' + r['url']) if r.get('url') else ''}".rstrip())
    else:
        lines.append("Available sources: (none — local documents / general knowledge only)")

    # Give the reviewer the ACTUAL evidence text so it can verify claims, not just
    # the source list. Truncated per section to keep the prompt bounded.
    lines.append("\nSection evidence (truncated):")
    for sec in outline:
        ev = evidence_map.get(sec["title"]) or {}
        snippet = (ev.get("context") or ev.get("web_context") or "").strip()
        if snippet:
            lines.append(f"\n### {sec['title']}\n{snippet[:_EVIDENCE_SNIPPET_CHARS]}")
        else:
            lines.append(f"\n### {sec['title']}\n(no evidence gathered)")
    return "\n".join(lines)


async def report_reviewer_node(state: ReportState) -> dict:
    stream = get_stream_writer()
    stream({"event": "status", "data": {"message": "Reviewing the draft…"}})

    llm = get_llm()
    review_input = (
        f"Topic / Instructions: {state.get('topic', '')}\n\n"
        f"{_evidence_digest(state)}\n\n"
        f"DRAFT:\n{state.get('draft', '')}"
    )
    resp = await llm.ainvoke(
        [SystemMessage(content=REVIEWER_SYSTEM_PROMPT), HumanMessage(content=review_input)]
    )
    decision = _parse_review(resp.content)

    revision_count = state.get("revision_count", 0)

    # Approve if the reviewer approved, or if there is nothing actionable.
    if decision["approved"] or (not decision["section_feedback"] and not decision["research_gaps"]):
        print(f"[ReportReviewer] Approved (revisions: {revision_count}).")
        return {
            "reviewed": True,
            "review_feedback": "",
            "section_feedback": {},
            "research_gaps": [],
            "next_agent": "finalize",
        }

    feedback = decision["feedback"] or "Revisions requested."
    gaps = decision["research_gaps"]
    print(f"[ReportReviewer] Revision {revision_count + 1}: {feedback[:120]} "
          f"(gaps={gaps}, sections={list(decision['section_feedback'])})")
    stream({"event": "status", "data": {"message": f"Revising: {feedback[:80]}"}})

    return {
        "reviewed": True,
        "review_feedback": feedback,
        "section_feedback": decision["section_feedback"],
        "research_gaps": gaps,
        "revision_count": revision_count + 1,
        "next_agent": "researcher" if gaps else "writer",
    }
