"""Report Writer — turns per-section evidence into a professional Markdown draft.

Writes sections CONCURRENTLY (bounded). On a revision pass it rewrites only the
sections the reviewer flagged in `section_feedback`, leaving the rest intact,
then re-assembles the draft body.
"""

import asyncio
import os
import re

from langchain_core.messages import SystemMessage
from langgraph.config import get_stream_writer

from RAG.agents.report_state import ReportState
from RAG.agents.supervisor import get_llm

_CONCURRENCY = int(os.getenv("REPORT_SECTION_CONCURRENCY", "6"))
_SECTION_MAX_TOKENS = int(os.getenv("REPORT_SECTION_MAX_TOKENS", "800"))


def _section_prompt(topic: str, sec: dict, evidence: dict, feedback: str) -> str:
    context = (evidence.get("context") or "").strip()
    web_context = (evidence.get("web_context") or "").strip()
    web_block = f"\n\nWEB SEARCH RESULTS (from the internet):\n{web_context}" if web_context else ""
    fix_block = (
        f"\n\nREVISION REQUIRED — the reviewer asked you to fix this section:\n{feedback}\n"
        "Address the feedback precisely while keeping what was already correct."
        if feedback else ""
    )
    return f"""You are a professional report writer producing ONE section of a report. Write accurate, well-structured, professional markdown.

Topic / Instructions: {topic}
Section Title: {sec['title']}
Section Focus: {sec.get('description', '')}

LOCAL DOCUMENT CONTEXT:
{context or '(no local documents matched — rely on the web results and general knowledge)'}{web_block}{fix_block}

Rules:
1. Respond in the SAME LANGUAGE as the Topic / Instructions (Turkish or English).
2. Prefer the provided web results and local context; you may add well-known general knowledge, but do not invent specific statistics.
3. Start with the section heading, e.g. `## {sec['title']}`. Do NOT repeat the report's main title. Keep it focused (2-4 short paragraphs, lists/tables where useful).
4. When the section presents numeric trends or comparisons AND you have concrete figures, include ONE chart by emitting a fenced code block tagged `chart` with JSON:
```chart
{{"type": "bar|line|pie", "title": "…", "labels": ["…", "…"], "values": [12.3, 45.6]}}
```
   Put it right after the paragraph discussing those numbers. Only use real numbers; at most one chart per section.
5. Output raw markdown only (do not wrap the whole answer in a code fence)."""


def _strip_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:markdown|md)?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


async def report_writer_node(state: ReportState) -> dict:
    stream = get_stream_writer()
    outline = state.get("outline", [])
    topic = state.get("topic", "")
    evidence_map = state.get("section_evidence") or {}
    feedback_map = state.get("section_feedback") or {}
    sections = dict(state.get("sections") or {})

    section_llm = get_llm().bind(max_tokens=_SECTION_MAX_TOKENS)

    # Revision pass → only rewrite flagged sections; first pass → write all.
    if feedback_map:
        to_write = [(i, s) for i, s in enumerate(outline) if s["title"] in feedback_map]
        stream({"event": "status", "data": {"message": f"Revising {len(to_write)} section(s)…"}})
    else:
        to_write = list(enumerate(outline))
        stream({"event": "status", "data": {"message": f"Writing {len(to_write)} sections in parallel…"}})

    sem = asyncio.Semaphore(min(max(len(to_write), 1), _CONCURRENCY))

    async def _write(idx: int, sec: dict) -> tuple[int, str, str]:
        evidence = evidence_map.get(sec["title"], {})
        feedback = feedback_map.get(sec["title"], "")
        async with sem:
            resp = await section_llm.ainvoke(
                [SystemMessage(content=_section_prompt(topic, sec, evidence, feedback))]
            )
        return idx, sec["title"], _strip_fence(resp.content)

    for coro in asyncio.as_completed([_write(i, s) for i, s in to_write]):
        idx, title, text = await coro
        sections[idx] = text
        stream({"event": "section_complete", "data": {"title": title}})

    draft = "\n\n".join(sections[i] for i in sorted(sections) if sections.get(i))

    return {
        "sections": sections,
        "draft": draft,
        "section_feedback": {},   # consumed
        "reviewed": False,        # new draft awaits review
        "next_agent": "reviewer",
    }
