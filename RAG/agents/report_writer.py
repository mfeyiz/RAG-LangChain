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
from RAG.services import paths

_CONCURRENCY = int(os.getenv("REPORT_SECTION_CONCURRENCY", "6"))
_SECTION_MAX_TOKENS = int(os.getenv("REPORT_SECTION_MAX_TOKENS", "800"))


def _figure_catalogue(figures: list) -> str:
    """Ready-to-embed markdown image URLs for figures anchored to this section."""
    lines = []
    for fig in figures or []:
        source = fig.get("source", "")
        name = fig.get("name", "")
        channel = fig.get("channel", "workspace")
        if not source or not name:
            continue
        stem = paths.stem_of(source)
        lines.append(f"- ![{name}](/images/{channel}/{stem}/{name})")
    return "\n".join(lines)


def _section_prompt(topic: str, sec: dict, evidence: dict, feedback: str,
                    language: str = "", tone: str = "", audience: str = "") -> str:
    context = (evidence.get("context") or "").strip()
    web_context = (evidence.get("web_context") or "").strip()
    sources_block = (evidence.get("sources_block") or "").strip()
    figures_block = _figure_catalogue(evidence.get("figures"))

    web_block = f"\n\nWEB SEARCH RESULTS (from the internet):\n{web_context}" if web_context else ""
    src_block = (
        f"\n\nAVAILABLE SOURCES (cite these EXACT numbers with [n]):\n{sources_block}"
        if sources_block else ""
    )
    fig_block = (
        f"\n\nAVAILABLE FIGURES (embed a relevant one by copying its markdown line verbatim):\n{figures_block}"
        if figures_block else ""
    )
    fix_block = (
        f"\n\nREVISION REQUIRED — the reviewer asked you to fix this section:\n{feedback}\n"
        "Address the feedback precisely while keeping what was already correct."
        if feedback else ""
    )
    lang_rule = (
        f"Write the entire section in {language}."
        if language and language.lower() not in ("auto", "")
        else "Respond in the SAME LANGUAGE as the Topic / Instructions."
    )
    style_bits = []
    if tone:
        style_bits.append(f"tone: {tone}")
    if audience:
        style_bits.append(f"written for: {audience}")
    style_line = (f"\nStyle — {', '.join(style_bits)}." if style_bits else "")
    return f"""You are a professional report writer producing ONE section of a report. Write accurate, well-structured, professional markdown.
{style_line}
Topic / Instructions: {topic}
Section Title: {sec['title']}
Section Focus: {sec.get('description', '')}

LOCAL DOCUMENT CONTEXT:
{context or '(no local documents matched — rely on the web results and general knowledge)'}{web_block}{src_block}{fig_block}{fix_block}

Rules:
1. {lang_rule}
2. Prefer the provided web results and local context; you may add well-known general knowledge, but do not invent specific statistics.
3. CITE your sources: after a factual claim drawn from the sources above, add its bracketed number, e.g. "revenue grew 12% [1]". Use ONLY the numbers listed under AVAILABLE SOURCES; never invent citation numbers. If no sources are listed, write without citations.
4. Start with the section heading, e.g. `## {sec['title']}`. Do NOT repeat the report's main title. Keep it focused (2-4 short paragraphs, lists/tables where useful).
5. For metric-heavy sections (KPIs, comparisons, results), present the figures as a markdown pipe table so they export cleanly to Word/PDF.
6. When the section presents numeric trends or comparisons AND you have concrete figures, include ONE chart by emitting a fenced code block tagged `chart` with JSON:
```chart
{{"type": "bar|line|pie|area|radar", "title": "…", "xlabel": "…", "ylabel": "…", "labels": ["…", "…"], "values": [12.3, 45.6]}}
```
   `xlabel`/`ylabel` are optional (ignored for pie/radar). Put the chart right after the paragraph discussing those numbers. Only use real numbers; at most one chart per section.
7. If a listed figure is directly relevant, embed it by copying its markdown image line verbatim; otherwise omit figures.
8. Output raw markdown only (do not wrap the whole answer in a code fence)."""


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
    language = state.get("language", "") or ""
    tone = state.get("tone", "") or ""
    audience = state.get("audience", "") or ""
    evidence_map = state.get("section_evidence") or {}
    feedback_map = state.get("section_feedback") or {}
    sections = dict(state.get("sections") or {})

    # Section length budget is caller-controllable (brief/standard/detailed → tokens).
    max_tokens = int(state.get("section_max_tokens") or _SECTION_MAX_TOKENS)
    section_llm = get_llm().bind(max_tokens=max_tokens)

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
        # Guard each section so one LLM hiccup can't abort the whole report.
        try:
            async with sem:
                resp = await section_llm.ainvoke(
                    [SystemMessage(content=_section_prompt(topic, sec, evidence, feedback, language, tone, audience))]
                )
            return idx, sec["title"], _strip_fence(resp.content)
        except Exception as exc:
            print(f"[ReportWriter] section '{sec['title']}' failed: {exc}")
            return idx, sec["title"], f"## {sec['title']}\n\n_(This section could not be generated.)_"

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
