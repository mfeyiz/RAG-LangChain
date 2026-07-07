"""Report Researcher — deep research per outline subtopic.

For each section the supervisor planned, runs a local RAG retrieval and (when
available) a Tavily web search CONCURRENTLY, records the evidence per section and
accumulates de-duplicated web sources. When the reviewer flags `research_gaps`,
only those sections are re-researched.

Read-only against retrieval — never writes back to any channel (that is the
Editor's job, which this pipeline stays clear of).
"""

import asyncio
import os

from langgraph.config import get_stream_writer

from RAG.agents.report_state import ReportState
from RAG.services.retrieval import retrieve_context_async
from RAG.services.web_search import web_search, web_search_available

_CONCURRENCY = int(os.getenv("REPORT_SECTION_CONCURRENCY", "6"))
_RAG_TOP_K = int(os.getenv("REPORT_RAG_TOP_K", "6"))
_WEB_CONTEXT_CHARS = 3500


async def _research_section(topic: str, sec: dict, sem: asyncio.Semaphore) -> tuple[str, dict, list]:
    """Return (section_title, {context, web_context}, web_sources)."""
    title = sec["title"]
    desc = sec.get("description", "")
    query = f"{title} {desc} {topic}".strip()

    async with sem:
        # RAG + web run concurrently; both degrade to empty on failure.
        async def _rag() -> str:
            try:
                context, _ = await retrieve_context_async(query, top_k=_RAG_TOP_K)
                return context or ""
            except Exception as exc:
                print(f"[ReportResearcher] RAG failed for '{title}': {exc}")
                return ""

        async def _web() -> tuple[str, list]:
            if not web_search_available():
                return "", []
            try:
                web = await asyncio.to_thread(web_search, query)
                return (web.get("context", "") or "")[:_WEB_CONTEXT_CHARS], web.get("sources", []) or []
            except Exception as exc:
                print(f"[ReportResearcher] web search failed for '{title}': {exc}")
                return "", []

        context, (web_context, web_sources) = await asyncio.gather(_rag(), _web())

    return title, {"context": context, "web_context": web_context}, web_sources


async def report_researcher_node(state: ReportState) -> dict:
    stream = get_stream_writer()
    outline = state.get("outline", [])
    topic = state.get("topic", "")
    gaps = set(state.get("research_gaps") or [])

    if gaps:
        targets = [s for s in outline if s["title"] in gaps]
        stream({"event": "status", "data": {"message": f"Re-researching {len(targets)} section(s)…"}})
    else:
        targets = outline
        stream({"event": "status", "data": {"message": "Gathering evidence…"}})

    sem = asyncio.Semaphore(min(max(len(targets), 1), _CONCURRENCY))
    results = await asyncio.gather(*[_research_section(topic, s, sem) for s in targets])

    # Merge new evidence over any existing (keeps untouched sections on a gap re-run).
    section_evidence = dict(state.get("section_evidence") or {})
    sources = list(state.get("sources") or [])
    seen_urls = {s.get("url") for s in sources}

    for title, evidence, web_sources in results:
        section_evidence[title] = evidence
        for s in web_sources:
            url = s.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({"title": s.get("title") or url, "url": url})

    print(f"[ReportResearcher] evidence for {len(results)} section(s), {len(sources)} total sources.")

    return {
        "section_evidence": section_evidence,
        "sources": sources,
        "research_gaps": [],   # gaps addressed
        "next_agent": "writer",
    }
