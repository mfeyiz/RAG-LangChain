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
from RAG.services.retrieval import DEFAULT_CHANNEL, retrieve_context_async
from RAG.services.web_search import web_search, web_search_available

_CONCURRENCY = int(os.getenv("REPORT_SECTION_CONCURRENCY", "6"))
_RAG_TOP_K = int(os.getenv("REPORT_RAG_TOP_K", "6"))
_WEB_CONTEXT_CHARS = int(os.getenv("REPORT_WEB_CONTEXT_CHARS", "3500"))
_MAX_SECTION_FIGURES = int(os.getenv("REPORT_MAX_SECTION_FIGURES", "2"))


def _collect_from_metadata(metadata: list[dict]) -> tuple[list[str], list[dict]]:
    """From retrieval metadata, pull (distinct local doc sources, anchored figures)."""
    local_sources: list[str] = []
    figures: list[dict] = []
    seen_fig: set[tuple[str, str]] = set()
    for entry in metadata or []:
        source = entry.get("source", "")
        if source and source != "unknown" and source not in local_sources:
            local_sources.append(source)
        for name in entry.get("images", []) or []:
            key = (source, name)
            if key in seen_fig:
                continue
            seen_fig.add(key)
            figures.append({"source": source, "name": name, "channel": DEFAULT_CHANNEL})
            if len(figures) >= _MAX_SECTION_FIGURES:
                break
    return local_sources, figures


async def _research_section(topic: str, sec: dict, sem: asyncio.Semaphore) -> tuple[str, dict, list]:
    """Return (section_title, evidence_dict, web_sources).

    evidence_dict carries: context, web_context, local_sources, figures.
    """
    title = sec["title"]
    desc = sec.get("description", "")
    query = f"{title} {desc} {topic}".strip()

    async with sem:
        # RAG + web run concurrently; both degrade to empty on failure.
        async def _rag() -> tuple[str, list[str], list[dict]]:
            try:
                context, metadata = await retrieve_context_async(query, top_k=_RAG_TOP_K)
                local_sources, figures = _collect_from_metadata(metadata)
                return context or "", local_sources, figures
            except Exception as exc:
                print(f"[ReportResearcher] RAG failed for '{title}': {exc}")
                return "", [], []

        async def _web() -> tuple[str, list]:
            if not web_search_available():
                return "", []
            try:
                web = await asyncio.to_thread(web_search, query)
                return (web.get("context", "") or "")[:_WEB_CONTEXT_CHARS], web.get("sources", []) or []
            except Exception as exc:
                print(f"[ReportResearcher] web search failed for '{title}': {exc}")
                return "", []

        (context, local_sources, figures), (web_context, web_sources) = await asyncio.gather(_rag(), _web())

    evidence = {
        "context": context,
        "web_context": web_context,
        "local_sources": local_sources,
        "figures": figures,
    }
    return title, evidence, web_sources


def _build_references(outline: list, section_evidence: dict) -> tuple[list, dict]:
    """Assign stable [n] numbers across all sections (web + local docs), in outline
    order. Returns (references, per_title_number_map). references := [{n,label,url,kind}].
    """
    references: list = []
    index: dict = {}          # key -> n
    per_title: dict = {}      # title -> [n, ...]

    def _add(key: str, label: str, url: str, kind: str) -> int:
        if key in index:
            return index[key]
        n = len(references) + 1
        index[key] = n
        references.append({"n": n, "label": label, "url": url, "kind": kind})
        return n

    for sec in outline:
        title = sec["title"]
        ev = section_evidence.get(title) or {}
        nums: list[int] = []
        for s in ev.get("web_sources", []) or []:
            url = s.get("url") or ""
            label = s.get("title") or url or "Source"
            nums.append(_add(url or label, label, url, "web"))
        for src in ev.get("local_sources", []) or []:
            nums.append(_add(f"doc:{src}", src, "", "document"))
        per_title[title] = nums
    return references, per_title


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
    for title, evidence, web_sources in results:
        # Stash the raw web sources on the evidence so reference numbering (below)
        # can run over the full, merged section set — not just this batch.
        evidence["web_sources"] = web_sources
        section_evidence[title] = evidence

    # Build stable [n] citation numbers across every section, in outline order.
    references, per_title = _build_references(outline, section_evidence)

    # Flat de-duplicated source list (kept for backward-compat with app.py) and a
    # ready-to-drop "sources block" per section for the writer prompt.
    sources = [{"title": r["label"], "url": r["url"]} for r in references]
    for sec in outline:
        title = sec["title"]
        ev = section_evidence.get(title) or {}
        nums = per_title.get(title, [])
        by_n = {r["n"]: r for r in references}
        lines = []
        for n in nums:
            r = by_n.get(n)
            if not r:
                continue
            lines.append(f"[{n}] {r['label']}{(' — ' + r['url']) if r['url'] else ''}")
        ev["sources_block"] = "\n".join(lines)
        section_evidence[title] = ev

    print(f"[ReportResearcher] evidence for {len(results)} section(s), {len(references)} references.")

    return {
        "section_evidence": section_evidence,
        "sources": sources,
        "references": references,
        "research_gaps": [],   # gaps addressed
        "next_agent": "writer",
    }
