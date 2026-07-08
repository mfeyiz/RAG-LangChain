"""Shared report-generation execution core.

Drives the report graph exactly like the original inline ``/documents/generate``
handler did, but as a plain async generator decoupled from HTTP/SSE and from
Kafka/Redis — so both the request-time fallback path (``RAG/app.py``, used
when Kafka is unavailable) and the Kafka worker (``RAG/worker.py``) run the
SAME logic instead of two diverging copies.

``run_report_job`` yields ``{"event": str, "data": dict}`` — plain dicts, not
pre-serialized JSON. Callers decide how to deliver them:
- ``RAG/app.py``'s inline fallback ``json.dumps()``s ``data`` for
  ``EventSourceResponse``.
- ``RAG/worker.py`` publishes them via
  ``RAG.services.job_events.publish_event()`` (which does its own
  serialization for the Redis transport).
"""
import asyncio
import hashlib

from RAG.services import paths, report_charts, report_registry


def unique_workspace_source(title: str) -> str:
    """Slugify `title` into a `<stem>.md` that doesn't collide with an existing
    workspace document (suffixing -2, -3, … on conflict)."""
    base_stem = paths.sanitize_stem(title)
    stem = base_stem
    n = 2
    while paths.workspace_md_path(paths.source_for(stem)).exists():
        stem = f"{base_stem}-{n}"
        n += 1
    return paths.source_for(stem)


def index_and_export_workspace(source: str, commit_msg: str) -> str:
    """Reindex a workspace source, regenerate PDF/DOCX (best-effort), commit."""
    from RAG.services.document_manager import reindex_workspace_source
    from RAG.agents.editor import _render_pdf_safe
    from RAG.services.version_control import commit_change

    reindex_workspace_source(source)
    _render_pdf_safe(source)
    try:
        from RAG.services.docx_exporter import render as render_docx
        render_docx(source)
    except Exception as exc:
        print(f"[report_jobs] DOCX generation failed for {source}: {exc}")
    return commit_change(source, commit_msg)


async def run_report_job(
    report_graph,
    *,
    title: str,
    topic: str,
    template: str,
    trace_id: str,
    language: str = "auto",
    tone: str = "",
    audience: str = "",
    length: str = "standard",
    sections: list | None = None,
):
    """Drive the report graph end-to-end. Async generator of {"event", "data"} dicts.

    Terminal events: "complete" (success, carries {source, markdown}) followed
    by a "status" + "done", or "error" (carries {error}) on failure.
    """
    from RAG.agents.report_supervisor import length_profile

    try:
        source = unique_workspace_source(title)
        stem = paths.stem_of(source)

        def _embed_charts(md: str) -> str:
            """Replace ```chart JSON blocks with rendered PNG image links."""
            imgdir = paths.workspace_images_dir(source)

            def repl(m):
                spec = report_charts.parse_spec(m.group(1))
                if not spec:
                    return ""
                # Content-hash filename → deterministic across runs + dedups
                # identical charts (unlike the salted builtin hash()).
                digest = hashlib.sha1(m.group(1).encode("utf-8")).hexdigest()[:12]
                name = f"chart_{digest}.png"
                if report_charts.render_chart_spec(spec, imgdir / name):
                    alt = str(spec.get("title") or "chart")
                    return f"![{alt}](/images/workspace/{stem}/{name})"
                return ""

            return report_charts.CHART_BLOCK_RE.sub(repl, md)

        # Drive the multi-agent report graph. Node progress arrives on the
        # "custom" stream (status / section_start / section_complete events);
        # the final assembled state arrives on the "values" stream.
        _, section_max_tokens = length_profile(length)

        initial_state = {
            "topic": topic,
            "title": title,
            "template": template,
            "trace_id": trace_id,
            "revision_count": 0,
            "language": language or "auto",
            "tone": tone or "",
            "audience": audience or "",
            "length": length or "standard",
            "section_max_tokens": section_max_tokens,
        }
        # Honour a caller-supplied outline (two-step: plan → edit → generate).
        if sections:
            clean = [
                {"title": str(s.get("title", "")).strip(),
                 "description": str(s.get("description", "")).strip()}
                for s in sections
                if isinstance(s, dict) and str(s.get("title", "")).strip()
            ]
            if clean:
                initial_state["outline"] = clean

        final_state: dict = {}
        async for mode, payload in report_graph.astream(
            initial_state,
            stream_mode=["custom", "values"],
            config={"recursion_limit": 50},
        ):
            if mode == "custom":
                yield {"event": payload.get("event", "status"), "data": payload.get("data", {})}
            elif mode == "values":
                final_state = payload

        body = final_state.get("final_markdown") or final_state.get("draft") or ""
        references = final_state.get("references") or []
        web_sources = final_state.get("sources") or []

        # Embed charts sequentially (matplotlib/pyplot is not thread-safe).
        yield {"event": "status", "data": {"message": "Rendering charts…"}}
        if "```chart" in body:
            body = await asyncio.to_thread(_embed_charts, body)

        # Append a numbered References section aligned with the inline [n]
        # citations the Writer emitted. `references` carries stable numbers;
        # fall back to the flat `sources` list for older graph runs.
        if references:
            lines = ["## Kaynaklar / References"]
            for r in references:
                n = r.get("n")
                label = r.get("label") or r.get("url") or f"Source {n}"
                url = r.get("url") or ""
                lines.append(f"{n}. [{label}]({url})" if url else f"{n}. {label}")
            body += "\n\n" + "\n".join(lines)
        elif web_sources:
            lines = ["## Kaynaklar / Sources"]
            for i, s in enumerate(web_sources, 1):
                t = s.get("title") or s.get("url") or f"Source {i}"
                u = s.get("url") or ""
                lines.append(f"{i}. [{t}]({u})" if u else f"{i}. {t}")
            body += "\n\n" + "\n".join(lines)

        full_markdown = f"# {title}\n\n" + body

        # Write the markdown FIRST and signal completion so the report is
        # viewable immediately; the heavier reindex + PDF/DOCX + commit runs
        # afterwards while the caller's stream stays open.
        paths.workspace_md_path(source).write_text(full_markdown, encoding="utf-8")
        report_registry.record(source, title, template, generated=True)

        yield {"event": "complete", "data": {"source": source, "markdown": full_markdown}}
        yield {"event": "status", "data": {"message": "Indexing & exporting…"}}
        try:
            await asyncio.to_thread(index_and_export_workspace, source, "generate report")
        except Exception as exc:
            print(f"[report_jobs] post-save index/export failed: {exc}")
        yield {"event": "done", "data": {}}

    except Exception as e:
        print(f"[report_jobs] Report Generation Error: {e}")
        import traceback
        traceback.print_exc()
        yield {"event": "error", "data": {"error": str(e)}}
