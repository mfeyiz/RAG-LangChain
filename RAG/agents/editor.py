"""Editor agent — the write-back half of the bi-directional RAG.

Triggered by an `@update ...` chat message. It locates the relevant workspace
Markdown file and applies the user's instruction as a SMALL PATCH (append a new
section, or replace a short existing snippet) — the LLM only ever emits the
changed fragment, never the whole file, so large documents can't be truncated
or destroyed. It then re-indexes the workspace channel and regenerates the
workspace PDF. The read-only `originals` channel is never touched.
"""
import asyncio
import re

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from RAG.agents.state import AgentState
from RAG.agents.supervisor import get_llm
from RAG.services import paths
from RAG.services.document_manager import reindex_workspace_source
from RAG.services.pending_edits import stash as stash_pending_edit
from RAG.services.retrieval import retrieve_context
from RAG.services.tracing import invoke_with_langfuse, trace_event, traced_observation
from RAG.services.version_control import commit_change

_DEFAULT_NOTES_SOURCE = "notes.md"

SECTION_EDITOR_SYSTEM_PROMPT = """You maintain a Markdown knowledge base. You are given ONE SECTION of a document and an UPDATE INSTRUCTION. Apply the instruction to that section and return the WHOLE, REWRITTEN SECTION.

Hard rules:
- Return ONLY this one section's Markdown (including its heading line). NEVER return other sections or the whole document, and never add commentary or code fences.
- If the instruction CORRECTS or CHANGES existing information, edit that information IN PLACE — modify the existing sentence/line/list item. Do NOT also keep the old version. No duplicated or contradictory statements.
- If the instruction ADDS genuinely new information, place it at the most logical spot within the section (e.g. the next item of the relevant list), not blindly at the end.
- Preserve everything in the section that the instruction does not touch, byte-for-byte where possible (keep existing headings, image links like ![](figure.png), tables, and formatting).
- Use the same language as the section. Keep well-formed Markdown.
- Output ONLY the rewritten section Markdown."""


def strip_update_tag(query: str) -> str:
    """Remove a leading/embedded @update tag, returning the bare instruction."""
    return re.sub(r"@update\b", "", query, count=1, flags=re.IGNORECASE).strip()


async def editor_node(state: AgentState) -> dict:
    instruction = strip_update_tag(state["query"])

    with traced_observation("editor", input_payload={"instruction": instruction}) as span:
        if not instruction:
            msg = "Lütfen @update etiketinden sonra ne eklemek/güncellemek istediğinizi yazın."
            span.update(output={"error": "empty_instruction"})
            return _finish(msg)

        # The instruction alone ("add a step 8 …") often doesn't name the topic,
        # so locating the target purely from it picks the wrong file. Enrich the
        # search with the recent conversation (e.g. the "cyber kill chain"
        # question this update follows) so the edit lands in the right document.
        history = state.get("conversation_history") or []
        recent_context = " ".join((turn.get("query") or "") for turn in history[-2:]).strip()
        search_text = f"{recent_context} {instruction}".strip()

        # Locate the target file AND the full SECTION (parent) retrieval returns
        # for this topic. We hand the whole section to the LLM, let it rewrite the
        # section in place, then swap that exact section back into the file. This
        # keeps the edit in the right place (no end-of-file dumping), lets the LLM
        # update existing facts instead of duplicating them, and never rewrites
        # the whole document.
        source, section, heading_path = await asyncio.to_thread(_locate_target, search_text)
        md_path = paths.workspace_md_path(source)
        current = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

        context_note = (
            f"\n\nCONVERSATION CONTEXT (what the user was asking about):\n{recent_context}"
            if recent_context else ""
        )

        # When the instruction asks to add a CHART/graph to the document, generate
        # the PNG up front (code interpreter over the stored tables) and tell the
        # editor LLM to embed it inline via ![](/images/workspace/<stem>/<file>).
        chart_embed = await _maybe_generate_chart_for_edit(instruction, source, span)

        # Resolve the exact on-disk section slice to rewrite IN PLACE. Prefer an
        # exact match of the indexed section text; otherwise relocate the section
        # by its heading trail — the indexed `parent_content` is whitespace-
        # normalised and often not byte-for-byte in the file, and relying on an
        # exact match wrongly forced updates to append at the end of the file.
        section_bounds: tuple[int, int] | None = None
        section_text = ""
        if section and section in current:
            idx = current.index(section)
            section_bounds = (idx, idx + len(section))
            section_text = section
        elif heading_path:
            found = _find_section_span(current, heading_path)
            if found:
                section_bounds = found
                section_text = current[found[0]:found[1]]

        # No existing section located (empty workspace / brand-new topic): fall
        # back to the legacy small-patch path so the user's content is captured.
        if section_bounds is None or not section_text.strip():
            updated, change = await _legacy_patch(
                instruction, context_note, source, current, span, chart_embed
            )
        else:
            llm = get_llm()
            messages = [
                SystemMessage(content=SECTION_EDITOR_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"UPDATE INSTRUCTION:\n{instruction}{context_note}"
                        + (f"\n\nA chart figure was generated for this update. Embed it in the rewritten "
                           f"section with this exact Markdown image link at the right place:\n{chart_embed['markdown']}"
                           if chart_embed else "")
                        + f"\n\nSECTION TO EDIT (from {source}) — return this whole section, rewritten:\n{section_text}"
                    )
                ),
            ]
            response = await invoke_with_langfuse(llm, messages)
            new_section = _clean_markdown(response.content).strip()
            if not new_section:
                span.update(output={"error": "empty_output"})
                return _finish("Güncelleme üretilemedi, lütfen talimatı netleştirin.")
            # Ensure the chart link is present even if the model dropped it.
            if chart_embed and chart_embed["markdown"] not in new_section:
                new_section += "\n\n" + chart_embed["markdown"]
            # Deterministic in-place splice of the located section slice.
            start, end = section_bounds
            updated = current[:start] + new_section + current[end:]
            change = "section_rewrite"

        if updated is None or updated.strip() == current.strip():
            span.update(output={"error": "no_change"})
            return _finish(
                "Belgede bir değişiklik yapılmadı. Lütfen talimatı daha açık yazın."
            )

        # ── Human-in-the-loop: do NOT persist yet. Stash the diff and ask the
        # user to approve it. The frontend renders a red/green diff preview; on
        # "Onayla" it POSTs the edit_token to /update/apply which persists,
        # re-indexes, regenerates the PDF, and creates a Git commit. On "Reddet"
        # it calls /update/reject to discard the stashed edit.
        token = stash_pending_edit({
            "source": source,
            "instruction": instruction,
            "before": current,
            "after": updated,
            "change_kind": change,
        })
        preview_lines = _diff_preview_lines(current, updated)

        await trace_event(
            state["trace_id"],
            "editor.preview",
            {"source": source, "change_kind": change, "preview_lines": len(preview_lines)},
        )
        span.update(output={"source": source, "change_kind": change, "preview": True})

        return {
            "next_agent": "finish",
            "needs_edit_approval": True,
            "edit_pending": True,
            "edit_token": token,
            "edit_preview": {
                "source": source,
                "instruction": instruction,
                "before": current,
                "after": updated,
                "change_kind": change,
                "diff": preview_lines,
            },
            "edit_target_file": source,
            "edit_instruction": instruction,
            "final_response": "",  # surfaced via the edit_preview SSE event, not a message
            "messages": [AIMessage(content="Değişiklik örneği hazır — onayınız bekleniyor.")],
        }


def _locate_target(search_text: str) -> tuple[str, str, str]:
    """Find the workspace file and the full parent SECTION for this topic.

    Returns ``(source, parent_content, heading_path)`` where ``parent_content``
    is the verbatim Markdown of the heading section the top match belongs to —
    the unit the editor rewrites in place — and ``heading_path`` is that
    section's heading trail (e.g. ``"Security > Threats"``) used to relocate the
    section in the file even if ``parent_content`` drifted from the on-disk text.
    Falls back to a shared notes file (empty section) when the workspace is empty
    or nothing relevant is found.
    """
    try:
        _, metadata = retrieve_context(search_text, top_k=5, channel="workspace")
    except Exception as exc:
        print(f"[Editor] Retrieval failed while locating target: {exc}")
        metadata = []

    for item in metadata:
        src = item.get("source")
        if src and paths.workspace_md_path(src).exists():
            return (
                src,
                (item.get("parent_content") or item.get("content") or ""),
                (item.get("heading_path") or ""),
            )
    return _DEFAULT_NOTES_SOURCE, "", ""


def _norm_heading(text: str) -> str:
    """Normalise a heading for comparison: drop leading #'s, trailing #'s, and
    collapse whitespace/case so on-disk headings match indexed heading paths."""
    text = re.sub(r"^#+\s*", "", (text or "").strip())
    text = re.sub(r"\s*#+\s*$", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _find_section_span(current: str, heading_path: str) -> tuple[int, int] | None:
    """Locate a heading section in ``current`` by its heading trail and return
    the ``(start, end)`` char offsets spanning from that heading line up to the
    next heading of the same-or-higher level (or EOF).

    This is deterministic and immune to the whitespace normalisation that makes
    the indexed ``parent_content`` differ from the on-disk file, so an update
    lands in the right section instead of being appended at the end.
    """
    if not heading_path.strip():
        return None

    components = [_norm_heading(c) for c in heading_path.split(">")]
    target = components[-1] if components else ""
    if not target:
        return None

    # All headings in the file: (start_offset, level, normalised text).
    headings: list[tuple[int, int, str]] = []
    for m in re.finditer(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", current):
        headings.append((m.start(), len(m.group(1)), _norm_heading(m.group(2))))
    if not headings:
        return None

    def ancestor_chain(idx: int) -> list[str]:
        """Heading texts from the document root down to headings[idx]."""
        _, level, _ = headings[idx]
        chain: list[str] = []
        cur_level = level
        for j in range(idx, -1, -1):
            _, lvl, txt = headings[j]
            if lvl < cur_level or j == idx:
                chain.append(txt)
                cur_level = lvl
                if lvl == 1:
                    break
        return list(reversed(chain))

    candidates = [i for i, (_, _, txt) in enumerate(headings) if txt == target]
    if not candidates:
        return None

    # Prefer a candidate whose full ancestor chain matches the indexed path
    # (disambiguates repeated heading names); else take the first text match.
    chosen = candidates[0]
    for i in candidates:
        if ancestor_chain(i) == components:
            chosen = i
            break

    start, level, _ = headings[chosen]
    end = len(current)
    for j in range(chosen + 1, len(headings)):
        nxt_start, nxt_level, _ = headings[j]
        if nxt_level <= level:
            end = nxt_start
            break
    return start, end


_LEGACY_SYSTEM_PROMPT = """You maintain a Markdown knowledge base. Turn the UPDATE INSTRUCTION into a small, self-contained Markdown block (a short heading plus content) suitable to add to a notes file. Output ONLY the Markdown — no commentary, no code fences."""


async def _legacy_patch(
    instruction: str, context_note: str, source: str, current: str, span, chart_embed: dict | None = None
) -> tuple[str | None, str]:
    """Fallback when no existing section was located (empty workspace / new topic).

    Asks the model for a self-contained Markdown block and appends it, so the
    user's content is captured even with nothing to rewrite in place.
    """
    llm = get_llm()
    prompt = f"UPDATE INSTRUCTION:\n{instruction}{context_note}"
    if chart_embed:
        prompt += "\n\nInclude this generated chart image at the end, exactly:\n" + chart_embed["markdown"]
    response = await invoke_with_langfuse(
        llm,
        [
            SystemMessage(content=_LEGACY_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ],
    )
    block = _clean_markdown(response.content).strip()
    if not block:
        return None, "append"
    if chart_embed and chart_embed["markdown"] not in block:
        block += "\n\n" + chart_embed["markdown"]
    if not current.strip():
        return block, "append"
    separator = "" if current.endswith("\n") else "\n"
    return f"{current}{separator}\n{block}\n", "append"


_CHART_HINTS = ("grafik", "grafiğ", "pasta", "bar chart", "chart", "plot", "histogram", "scatter", "çiz")


async def _maybe_generate_chart_for_edit(instruction: str, source: str, span) -> dict | None:
    """If the @update instruction asks for a chart and stored tables exist,
    generate a PNG into the workspace images dir and return a metadata dict the
    editor uses to embed the figure inline. Returns None when no chart is needed.
    """
    instr = instruction.lower()
    if not any(h in instr for h in _CHART_HINTS):
        return None
    try:
        from RAG.services.table_store import list_all_tables

        tables = list_all_tables()
        # Prefer the edited document's tables.
        from RAG.services import paths as _paths
        stem = _paths.stem_of(source)
        own_tables = [t for t in tables if t["source"] == source]
        relevant = own_tables or tables
        if not relevant:
            return None

        from RAG.agents.code_interpreter import _generate_snippet, _run_snippet, _slug

        chart_dir = _paths.workspace_images_dir(source)
        chart_dir.mkdir(parents=True, exist_ok=True)
        chart_name = f"chart_{_slug(instruction)}.png"
        chart_path = chart_dir / chart_name

        # Drive the code interpreter with the current @update instruction so it
        # produces the requested chart type.
        pseudo_state = {"query": instruction, "trace_id": ""}
        snippet = await _generate_snippet(pseudo_state, relevant, str(chart_path))
        _result_text, chart_saved = _run_snippet(
            snippet, relevant, str(chart_path),
        )
        if not chart_saved or not chart_path.exists():
            return None
        url = f"/images/workspace/{stem}/{chart_name}"
        return {"name": chart_name, "path": str(chart_path), "markdown": f"![chart]({url})"}
    except Exception as exc:
        print(f"[Editor] chart generation failed: {exc}")
        return None


def _render_pdf_safe(source: str) -> bool:
    try:
        from RAG.services.pdf_renderer import render

        render(source)
        return True
    except Exception as exc:
        print(f"[Editor] PDF regeneration failed for {source}: {exc}")
        return False


def _clean_markdown(text: str) -> str:
    text = (text or "").strip()
    # Strip an accidental surrounding ```markdown ... ``` fence.
    fence = re.match(r"^```(?:markdown|md)?\s*\n(.*)\n```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return text


def _finish(message: str) -> dict:
    return {
        "next_agent": "finish",
        "final_response": message,
        "edit_summary": message,
        "messages": [AIMessage(content=message)],
    }


def apply_pending_edit(edit: dict) -> dict:
    """Persist a previously-stashed, user-approved @update edit.

    Called by the /update/apply HTTP endpoint after the user approves the diff
    preview. Performs write → reindex → render PDF → Git commit. Synchronous
    because it already runs in a thread via asyncio.to_thread at the call site.
    """
    source = edit["source"]
    after = edit["after"]
    instruction = edit.get("instruction", "")

    md_path = paths.workspace_md_path(source)
    paths.ensure_dirs()
    md_path.write_text(after, encoding="utf-8")

    reindex = reindex_workspace_source(source)
    pdf_ok = _render_pdf_safe(source)
    commit_sha = commit_change(source, instruction[:120])

    action_label = {
        "section_rewrite": "ilgili bölüm yerinde güncellendi",
        "append": "belgenin sonuna eklendi",
    }.get(edit.get("change_kind", ""), "güncellendi")
    summary = f"'{source}' — {action_label} ({reindex.get('chunks_added', 0)} parça yeniden indekslendi)."
    reply = summary
    if pdf_ok:
        reply += " Güncel PDF indirilebilir."
    else:
        reply += " (PDF yeniden üretilemedi.)"
    if commit_sha:
        reply += f" Sürüm kaydedildi ({commit_sha[:7]})."

    return {
        "source": source,
        "summary": summary,
        "reply": reply,
        "pdf_ok": pdf_ok,
        "git_sha": commit_sha,
        "chunks_added": reindex.get("chunks_added", 0),
    }


def _diff_preview_lines(before: str, after: str, context: int = 2) -> list[dict]:
    """A compact line-level diff for the frontend diff viewer.

    Returns a list of {type: same|added|removed, before, after}. Uses difflib to
    keep it dependency-free; the frontend renders red/green rows from this.
    """
    import difflib

    a = before.splitlines()
    b = after.splitlines()
    # Use ndiff but project to added/removed/equal ops with limited context.
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    rows: list[dict] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                rows.append({"type": "same", "before": a[k], "after": a[k]})
        elif tag == "replace":
            for k in range(i1, i2):
                rows.append({"type": "removed", "before": a[k], "after": None})
            for k in range(j1, j2):
                rows.append({"type": "added", "before": None, "after": b[k]})
        elif tag == "delete":
            for k in range(i1, i2):
                rows.append({"type": "removed", "before": a[k], "after": None})
        elif tag == "insert":
            for k in range(j1, j2):
                rows.append({"type": "added", "before": None, "after": b[k]})
    # Trim long unchanged runs to keep the preview lightweight for the UI.
    return _trim_context(rows, context)


def _trim_context(rows: list[dict], context: int) -> list[dict]:
    """Collapse runs of `same` longer than 2*context into a single ellipsis row."""
    if context <= 0:
        return rows
    out: list[dict] = []
    i = 0
    while i < len(rows):
        if rows[i]["type"] != "same":
            out.append(rows[i]); i += 1; continue
        j = i
        while j < len(rows) and rows[j]["type"] == "same":
            j += 1
        run = rows[i:j]
        if len(run) <= 2 * context:
            out.extend(run)
        else:
            out.extend(run[:context])
            out.append({"type": "ellipsis", "before": None, "after": None})
            out.extend(run[-context:])
        i = j
    return out
