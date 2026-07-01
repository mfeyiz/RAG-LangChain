"""Editor agent — the write-back half of the bi-directional RAG.

Triggered by an `@update ...` chat message. It locates the relevant workspace
Markdown file and applies the user's instruction as a SMALL PATCH (append a new
section, or replace a short existing snippet) — the LLM only ever emits the
changed fragment, never the whole file, so large documents can't be truncated
or destroyed. It then re-indexes the workspace channel and regenerates the
workspace PDF. The read-only `originals` channel is never touched.

Quoting syntax: text in "double quotes" within the @update instruction is
treated as verbatim content that must appear byte-for-byte in the output.
Unquoted text is the LLM instruction (what/where to change).
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

# A located section larger than this is treated as "too coarse to rewrite whole"
# (typically a heading-less document whose only section is the entire file). The
# editor then narrows the edit to the matched passage instead.
MAX_SECTION_CHARS = 3500

SECTION_EDITOR_SYSTEM_PROMPT = """You maintain a Markdown knowledge base. You are given ONE SECTION of a document and an UPDATE INSTRUCTION. Apply the instruction to that section and return the WHOLE, REWRITTEN SECTION.

Hard rules:
- Return ONLY this one section's Markdown (including its heading line). NEVER return other sections or the whole document, and never add commentary or code fences.
- If the instruction CORRECTS or CHANGES existing information, edit that information IN PLACE — modify the existing sentence/line/list item. Do NOT also keep the old version. No duplicated or contradictory statements.
- If the instruction ADDS genuinely new information, place it at the most logical spot within the section (e.g. the next item of the relevant list), not blindly at the end.
- Preserve everything in the section that the instruction does not touch, byte-for-byte where possible (keep existing headings, image links like ![](figure.png), tables, and formatting).
- Any [VERBATIM_N] placeholder in the instruction refers to an exact string listed in the VERBATIM STRINGS block. Copy that string into your output character-for-character. Never rephrase, shorten, translate, or modify verbatim strings.
- Keep well-formed Markdown.
- Output ONLY the rewritten section Markdown."""


def strip_update_tag(query: str) -> str:
    """Remove a leading/embedded @update tag, returning the bare instruction."""
    return re.sub(r"@update\b", "", query, count=1, flags=re.IGNORECASE).strip()


def _parse_verbatim_quotes(instruction: str) -> tuple[str, list[str]]:
    """Extract "quoted" strings from the @update instruction.

    Quoted strings are verbatim content that must appear unchanged in the
    output Markdown. Unquoted text is the LLM instruction (where/how to edit).

    Returns (processed_instruction, verbatim_list):
    - processed_instruction: instruction with each quoted string replaced by
      a [VERBATIM_N] marker so the LLM knows its position in the instruction.
    - verbatim_list: the original quoted strings in order.
    """
    verbatim: list[str] = []

    def replacer(m: re.Match) -> str:
        verbatim.append(m.group(1))
        return f"[VERBATIM_{len(verbatim)}]"

    processed = re.sub(r'"([^"]+)"', replacer, instruction)
    return processed, verbatim


def _verbatim_block(verbatim_list: list[str]) -> str:
    """Build the VERBATIM STRINGS appendix for the LLM HumanMessage."""
    if not verbatim_list:
        return ""
    lines = "\n".join(f"  [VERBATIM_{i + 1}]: {v}" for i, v in enumerate(verbatim_list))
    return f"\n\nVERBATIM STRINGS — copy these character-for-character into your output:\n{lines}"


async def editor_node(state: AgentState) -> dict:
    raw_instruction = strip_update_tag(state["query"])
    instruction, verbatim_list = _parse_verbatim_quotes(raw_instruction)

    with traced_observation("editor", input_payload={"instruction": instruction}) as span:
        if not instruction.strip() and not verbatim_list:
            msg = "Please write what you want to add/update after the @update tag."
            span.update(output={"error": "empty_instruction"})
            return _finish(msg)

        # Enrich the search with recent conversation context so the edit lands
        # in the right document even when the instruction alone ("add step 8")
        # doesn't name the topic.
        history = state.get("conversation_history") or []
        recent_context = " ".join((turn.get("query") or "") for turn in history[-2:]).strip()
        # Use the raw (un-marker-replaced) instruction for retrieval so quoted
        # strings contribute to the semantic search.
        search_text = f"{recent_context} {raw_instruction}".strip()

        source, section, heading_path, child_anchor = await asyncio.to_thread(_locate_target, search_text)
        md_path = paths.workspace_md_path(source)
        current = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

        context_note = (
            f"\n\nCONVERSATION CONTEXT (what the user was asking about):\n{recent_context}"
            if recent_context else ""
        )

        chart_embed = await _maybe_generate_chart_for_edit(raw_instruction, source, span)

        # ── 4-level section resolution ──────────────────────────────────────
        # Level 1: exact match of indexed parent_content in file
        section_bounds: tuple[int, int] | None = None
        section_text = ""

        if section and section in current:
            idx = current.index(section)
            section_bounds = (idx, idx + len(section))
            section_text = section

        # Level 1.5: whitespace-normalized match of the whole parent_content.
        # MarkdownHeaderTextSplitter normalizes whitespace, so the stored
        # parent_content usually differs from the on-disk section only by spacing
        # (hard-break "  \n", collapsed blank lines). Without this, heading-less
        # sections (empty heading_path) fell through every level and the edit was
        # blindly appended at the end of the file instead of rewritten in place.
        if not section_bounds and section:
            found = _find_section_by_normalized_match(current, section)
            if found:
                section_bounds = found
                section_text = current[found[0]:found[1]]

        # Level 2: locate section via stored heading_path
        if not section_bounds and heading_path:
            found = _find_section_span(current, heading_path)
            if not found:
                # Try each component individually (leaf → root) in case the
                # full ancestor chain doesn't match but the leaf heading does.
                for component in reversed(heading_path.split(">")):
                    component = component.strip()
                    if component:
                        found = _find_section_span(current, component)
                        if found:
                            break
            if found:
                section_bounds = found
                section_text = current[found[0]:found[1]]

        # Level 3: scan every heading line inside parent_content and try each
        # one — do NOT break on the first heading regardless of whether it matched.
        if not section_bounds and section:
            for line in section.splitlines():
                stripped = line.strip()
                if re.match(r"^#{1,6}\s+", stripped):
                    found = _find_section_span(current, _norm_heading(stripped))
                    if found:
                        section_bounds = found
                        section_text = current[found[0]:found[1]]
                        break   # only stop when we actually found a match

        # Level 4: anchor-text search — find a distinctive line from
        # parent_content in the file, then derive its containing section.
        if not section_bounds and section:
            found = _find_section_by_anchor(current, section)
            if found:
                section_bounds = found
                section_text = current[found[0]:found[1]]

        # Level 5: word-overlap — find the section whose vocabulary overlaps
        # most with parent_content. Catches translated / reformatted documents
        # where none of the text-based levels matched.
        if not section_bounds and section:
            found = _find_section_by_overlap(current, section)
            if found:
                section_bounds = found
                section_text = current[found[0]:found[1]]

        # ── Narrow oversized sections to the matched passage ────────────────
        # Heading-less documents have no heading tree, so the "parent section"
        # is the ENTIRE file. Rewriting a whole 25k-char document through the LLM
        # risks truncation and is what the small-patch design exists to avoid.
        # When the located section is too large, or nothing was located at all,
        # fall back to the specific matched passage (the child chunk) and rewrite
        # just the paragraph around it — so the edit lands in the right place
        # instead of being appended at the end of the file.
        narrowed_to_passage = False
        too_large = section_bounds is not None and (section_bounds[1] - section_bounds[0]) > MAX_SECTION_CHARS
        if (section_bounds is None or too_large) and child_anchor:
            passage = _find_passage_around_anchor(current, child_anchor)
            if passage:
                section_bounds = passage
                section_text = current[passage[0]:passage[1]]
                narrowed_to_passage = True

        # ── Build LLM prompt ────────────────────────────────────────────────
        verbatim_suffix = _verbatim_block(verbatim_list)
        chart_suffix = (
            f"\n\nA chart figure was generated for this update. Embed it in the "
            f"rewritten section with this exact Markdown image link at the right "
            f"place:\n{chart_embed['markdown']}"
            if chart_embed else ""
        )

        if section_bounds is None or not section_text.strip():
            updated, change = await _create_new_section(
                instruction, verbatim_suffix, context_note, source, current, span, chart_embed
            )
        else:
            llm = get_llm()
            messages = [
                SystemMessage(content=SECTION_EDITOR_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"UPDATE INSTRUCTION:\n{instruction}{context_note}"
                        f"{chart_suffix}"
                        f"{verbatim_suffix}"
                        f"\n\nSECTION TO EDIT (from {source}) — return this whole section, rewritten:\n{section_text}"
                    )
                ),
            ]
            response = await invoke_with_langfuse(llm, messages)
            new_section = _clean_markdown(response.content).strip()
            if not new_section:
                span.update(output={"error": "empty_output"})
                return _finish("Could not generate update, please clarify the instruction.")
            if chart_embed and chart_embed["markdown"] not in new_section:
                new_section += "\n\n" + chart_embed["markdown"]
            start, end = section_bounds
            updated = current[:start] + new_section + current[end:]
            change = "passage_rewrite" if narrowed_to_passage else "section_rewrite"

        if updated is None or updated.strip() == current.strip():
            span.update(output={"error": "no_change"})
            return _finish(
                "No change was made to the document. Please write a clearer instruction."
            )

        token = stash_pending_edit({
            "source": source,
            "instruction": raw_instruction,
            "before": current,
            "after": updated,
            "change_kind": change,
            "heading_path": heading_path,
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
                "instruction": raw_instruction,
                "before": current,
                "after": updated,
                "change_kind": change,
                "diff": preview_lines,
                "heading_path": heading_path,
            },
            "edit_target_file": source,
            "edit_instruction": raw_instruction,
            "final_response": "",
            "messages": [AIMessage(content="Change preview ready — waiting for your approval.")],
        }


def _locate_target(search_text: str) -> tuple[str, str, str, str]:
    """Find the workspace file and the full parent SECTION for this topic.

    Returns ``(source, parent_content, heading_path, child_anchor)`` where
    ``parent_content`` is the verbatim Markdown of the heading section the top
    match belongs to — the unit the editor rewrites in place — ``heading_path``
    is that section's heading trail (e.g. ``"Security > Threats"``) used to
    relocate the section even if ``parent_content`` drifted from the on-disk
    text, and ``child_anchor`` is the specific matched passage. The anchor lets
    the editor narrow the edit to the relevant paragraph when the parent section
    is the whole document (heading-less files) instead of rewriting everything.
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
                (item.get("content") or ""),
            )
    return _DEFAULT_NOTES_SOURCE, "", "", ""


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
    """
    if not heading_path.strip():
        return None

    components = [_norm_heading(c) for c in heading_path.split(">")]
    target = components[-1] if components else ""
    if not target:
        return None

    headings: list[tuple[int, int, str]] = []
    for m in re.finditer(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", current):
        headings.append((m.start(), len(m.group(1)), _norm_heading(m.group(2))))
    if not headings:
        return None

    def ancestor_chain(idx: int) -> list[str]:
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


def _locate_line(current: str, line: str) -> int | None:
    """Return the offset of ``line`` in ``current`` (exact, else whitespace-
    normalized with the offset mapped back to the original text)."""
    if line in current:
        return current.index(line)
    norm_line = re.sub(r"\s+", " ", line).strip()
    if len(norm_line) < 12:
        return None
    norm_current = re.sub(r"\s+", " ", current)
    idx = norm_current.find(norm_line)
    if idx == -1:
        return None
    return _norm_to_orig_offset(current, idx)


def _find_passage_around_anchor(current: str, anchor: str) -> tuple[int, int] | None:
    """Locate the matched child passage in ``current`` and return the bounds of
    the paragraph (blank-line delimited) containing it.

    Used to narrow an edit to the relevant paragraph when the parent section is
    the whole file (heading-less documents). ``anchor`` is the indexed child
    ``content`` — for headed docs it is prefixed with the heading path, so we try
    each of its lines (skipping headings / the heading-path prefix / short lines)
    until one is found in the document.
    """
    if not anchor or not current:
        return None
    for line in anchor.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or len(stripped) < 20:
            continue
        pos = _locate_line(current, stripped)
        if pos is None:
            continue
        para_start = current.rfind("\n\n", 0, pos)
        para_start = 0 if para_start == -1 else para_start + 2
        para_end = current.find("\n\n", pos)
        para_end = len(current) if para_end == -1 else para_end
        if para_end > para_start:
            return para_start, para_end
    return None


def _find_section_by_normalized_match(current: str, section: str) -> tuple[int, int] | None:
    """Locate ``section`` (an indexed parent_content) in ``current`` when the two
    differ only by whitespace, returning the ``(start, end)`` offsets in the
    ORIGINAL text so the section can be rewritten in place.

    The indexed parent_content comes from MarkdownHeaderTextSplitter, which
    collapses/normalizes whitespace, so a byte-exact ``in`` check misses even
    though the section is clearly present. We normalize both sides, find the
    span, then map the normalized offsets back onto the original string.
    """
    norm_section = re.sub(r"\s+", " ", section).strip()
    if len(norm_section) < 12:
        return None
    norm_current = re.sub(r"\s+", " ", current)
    idx = norm_current.find(norm_section)
    if idx == -1:
        return None
    start = _norm_to_orig_offset(current, idx)
    end = _norm_to_orig_offset(current, idx + len(norm_section))
    if end <= start:
        end = len(current)
    return start, end


def _find_section_by_anchor(current: str, section: str) -> tuple[int, int] | None:
    """Level-4 fallback: find a distinctive non-heading line from ``section``
    in ``current``, then return the heading-section boundaries that contain it.

    This handles the case where ``parent_content`` has different whitespace
    from the on-disk file and no heading path is available.
    """
    if not section or not current:
        return None

    # Collect headings with their positions and levels for boundary calculation.
    headings: list[tuple[int, int]] = []  # (start_offset, level)
    for m in re.finditer(r"(?m)^(#{1,6})[ \t]+", current):
        headings.append((m.start(), len(m.group(1))))

    # Find the first distinctive (non-heading) line from parent_content.
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or len(stripped) < 20:
            continue
        # Try exact and whitespace-normalized match.
        anchor = None
        if stripped in current:
            anchor = current.index(stripped)
        else:
            norm_stripped = re.sub(r"\s+", " ", stripped)
            norm_current = re.sub(r"\s+", " ", current)
            if norm_stripped in norm_current:
                # Map normalized index back to original by walking the string.
                norm_idx = norm_current.index(norm_stripped)
                orig_idx = _norm_to_orig_offset(current, norm_idx)
                anchor = orig_idx

        if anchor is None:
            continue

        # Derive the section that contains this anchor.
        section_start = 0
        section_level = 1
        for h_start, h_level in headings:
            if h_start > anchor:
                break
            section_start = h_start
            section_level = h_level

        section_end = len(current)
        for h_start, h_level in headings:
            if h_start > section_start and h_level <= section_level:
                section_end = h_start
                break

        return section_start, section_end

    return None


def _find_section_by_overlap(current: str, section: str) -> tuple[int, int] | None:
    """Level-5 fallback: find the heading-section in ``current`` whose vocabulary
    overlaps most with ``section`` (the indexed parent_content).

    Handles translated or heavily reformatted documents where neither exact
    match, heading-path lookup, nor anchor-text search can locate the section.
    Requires at least 30 % word overlap to avoid false positives.
    """
    if not section or not current:
        return None

    headings = list(re.finditer(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", current))
    if not headings:
        return None

    # Content words (≥4 chars) from the indexed chunk.
    section_words = set(re.findall(r"\b\w{4,}\b", section.lower()))
    if not section_words:
        return None

    best_score = 0.0
    best_bounds: tuple[int, int] | None = None

    for i, h in enumerate(headings):
        start = h.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(current)
        sec_text = current[start:end]
        sec_words = set(re.findall(r"\b\w{4,}\b", sec_text.lower()))
        overlap = len(section_words & sec_words)
        score = overlap / max(1, len(section_words))
        if score > best_score:
            best_score = score
            best_bounds = (start, end)

    if best_score >= 0.3 and best_bounds:
        return best_bounds
    return None


def _norm_to_orig_offset(text: str, norm_idx: int) -> int:
    """Map a character offset in whitespace-normalized text back to ``text``."""
    count = 0
    in_space = False
    for orig_i, ch in enumerate(text):
        if ch in " \t\n\r":
            if not in_space:
                if count == norm_idx:
                    return orig_i
                count += 1
                in_space = True
        else:
            if count == norm_idx:
                return orig_i
            count += 1
            in_space = False
    return len(text)


_NEW_SECTION_SYSTEM_PROMPT = """You maintain a Markdown knowledge base. Turn the UPDATE INSTRUCTION into a small, self-contained Markdown block (a short heading plus content) suitable to add to a notes file. Output ONLY the Markdown — no commentary, no code fences.

Hard rule: Any [VERBATIM_N] placeholder in the instruction refers to an exact string listed in the VERBATIM STRINGS block. Copy it character-for-character into your output."""


async def _create_new_section(
    instruction: str,
    verbatim_suffix: str,
    context_note: str,
    source: str,
    current: str,
    span,
    chart_embed: dict | None = None,
) -> tuple[str | None, str]:
    """Create a new Markdown section when no existing section was located.

    Generates a self-contained Markdown block with a proper heading and inserts
    it at the end of the document with a blank-line separator. Unlike the old
    _legacy_patch, this never concatenates raw text without a heading.
    """
    llm = get_llm()
    prompt = f"UPDATE INSTRUCTION:\n{instruction}{context_note}{verbatim_suffix}"
    if chart_embed:
        prompt += "\n\nInclude this generated chart image at the end, exactly:\n" + chart_embed["markdown"]
    response = await invoke_with_langfuse(
        llm,
        [
            SystemMessage(content=_NEW_SECTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ],
    )
    block = _clean_markdown(response.content).strip()
    if not block:
        return None, "new_section"
    if chart_embed and chart_embed["markdown"] not in block:
        block += "\n\n" + chart_embed["markdown"]
    if not current.strip():
        return block, "new_section"
    separator = "" if current.endswith("\n") else "\n"
    return f"{current}{separator}\n{block}\n", "new_section"


_CHART_HINTS = (
    "chart", "graph", "plot", "histogram", "scatter", "bar chart", "pie chart",
    "line chart", "area chart", "draw",
)


async def _maybe_generate_chart_for_edit(instruction: str, source: str, span) -> dict | None:
    """If the @update instruction asks for a chart and stored tables exist,
    generate a PNG and return metadata for inline embedding. Returns None when
    no chart is needed.
    """
    instr = instruction.lower()
    if not any(h in instr for h in _CHART_HINTS):
        return None
    try:
        from RAG.services.table_store import list_all_tables

        tables = list_all_tables()
        from RAG.services import paths as _paths
        own_tables = [t for t in tables if t["source"] == source]
        relevant = own_tables or tables
        if not relevant:
            return None

        from RAG.agents.code_interpreter import _generate_snippet, _run_snippet, _slug

        chart_dir = _paths.workspace_images_dir(source)
        chart_dir.mkdir(parents=True, exist_ok=True)
        chart_name = f"chart_{_slug(instruction)}.png"
        chart_path = chart_dir / chart_name

        pseudo_state = {"query": instruction, "trace_id": ""}
        snippet = await _generate_snippet(pseudo_state, relevant, str(chart_path))
        _result_text, chart_saved = _run_snippet(snippet, relevant, str(chart_path))
        if not chart_saved or not chart_path.exists():
            return None
        stem = _paths.stem_of(source)
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
    before = edit.get("before", "")

    md_path = paths.workspace_md_path(source)
    paths.ensure_dirs()

    current_content = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    if current_content.strip() != before.strip():
        raise ValueError(
            "The document was modified by another edit since the preview was generated. "
            "Please re-request the change."
        )

    md_path.write_text(after, encoding="utf-8")

    reindex = reindex_workspace_source(source)
    pdf_ok = _render_pdf_safe(source)

    try:
        from RAG.services.docx_exporter import render as render_docx
        render_docx(source)
    except Exception as exc:
        print(f"[Editor] DOCX regeneration failed for {source} during apply: {exc}")

    commit_sha = commit_change(source, instruction[:120])

    action_label = {
        "section_rewrite": "relevant section updated in place",
        "passage_rewrite": "relevant passage updated in place",
        "new_section": "new section added to document",
        "append": "appended to document",
    }.get(edit.get("change_kind", ""), "updated")
    summary = (
        f"'{source}' — {action_label} "
        f"({reindex.get('chunks_added', 0)} chunks re-indexed)."
    )
    reply = summary
    if pdf_ok:
        reply += " Updated PDF is available for download."
    else:
        reply += " (PDF could not be regenerated.)"
    if commit_sha:
        reply += f" Version saved ({commit_sha[:7]})."

    return {
        "source": source,
        "summary": summary,
        "reply": reply,
        "pdf_ok": pdf_ok,
        "git_sha": commit_sha,
        "chunks_added": reindex.get("chunks_added", 0),
    }


def _diff_preview_lines(before: str, after: str, context: int = 2) -> list[dict]:
    """A compact line-level diff for the frontend diff viewer."""
    import difflib

    a = before.splitlines()
    b = after.splitlines()
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


async def direct_edit_markdown(current_markdown: str, query: str) -> str:
    """Directly edit markdown content using instructions and a direct LLM call."""
    llm = get_llm()
    system_prompt = """You are a precise Markdown document editor.
You are given a Markdown document and a user's instruction to edit it.
Modify the document according to the instruction.

Hard rules:
1. Return ONLY the final, complete, updated Markdown document.
2. Do NOT add any introductory or concluding text, explanations, or commentary.
3. Do NOT wrap the output in markdown code blocks or code fences (e.g. do NOT use ```markdown). Output raw markdown content directly.
4. Keep all existing structure, headings, tables, image tags, and text that are not affected by the instruction.
"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"--- DOCUMENT BEGIN ---\n{current_markdown}\n--- DOCUMENT END ---\n\nINSTRUCTION: {query}")
    ]
    response = await llm.ainvoke(messages)
    content = response.content.strip()
    if content.startswith("```markdown"):
        content = content[11:].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    return content
