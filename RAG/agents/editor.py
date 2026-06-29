"""Editor agent — the write-back half of the bi-directional RAG.

Triggered by an `@update ...` chat message. It locates the relevant workspace
Markdown file and applies the user's instruction as a SMALL PATCH (append a new
section, or replace a short existing snippet) — the LLM only ever emits the
changed fragment, never the whole file, so large documents can't be truncated
or destroyed. It then re-indexes the workspace channel and regenerates the
workspace PDF. The read-only `originals` channel is never touched.
"""
import asyncio
import json
import re

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from RAG.agents.state import AgentState
from RAG.agents.supervisor import get_llm
from RAG.services import paths
from RAG.services.document_manager import reindex_workspace_source
from RAG.services.retrieval import retrieve_context
from RAG.services.tracing import invoke_with_langfuse, trace_event, traced_observation

_DEFAULT_NOTES_SOURCE = "notes.md"

EDITOR_SYSTEM_PROMPT = """You maintain a Markdown knowledge base. Apply the user's UPDATE INSTRUCTION as a SMALL PATCH. NEVER rewrite or output the whole document.

Return ONLY a JSON object with this shape:
{
  "action": "insert_after" | "replace" | "append",
  "find": "<verbatim snippet copied from the current file>",
  "markdown": "<the new or rewritten Markdown to insert>"
}

Choosing the action:
- "insert_after" (PREFERRED for adding related info): place the new content right after a specific existing passage. Copy that passage into "find" (e.g. the LAST item of a list when adding the next item) and the new content into "markdown". This keeps related information together so it stays retrievable as one unit.
- "replace": to change/correct existing text — copy the EXACT existing text into "find" and the rewritten version into "markdown".
- "append": only when the new information has no related location — it goes to the end of the file. Leave "find" empty.

Rules:
- When the instruction extends an existing list/section (e.g. "add a step 8"), use "insert_after" anchored on the most relevant existing item — do NOT append to the end of the document.
- "find" must be copied VERBATIM from the current file, as short as possible while still unique.
- "markdown" must contain ONLY the added/changed content — never the whole document.
- Use the same language as the existing document. Keep well-formed Markdown.
- Output ONLY the JSON object — no commentary, no code fences."""


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

        # Locate the target file AND the exact passage retrieval returns for this
        # topic. Anchoring the insertion on that passage keeps the new content in
        # the same chunk that answers the query — otherwise (on messy OCR docs
        # where a topic appears in several places) it can land in an unrelated
        # chunk and never surface alongside the original content.
        source, anchor = await asyncio.to_thread(_locate_target, search_text)
        md_path = paths.workspace_md_path(source)
        current = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

        context_note = (
            f"\n\nCONVERSATION CONTEXT (what the user was asking about):\n{recent_context}"
            if recent_context else ""
        )
        anchor_note = (
            "\n\nRELEVANT PASSAGE — this is the exact text users retrieve for this topic. "
            "To add related info (e.g. a new list item), use \"insert_after\" and copy a short "
            "VERBATIM snippet from THIS passage into \"find\", so the new content stays in the "
            f"same retrievable chunk:\n{anchor}"
            if anchor else ""
        )

        llm = get_llm()
        messages = [
            SystemMessage(content=EDITOR_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"UPDATE INSTRUCTION:\n{instruction}{context_note}{anchor_note}\n\n"
                    f"CURRENT FILE ({source}):\n{current or '(empty file — create the content)'}"
                )
            ),
        ]
        response = await invoke_with_langfuse(llm, messages)
        patch = _parse_patch(response.content)

        if patch is None or not (patch.get("markdown") or "").strip():
            span.update(output={"error": "empty_output"})
            return _finish("Güncelleme üretilemedi, lütfen talimatı netleştirin.")

        # Deterministically drop the new content INSIDE the passage retrieval
        # returns for this topic, so it lands in that same high-ranking chunk
        # (the LLM tends to anchor at a section's end, which spills into a
        # separate, low-ranking chunk). Fall back to the LLM's own patch when the
        # passage can't be located or this is a "replace".
        new_md = (patch.get("markdown") or "").strip()
        placed = None
        if anchor and (patch.get("action") or "").lower() != "replace":
            placed = _insert_near_anchor(current, new_md, anchor)
        if placed is not None:
            updated, change = placed, "insert_after"
        else:
            updated, change = _apply_patch(current, patch)

        if updated.strip() == current.strip():
            span.update(output={"error": "no_change"})
            return _finish(
                "Belgede bir değişiklik yapılmadı. Lütfen talimatı daha açık yazın."
            )

        # Persist the edited workspace markdown, then re-index + regenerate PDF.
        paths.ensure_dirs()
        md_path.write_text(updated, encoding="utf-8")

        reindex = await asyncio.to_thread(reindex_workspace_source, source)

        pdf_ok = await asyncio.to_thread(_render_pdf_safe, source)

        await trace_event(
            state["trace_id"],
            "editor.applied",
            {"source": source, "chunks": reindex.get("chunks_added"), "pdf": pdf_ok},
        )
        span.update(output={"source": source, "chunks": reindex.get("chunks_added"), "pdf": pdf_ok})

        action_label = {
            "insert_after": "ilgili yerin hemen ardına eklendi",
            "replace": "ilgili bölüm güncellendi",
            "append": "belgenin sonuna eklendi",
        }.get(change, "güncellendi")
        summary = f"'{source}' — {action_label} ({reindex.get('chunks_added', 0)} parça yeniden indekslendi)."
        reply = summary
        if pdf_ok:
            reply += " Güncel PDF indirilebilir."
        else:
            reply += " (PDF yeniden üretilemedi.)"

        return {
            "next_agent": "finish",
            "final_response": reply,
            "edit_instruction": instruction,
            "edit_target_file": source,
            "edit_summary": summary,
            "regenerated_pdf": source if pdf_ok else "",
            "messages": [AIMessage(content=reply)],
        }


def _locate_target(search_text: str) -> tuple[str, str]:
    """Find the workspace file and the passage retrieval returns for this topic.

    Returns (source, anchor) where `anchor` is the top matching chunk's text —
    the same passage a query would surface — so the editor can insert new
    content right beside it. Falls back to a shared notes file (and empty anchor)
    when the workspace is empty or nothing relevant is found.
    """
    try:
        _, metadata = retrieve_context(search_text, top_k=5, channel="workspace")
    except Exception as exc:
        print(f"[Editor] Retrieval failed while locating target: {exc}")
        metadata = []

    for item in metadata:
        src = item.get("source")
        if src and paths.workspace_md_path(src).exists():
            return src, (item.get("content") or "")
    return _DEFAULT_NOTES_SOURCE, ""


def _parse_patch(raw: str) -> dict | None:
    """Parse the model's JSON patch, tolerating code fences and surrounding prose."""
    text = _clean_markdown(raw)
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fall back to the first {...} block if the model added stray text.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _insert_near_anchor(current: str, new_md: str, anchor: str) -> str | None:
    """Insert `new_md` inside the retrieved passage so it shares that chunk.

    Anchors on the tail of the passage retrieval returned (a verbatim prefix of
    the top-ranked chunk). Inserting there keeps the new content within the same
    high-ranking chunk instead of spilling into a separate, low-ranking one.
    Returns the updated text, or None if the passage can't be located verbatim.
    """
    core = anchor.strip()
    if core.endswith("..."):
        core = core[:-3].rstrip()
    if not core:
        return None

    for tail_len in (200, 120, 60):
        tail = core[-tail_len:]
        pos = current.find(tail)
        if pos != -1:
            idx = pos + len(tail)
            return f"{current[:idx]}\n\n{new_md}\n{current[idx:]}"
    return None


def _apply_patch(current: str, patch: dict) -> tuple[str, str]:
    """Apply a small patch to the markdown. Returns (updated_text, change_kind).

    Never rewrites the whole document. Supported actions:
      - insert_after: place new content right after a verbatim anchor (keeps
        related info — e.g. a new list item — adjacent so it stays retrievable).
      - replace: swap a verbatim snippet for a rewritten version.
      - append: add to the end of the file.
    Any action whose anchor can't be located falls back to "append" so the
    user's content is never silently dropped.
    """
    new_md = (patch.get("markdown") or "").strip()
    action = (patch.get("action") or "append").lower()
    find = patch.get("find") or ""

    if action == "insert_after" and find and find in current:
        idx = current.find(find) + len(find)
        return f"{current[:idx]}\n\n{new_md}\n{current[idx:]}", "insert_after"

    if action == "replace" and find and find in current:
        return current.replace(find, new_md, 1), "replace"

    # append (default, and fallback when the anchor can't be located)
    if not current.strip():
        return new_md, "append"
    separator = "" if current.endswith("\n") else "\n\n"
    return f"{current}{separator}\n{new_md}\n", "append"


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
