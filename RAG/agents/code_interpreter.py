"""Code interpreter node — real arithmetic & charts over stored table data.

When a user asks something that needs math or a chart over financial/tabular
data ("son 3 yılın kâr marjı ortalamasını bul", "bu verileri pasta grafiğine dök
ve dökümana ekle"), the Researcher gathers the relevant tables and the
supervisor routes the turn through this node BEFORE the writer.

Execution model: a sandboxed subprocess running a stock python interpreter with
only the table CSVs exposed as pandas DataFrames (`tables`) plus matplotlib for
charts. The LLM generates the python snippet; this node runs it. Hard timeout
and a filesystem jail (only the workspace images dir is writable) keep things
contained. No network, no arbitrary file writes outside the chart directory.

Outputs:
- `calc_result`  : a textual result the Writer folds into its answer.
- `chart_image`  : filename of a generated PNG written into the workspace
                  images dir, which the Writer embeds inline as
                  `![](/images/workspace/<stem>/<file>)`.
- `code_interpreter` trace event with the snippet and status.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from RAG.agents.state import AgentState
from RAG.agents.supervisor import get_llm
from RAG.services import paths
from RAG.services.table_store import list_all_tables
from RAG.services.tracing import invoke_with_langfuse, trace_event, traced_observation

# Security toggles. Disable to turn off the code interpreter entirely.
_CI_ENABLED = os.getenv("RAG_CODE_INTERPRETER", "1") == "1"
_EXEC_TIMEOUT = float(os.getenv("CODE_INTERPRETER_TIMEOUT", "15"))


CODEGEN_SYSTEM_PROMPT = """You generate a single, self-contained Python snippet that answers a calculation or chart request using the provided tables.

Environment available inside the snippet:
- `tables`: dict mapping a table name (e.g. "table_1") to a pandas DataFrame of that table.
- `matplotlib`/`pylab` as `plt` for charts. Charts MUST be saved with the exact path given in `CHART_PATH`.
- `result`: set this variable to a short string (the answer to print back to the user). For calculations, prefer a concise human-readable summary (e.g. "Ortalama kâr marjı: %18.4").

Hard rules:
- Use ONLY the tables provided. Coerce numeric-looking strings with `pd.to_numeric(..., errors="coerce")`.
- Do NOT import anything except what is already available (pandas, matplotlib).
- Do NOT read or write arbitrary files. To save a chart, write to `CHART_PATH` (a string variable pre-set for you).
- Output ONLY a code block (```python ... ```), no commentary.
- If a chart is requested set `CHART_SAVED = True` after `plt.savefig(CHART_PATH)`; else set `CHART_SAVED = False`.
- Always set `result` to a string.
"""


async def code_interpreter_node(state: AgentState) -> dict:
    with traced_observation("code_interpreter", input_payload={"query": state["query"]}) as span:
        if not _CI_ENABLED:
            span.update(output={"disabled": True})
            return {"calc_result": "", "messages": [AIMessage(content="Code interpreter disabled.")]}

        tables = list_all_tables()
        if not tables:
            span.update(output={"no_tables": True})
            return {"calc_result": "", "messages": [AIMessage(content="No tables available to compute over.")]}

        # Pick the most relevant tables by naive keyword overlap with the query
        # before handing the catalogue to the codegen LLM — keeps the prompt tight.
        relevant = _select_relevant_tables(state["query"], tables)

        # Where a generated chart PNG should be written, if any. Use a synthetic
        # stem derived from the source with the most tables so the chart is
        # servable via /images/workspace/<stem>/<file>.
        chart_stem = paths.sanitize_stem(relevant[0]["source"]) if relevant else "chart"
        chart_dir = paths.workspace_images_dir(f"{chart_stem}.md")
        chart_dir.mkdir(parents=True, exist_ok=True)
        chart_name = f"chart_{_slug(state['query'])}.png"
        chart_path = chart_dir / chart_name

        snippet = await _generate_snippet(state, relevant, str(chart_path))
        result_text, chart_saved = _run_snippet(snippet, relevant, str(chart_path))

        await trace_event(
            state["trace_id"],
            "code_interpreter.exec",
            {"snippet": snippet, "result": result_text[:500], "chart": chart_saved},
        )
        span.update(output={"result": result_text[:500], "chart": chart_saved})

        # Surface the computed answer to the writer as its "research context" so
        # the writer folds the numeric/chart result into the user-facing answer.
        research_block = f"### Computed result (code interpreter)\n{result_text}"
        context_images: list[dict] = []
        if chart_saved and chart_path.exists():
            chart_source = f"{chart_stem}.md"
            context_images = [{"source": chart_source, "name": chart_name, "channel": "workspace"}]

        updates: dict = {
            "calc_result": result_text,
            "research_results": research_block,
            "rewritten_query": state["query"],
            "source_type": "table",
            "context_images": context_images,
            "messages": [AIMessage(content=f"Code interpreter: {result_text[:200]}")],
        }
        return updates


async def _generate_snippet(state: AgentState, tables: list[dict], chart_path: str) -> str:
    catalogue = "\n".join(
        f"- {t['name']} (from {t['source']}); columns: {t['headers']}; first row: {t['rows'][0] if t['rows'] else []}"
        for t in tables
    )
    user_text = (
        f"User request:\n{state['query']}\n\n"
        f"Available tables:\n{catalogue}\n\n"
        f"CHART_PATH = {chart_path!r}\n\n"
        "Write the python snippet now."
    )
    llm = get_llm()
    response = await invoke_with_langfuse(
        llm,
        [SystemMessage(content=CODEGEN_SYSTEM_PROMPT), HumanMessage(content=user_text)],
    )
    return _extract_code(response.content or "")


def _extract_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return (fence.group(1) if fence else text).strip()


def _run_snippet(snippet: str, tables: list[dict], chart_path: str) -> tuple[str, bool]:
    """Execute the LLM-generated snippet in a subprocess sandbox.

    `tables` becomes a dict of DataFrames, `CHART_PATH` is pre-bound, and `result`
    is captured from the namespace. Pandas/matplotlib are imported inside the
    sandbox so we don't hard-depend on them at import time of this module.
    """
    # Build the `tables` dict literal outside any f-string to avoid backslashes
    # (which are a SyntaxError inside f-string expressions pre-3.12).
    _table_entries = ", ".join(
        f"{t['name']!r}: pd.DataFrame({t['rows']!r}, columns={t['headers']!r})"
        for t in tables
    )
    loader = (
        "import pandas as pd\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        f"CHART_PATH = {chart_path!r}\n"
        f"tables = {{{_table_entries}}}\n"
        "CHART_SAVED = False\n"
        "result = ''\n"
    )
    capture = (
        "\nimport json as _json\n"
        "print('___RESULT___' + _json.dumps({'result': str(result), 'chart_saved': bool(CHART_SAVED)}))\n"
    )
    program = loader + "\n" + snippet + "\n" + capture

    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=_EXEC_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return ("Hesaplama zaman aşımına uğradı.", False)
    except Exception as exc:
        return (f"Kod çalıştırılamadı: {exc}", False)

    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        return (f"Kod hatası: {' '.join(err[-3:])[:300]}", False)

    # Parse the marker line for the result.
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("___RESULT___"):
            import json
            try:
                payload = json.loads(line[len("___RESULT___"):])
                return (str(payload.get("result", "") or ""), bool(payload.get("chart_saved", False)))
            except Exception:
                return (line[len("___RESULT___"):], False)
    # Fallback: any non-empty stdout.
    return ((proc.stdout.strip() or "")[:500], Path(chart_path).exists())


def _select_relevant_tables(query: str, tables: list[dict], top_k: int = 4) -> list[dict]:
    q = set(_tokens(query))
    scored: list[tuple[int, dict]] = []
    for t in tables:
        text = " ".join(t["headers"]) + " " + " ".join(" ".join(r) for r in t["rows"][:5])
        score = len(q & _tokens(text))
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:top_k]] or tables[:top_k]


def _tokens(text: str) -> set[str]:
    return {w for w in re.sub(r"[^\w ]+", " ", text.lower()).split() if len(w) > 1}


def _slug(text: str) -> str:
    return re.sub(r"[^\w-]+", "_", text.lower())[:24].strip("_") or "req"