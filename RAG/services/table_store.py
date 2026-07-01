"""Structured table sidecars for the workspace channel.

Markdown tables are great for display, but doing math over them ("average profit
margin over the last 3 years") needs structured data. During ingestion we parse every pipe
table out of the converted Markdown and persist it as both CSV and JSON under
``data/workspace/tables/<stem>/table_<n>.{csv,json}``.

The code-interpreter node loads these CSVs into a pandas DataFrame so the
Researcher can hand arithmetic questions off to a real Python REPL instead of
asking the LLM to do mental math.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from RAG.services import paths

TABLES_DIR_BASE = paths.WORKSPACE_MD_DIR.parent / "tables"
ORIGINALS_TABLES_DIR = TABLES_DIR_BASE  # tables are derived; kept only in workspace for editability


def _tables_dir(source: str) -> Path:
    d = TABLES_DIR_BASE / paths.stem_of(source)
    d.mkdir(parents=True, exist_ok=True)
    return d


# A Markdown table is a contiguous run of lines, each containing a leading pipe
# pattern with ≥2 cells, immediately followed by a separator row of ---/:---.
_TABLE_ROW_RE = re.compile(r"^\s*\|?.*\|.*\|\s*$")
_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]+\|?\s*$")


def extract_tables(markdown: str, source: str) -> list[dict]:
    """Parse pipe tables from `markdown` and write CSV+JSON sidecars for `source`.

    Returns a list of {name, headers, rows, csv_path, json_path} for in-memory
    use by the code interpreter. Idempotent: clears prior sidecars for the
    source first so a re-ingest doesn't double up.
    """
    tables: list[dict] = []
    lines = markdown.splitlines()
    i = 0
    out_dir = _tables_dir(source)
    # clean stale sidecars
    for f in out_dir.glob("table_*.csv"):
        f.unlink(missing_ok=True)
    for f in out_dir.glob("table_*.json"):
        f.unlink(missing_ok=True)

    idx = 0
    while i < len(lines):
        # Look for a header row followed immediately by a separator row.
        if i + 1 < len(lines) and _TABLE_ROW_RE.match(lines[i]) and _SEP_RE.match(lines[i + 1]):
            headers = _parse_row(lines[i])
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines) and _TABLE_ROW_RE.match(lines[j]) and not _SEP_RE.match(lines[j]):
                cells = _parse_row(lines[j])
                if cells:
                    rows.append(cells)
                j += 1
            if headers and rows:
                name = f"table_{idx + 1}"
                rec = {"name": name, "headers": headers, "rows": rows}
                _write_sidecar(out_dir, name, rec)
                rec["csv_path"] = str(out_dir / f"{name}.csv")
                rec["json_path"] = str(out_dir / f"{name}.json")
                tables.append(rec)
                idx += 1
                i = j
                continue
        i += 1
    return tables


def _parse_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _write_sidecar(out_dir: Path, name: str, rec: dict) -> None:
    csv_path = out_dir / f"{name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(rec["headers"])
        for row in rec["rows"]:
            w.writerow(row)
    json_path = out_dir / f"{name}.json"
    json_path.write_text(
        json.dumps({"headers": rec["headers"], "rows": rec["rows"]}, ensure_ascii=False),
        encoding="utf-8",
    )


def load_tables(source: str) -> list[dict]:
    """Load the persisted JSON sidecars for a source back into memory."""
    d = TABLES_DIR_BASE / paths.stem_of(source)
    out: list[dict] = []
    if not d.exists():
        return out
    for p in sorted(d.glob("table_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "name": p.stem,
                "headers": data.get("headers", []),
                "rows": data.get("rows", []),
                "csv_path": str(p.with_suffix(".csv")),
            })
        except Exception as exc:
            print(f"[TableStore] failed to load {p.name}: {exc}")
    return out


def list_all_tables() -> list[dict]:
    """All stored tables across sources — used by the code interpreter to make
    every document's CSV available as `tables[<name>]`."""
    out: list[dict] = []
    if not TABLES_DIR_BASE.exists():
        return out
    for stem_dir in sorted(TABLES_DIR_BASE.iterdir()):
        if not stem_dir.is_dir():
            continue
        for p in sorted(stem_dir.glob("table_*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                out.append({
                    "name": p.stem,
                    "source": f"{stem_dir.name}.md",
                    "headers": data.get("headers", []),
                    "rows": data.get("rows", []),
                    "csv_path": str(p.with_suffix(".csv")),
                })
            except Exception:
                continue
    return out