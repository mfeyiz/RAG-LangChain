"""Lightweight registry distinguishing generated reports from uploaded docs.

Both live in the workspace channel as ``<stem>.md`` and are otherwise
indistinguishable in ``list_documents("workspace")``. This registry records the
sources that were *created in the studio* (blank/template/AI-generated) plus a
little metadata (title, template, timestamps), so the UI can group "Reports"
apart from "Uploaded" documents and show friendly titles.

Stored as a single JSON file (``data/reports.json``). Process-local writes are
serialized with a lock; this matches the app's single-instance deployment.
"""
from __future__ import annotations

import json
import time
from threading import Lock

from RAG.services import paths

_PATH = paths.DATA_DIR / "reports.json"
_lock = Lock()


def _load() -> dict:
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _write(data: dict) -> None:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    _PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def record(source: str, title: str, template: str = "blank", *, generated: bool = False) -> None:
    """Register (or refresh) a studio-created report."""
    with _lock:
        data = _load()
        now = time.time()
        existing = data.get(source, {})
        data[source] = {
            "title": title or existing.get("title") or paths.stem_of(source),
            "template": template or existing.get("template", "blank"),
            "generated": bool(generated or existing.get("generated", False)),
            "created_at": existing.get("created_at", now),
            "updated_at": now,
        }
        _write(data)


def remove(source: str) -> None:
    with _lock:
        data = _load()
        if data.pop(source, None) is not None:
            _write(data)


def rename(old_source: str, new_source: str, title: str | None = None) -> None:
    """Move a registry entry to a new source key (used by rename/duplicate)."""
    with _lock:
        data = _load()
        entry = data.pop(old_source, None)
        if entry is None:
            entry = {"template": "blank", "generated": False, "created_at": time.time()}
        if title is not None:
            entry["title"] = title
        entry["updated_at"] = time.time()
        data[new_source] = entry
        _write(data)


def get(source: str) -> dict | None:
    return _load().get(source)


def all_reports() -> dict:
    return _load()
