"""In-memory stash for pending @update edits awaiting human approval.

The editor produces a diff preview and stops the graph (needs_edit_approval).
The frontend shows the diff and asks the user to Approve or Reject. On Approve
it POSTs the returned `edit_token` to /update/apply, which pulls the stashed
{"source","before","after","instruction"} out of here and persists it
(write → reindex → render PDF → git commit).

Keys are short opaque tokens, not user-derived, and entries expire after a TTL
to avoid unbounded growth. This is intentionally process-local: it assumes a
single backend instance (sufficient for this app's deployment).
"""
from __future__ import annotations

import secrets
import time
from threading import Lock

_TTL_SECONDS = 30 * 60  # 30 minutes

_store: dict[str, dict] = {}
_lock = Lock()


def stash(edit: dict) -> str:
    """Store a pending edit and return an opaque token for /update/apply."""
    token = secrets.token_urlsafe(12)
    with _lock:
        _evict_expired_locked()
        _store[token] = {**edit, "_stashed_at": time.time()}
    return token


def get(token: str) -> dict | None:
    with _lock:
        entry = _store.get(token)
        if entry is None:
            return None
        if time.time() - entry.get("_stashed_at", 0) > _TTL_SECONDS:
            _store.pop(token, None)
            return None
        return {k: v for k, v in entry.items() if not k.startswith("_")}


def pop(token: str) -> dict | None:
    with _lock:
        entry = _store.pop(token, None)
        if entry is None:
            return None
        if time.time() - entry.get("_stashed_at", 0) > _TTL_SECONDS:
            return None
        return {k: v for k, v in entry.items() if not k.startswith("_")}


def reject(token: str) -> bool:
    with _lock:
        return _store.pop(token, None) is not None


def _evict_expired_locked() -> None:
    now = time.time()
    for tok in [t for t, e in _store.items() if now - e.get("_stashed_at", 0) > _TTL_SECONDS]:
        _store.pop(tok, None)