"""Git-backed version control for the workspace channel.

The editable workspace markdown (and extracted figures + tables) lives under
``data/workspace``. We keep a local Git repository there so every ``@update``
produces a commit the user can later inspect, diff, or roll back ("Son yaptığım
değişikliği geri al", "Dün saat 15:00'teki sürüme dön").

GitPython is imported lazily; when git is unavailable the module degrades to a
no-op so the rest of the app keeps working.
"""
from __future__ import annotations

from datetime import datetime

from RAG.services import paths

# The whole workspace tree is one repo. We commit markdown, regenerated PDFs,
# extracted figures, and the tables sidecars so a restore brings back the exact
# state of a document at that point in time.
REPO_DIR = paths.WORKSPACE_MD_DIR.parent  # data/workspace

try:
    import git as _git  # type: ignore
    _GIT_AVAILABLE = True
except Exception:  # pragma: no cover - GitPython optional
    _git = None
    _GIT_AVAILABLE = False


def _repo():
    if not _GIT_AVAILABLE:
        return None
    try:
        return _git.Repo(str(REPO_DIR))
    except Exception:
        return None


def ensure_repo() -> None:
    """Initialise the workspace Git repo on startup if it doesn't exist."""
    if not _GIT_AVAILABLE:
        return
    paths.ensure_dirs()
    repo = _repo()
    if repo is None:
        try:
            repo = _git.Repo.init(str(REPO_DIR))
        except Exception as exc:  # pragma: no cover
            print(f"[VersionControl] git init failed: {exc}")
            return

    # Make sure the workspace tree is tracked. We only commit the workspace
    # subdir; the workspace PDFs/images/tables live alongside markdown there.
    _gitignore = REPO_DIR / ".gitignore"
    if not _gitignore.exists():
        _gitignore.write_text(
            "# nothing ignored by default — the workspace is small and we want\n"
            "# full history of regenerated PDFs too.\n",
            encoding="utf-8",
        )

    # Seed an initial commit if the repo is empty so commits/branches work.
    if not repo.head.is_valid():
        try:
            repo.index.add([".gitignore"])
            repo.index.commit("workspace: initial commit", author=_default_author())
        except Exception as exc:  # pragma: no cover
            print(f"[VersionControl] initial commit failed: {exc}")


def _default_author():
    return _git.Actor("RAG Workspace", "rag@local")


def commit_change(source: str, summary: str) -> str | None:
    """Commit all current workspace changes attributed to an @update of `source`.

    Returns the commit SHA, or None if git is unavailable / nothing changed.
    """
    if not _GIT_AVAILABLE:
        return None
    repo = _repo()
    if repo is None:
        return None

    stem = paths.stem_of(source)
    # Track the doc's markdown, its regenerated PDF, its figures, and tables (only if they exist).
    patterns = []
    for rel in (f"markdown/{source}", f"pdf/{stem}.pdf", f"images/{stem}", f"tables/{stem}"):
        if (paths.WORKSPACE_MD_DIR.parent / rel).exists():
            patterns.append(rel)

    if not patterns:
        return None

    try:
        repo.git.add("--", *patterns)
    except Exception as exc:
        print(f"[VersionControl] git add failed: {exc}")
        return None

    if not repo.is_dirty(index=True, untracked_files=True):
        # Nothing staged for this commit — still, a previous @update may have
        # left staged changes; only commit if the index actually differs.
        try:
            if not repo.head.is_valid() or not repo.index.diff("HEAD"):
                return None
        except Exception:
            return None

    try:
        commit = repo.index.commit(
            f"@update {source}: {summary}"[:200],
            author=_default_author(),
        )
        return commit.hexsha
    except Exception as exc:  # pragma: no cover
        print(f"[VersionControl] commit failed: {exc}")
        return None


def history(source: str | None = None, limit: int = 50) -> list[dict]:
    """Return commit history. When `source` is given, only commits that touched
    that document's path are returned."""
    if not _GIT_AVAILABLE:
        return []
    repo = _repo()
    if repo is None or not repo.head.is_valid():
        return []

    path_filter = None
    if source:
        stem = paths.stem_of(source)
        path_filter = [
            f"markdown/{source}",
            f"pdf/{stem}.pdf",
            f"images/{stem}/",
            f"tables/{stem}/",
        ]

    out: list[dict] = []
    try:
        commits = repo.iter_commits(paths=path_filter, max_count=limit) if path_filter else repo.iter_commits(max_count=limit)
        for c in commits:
            out.append({
                "sha": c.hexsha,
                "short_sha": c.hexsha[:7],
                "message": (c.message or "").strip().splitlines()[0] if c.message else "",
                "author": str(c.author),
                "date": datetime.fromtimestamp(c.committed_date).isoformat(),
                "files": list(c.stats.files.keys()) if hasattr(c, "stats") else [],
            })
    except Exception as exc:  # pragma: no cover
        print(f"[VersionControl] history failed: {exc}")
    return out


def restore(source: str, ref: str) -> dict:
    """Restore one document's workspace artifacts to a given commit ref.

    Checks out just the paths belonging to `source` at `ref` into the working
    tree, then re-indexes the workspace channel for that source. The originals
    channel is never touched.
    Returns {source, sha, restored_files}.
    """
    if not _GIT_AVAILABLE:
        raise RuntimeError("Git backend not available (GitPython missing).")
    repo = _repo()
    if repo is None:
        raise RuntimeError("Workspace git repo not initialised.")

    stem = paths.stem_of(source)
    targets = [
        f"markdown/{source}",
        f"pdf/{stem}.pdf",
        f"images/{stem}",
        f"tables/{stem}",
    ]
    restored: list[str] = []
    repo.git.checkout("--", ref, "--", *targets)
    for t in targets:
        p = REPO_DIR / t
        if p.exists():
            restored.append(t)
    return {"source": source, "sha": ref, "restored_files": restored}


def diff_at(source: str, ref: str | None = None) -> dict:
    """Return a textual diff of a document at a given commit vs the working tree
    (or vs the previous commit when ref is the HEAD-ish tip)."""
    if not _GIT_AVAILABLE:
        return {"text": "", "available": False}
    repo = _repo()
    if repo is None:
        return {"text": "", "available": False}
    try:
        if ref is None:
            text = repo.git.diff("HEAD", "--", f"markdown/{source}")
        else:
            text = repo.git.show(f"{ref}", "--", f"markdown/{source}")
    except Exception as exc:
        return {"text": "", "available": False, "error": str(exc)}
    return {"text": text, "available": True}