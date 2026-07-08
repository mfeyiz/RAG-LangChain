"""Postgres-backed job store for the async job queue.

Jobs live in the ``jobs`` table (sharing the pgvector connection pool, same
pattern as ``users.py``). The table is intentionally generic — ``job_type`` +
jsonb ``payload``/``result`` — so any future job (bulk indexing, batch export,
...) reuses this same table/producer/worker loop; only a new handler needs to
be registered in ``RAG.worker``.

This table is what makes job state safe across multiple app/worker replicas
(unlike ``report_registry.py``'s flat JSON file, which assumes a single
instance and is out of scope here).
"""
import json

from RAG.services import vector_store

TABLE = "jobs"
_schema_ready = False


def ensure_jobs_schema() -> bool:
    """Create the jobs table if missing (idempotent). Returns True on success."""
    global _schema_ready
    if _schema_ready:
        return True

    pool = vector_store.get_pool()
    if pool is None:
        return False
    try:
        with pool.connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    job_id      uuid PRIMARY KEY,
                    job_type    text NOT NULL,
                    status      text NOT NULL DEFAULT 'queued',
                    payload     jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                    result      jsonb,
                    error       text,
                    trace_id    text,
                    created_at  timestamptz NOT NULL DEFAULT now(),
                    updated_at  timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS jobs_status_idx ON {TABLE} (status)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS jobs_job_type_idx ON {TABLE} (job_type)")
        _schema_ready = True
        return True
    except Exception as exc:
        print(f"[Jobs] ensure_jobs_schema failed: {exc}")
        return False


def create_job(job_id: str, job_type: str, payload: dict, trace_id: str = "") -> bool:
    """Insert a new job row with status 'queued'. Returns True on success."""
    if not ensure_jobs_schema():
        return False
    pool = vector_store.get_pool()
    try:
        with pool.connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {TABLE} (job_id, job_type, status, payload, trace_id)
                VALUES (%s, %s, 'queued', %s, %s)
                ON CONFLICT (job_id) DO NOTHING
                """,
                (job_id, job_type, json.dumps(payload, ensure_ascii=False), trace_id),
            )
        return True
    except Exception as exc:
        print(f"[Jobs] create_job failed ({job_id}): {exc}")
        return False


def mark_running(job_id: str) -> None:
    _update(job_id, "status = 'running'")


def mark_done(job_id: str, result: dict) -> None:
    _update(job_id, "status = 'done', result = %s", (json.dumps(result, ensure_ascii=False),))


def mark_error(job_id: str, error: str) -> None:
    _update(job_id, "status = 'error', error = %s", (error,))


def _update(job_id: str, set_clause: str, extra_params: tuple = ()) -> None:
    if not ensure_jobs_schema():
        return
    pool = vector_store.get_pool()
    if pool is None:
        return
    try:
        with pool.connection() as conn:
            conn.execute(
                f"UPDATE {TABLE} SET {set_clause}, updated_at = now() WHERE job_id = %s",
                (*extra_params, job_id),
            )
    except Exception as exc:
        print(f"[Jobs] update failed ({job_id}): {exc}")


def get_job(job_id: str) -> dict | None:
    if not ensure_jobs_schema():
        return None
    pool = vector_store.get_pool()
    if pool is None:
        return None
    try:
        with pool.connection() as conn:
            row = conn.execute(
                f"""
                SELECT job_id, job_type, status, payload, result, error, trace_id,
                       created_at, updated_at
                FROM {TABLE} WHERE job_id = %s
                """,
                (job_id,),
            ).fetchone()
    except Exception as exc:
        print(f"[Jobs] get_job failed ({job_id}): {exc}")
        return None

    if not row:
        return None
    (job_id_, job_type, status, payload, result, error, trace_id,
     created_at, updated_at) = row
    return {
        "job_id": str(job_id_),
        "job_type": job_type,
        "status": status,
        "payload": payload,
        "result": result,
        "error": error,
        "trace_id": trace_id,
        "created_at": created_at.isoformat() if created_at else None,
        "updated_at": updated_at.isoformat() if updated_at else None,
    }
