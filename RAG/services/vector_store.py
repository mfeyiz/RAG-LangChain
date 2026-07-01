"""Postgres + pgvector dense vector store.

Replaces the previous Qdrant backend. A single ``rag_chunks`` table holds every
channel's dense vectors; the ``channel`` column isolates the read-only
``originals`` index from the editable ``workspace`` index (the same split Qdrant
used with two collections). BM25 stays file-based in retrieval.py — only the
dense channel lives here.

Connection comes from ``DATABASE_URL``. If it is unset or unreachable the public
functions degrade gracefully (return ``None``/empty) exactly like the old
Qdrant-unavailable path, so dense search is simply skipped and BM25 still serves
results.
"""
import os
import threading

from RAG.services.embeddings import EMBEDDING_DIM

TABLE = "rag_chunks"

# Process-wide connection pool singleton. pgvector's local file lock is gone, but
# we still want one shared pool rather than a connection per call.
_pool = None
_pool_lock = threading.Lock()
_schema_ready = False


def get_pool():
    """Return the process-wide connection pool, opening it on first use.

    Returns ``None`` when ``DATABASE_URL`` is unset or the database can't be
    reached, so callers can skip dense search the way they skipped Qdrant before.
    """
    global _pool
    if _pool is not None:
        return _pool

    dsn = os.getenv("DATABASE_URL", "").strip()
    if not dsn:
        return None

    with _pool_lock:
        if _pool is not None:
            return _pool
        try:
            from psycopg_pool import ConnectionPool

            pool = ConnectionPool(
                conninfo=dsn,
                min_size=1,
                max_size=int(os.getenv("PG_POOL_MAX_SIZE", "10")),
                kwargs={"autocommit": True},
                open=True,
                configure=_configure_connection,
            )
            pool.wait(timeout=float(os.getenv("PG_CONNECT_TIMEOUT", "10")))
            _pool = pool
        except Exception as exc:
            print(f"[VectorStore] Postgres unavailable: {exc}")
            return None
    return _pool


def reset_pool() -> None:
    """Drop the cached pool so the next call reconnects (used after errors)."""
    global _pool, _schema_ready
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.close()
            except Exception:
                pass
        _pool = None
        _schema_ready = False


def _configure_connection(conn) -> None:
    """Register the pgvector type adapters on each pooled connection.

    ``register_vector`` looks up the ``vector`` type in the DB, so the extension
    must exist first — create it here (idempotent) so a brand-new database works
    on the very first connection, before ``ensure_schema`` has run.
    """
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        from pgvector.psycopg import register_vector

        register_vector(conn)
    except Exception as exc:
        print(f"[VectorStore] pgvector adapter registration failed: {exc}")


# ── Schema ────────────────────────────────────────────────────────────────────

def ensure_schema(dim: int | None = None) -> bool:
    """Create the extension, table and indexes if missing (idempotent).

    If the existing ``embedding`` column dimension differs from ``dim`` (e.g. the
    embedding model changed), the table is dropped and recreated — the pgvector
    equivalent of Qdrant's stale-collection rebuild. Returns True on success.
    """
    global _schema_ready
    dim = dim or EMBEDDING_DIM
    pool = get_pool()
    if pool is None:
        return False

    try:
        with pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            existing_dim = _column_dim(conn)
            if existing_dim is not None and existing_dim != dim:
                print(
                    f"[VectorStore] embedding dim mismatch (stored={existing_dim}, "
                    f"expected={dim}) — dropping {TABLE}."
                )
                conn.execute(f"DROP TABLE IF EXISTS {TABLE}")

            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    pk        bigserial PRIMARY KEY,
                    channel   text NOT NULL,
                    doc_id    text NOT NULL,
                    content   text NOT NULL,
                    metadata  jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({dim}) NOT NULL
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_channel_idx ON {TABLE} (channel)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_source_idx "
                f"ON {TABLE} ((metadata->>'source'))"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_embedding_idx "
                f"ON {TABLE} USING hnsw (embedding vector_cosine_ops)"
            )
        _schema_ready = True
        return True
    except Exception as exc:
        print(f"[VectorStore] ensure_schema failed: {exc}")
        return False


def _column_dim(conn) -> int | None:
    """Return the declared dimension of the embedding column, or None if absent."""
    row = conn.execute(
        """
        SELECT a.atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = %s AND a.attname = 'embedding'
        """,
        (TABLE,),
    ).fetchone()
    if not row or row[0] is None or row[0] < 0:
        return None
    # vector's typmod stores the dimension directly (no -4 VARLENA offset).
    return int(row[0])


def _ensure_ready() -> bool:
    if _schema_ready:
        return True
    return ensure_schema()


# ── Writes ────────────────────────────────────────────────────────────────────

def _to_vector(embedding):
    """Wrap a python list in pgvector's ``Vector`` so psycopg can adapt it.

    pgvector's psycopg dumper is registered only for ``Vector`` and
    ``numpy.ndarray`` — a bare ``list`` has no dumper and would fail to bind.
    """
    from pgvector import Vector

    return Vector(embedding)


def replace_channel(channel: str, rows: list[dict]) -> None:
    """Replace every row for ``channel`` with ``rows`` (full rebuild).

    ``rows`` items: ``{doc_id, content, metadata, embedding}``. An empty list
    just clears the channel.
    """
    pool = get_pool()
    if pool is None:
        print(f"[VectorStore] replace_channel skipped — Postgres unavailable ({channel}).")
        return
    if not _ensure_ready():
        return

    try:
        with pool.connection() as conn:
            with conn.transaction():
                conn.execute(f"DELETE FROM {TABLE} WHERE channel = %s", (channel,))
                _insert_rows(conn, channel, rows)
        print(f"[VectorStore] Wrote {len(rows)} rows to channel '{channel}'.")
    except Exception as exc:
        print(f"[VectorStore] replace_channel failed ({channel}): {exc}")
        reset_pool()


def upsert(channel: str, rows: list[dict]) -> None:
    """Append ``rows`` to ``channel`` (incremental add)."""
    pool = get_pool()
    if pool is None:
        print(f"[VectorStore] upsert skipped — Postgres unavailable ({channel}).")
        return
    if not _ensure_ready() or not rows:
        return

    try:
        with pool.connection() as conn:
            _insert_rows(conn, channel, rows)
    except Exception as exc:
        print(f"[VectorStore] upsert failed ({channel}): {exc}")
        reset_pool()


def _insert_rows(conn, channel: str, rows: list[dict]) -> None:
    import json

    sql = (
        f"INSERT INTO {TABLE} (channel, doc_id, content, metadata, embedding) "
        f"VALUES (%s, %s, %s, %s, %s)"
    )
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(
                sql,
                (
                    channel,
                    str(row["doc_id"]),
                    row["content"],
                    json.dumps(row.get("metadata", {}), ensure_ascii=False),
                    _to_vector(row["embedding"]),
                ),
            )


def delete_by_source(channel: str, source: str) -> int:
    """Delete every chunk for a given source document in a channel; return count."""
    pool = get_pool()
    if pool is None or not _ensure_ready():
        return 0
    try:
        with pool.connection() as conn:
            cur = conn.execute(
                f"DELETE FROM {TABLE} WHERE channel = %s AND metadata->>'source' = %s",
                (channel, source),
            )
            return cur.rowcount or 0
    except Exception as exc:
        print(f"[VectorStore] delete_by_source failed ({channel}): {exc}")
        reset_pool()
        return 0


# ── Reads ─────────────────────────────────────────────────────────────────────

def query(channel: str, vector: list[float], k: int) -> list[dict]:
    """Return the top-k nearest chunks for ``channel`` by cosine similarity.

    Each item: ``{doc_id, content, metadata, score}`` where ``score`` is the
    cosine similarity (``1 - distance``), matching Qdrant's COSINE score range.
    """
    pool = get_pool()
    if pool is None or not _ensure_ready():
        return []
    try:
        with pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT doc_id, content, metadata, 1 - (embedding <=> %s) AS score
                FROM {TABLE}
                WHERE channel = %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (_to_vector(vector), channel, _to_vector(vector), k),
            ).fetchall()
    except Exception as exc:
        print(f"[VectorStore] query failed ({channel}): {exc}")
        reset_pool()
        return []

    results = []
    for doc_id, content, metadata, score in rows:
        results.append(
            {
                "doc_id": str(doc_id or ""),
                "content": content or "",
                "metadata": metadata or {},
                "score": float(score) if score is not None else 0.0,
            }
        )
    return results


def channel_count(channel: str) -> int:
    pool = get_pool()
    if pool is None or not _ensure_ready():
        return 0
    try:
        with pool.connection() as conn:
            row = conn.execute(
                f"SELECT count(*) FROM {TABLE} WHERE channel = %s", (channel,)
            ).fetchone()
            return int(row[0]) if row else 0
    except Exception as exc:
        print(f"[VectorStore] channel_count failed ({channel}): {exc}")
        return 0


def channel_ready(channel: str) -> bool:
    return channel_count(channel) > 0
