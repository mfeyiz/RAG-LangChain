"""Postgres-backed user store for token authentication.

Users live in the ``auth_users`` table (sharing the pgvector connection pool).
Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes using only the stdlib —
no extra dependency. A bootstrap admin is seeded from ``ADMIN_USERNAME`` /
``ADMIN_PASSWORD`` on startup so a fresh deployment has someone who can log in.
"""
import hashlib
import hmac
import os
import secrets

from RAG.services import vector_store

TABLE = "auth_users"
_PBKDF2_ROUNDS = 200_000
_schema_ready = False


def ensure_users_schema() -> bool:
    """Create the users table if missing (idempotent). Returns True on success."""
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
                    username      text PRIMARY KEY,
                    password_hash text NOT NULL,
                    salt          text NOT NULL,
                    role          text NOT NULL DEFAULT 'user',
                    created_at    timestamptz NOT NULL DEFAULT now()
                )
                """
            )
        _schema_ready = True
        return True
    except Exception as exc:
        print(f"[Users] ensure_users_schema failed: {exc}")
        return False


def create_user(username: str, password: str, role: str = "user") -> bool:
    """Create (or overwrite) a user. Returns True on success."""
    username = (username or "").strip()
    if not username or not password:
        return False
    if not ensure_users_schema():
        return False

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    pool = vector_store.get_pool()
    try:
        with pool.connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {TABLE} (username, password_hash, salt, role)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (username)
                DO UPDATE SET password_hash = EXCLUDED.password_hash,
                              salt = EXCLUDED.salt,
                              role = EXCLUDED.role
                """,
                (username, password_hash, salt, role),
            )
        return True
    except Exception as exc:
        print(f"[Users] create_user failed: {exc}")
        return False


def verify_credentials(username: str, password: str) -> str | None:
    """Return the user's role if the password matches, else None."""
    username = (username or "").strip()
    if not username or not password:
        return None
    if not ensure_users_schema():
        return None

    pool = vector_store.get_pool()
    try:
        with pool.connection() as conn:
            row = conn.execute(
                f"SELECT password_hash, salt, role FROM {TABLE} WHERE username = %s",
                (username,),
            ).fetchone()
    except Exception as exc:
        print(f"[Users] verify_credentials failed: {exc}")
        return None

    if not row:
        return None
    stored_hash, salt, role = row
    candidate = _hash_password(password, salt)
    if hmac.compare_digest(candidate, stored_hash):
        return role
    return None


def count_users() -> int:
    """Number of registered users. Returns -1 if the store is unavailable."""
    if not ensure_users_schema():
        return -1
    pool = vector_store.get_pool()
    try:
        with pool.connection() as conn:
            row = conn.execute(f"SELECT count(*) FROM {TABLE}").fetchone()
        return int(row[0]) if row else 0
    except Exception as exc:
        print(f"[Users] count_users failed: {exc}")
        return -1


def ensure_admin_seed() -> None:
    """Seed/refresh the bootstrap admin from ADMIN_USERNAME / ADMIN_PASSWORD env."""
    username = os.getenv("ADMIN_USERNAME", "").strip()
    password = os.getenv("ADMIN_PASSWORD", "")
    if not username or not password:
        return
    if not ensure_users_schema():
        print("[Users] Admin seed skipped — Postgres unavailable.")
        return
    if create_user(username, password, role="admin"):
        print(f"[Users] Bootstrap admin '{username}' ensured.")


def _hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ROUNDS
    )
    return digest.hex()
