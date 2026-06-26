import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass

from fastapi import Request


@dataclass
class AuthResult:
    allowed: bool
    user_id: str = "anonymous"
    session_id: str = ""
    role: str = "user"
    error: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def authenticate_request(request: Request) -> AuthResult:
    secret = os.getenv("JWT_SECRET", "").strip()
    allow_insecure = os.getenv("ALLOW_INSECURE_DEV", "").strip() == "1"
    if not secret:
        if allow_insecure:
            print("[Auth] WARNING: JWT_SECRET is not set and ALLOW_INSECURE_DEV=1 — all requests are accepted without authentication.")
            return AuthResult(allowed=True, role="admin")  # dev mode: full access
        print("[Auth] ERROR: JWT_SECRET is not set — authentication required. Set ALLOW_INSECURE_DEV=1 to disable for development.")
        return AuthResult(allowed=False, error="Authentication not configured. Set JWT_SECRET or ALLOW_INSECURE_DEV=1 for development.")

    token = _bearer_token(request)
    if not token:
        return AuthResult(allowed=False, error="Missing bearer token.")

    payload = verify_jwt(token, secret)
    if payload is None:
        return AuthResult(allowed=False, error="Invalid bearer token.")

    user_id = str(payload.get("sub") or payload.get("user_id") or "authenticated")
    session_id = str(payload.get("sid") or payload.get("session_id") or "")
    role = str(payload.get("role") or "user")
    return AuthResult(allowed=True, user_id=user_id, session_id=session_id, role=role)


def verify_jwt(token: str, secret: str) -> dict | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None

    signing_input = ".".join(parts[:2]).encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    actual = _base64url_decode(parts[2])

    if not hmac.compare_digest(expected, actual):
        return None

    try:
        payload = json.loads(_base64url_decode(parts[1]).decode("utf-8"))
    except Exception:
        return None

    now = int(time.time())
    exp = payload.get("exp")
    if exp is None:
        return None
    if now > int(exp):
        return None

    nbf = payload.get("nbf")
    if nbf is not None and now < int(nbf):
        return None

    return payload


def _bearer_token(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)
