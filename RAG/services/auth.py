import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass

from fastapi import Request


DEFAULT_TOKEN_TTL = int(os.getenv("JWT_TTL_SECONDS", str(12 * 3600)))


@dataclass
class AuthResult:
    allowed: bool
    user_id: str = "anonymous"
    session_id: str = ""
    role: str = "user"
    error: str = ""
    authenticated: bool = False

    @property
    def is_admin(self) -> bool:
        return self.authenticated and self.role == "admin"


def auth_configured() -> bool:
    """True when tokens can be issued/verified (JWT_SECRET is set)."""
    return bool(os.getenv("JWT_SECRET", "").strip())


def authenticate_request(request: Request) -> AuthResult:
    """Resolve the caller.

    Reads are open to everyone, so a missing/invalid token yields an *anonymous*
    (but allowed) result rather than a 401 — only privileged actions such as
    ``@update`` check ``AuthResult.authenticated``. A valid bearer token promotes
    the caller to authenticated with the role embedded in the JWT.
    """
    secret = os.getenv("JWT_SECRET", "").strip()
    allow_insecure = os.getenv("ALLOW_INSECURE_DEV", "").strip() == "1"
    if not secret:
        if allow_insecure:
            print("[Auth] WARNING: JWT_SECRET is not set and ALLOW_INSECURE_DEV=1 — all requests are treated as an authenticated admin.")
            return AuthResult(allowed=True, role="admin", authenticated=True)  # dev mode: full access
        # No auth configured: read-only anonymous access. @update stays blocked
        # because no token can be issued or verified.
        return AuthResult(allowed=True, role="anonymous", authenticated=False)

    token = _bearer_token(request)
    if not token:
        return AuthResult(allowed=True, role="anonymous", authenticated=False)

    payload = verify_jwt(token, secret)
    if payload is None:
        return AuthResult(allowed=True, role="anonymous", authenticated=False, error="Invalid bearer token.")

    user_id = str(payload.get("sub") or payload.get("user_id") or "authenticated")
    session_id = str(payload.get("sid") or payload.get("session_id") or "")
    role = str(payload.get("role") or "user")
    return AuthResult(
        allowed=True,
        user_id=user_id,
        session_id=session_id,
        role=role,
        authenticated=True,
    )


def create_access_token(sub: str, role: str = "user", ttl_seconds: int | None = None) -> str:
    """Mint a signed JWT for ``sub`` with the given role. Raises if JWT_SECRET unset."""
    secret = os.getenv("JWT_SECRET", "").strip()
    if not secret:
        raise RuntimeError("JWT_SECRET is not set — cannot issue tokens.")
    now = int(time.time())
    ttl = DEFAULT_TOKEN_TTL if ttl_seconds is None else ttl_seconds
    payload = {
        "sub": sub,
        "role": role,
        "iat": now,
        "nbf": now,
        "exp": now + ttl,
    }
    return encode_jwt(payload, secret)


def encode_jwt(payload: dict, secret: str) -> str:
    """HMAC-SHA256 JWT encoder, symmetric with verify_jwt."""
    header = {"alg": "HS256", "typ": "JWT"}
    segments = [
        _base64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
        _base64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
    ]
    signing_input = ".".join(segments).encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    segments.append(_base64url_encode(signature))
    return ".".join(segments)


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


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")
