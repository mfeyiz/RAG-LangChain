"""Shared Redis client factory.

Each module that needs a lazy Redis connection calls `make_redis_loader(tag)`
once at import time to get its own `_load_redis_client` function with isolated
state (client cache + retry-after gate). This deduplicates the three near-
identical copies that previously lived in feedback/guardrails/retrieval.
"""
import os
import time

_RETRY_INTERVAL = 30.0


def make_redis_loader(tag: str):
    """Return a `(load_client, get_client)` pair bound to private module state.

    `load_client()` mirrors the original `_load_redis_client` contract: returns
    a connected `Redis` instance or `None` (and throttles reconnect attempts).
    """
    state = {"client": None, "retry_after": 0.0}

    def load_client():
        if state["client"] is not None:
            return state["client"]

        now = time.time()
        if now < state["retry_after"]:
            return None

        redis_url = os.getenv("REDIS_URL", "").strip()
        if not redis_url:
            return None

        try:
            from redis import Redis

            client = Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            state["client"] = client
            return client
        except Exception as exc:
            print(f"[{tag}] Redis unavailable, retrying in {_RETRY_INTERVAL}s: {exc}")
            state["retry_after"] = now + _RETRY_INTERVAL
            return None

    return load_client
