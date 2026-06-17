import json
import os
import time
import uuid

_REDIS_RETRY_INTERVAL = 30.0


class SessionStore:
    def __init__(self):
        self._memory: dict[str, dict] = {}
        self._redis = None
        self._redis_retry_after: float = 0.0

    def get_or_create(self, session_id: str = "") -> str:
        return session_id or str(uuid.uuid4())

    async def load(self, session_id: str) -> dict:
        redis = await self._get_redis()
        if redis is None:
            return self._memory.get(session_id, {})

        raw = await redis.get(self._key(session_id))
        if not raw:
            return {}
        return json.loads(raw)

    async def save(self, session_id: str, state: dict, ttl_seconds: int = 3600):
        payload = {**state, "updated_at": time.time()}

        redis = await self._get_redis()
        if redis is None:
            self._memory[session_id] = payload
            return

        await redis.setex(self._key(session_id), ttl_seconds, json.dumps(payload, ensure_ascii=False))

    async def close(self):
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def _get_redis(self):
        if self._redis is not None:
            return self._redis

        now = time.monotonic()
        if now < self._redis_retry_after:
            return None

        redis_url = os.getenv("REDIS_URL", "").strip()
        if not redis_url:
            return None

        try:
            from redis.asyncio import Redis

            client = Redis.from_url(redis_url, decode_responses=True)
            await client.ping()
            self._redis = client
            return client
        except Exception as exc:
            self._redis_retry_after = now + _REDIS_RETRY_INTERVAL
            print(f"[SessionStore] Redis unavailable, retrying in {_REDIS_RETRY_INTERVAL}s: {exc}")
            return None

    @staticmethod
    def _key(session_id: str) -> str:
        return f"rag:session:{session_id}"


session_store = SessionStore()
