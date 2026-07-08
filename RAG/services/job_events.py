"""Redis pub/sub bridge for live job progress.

Kafka decouples the API pod (which accepts the HTTP request) from the worker
pod (which actually runs the job), so progress events have to cross that
boundary through something both sides can reach — Redis already plays this
role for sessions/checkpointing in this codebase.

IMPORTANT — subscribe-before-enqueue: Redis pub/sub has no backlog/replay, so
a subscriber that connects after a message was published simply misses it.
Because Python async generators are lazy (their body doesn't run until first
iterated), a single `async def subscribe(): ... yield ...` generator would NOT
actually call `pubsub.subscribe()` until the caller starts consuming it —
too late to guarantee ordering against an enqueue that races ahead. This is
why subscribing is split into two steps:

    pubsub = await open_subscription(job_id)   # blocks until SUBSCRIBE acked
    ok = await job_queue.enqueue(...)          # now safe to enqueue
    async for item in iter_messages(pubsub):    # consume once ready
        ...

A late/reconnecting client (or one where Redis is simply unavailable) should
instead read persisted status from ``RAG.services.jobs.get_job()``.
"""
import asyncio
import json
import os
import time

_REDIS_RETRY_INTERVAL = 30.0
_DEFAULT_IDLE_TIMEOUT = float(os.getenv("JOB_EVENTS_IDLE_TIMEOUT", "600"))  # 10 min
_TERMINAL_EVENTS = {"complete", "error", "done"}

_redis = None
_redis_retry_after = 0.0


def _channel(job_id: str) -> str:
    return f"job:{job_id}:events"


async def _get_redis():
    global _redis, _redis_retry_after
    if _redis is not None:
        return _redis

    now = time.monotonic()
    if now < _redis_retry_after:
        return None

    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return None

    try:
        from redis.asyncio import Redis

        client = Redis.from_url(redis_url, decode_responses=True)
        await client.ping()
        _redis = client
        return client
    except Exception as exc:
        _redis_retry_after = now + _REDIS_RETRY_INTERVAL
        print(f"[JobEvents] Redis unavailable, retrying in {_REDIS_RETRY_INTERVAL}s: {exc}")
        return None


async def publish_event(job_id: str, event: str, data: dict) -> None:
    """Publish one progress event for ``job_id``. Called from the worker."""
    redis = await _get_redis()
    if redis is None:
        return
    try:
        await redis.publish(_channel(job_id), json.dumps({"event": event, "data": data}, ensure_ascii=False))
    except Exception as exc:
        print(f"[JobEvents] publish failed ({job_id}): {exc}")


async def open_subscription(job_id: str):
    """Subscribe to ``job_id``'s channel and return the pubsub object, or None
    if Redis is unavailable. Await this and get a result BEFORE enqueuing the
    job — see module docstring."""
    redis = await _get_redis()
    if redis is None:
        return None
    pubsub = redis.pubsub()
    try:
        await pubsub.subscribe(_channel(job_id))
        return pubsub
    except Exception as exc:
        print(f"[JobEvents] subscribe failed ({job_id}): {exc}")
        try:
            await pubsub.aclose()
        except Exception:
            pass
        return None


async def close_subscription(pubsub) -> None:
    if pubsub is None:
        return
    try:
        await pubsub.unsubscribe()
        await pubsub.aclose()
    except Exception:
        pass


async def iter_messages(pubsub, idle_timeout: float = _DEFAULT_IDLE_TIMEOUT):
    """Yield ``{"event", "data"}`` dicts from an already-subscribed ``pubsub``
    until a terminal event arrives or ``idle_timeout`` seconds pass with no
    message. Closes the subscription itself when done."""
    if pubsub is None:
        return
    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True, timeout=idle_timeout),
                    timeout=idle_timeout + 1,
                )
            except asyncio.TimeoutError:
                return
            if message is None:
                continue
            try:
                payload = json.loads(message["data"])
            except (TypeError, ValueError):
                continue
            yield payload
            if payload.get("event") in _TERMINAL_EVENTS:
                return
    finally:
        await close_subscription(pubsub)
