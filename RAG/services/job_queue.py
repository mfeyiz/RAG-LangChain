"""Kafka producer for the async job queue.

Mirrors ``_redis.py``'s lazy-singleton-with-retry-gate pattern, but async
(``asyncio.Lock`` instead of ``threading.Lock``) since ``aiokafka``'s
``start()``/``send_and_wait()`` are coroutines and every call site here is
already inside an async FastAPI request handler.

Graceful degradation, matching the rest of this codebase's convention
(Postgres pool -> None, Tavily key missing -> web search skipped): if
``KAFKA_BOOTSTRAP_SERVERS`` is unset, or the broker can't be reached, calls
here return ``None``/``False`` instead of raising — the caller is expected to
fall back to running the job inline (see ``RAG/app.py``'s
``/documents/generate`` handler).
"""
import asyncio
import json
import os
import time

_RETRY_INTERVAL = 30.0
TOPIC = "rag.jobs"

_producer = None
_producer_lock = asyncio.Lock()
_retry_after = 0.0


async def get_producer():
    """Return a started ``AIOKafkaProducer``, or None if Kafka is unavailable."""
    global _producer, _retry_after

    if _producer is not None:
        return _producer

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").strip()
    if not bootstrap:
        return None

    now = time.time()
    if now < _retry_after:
        return None

    async with _producer_lock:
        if _producer is not None:
            return _producer
        try:
            from aiokafka import AIOKafkaProducer

            producer = AIOKafkaProducer(bootstrap_servers=bootstrap)
            await producer.start()
            _producer = producer
            print(f"[JobQueue] Kafka producer connected ({bootstrap}).")
            return _producer
        except Exception as exc:
            print(f"[JobQueue] Kafka unavailable, retrying in {_RETRY_INTERVAL}s: {exc}")
            _retry_after = now + _RETRY_INTERVAL
            return None


async def enqueue(job_type: str, payload: dict, job_id: str, trace_id: str = "") -> bool:
    """Publish a job envelope to the ``rag.jobs`` topic. Returns True on success."""
    producer = await get_producer()
    if producer is None:
        return False

    envelope = {
        "job_id": job_id,
        "job_type": job_type,
        "trace_id": trace_id,
        "payload": payload,
    }
    try:
        await producer.send_and_wait(
            TOPIC,
            key=job_id.encode("utf-8"),
            value=json.dumps(envelope, ensure_ascii=False).encode("utf-8"),
        )
        return True
    except Exception as exc:
        print(f"[JobQueue] enqueue failed ({job_id}): {exc}")
        return False


async def close_producer() -> None:
    global _producer
    if _producer is not None:
        try:
            await _producer.stop()
        except Exception as exc:
            print(f"[JobQueue] producer close failed: {exc}")
        _producer = None
