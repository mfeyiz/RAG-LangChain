"""Async job worker — consumes `rag.jobs` from Kafka and executes jobs.

Run as its own process/pod, sharing the same Docker image as the API but with
a different entrypoint::

    uv run --no-sync python -m RAG.worker

Deliberately does NOT import ``RAG.app`` (that would drag in the whole FastAPI
app object and its route decorators) — job execution logic lives in
``RAG.services.report_jobs`` and is shared by both this worker and the API's
inline (no-Kafka) fallback path.

Handler registry (``HANDLERS``) is the pluggable point: a future job type
(e.g. bulk indexing) just registers a new async handler here — the consume
loop, offset-commit and error handling are generic.
"""
import asyncio
import json
import os
import signal

from RAG.services import job_events, job_queue, jobs, report_jobs
from RAG.agents.report_graph import create_report_graph

GROUP_ID = "rag-workers"


async def handle_report_generate(envelope: dict) -> None:
    job_id = envelope["job_id"]
    trace_id = envelope.get("trace_id", "")
    payload = envelope.get("payload", {})

    jobs.mark_running(job_id)
    await job_events.publish_event(job_id, "status", {"message": "Starting…"})

    report_graph = create_report_graph()
    result: dict = {}
    error: str | None = None

    async for item in report_jobs.run_report_job(
        report_graph,
        title=(payload.get("title") or "").strip(),
        topic=(payload.get("topic") or "").strip(),
        template=(payload.get("template") or "business-report").strip(),
        trace_id=trace_id,
        language=payload.get("language", "auto"),
        tone=payload.get("tone", ""),
        audience=payload.get("audience", ""),
        length=payload.get("length", "standard"),
        sections=payload.get("sections"),
    ):
        await job_events.publish_event(job_id, item["event"], item["data"])
        if item["event"] == "complete":
            result = item["data"]
        elif item["event"] == "error":
            error = item["data"].get("error", "Generation error")

    if error is not None:
        jobs.mark_error(job_id, error)
    else:
        jobs.mark_done(job_id, result)


# Pluggable dispatch point — register new job types here.
HANDLERS = {
    "report_generate": handle_report_generate,
}


async def _process_message(envelope: dict) -> None:
    job_type = envelope.get("job_type", "")
    handler = HANDLERS.get(job_type)
    job_id = envelope.get("job_id", "?")
    if handler is None:
        print(f"[Worker] No handler for job_type={job_type!r} (job {job_id}) — skipping.")
        jobs.mark_error(job_id, f"Unknown job_type: {job_type}")
        return
    try:
        await handler(envelope)
        print(f"[Worker] job {job_id} ({job_type}) completed.")
    except Exception as exc:
        # A permanently-broken job payload should not become a poison-pill
        # that blocks the partition forever — record the failure and move on
        # (the offset is still committed by the caller after this returns).
        print(f"[Worker] job {job_id} ({job_type}) failed: {exc}")
        import traceback
        traceback.print_exc()
        jobs.mark_error(job_id, str(exc))


async def run() -> None:
    from aiokafka import AIOKafkaConsumer

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").strip()
    if not bootstrap:
        print("[Worker] KAFKA_BOOTSTRAP_SERVERS not set — nothing to consume, exiting.")
        return

    if not jobs.ensure_jobs_schema():
        print("[Worker] WARNING: jobs schema not ready (Postgres unavailable) — "
              "job status writes will be skipped until it recovers.")

    consumer = AIOKafkaConsumer(
        job_queue.TOPIC,
        bootstrap_servers=bootstrap,
        group_id=GROUP_ID,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )
    await consumer.start()
    print(f"[Worker] Connected to Kafka ({bootstrap}), consuming '{job_queue.TOPIC}' as group '{GROUP_ID}'.")

    stop_event = asyncio.Event()

    def _handle_signal(*_args):
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass  # Windows / restricted environments — best-effort only.

    try:
        while not stop_event.is_set():
            try:
                batch = await consumer.getmany(timeout_ms=1000, max_records=10)
            except Exception as exc:
                print(f"[Worker] consumer poll error: {exc}")
                await asyncio.sleep(1)
                continue

            for tp, messages in batch.items():
                for msg in messages:
                    try:
                        envelope = json.loads(msg.value.decode("utf-8"))
                    except Exception as exc:
                        print(f"[Worker] malformed message, skipping: {exc}")
                        continue
                    await _process_message(envelope)
                if messages:
                    await consumer.commit({tp: messages[-1].offset + 1})
    finally:
        await consumer.stop()
        print("[Worker] Stopped.")


if __name__ == "__main__":
    asyncio.run(run())
