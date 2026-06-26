import asyncio
import json
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler


TRACE_DIR = Path(__file__).resolve().parent.parent / "traces"
_CURRENT_LANGFUSE_HANDLER: ContextVar[CallbackHandler | None] = ContextVar(
    "current_langfuse_handler",
    default=None,
)


def new_trace_id() -> str:
    return uuid.uuid4().hex


@lru_cache(maxsize=1)
def get_langfuse_client():
    return get_client()


def get_langfuse_handler() -> CallbackHandler | None:
    return _CURRENT_LANGFUSE_HANDLER.get()


@contextmanager
def start_request_trace(
    *,
    trace_name: str,
    trace_id: str,
    user_id: str,
    session_id: str,
    input_payload: dict,
):
    langfuse = get_langfuse_client()
    with langfuse.start_as_current_observation(
        as_type="span",
        name=trace_name,
        input=input_payload,
        trace_context={"trace_id": trace_id},
    ) as root_span:
        with propagate_attributes(
            trace_name=trace_name,
            user_id=user_id,
            session_id=session_id,
        ):
            handler = CallbackHandler()
            token = _CURRENT_LANGFUSE_HANDLER.set(handler)
            try:
                yield root_span
            finally:
                _CURRENT_LANGFUSE_HANDLER.reset(token)


@contextmanager
def traced_observation(
    name: str,
    *,
    input_payload: dict | None = None,
    as_type: str = "span",
):
    langfuse = get_langfuse_client()
    observation_kwargs = {"as_type": as_type, "name": name}
    if input_payload is not None:
        observation_kwargs["input"] = input_payload

    with langfuse.start_as_current_observation(**observation_kwargs) as observation:    
        yield observation


async def invoke_with_langfuse(llm, messages):
    # The Langfuse handler is attached at the graph level (see app.py) so it
    # propagates through the run tree to every node. We must NOT pass an explicit
    # callbacks config here — doing so detaches this call from the parent run and
    # prevents astream_events from capturing on_chat_model_stream token events.
    return await llm.ainvoke(messages)


async def trace_event(trace_id: str, name: str, payload: dict | None = None):
    await asyncio.to_thread(_trace_event_sync, trace_id, name, payload)


def _trace_event_sync(trace_id: str, name: str, payload: dict | None = None):
    record = {
        "trace_id": trace_id,
        "event": name,
        "timestamp": time.time(),
        "payload": payload or {},
    }

    if _send_to_langfuse(record):
        return

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    with (TRACE_DIR / f"{trace_id}.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_trace(trace_id: str) -> list[dict]:
    path = TRACE_DIR / f"{trace_id}.jsonl"
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _send_to_langfuse(record: dict) -> bool:
    if os.getenv("LANGFUSE_ENABLED", "0") != "1":
        return False

    try:
        client = get_langfuse_client()
        with client.start_as_current_observation(
            as_type="event",
            name=record["event"],
            input=record["payload"],
        ):
            pass
        return True
    except Exception as exc:
        print(f"[Tracing] Langfuse unavailable, writing local trace: {exc}")
        return False
