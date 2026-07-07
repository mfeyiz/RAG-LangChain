import json
import os
import time
from pathlib import Path

from RAG.services._redis import make_redis_loader

FEEDBACK_PATH = Path(__file__).resolve().parent.parent / "traces" / "feedback.jsonl"

_load_redis_client = make_redis_loader("Feedback")


def _feedback_key(trace_id: str, session_id: str) -> str:
    return f"feedback:{trace_id}:{session_id}"


def save_feedback(
    session_id: str,
    trace_id: str,
    rating: int,        # 1 = thumbs up, -1 = thumbs down
    query: str = "",
    comment: str = "",
    user_id: str = "anonymous",
) -> None:
    record = {
        "ts": time.time(),
        "session_id": session_id,
        "trace_id": trace_id,
        "user_id": user_id,
        "query": query,
        "rating": rating,
        "comment": comment,
    }

    client = _load_redis_client()
    if client is not None:
        try:
            key = _feedback_key(trace_id, session_id)
            client.set(key, json.dumps(record, ensure_ascii=False))
            client.sadd("feedback:all", key)
            return
        except Exception as exc:
            print(f"[Feedback] Redis save failed, falling back to file: {exc}")

    # Fallback to file
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_feedback_comment(trace_id: str, session_id: str, comment: str) -> bool:
    """Attach a comment to the most recent feedback record for a trace/session.

    Lets the UI capture the rating immediately on click, then enrich it with a
    comment afterwards — without creating a duplicate (double-counted) record.
    """
    if not comment:
        return False

    client = _load_redis_client()
    if client is not None:
        try:
            key = _feedback_key(trace_id, session_id)
            raw = client.get(key)
            if raw is not None:
                record = json.loads(raw)
                record["comment"] = comment
                client.set(key, json.dumps(record, ensure_ascii=False))
                return True
            return False
        except Exception as exc:
            print(f"[Feedback] Redis update failed, falling back to file: {exc}")

    # Fallback to file
    if not FEEDBACK_PATH.exists():
        return False

    records: list[dict] = []
    with FEEDBACK_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))

    for rec in reversed(records):
        if rec.get("trace_id") == trace_id and rec.get("session_id") == session_id:
            rec["comment"] = comment
            with FEEDBACK_PATH.open("w", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            return True
    return False


def get_feedback_stats() -> dict:
    client = _load_redis_client()
    if client is not None:
        try:
            keys = client.smembers("feedback:all")
            total = positive = negative = 0
            for key in keys:
                raw = client.get(key)
                if raw:
                    rec = json.loads(raw)
                    total += 1
                    if rec.get("rating", 0) > 0:
                        positive += 1
                    else:
                        negative += 1
            score = round(positive / total, 3) if total else None
            return {"total": total, "positive": positive, "negative": negative, "score": score}
        except Exception as exc:
            print(f"[Feedback] Redis stats failed, falling back to file: {exc}")

    # Fallback to file
    if not FEEDBACK_PATH.exists():
        return {"total": 0, "positive": 0, "negative": 0, "score": None}

    total = positive = negative = 0
    with FEEDBACK_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            if rec.get("rating", 0) > 0:
                positive += 1
            else:
                negative += 1

    score = round(positive / total, 3) if total else None
    return {"total": total, "positive": positive, "negative": negative, "score": score}


def list_feedback(limit: int = 50) -> list[dict]:
    client = _load_redis_client()
    if client is not None:
        try:
            keys = client.smembers("feedback:all")
            records = []
            for key in keys:
                raw = client.get(key)
                if raw:
                    records.append(json.loads(raw))
            records.sort(key=lambda r: r.get("ts", 0), reverse=True)
            return records[:limit]
        except Exception as exc:
            print(f"[Feedback] Redis list failed, falling back to file: {exc}")

    # Fallback to file
    if not FEEDBACK_PATH.exists():
        return []

    records: list[dict] = []
    with FEEDBACK_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))

    return records[-limit:]
