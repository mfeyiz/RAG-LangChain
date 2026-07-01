import os
import re
import time
from dataclasses import dataclass

# ── 1. Query length limit ────────────────────────────────────────────────────
MAX_QUERY_LENGTH = int(os.getenv("GUARDRAIL_MAX_QUERY_LENGTH", "1000"))

# ── 2. Rate limiting ─────────────────────────────────────────────────────────
RATE_LIMIT_MAX = int(os.getenv("GUARDRAIL_RATE_LIMIT_MAX", "15"))
RATE_LIMIT_WINDOW = int(os.getenv("GUARDRAIL_RATE_LIMIT_WINDOW", "60"))

# ── 3. Prompt injection patterns ─────────────────────────────────────────────
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(everything|all)\s+",
    r"you\s+are\s+now\s+(a\s+)?",
    r"(act|pretend|roleplay|behave)\s+as\s+",
    r"disregard\s+(your|all)\s+(previous|prior|system|instructions)",
    r"bypass\s+(your\s+)?(safety|filter|guardrail|restriction)",
    r"\bjailbreak\b",
    r"\bDAN\s+mode\b",
    r"<\|im_start\|>|<\|im_end\|>|\[INST\]|\[\/INST\]",
    r"system\s*prompt\s*:",
    r"new\s+instructions\s*:",
]
INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# ── 4. Harmful content patterns ───────────────────────────────────────────────
_HARMFUL_PATTERNS = [
    r"\bhow\s+to\s+(make|create|build|synthesize)\s+(a\s+)?(bomb|explosive|weapon|poison|malware|ransomware|virus|trojan)\b",
    r"\b(suicide|self.?harm)\s+(method|instruction|guide|tutorial|step)\b",
    r"\bchild\s+(porn|pornography|abuse|exploitation|grooming)\b",
    r"\b(produce|generate|create)\s+(CSAM|child\s+sexual)\b",
    r"\bstep.by.step\s+(guide|instructions?)\s+(to\s+)?(kill|murder|attack|harm)\b",
]
HARMFUL_RE = [re.compile(p, re.IGNORECASE) for p in _HARMFUL_PATTERNS]

# ── 5. Off-topic patterns (topic restriction) ─────────────────────────────────
_OFFTOPIC_PATTERNS = [
    r"^(tell\s+me\s+a?\s*joke|write\s+me\s+a?\s*(poem|song|story|essay|novel|fiction))\b",
    r"^(what\s+is\s+your\s+(favorite|name|age|gender)|who\s+(are|created)\s+you)\b",
    r"^(draw|generate|create|paint)\s+(me\s+)?(an?\s+)?(image|picture|photo|illustration)\b",
    r"^(translate\s+this\s+(to|into)|convert\s+this\s+to)\b",
    r"^(play\s+a\s+game|let'?s\s+play|entertain\s+me)\b",
]
OFFTOPIC_RE = [re.compile(p, re.IGNORECASE) for p in _OFFTOPIC_PATTERNS]

_redis_client = None
_redis_retry_after: float = 0.0
_REDIS_RETRY_INTERVAL = 30.0


def _load_redis_client():
    global _redis_client, _redis_retry_after
    if _redis_client is not None:
        return _redis_client

    now = time.time()
    if now < _redis_retry_after:
        return None

    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return None

    try:
        from redis import Redis

        client = Redis.from_url(redis_url, decode_responses=True)
        client.ping()
        _redis_client = client
        return client
    except Exception as exc:
        print(f"[Guardrails] Redis unavailable, retrying in {_REDIS_RETRY_INTERVAL}s: {exc}")
        _redis_retry_after = now + _REDIS_RETRY_INTERVAL
        return None


@dataclass
class GuardrailResult:
    allowed: bool
    error: str = ""
    status_code: int = 200


class GuardrailService:
    # 1. Length
    def check_length(self, query: str) -> GuardrailResult:
        if len(query) > MAX_QUERY_LENGTH:
            return GuardrailResult(
                allowed=False,
                error=f"Query too long. Maximum {MAX_QUERY_LENGTH} characters allowed (got {len(query)}).",
                status_code=400,
            )
        if not query.strip():
            return GuardrailResult(allowed=False, error="Query cannot be empty.", status_code=400)
        return GuardrailResult(allowed=True)

    # 2. Rate limiting (sliding window, per user) — backed by Redis for multi-pod correctness
    def check_rate_limit(self, user_id: str) -> GuardrailResult:
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW
        key = f"ratelimit:{user_id}"

        client = _load_redis_client()
        if client is not None:
            try:
                pipe = client.pipeline()
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zcard(key)
                pipe.zadd(key, {str(now): now})
                pipe.expire(key, RATE_LIMIT_WINDOW + 1)
                results = pipe.execute()
                current_count = results[1]
                if current_count >= RATE_LIMIT_MAX:
                    return GuardrailResult(
                        allowed=False,
                        error=f"Rate limit exceeded: max {RATE_LIMIT_MAX} requests per {RATE_LIMIT_WINDOW}s. Please wait.",
                        status_code=429,
                    )
                return GuardrailResult(allowed=True)
            except Exception as exc:
                print(f"[Guardrails] Redis rate limit failed, falling back to in-memory: {exc}")

        # Fallback to in-memory (single-process only)
        if not hasattr(self, "_rate_store"):
            from collections import defaultdict
            self._rate_store = defaultdict(list)
        bucket = self._rate_store[user_id]
        bucket[:] = [t for t in bucket if t > window_start]
        if len(bucket) >= RATE_LIMIT_MAX:
            return GuardrailResult(
                allowed=False,
                error=f"Rate limit exceeded: max {RATE_LIMIT_MAX} requests per {RATE_LIMIT_WINDOW}s. Please wait.",
                status_code=429,
            )
        bucket.append(now)
        return GuardrailResult(allowed=True)

    # 3. Prompt injection
    def check_injection(self, query: str) -> GuardrailResult:
        for pattern in INJECTION_RE:
            if pattern.search(query):
                return GuardrailResult(
                    allowed=False,
                    error="Query contains disallowed patterns (possible prompt injection).",
                    status_code=400,
                )
        return GuardrailResult(allowed=True)

    # 4. Harmful content
    def check_harmful(self, query: str) -> GuardrailResult:
        for pattern in HARMFUL_RE:
            if pattern.search(query):
                return GuardrailResult(
                    allowed=False,
                    error="Query contains content that is not permitted.",
                    status_code=400,
                )
        return GuardrailResult(allowed=True)

    # 5. Topic restriction (heuristic — catches obviously off-topic requests)
    def check_topic(self, query: str) -> GuardrailResult:
        for pattern in OFFTOPIC_RE:
            if pattern.match(query.strip()):
                return GuardrailResult(
                    allowed=False,
                    error="This assistant only answers questions based on its knowledge base documents.",
                    status_code=400,
                )
        return GuardrailResult(allowed=True)

    def check_all(self, query: str, user_id: str) -> GuardrailResult:
        for check in (
            lambda: self.check_length(query),
            lambda: self.check_rate_limit(user_id),
            lambda: self.check_injection(query),
            lambda: self.check_harmful(query),
            lambda: self.check_topic(query),
        ):
            result = check()
            if not result.allowed:
                return result
        return GuardrailResult(allowed=True)


guardrails = GuardrailService()
