import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

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

# ── In-memory rate limit store ────────────────────────────────────────────────
_rate_store: dict[str, list[float]] = defaultdict(list)


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

    # 2. Rate limiting (sliding window, per user)
    def check_rate_limit(self, user_id: str) -> GuardrailResult:
        now = time.monotonic()
        window_start = now - RATE_LIMIT_WINDOW
        bucket = _rate_store[user_id]
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
