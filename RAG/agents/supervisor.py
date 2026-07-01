import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import httpx

from RAG.agents.state import AgentState
from RAG.services.tracing import invoke_with_langfuse, trace_event, traced_observation


env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


# Provider is OpenAI-compatible and selected via env so we can move off OpenRouter
# (e.g. to OpenCode Go) without code changes. LLM_API_KEY is the generic key;
# OPENROUTER_API_KEY is honoured as a fallback for older configs.
_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
_LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")


def _provider_extra_body() -> dict:
    """OpenRouter-only routing hint. Other gateways (OpenCode Go) reject unknown
    body fields, so only send it when actually talking to OpenRouter."""
    if "openrouter.ai" in _LLM_BASE_URL:
        # Route to the lowest-latency OpenRouter provider — cut writer time
        # from ~10s to ~3s in local benchmarks.
        return {"provider": {"sort": "latency"}}
    return {}


def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
        base_url=_LLM_BASE_URL,
        api_key=_LLM_API_KEY,
        temperature=0,
        streaming=True,
        extra_body=_provider_extra_body(),
        http_client=httpx.Client(verify=False),
        http_async_client=httpx.AsyncClient(verify=False),
    )


# ── Fast-track (lightweight classifier) model ─────────────────────────────────
# Defaults to the same gateway/model as get_llm() but can be pointed at a cheaper
# "flash" model (Gemini Flash, Llama 3 Mild, …) via FAST_LLM_* env vars so the
# supervisor can route greetings / single-step questions without the full
# researcher → reviewer pipeline.
_FAST_LLM_BASE_URL = os.getenv("FAST_LLM_BASE_URL") or _LLM_BASE_URL
_FAST_LLM_API_KEY = os.getenv("FAST_LLM_API_KEY") or _LLM_API_KEY
_FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL") or os.getenv("LLM_MODEL", "deepseek-v4-flash")

# Set RAG_FAST_TRACK=0 to disable LLM-based fast tracking entirely.
_FAST_TRACK_ENABLED = os.getenv("RAG_FAST_TRACK", "1") == "1"


def get_fast_llm():
    """Cheaper/lighter model used only for intent classification in the
    supervisor. Falls back to get_llm() when not configured separately."""
    if (
        _FAST_LLM_BASE_URL == _LLM_BASE_URL
        and _FAST_LLM_MODEL == os.getenv("LLM_MODEL", "deepseek-v4-flash")
        and _FAST_LLM_API_KEY == _LLM_API_KEY
    ):
        return get_llm()
    return ChatOpenAI(
        model=_FAST_LLM_MODEL,
        base_url=_FAST_LLM_BASE_URL,
        api_key=_FAST_LLM_API_KEY,
        temperature=0,
        streaming=False,
        extra_body={"provider": {"sort": "latency"}} if "openrouter.ai" in _FAST_LLM_BASE_URL else {},
        http_client=httpx.Client(verify=False),
        http_async_client=httpx.AsyncClient(verify=False),
    )


def get_vision_llm(streaming: bool = True):
    """Vision-capable model used by the writer (and ingest captioning) when images
    are in play. Kept separate from `get_llm()` so routing, query rewriting, and
    review stay on the cheaper text model. Shares the same gateway/key by default,
    but VISION_LLM_BASE_URL / VISION_LLM_API_KEY can point it elsewhere."""
    base_url = os.getenv("VISION_LLM_BASE_URL", _LLM_BASE_URL)
    extra_body = {"provider": {"sort": "latency"}} if "openrouter.ai" in base_url else {}
    return ChatOpenAI(
        model=os.getenv("VISION_LLM_MODEL", "qwen3.6-plus"),
        base_url=base_url,
        api_key=os.getenv("VISION_LLM_API_KEY") or _LLM_API_KEY,
        temperature=0,
        streaming=streaming,
        extra_body=extra_body,
        http_client=httpx.Client(verify=False),
        http_async_client=httpx.AsyncClient(verify=False),
    )


async def supervisor_node(state: AgentState) -> dict:
    with traced_observation(
        "supervisor",
        input_payload={
            "query": state["query"],
            "has_research": bool(state.get("research_results")),
            "has_draft": bool(state.get("draft_response")),
            "has_feedback": bool(state.get("review_feedback")),
            "revision_count": state.get("revision_count", 0),
        },
    ) as span:
        if state.get("final_response"):
            print("[Supervisor] Decision: finish")
            await trace_event(state["trace_id"], "supervisor.decision", {"next_agent": "finish"})
            span.update(output={"next_agent": "finish"})
            return {"next_agent": "finish"}

        # Researcher found weak RAG results and is asking the user before going to
        # the web — stop the graph and let the UI surface the approval prompt.
        if state.get("needs_web_approval"):
            print("[Supervisor] Decision: finish (awaiting web search approval)")
            await trace_event(state["trace_id"], "supervisor.decision", {"next_agent": "finish"})
            span.update(output={"next_agent": "finish"})
            return {"next_agent": "finish"}

        # Editor produced a diff preview and is awaiting human approval before
        # the edit is persisted — stop the graph; the UI drives /update/apply.
        if state.get("needs_edit_approval"):
            print("[Supervisor] Decision: finish (awaiting @update approval)")
            await trace_event(state["trace_id"], "supervisor.decision", {"next_agent": "finish"})
            span.update(output={"next_agent": "finish"})
            return {"next_agent": "finish"}

        if state.get("draft_response"):
            print("[Supervisor] Decision: reviewer")
            await trace_event(state["trace_id"], "supervisor.decision", {"next_agent": "reviewer"})
            span.update(output={"next_agent": "reviewer"})
            return {"next_agent": "reviewer"}

        if state.get("review_feedback"):
            print("[Supervisor] Decision: writer")
            await trace_event(state["trace_id"], "supervisor.decision", {"next_agent": "writer"})
            span.update(output={"next_agent": "writer"})
            return {"next_agent": "writer"}

        if state.get("research_results"):
            print("[Supervisor] Decision: writer")
            await trace_event(state["trace_id"], "supervisor.decision", {"next_agent": "writer"})
            span.update(output={"next_agent": "writer"})
            return {"next_agent": "writer"}

        # Initial state: nothing set yet. An @update message is a write-back
        # request and goes to the editor instead of the read-only RAG path.
        if re.search(r"@update\b", state["query"], flags=re.IGNORECASE):
            print("[Supervisor] Decision: editor")
            await trace_event(state["trace_id"], "supervisor.decision", {"next_agent": "editor"})
            span.update(output={"next_agent": "editor"})
            return {"next_agent": "editor"}

        # Route to researcher unless it is a short social phrase that needs no
        # document lookup.
        _SOCIAL = frozenset([
            "hello", "hi", "hey", "thanks", "thank", "bye", "goodbye",
        ])
        words = state["query"].lower().split()
        if len(words) <= 6 and _SOCIAL.intersection(words):
            decision = "writer"
            fast_track = True
        else:
            fast_track, decision = await _classify_intent(state)

        # Fast-track: simple / single-step questions bypass the researcher and
        # reviewer entirely and go straight to the writer, which will return the
        # final response in one shot.
        updates = {"next_agent": decision}
        if fast_track:
            updates["fast_track"] = True

        print(f"[Supervisor] Decision: {decision}")
        await trace_event(
            state["trace_id"],
            "supervisor.decision",
            {"next_agent": decision, "fast_track": fast_track},
        )
        span.update(output={"next_agent": decision, "fast_track": fast_track})

        return updates


_FAST_CLASSIFY_PROMPT = """You are a router for a multi-agent document QA system. Classify the user message into ONE route.

Return ONLY a single word, no punctuation, no explanation:
- "simple" : greetings, chitchat, thanks, or a single-step factual question answerable from general knowledge WITHOUT the document store (e.g. "who wrote Hamlet?").
- "calc" : the message asks for an arithmetic calculation, an average/sum/percentage, a comparison of numbers, OR asks to generate a chart/graph from data.
- "research" : anything that needs the knowledge base / documents to answer (most questions about the uploaded corpus), multi-step questions, comparisons, or requests that mention @update.

User message:
\"\"\"{query}\"\"\""""


async def _classify_intent(state: AgentState) -> tuple[bool, str]:
    """Use the lightweight fast-LLM to decide whether the query is simple enough
    to bypass the researcher+reviewer pipeline. Falls back to "research" on any
    error or when fast-track is disabled.

    Returns ``(fast_track, next_agent)``:
    - ("simple"  ) → (True, "writer")
    - ("calc"    ) → (False, "code_interpreter")
    - ("research") → (False, "researcher")
    """
    query = state["query"]
    if not _FAST_TRACK_ENABLED or not query or len(query.split()) < 2:
        return False, "researcher"
    # No API key configured → a classifier round-trip would just fail; skip it
    # (keeps offline unit tests from making live network calls).
    if not _LLM_API_KEY and not os.getenv("OPENROUTER_API_KEY"):
        return False, "researcher"

    # Cheap regex pre-filter for calculation/chart intent so we can route to the
    # code interpreter without always paying for an LLM round-trip.
    if _looks_like_calculation(query):
        print("[Supervisor] Fast-track: calculation/chart request → code_interpreter.")
        return False, "code_interpreter"

    try:
        llm = get_fast_llm()
        response = await invoke_with_langfuse(
            llm,
            [
                SystemMessage(content=_FAST_CLASSIFY_PROMPT.format(query=query)),
                HumanMessage(content=query),
            ],
        )
        verdict = (response.content or "").strip().strip("`\"").lower()
        if verdict.startswith("simple"):
            print("[Supervisor] Fast-track: simple question → writer (bypassing researcher/reviewer).")
            return True, "writer"
        if verdict.startswith("calc"):
            print("[Supervisor] Fast-track: calculation/chart request → code_interpreter.")
            return False, "code_interpreter"
        return False, "researcher"
    except Exception as exc:
        print(f"[Supervisor] Fast-track classifier failed ({exc}); routing to researcher.")
        return False, "researcher"


_CALC_HINTS = (
    "bar chart", "scatter", "histogram", "plot", "calculate", "average", "sum",
    "ratio", "percentage", "compound", "cagr", "wacc", "ebitda", "regression",
    "correlation", "pie chart",
)

# Arithmetic between two numbers: "5 + 3", "12*4", "80 / 4", "50 % 2", "3 x 4".
_ARITH_RE = re.compile(r"\d+(?:\.\d+)?\s*[-+*/x×÷^%]\s*\d+(?:\.\d+)?")
# A series of 3+ numbers separated by commas/semicolons — chart/plot data such
# as "plot 10, 20, 30". A lone multi-digit number (a year, version, or figure)
# must NOT match, which the previous "\d+ ... \d+" pattern did (it split "2022"
# into "202" + "2" and mis-routed any dated question to the code interpreter).
_NUM_SERIES_RE = re.compile(r"\b\d+(?:\.\d+)?\b(?:\s*[,;]\s*\d+(?:\.\d+)?){2,}")


def _looks_like_calculation(query: str) -> bool:
    q = query.lower()
    if any(h in q for h in _CALC_HINTS):
        return True
    return bool(_ARITH_RE.search(q) or _NUM_SERIES_RE.search(q))
