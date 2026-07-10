"""Tavily web search fallback — used when the RAG corpus has no relevant answer."""
import os

import httpx

TAVILY_API_URL = "https://api.tavily.com/search"
_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
_TIMEOUT = float(os.getenv("TAVILY_TIMEOUT_SECONDS", "8"))
_MAX_QUERY_CHARS = 400  # Tavily hard limit; longer queries get a 400 Bad Request.


def web_search_available() -> bool:
    return bool(os.getenv("TAVILY_API_KEY", "").strip())


def web_search(query: str) -> dict:
    """Search the web via Tavily. Returns {context, sources, answer}.

    sources: [{title, url, content}, ...] for the evidence panel.
    answer:  Tavily's own synthesized answer (may be empty).
    Returns empty structure on any failure so the caller can degrade gracefully.
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return {"context": "", "sources": [], "answer": ""}

    payload = {
        "query": query[:_MAX_QUERY_CHARS],
        "max_results": _MAX_RESULTS,
        "search_depth": "basic",
        "include_answer": True,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    try:
        response = httpx.post(TAVILY_API_URL, json=payload, headers=headers, timeout=_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        print(f"[WebSearch] Tavily request failed: {exc}")
        return {"context": "", "sources": [], "answer": ""}

    sources = []
    blocks = []
    for index, item in enumerate(data.get("results", []), start=1):
        title = item.get("title", "") or item.get("url", "")
        url = item.get("url", "")
        content = (item.get("content", "") or "").strip()
        sources.append({"title": title, "url": url, "content": content})
        blocks.append(f"[{index}] {title} ({url})\n{content}")

    return {
        "context": "\n\n".join(blocks),
        "sources": sources,
        "answer": (data.get("answer") or "").strip(),
    }
