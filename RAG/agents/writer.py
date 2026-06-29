import base64
import mimetypes
import os

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from RAG.agents.state import AgentState
from RAG.agents.supervisor import get_llm, get_vision_llm
from RAG.services import paths
from RAG.services.tracing import invoke_with_langfuse, trace_event, traced_observation

# Reviewer adds a ~1s LLM round-trip — cheap, so kept on by default for quality.
# The real latency bottleneck is retrieval (reranker), not the reviewer.
_REVIEWER_ENABLED = os.getenv("RAG_ENABLE_REVIEWER", "1") == "1"
_MAX_IMAGES = int(os.getenv("MAX_CONTEXT_IMAGES", "4"))


def _file_to_data_url(path) -> str | None:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    try:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:{mime};base64,{data}"


def _image_blocks(state: AgentState) -> list[dict]:
    """Build OpenAI-style image_url content blocks from user-attached images and
    figures anchored to the retrieved context. Bounded by _MAX_IMAGES."""
    blocks: list[dict] = []

    for data_url in state.get("query_images") or []:
        if data_url:
            blocks.append({"type": "image_url", "image_url": {"url": data_url}})

    for img in state.get("context_images") or []:
        path = paths.image_path(
            img.get("channel", "workspace"), paths.stem_of(img.get("source", "")), img.get("name", "")
        )
        if not path:
            continue
        data_url = _file_to_data_url(path)
        if data_url:
            blocks.append({"type": "image_url", "image_url": {"url": data_url}})

    return blocks[:_MAX_IMAGES]


WRITER_SYSTEM_PROMPT = """You are a document-grounded QA assistant. Prepare an accurate response to the user's question using the provided top-5 retrieved context.

Rules:
- If research results are available, base your answer only on them.
- Cite supporting context with bracket numbers like [1] or [2].
- If the retrieved context does not contain enough evidence, say the documents do not contain enough information.
- Do not add facts from general knowledge.
- Be polite and professional in social interactions like greetings.
- Respond in the same language as the user's query.
- Write your response in a clear and structured way."""


WRITER_WEB_SYSTEM_PROMPT = """You are a QA assistant answering from WEB SEARCH results because the local knowledge base did not contain the answer.

Rules:
- Base your answer only on the provided web search results.
- Start your answer by stating that this information was found on the internet (not in the local documents).
- Cite supporting sources with bracket numbers like [1] or [2].
- Respond in the same language as the user's query.
- Write your response in a clear and structured way."""


_NO_DOCS_REPLY = (
    "I could not find any relevant information in the knowledge base to answer your question. "
    "Please rephrase your query or ask about topics covered by the available documents."
)


async def writer_node(state: AgentState) -> dict:
    with traced_observation("writer", input_payload={"query": state["query"]}) as span:
        # Topic restriction: researcher ran but found nothing — skip LLM, return early.
        research = state.get("research_results", "")
        researcher_ran = bool(state.get("rewritten_query") or research)
        no_docs = not research or research.strip() == "No relevant documents found."
        is_revision = bool(state.get("review_feedback"))
        # A user-attached image is itself answerable content — don't short-circuit
        # to the "no documents" reply when there's an image to reason over.
        has_user_images = bool(state.get("query_images"))

        if researcher_ran and no_docs and not is_revision and not has_user_images:
            await trace_event(state["trace_id"], "writer.no_docs", {"query": state["query"]})
            span.update(output={"no_docs": True})
            print("[Writer] No documents found — returning early.")
            return {
                "final_response": _NO_DOCS_REPLY,
                "draft_response": _NO_DOCS_REPLY,
                "messages": [AIMessage(content=_NO_DOCS_REPLY)],
            }

        is_web = state.get("source_type") == "web"
        system_prompt = WRITER_WEB_SYSTEM_PROMPT if is_web else WRITER_SYSTEM_PROMPT
        messages = [SystemMessage(content=system_prompt)]

        user_content = f"Question: {state['query']}"

        history = state.get("conversation_history") or []
        if history:
            history_text = "\n".join(
                f"User: {turn['query']}\nAssistant: {turn['response']}" for turn in history
            )
            user_content = f"Conversation history (for context only):\n{history_text}\n\n{user_content}"

        if research and research.strip() != "No relevant documents found.":
            user_content += f"\n\nResearch Results:\n{research}"

        if state.get("review_feedback"):
            user_content += f"\n\nPrevious Revision Feedback:\n{state['review_feedback']}"
            user_content += f"\n\nPrevious Draft:\n{state['draft_response']}"
            user_content += "\n\nPlease revise your response taking the feedback into account."

        # Multimodal: when figures (from retrieval) or user-attached images are
        # present, send them to a vision model as image_url blocks. Otherwise
        # keep the cheaper text model and a plain string message.
        image_blocks = _image_blocks(state)
        if image_blocks:
            llm = get_vision_llm()
            messages.append(
                HumanMessage(content=[{"type": "text", "text": user_content}, *image_blocks])
            )
            print(f"[Writer] Vision mode — {len(image_blocks)} image(s) attached.")
        else:
            llm = get_llm()
            messages.append(HumanMessage(content=user_content))

        response = await invoke_with_langfuse(llm, messages)
        draft = response.content
        await trace_event(
            state["trace_id"],
            "writer.response",
            {
                "prompt_length": len(user_content),
                "response_length": len(draft),
                "has_research": bool(research),
                "answer": draft,
            },
        )
        span.update(output={"response_length": len(draft)})

        print(f"[Writer] Response prepared ({len(draft)} characters)")

        # When reviewer is enabled, return only the draft so the reviewer can
        # check it. When disabled (default), the draft is the final answer —
        # set final_response so the supervisor finishes without an extra LLM call.
        if _REVIEWER_ENABLED:
            return {
                "draft_response": draft,
                "messages": [AIMessage(content="Response draft prepared.")],
            }

        return {
            "draft_response": draft,
            "final_response": draft,
            "messages": [AIMessage(content="Response prepared.")],
        }
