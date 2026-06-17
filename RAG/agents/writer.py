from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from RAG.agents.state import AgentState
from RAG.agents.supervisor import get_llm
from RAG.services.tracing import invoke_with_langfuse, trace_event, traced_observation


WRITER_SYSTEM_PROMPT = """You are a document-grounded QA assistant. Prepare an accurate response to the user's question using the provided top-5 retrieved context.

Rules:
- If research results are available, base your answer only on them.
- Cite supporting context with bracket numbers like [1] or [2].
- If the retrieved context does not contain enough evidence, say the documents do not contain enough information.
- Do not add facts from general knowledge.
- Be polite and professional in social interactions like greetings.
- Respond in the same language as the user's query.
- Write your response in a clear and structured way."""


_NO_DOCS_REPLY = (
    "I could not find any relevant information in the knowledge base to answer your question. "
    "Please rephrase your query or ask about topics covered by the available documents."
)


async def writer_node(state: AgentState) -> dict:
    llm = get_llm()

    with traced_observation("writer", input_payload={"query": state["query"]}) as span:
        # Topic restriction: researcher ran but found nothing — skip LLM, return early.
        research = state.get("research_results", "")
        researcher_ran = bool(state.get("rewritten_query") or research)
        no_docs = not research or research.strip() == "No relevant documents found."
        is_revision = bool(state.get("review_feedback"))

        if researcher_ran and no_docs and not is_revision:
            await trace_event(state["trace_id"], "writer.no_docs", {"query": state["query"]})
            span.update(output={"no_docs": True})
            print("[Writer] No documents found — returning early.")
            return {
                "final_response": _NO_DOCS_REPLY,
                "draft_response": _NO_DOCS_REPLY,
                "messages": [AIMessage(content=_NO_DOCS_REPLY)],
            }

        messages = [SystemMessage(content=WRITER_SYSTEM_PROMPT)]

        user_content = f"Question: {state['query']}"

        if research and research.strip() != "No relevant documents found.":
            user_content += f"\n\nResearch Results:\n{research}"

        if state.get("review_feedback"):
            user_content += f"\n\nPrevious Revision Feedback:\n{state['review_feedback']}"
            user_content += f"\n\nPrevious Draft:\n{state['draft_response']}"
            user_content += "\n\nPlease revise your response taking the feedback into account."

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

        return {
            "draft_response": draft,
            "messages": [AIMessage(content="Response draft prepared.")],
        }
