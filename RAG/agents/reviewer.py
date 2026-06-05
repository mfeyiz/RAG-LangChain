import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from RAG.agents.state import AgentState
from RAG.agents.supervisor import get_llm


REVIEWER_SYSTEM_PROMPT = """You are a groundedness quality assurance specialist. Evaluate the prepared response.

Evaluation criteria:
1. Accuracy: Is the response consistent with the research results?
2. Completeness: Is the question fully answered?
3. Clarity: Is the response clear and well-structured?
4. Citation grounding: Are factual claims supported by retrieved context citations?
5. Language: Is the grammar and language correct?

Return ONLY valid JSON:
{
  "approved": true,
  "feedback": "",
  "unsupported_claims": []
}

If insufficient, set approved to false and keep feedback under 2 sentences."""


def reviewer_node(state: AgentState) -> dict:
    llm = get_llm()

    messages = [SystemMessage(content=REVIEWER_SYSTEM_PROMPT)]

    review_content = f"Question: {state['query']}"

    if state.get("research_results"):
        review_content += f"\n\nResearch Results:\n{state['research_results']}"

    review_content += f"\n\nPrepared Response:\n{state['draft_response']}"

    messages.append(HumanMessage(content=review_content))

    response = llm.invoke(messages)
    decision = _parse_review(response.content)
    feedback = decision["feedback"]

    revision_count = state.get("revision_count", 0)

    if decision["approved"] or revision_count >= 2:
        print(f"[Reviewer] Approved (revisions: {revision_count})")
        return {
            "final_response": state["draft_response"],
            "review_feedback": "",
            "messages": [AIMessage(content="Response approved.")],
        }

    print(f"[Reviewer] Revision requested: {feedback[:100]}")
    return {
        "review_feedback": feedback,
        "revision_count": revision_count + 1,
        "draft_response": "",
        "messages": [AIMessage(content=f"Revision requested: {feedback[:100]}")],
    }


def _parse_review(content: str) -> dict:
    try:
        parsed = json.loads(content)
        return {
            "approved": bool(parsed.get("approved", False)),
            "feedback": str(parsed.get("feedback", "")).strip(),
            "unsupported_claims": parsed.get("unsupported_claims", []),
        }
    except Exception:
        normalized = content.strip()
        return {
            "approved": normalized.upper() == "APPROVED",
            "feedback": normalized,
            "unsupported_claims": [],
        }
