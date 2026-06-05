from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from RAG.agents.state import AgentState
from RAG.agents.supervisor import get_llm


WRITER_SYSTEM_PROMPT = """You are a document-grounded QA assistant. Prepare an accurate response to the user's question using the provided top-5 retrieved context.

Rules:
- If research results are available, base your answer only on them.
- Cite supporting context with bracket numbers like [1] or [2].
- If the retrieved context does not contain enough evidence, say the documents do not contain enough information.
- Do not add facts from general knowledge.
- Be polite and professional in social interactions like greetings.
- Respond in the same language as the user's query.
- Write your response in a clear and structured way."""


def writer_node(state: AgentState) -> dict:
    llm = get_llm()
    
    messages = [SystemMessage(content=WRITER_SYSTEM_PROMPT)]
    
    user_content = f"Question: {state['query']}"
    
    if state.get("research_results") and state["research_results"] != "No relevant documents found.":
        user_content += f"\n\nResearch Results:\n{state['research_results']}"
    
    if state.get("review_feedback"):
        user_content += f"\n\nPrevious Revision Feedback:\n{state['review_feedback']}"
        user_content += f"\n\nPrevious Draft:\n{state['draft_response']}"
        user_content += "\n\nPlease revise your response taking the feedback into account."
    
    messages.append(HumanMessage(content=user_content))
    
    response = llm.invoke(messages)
    draft = response.content
    
    print(f"[Writer] Response prepared ({len(draft)} characters)")
    
    return {
        "draft_response": draft,
        "messages": [AIMessage(content="Response draft prepared.")],
    }
