import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from RAG.agents.state import AgentState, AgentRole


env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


def get_llm():
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )


SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor agent. Your task is to analyze the user's question and decide which agent should run next.

Available agents:
- researcher: Searches the document database. Use for technical questions, product information, or definitions.
- writer: Prepares a response for the user using research results.
- reviewer: Checks the quality of the prepared response.

Decision rules:
1. If the user asks a technical question or requests information -> "researcher"
2. Social interactions like greetings, thanks -> "writer" (no research needed)
3. If research is done -> "writer"
4. If a response is drafted -> "reviewer"
5. If reviewer approved or revision count reached 2 -> "FINISH"

Return ONLY one of these values: researcher, writer, reviewer, FINISH"""


def supervisor_node(state: AgentState) -> dict:
    if state.get("final_response"):
        print("[Supervisor] Decision: finish")
        return {"next_agent": "finish"}

    if state.get("draft_response"):
        print("[Supervisor] Decision: reviewer")
        return {"next_agent": "reviewer"}

    if state.get("review_feedback"):
        print("[Supervisor] Decision: writer")
        return {"next_agent": "writer"}

    if state.get("research_results"):
        print("[Supervisor] Decision: writer")
        return {"next_agent": "writer"}

    llm = get_llm()
    
    messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)]
    
    context_parts = [f"User query: {state['query']}"]
    
    if state.get("research_results"):
        context_parts.append(f"Research completed: {len(state['research_results'])} characters of results found.")
    
    if state.get("draft_response"):
        context_parts.append(f"Response drafted: {len(state['draft_response'])} characters.")
    
    if state.get("review_feedback"):
        context_parts.append(f"Reviewer feedback: {state['review_feedback']}")
    
    context_parts.append(f"Revision count: {state.get('revision_count', 0)}")
    
    messages.append(HumanMessage(content="\n".join(context_parts)))
    
    response = llm.invoke(messages)
    decision = response.content.strip().lower()
    
    valid_decisions = ["researcher", "writer", "reviewer", "finish"]
    if decision not in valid_decisions:
        for d in valid_decisions:
            if d in decision:
                decision = d
                break
        else:
            decision = "writer"
    
    print(f"[Supervisor] Decision: {decision}")
    
    return {"next_agent": decision}
