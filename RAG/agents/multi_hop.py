"""Multi-hop query decomposition and chaining."""
from langchain_core.messages import HumanMessage, SystemMessage
from RAG.agents.supervisor import get_llm
from RAG.services.tracing import invoke_with_langfuse


DECOMPOSE_PROMPT = """Analyze the user's question and decompose it into sub-questions if it requires multiple reasoning steps.

Rules:
1. If the question is simple (can be answered directly), return: {"type": "single", "query": "<original query>"}
2. If the question has multiple hops (e.g., "X film's director's birthplace"), decompose into steps:
   - Return: {"type": "multi_hop", "steps": [{"number": 1, "query": "..."}, {"number": 2, "query": "...", "uses_answer_from": 1}]}
3. Each step after the first should reference which previous step's answer it needs
4. Be specific about what information each step extracts

Return ONLY valid JSON."""


ANSWER_EXTRACTION_PROMPT = """From the provided context, extract the specific information needed to answer the query.

Query: {query}
Context: {context}

Extract the key entity or fact that will be used in the next step of reasoning.
Return ONLY the extracted information as a concise string."""


async def decompose_question(query: str) -> dict:
    """Decompose a question into sub-questions if needed."""
    try:
        llm = get_llm()
        response = await invoke_with_langfuse(
            llm,
            [
                SystemMessage(content=DECOMPOSE_PROMPT),
                HumanMessage(content=query),
            ],
        )
        
        import json
        result = json.loads(response.content.strip())
        return result
    except Exception as exc:
        print(f"[MultiHop] Question decomposition failed: {exc}")
        return {"type": "single", "query": query}


async def extract_answer_for_next_step(query: str, context: str) -> str:
    """Extract relevant answer from context for use in next query step."""
    try:
        llm = get_llm()
        prompt = ANSWER_EXTRACTION_PROMPT.format(query=query, context=context[:2000])
        
        response = await invoke_with_langfuse(
            llm,
            [
                SystemMessage(content="You are an information extractor."),
                HumanMessage(content=prompt),
            ],
        )
        
        return response.content.strip()
    except Exception as exc:
        print(f"[MultiHop] Answer extraction failed: {exc}")
        return ""


async def build_chained_query(original_query: str, previous_answer: str, next_step: dict) -> str:
    """Build the next query incorporating the previous answer."""
    try:
        llm = get_llm()
        prompt = f"""The previous step found: "{previous_answer}"
        
Now answer this related question: "{next_step['query']}"
Use the information from the previous step in your search query."""
        
        response = await invoke_with_langfuse(
            llm,
            [
                SystemMessage(content="You are a query refinement agent."),
                HumanMessage(content=prompt),
            ],
        )
        
        return response.content.strip()
    except Exception as exc:
        print(f"[MultiHop] Query chaining failed: {exc}")
        return next_step['query']
