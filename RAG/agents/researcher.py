from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from RAG.agents.state import AgentState
from RAG.agents.supervisor import get_llm
from RAG.services.retrieval import FINAL_CONTEXT_K, retrieve_context


QUERY_REWRITE_PROMPT = """Rewrite the user's question into a concise retrieval query.

Rules:
- Preserve names, dates, entities, and comparison targets.
- Expand pronouns only when the question itself provides the referent.
- Return only the rewritten query.
- Use the same language as the user unless proper nouns require otherwise."""


def researcher_node(state: AgentState) -> dict:
    original_query = state["query"]
    rewritten_query = rewrite_query(original_query)

    context, metadata = retrieve_context(rewritten_query, top_k=FINAL_CONTEXT_K)
    result_count = len(metadata)

    print(f"\n[Researcher] Query: {original_query}")
    print(f"[Researcher] Rewritten: {rewritten_query}")
    print(f"[Researcher] {result_count} context chunks selected.")

    return {
        "research_results": context,
        "search_metadata": metadata,
        "messages": [
            AIMessage(
                content=(
                    "Research completed: "
                    f"{result_count} chunks found for query: {rewritten_query}"
                )
            )
        ],
    }


def rewrite_query(query: str) -> str:
    try:
        llm = get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=QUERY_REWRITE_PROMPT),
                HumanMessage(content=query),
            ]
        )
        rewritten = response.content.strip().strip('"')
        return rewritten or query
    except Exception as exc:
        print(f"[Researcher] Query rewrite skipped: {exc}")
        return query
