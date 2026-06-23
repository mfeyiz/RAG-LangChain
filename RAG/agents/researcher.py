import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from RAG.agents.multi_hop import decompose_question, extract_answer_for_next_step, build_chained_query
from RAG.agents.state import AgentState
from RAG.agents.supervisor import get_llm
from RAG.services.retrieval import FINAL_CONTEXT_K, retrieve_context_async
from RAG.services.tracing import invoke_with_langfuse, trace_event, traced_observation

# Set RAG_ENABLE_MULTIHOP=1 to activate multi-hop decomposition (adds ~30s per query).
_MULTIHOP_ENABLED = os.getenv("RAG_ENABLE_MULTIHOP", "0") == "1"
# Rewrite query only when it is long enough to benefit from reformulation.
_REWRITE_MIN_WORDS = int(os.getenv("RAG_REWRITE_MIN_WORDS", "7"))

QUERY_REWRITE_PROMPT = """Rewrite the user's question into a concise retrieval query.

Rules:
- Preserve names, dates, entities, and comparison targets.
- Expand pronouns only when the question itself provides the referent.
- Return only the rewritten query.
- Use the same language as the user unless proper nouns require otherwise."""


async def researcher_node(state: AgentState) -> dict:
    original_query = state["query"]

    with traced_observation("researcher", input_payload={"query": original_query}) as span:
        if _MULTIHOP_ENABLED:
            decomposed = await decompose_question(original_query)
            if decomposed.get("type") == "multi_hop":
                return await _multi_hop_retrieve(state, decomposed, span)

        return await _single_hop_retrieve(state, original_query, span)


async def _single_hop_retrieve(state: AgentState, original_query: str, span) -> dict:
    if len(original_query.split()) >= _REWRITE_MIN_WORDS:
        rewritten_query = await rewrite_query(original_query)
    else:
        rewritten_query = original_query
    context, metadata = await retrieve_context_async(rewritten_query, top_k=FINAL_CONTEXT_K)
    result_count = len(metadata)

    await trace_event(
        state["trace_id"],
        "researcher.retrieval",
        {
            "query": original_query,
            "rewritten_query": rewritten_query,
            "result_count": result_count,
            "results": metadata,
        },
    )
    span.update(output={"rewritten_query": rewritten_query, "result_count": result_count})

    print(f"\n[Researcher] Query: {original_query}")
    print(f"[Researcher] Rewritten: {rewritten_query}")
    print(f"[Researcher] {result_count} context chunks selected.")

    return {
        "research_results": context,
        "search_metadata": metadata,
        "rewritten_query": rewritten_query,
        "hop_steps": [],
        "hop_context": "",
        "messages": [
            AIMessage(content=f"Research completed: {result_count} chunks found for query: {rewritten_query}")
        ],
    }


async def _multi_hop_retrieve(state: AgentState, decomposed: dict, span) -> dict:
    steps = decomposed.get("steps", [])
    print(f"\n[Researcher] Multi-hop query detected — {len(steps)} steps.")

    all_metadata: list[dict] = []
    context_blocks: list[str] = []
    previous_answer = ""
    completed_steps: list[dict] = []

    for step in steps:
        step_num = step.get("number", 0)
        step_query = step.get("query", "")

        if step.get("uses_answer_from") and previous_answer:
            step_query = await build_chained_query(state["query"], previous_answer, step)

        print(f"[Researcher] Hop {step_num}: {step_query}")
        context, metadata = await retrieve_context_async(step_query, top_k=FINAL_CONTEXT_K)
        all_metadata.extend(metadata)

        hop_answer = await extract_answer_for_next_step(step_query, context)
        previous_answer = hop_answer

        context_blocks.append(f"### Step {step_num}: {step_query}\n{context}")
        completed_steps.append({"number": step_num, "query": step_query, "answer": hop_answer})

        await trace_event(
            state["trace_id"],
            f"researcher.hop.{step_num}",
            {"query": step_query, "result_count": len(metadata), "extracted_answer": hop_answer},
        )

    aggregated_context = "\n\n".join(context_blocks)
    span.update(output={"hops": len(steps), "total_chunks": len(all_metadata)})

    return {
        "research_results": aggregated_context,
        "search_metadata": all_metadata,
        "rewritten_query": state["query"],
        "hop_steps": completed_steps,
        "hop_context": aggregated_context,
        "messages": [
            AIMessage(content=f"Multi-hop research completed: {len(steps)} hops, {len(all_metadata)} chunks.")
        ],
    }


async def rewrite_query(query: str) -> str:
    try:
        llm = get_llm()
        response = await invoke_with_langfuse(
            llm,
            [
                SystemMessage(content=QUERY_REWRITE_PROMPT),
                HumanMessage(content=query),
            ],
        )
        rewritten = response.content.strip().strip('"')
        return rewritten or query
    except Exception as exc:
        print(f"[Researcher] Query rewrite skipped: {exc}")
        return query
