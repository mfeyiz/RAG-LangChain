
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_api_key_config():
    key = os.getenv("OPENROUTER_API_KEY")
    if key is None:
        pytest.skip("OPENROUTER_API_KEY is only required for live LLM integration tests.")
    assert key is not None, "API Key not defined as secret!"
    assert key.startswith("sk-"), "API Key appears to be in wrong format!"


def test_setup_works():
    assert True


def test_graph_creation():
    from RAG.agents.graph import create_graph
    graph = create_graph()
    assert graph is not None


def test_agent_state_structure():
    from RAG.agents.state import AgentState
    state = AgentState(
        messages=[],
        next_agent="supervisor",
        query="test",
        research_results="",
        draft_response="",
        final_response="",
        review_feedback="",
        revision_count=0,
        search_metadata=[],
    )
    assert state["query"] == "test"
    assert state["next_agent"] == "supervisor"


def test_fastapi_app_import():
    from RAG.app import app
    assert app is not None


def test_reviewer_json_parse():
    from RAG.agents.reviewer import _parse_review

    decision = _parse_review('{"approved": true, "feedback": "", "unsupported_claims": []}')
    assert decision["approved"] is True

    rejected = _parse_review('{"approved": false, "feedback": "Missing citation."}')
    assert rejected["approved"] is False


def test_retrieval_metadata_shape():
    from RAG.services.retrieval import RetrievalCandidate, search_metadata

    metadata = search_metadata(
        [
            RetrievalCandidate(
                doc_id="1",
                content="Arthur's Magazine was published earlier.",
                metadata={"source": "sample.json", "title": "Arthur's Magazine", "kind": "context"},
                dense_score=0.8,
                bm25_score=0.4,
            )
        ]
    )
    assert metadata[0]["source"] == "sample.json"
    assert metadata[0]["title"] == "Arthur's Magazine"
    assert metadata[0]["score"] == pytest.approx(1.2)
