
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
    import inspect

    from RAG.agents.graph import create_graph

    assert inspect.iscoroutinefunction(create_graph)


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
        user_id="anonymous",
        session_id="test-session",
        trace_id="test-trace",
        rewritten_query="",
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


def test_jwt_verification_with_hmac_sha256():
    import base64
    import hashlib
    import hmac
    import json
    import time

    from RAG.services.auth import verify_jwt

    secret = "test-secret"
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": "user-1", "sid": "session-1", "exp": int(time.time()) + 3600}

    def encode(data):
        raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    signing_input = f"{encode(header)}.{encode(payload)}"
    signature = hmac.new(secret.encode("utf-8"), signing_input.encode("utf-8"), hashlib.sha256).digest()
    token = f"{signing_input}.{base64.urlsafe_b64encode(signature).decode('utf-8').rstrip('=')}"

    assert verify_jwt(token, secret)["sub"] == "user-1"


def test_retrieval_candidate_round_trip():
    from RAG.services.retrieval import RetrievalCandidate

    candidate = RetrievalCandidate(
        doc_id="doc-1",
        content="content",
        metadata={"source": "source.txt"},
        dense_score=0.5,
        bm25_score=0.25,
        rerank_score=0.9,
    )

    restored = RetrievalCandidate.from_dict(candidate.to_dict())
    assert restored.doc_id == "doc-1"
    assert restored.final_score == pytest.approx(0.9)
