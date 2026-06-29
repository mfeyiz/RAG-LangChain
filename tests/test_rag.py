
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


def test_heading_aware_chunking_preserves_section_context():
    from langchain_core.documents import Document
    from RAG.services.rag_service import semantic_chunk_documents

    markdown = (
        "# Information Security\n\n"
        "Intro paragraph about security.\n\n"
        "## Threats\n\n"
        "Threats include malware.\n\n"
        "### Malware\n\n"
        "Malware is harmful software.\n\n"
        "![A virus diagram](images/sample/figure1.png)\n"
    )
    doc = Document(
        page_content=markdown,
        metadata={"source": "sample.md", "kind": "markdown", "title": "sample"},
    )

    chunks = semantic_chunk_documents([doc])
    assert chunks, "expected at least one chunk"

    # Every chunk carries the new chunking method and original doc metadata.
    assert all(c.metadata["chunking"] == "heading-aware-markdown" for c in chunks)
    assert all(c.metadata["source"] == "sample.md" for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))

    # The deepest section keeps its full heading path in metadata and content.
    malware = next(c for c in chunks if c.metadata.get("h3") == "Malware")
    assert malware.metadata["h1"] == "Information Security"
    assert malware.metadata["h2"] == "Threats"
    assert malware.metadata["heading_path"] == "Information Security > Threats > Malware"
    assert malware.page_content.startswith("Information Security > Threats > Malware")
    # The figure in that section is anchored to the chunk for text-anchored retrieval.
    assert malware.metadata["images"] == ["figure1.png"]


def test_chunking_handles_markdown_without_headings():
    from langchain_core.documents import Document
    from RAG.services.rag_service import semantic_chunk_documents

    doc = Document(
        page_content="Just a flat paragraph with no headings at all.",
        metadata={"source": "flat.md", "kind": "markdown", "title": "flat"},
    )
    chunks = semantic_chunk_documents([doc])
    assert len(chunks) == 1
    assert chunks[0].metadata["images"] == []
    assert "heading_path" not in chunks[0].metadata


def test_image_path_resolves_and_guards_traversal(tmp_path, monkeypatch):
    from RAG.services import paths

    monkeypatch.setattr(paths, "WORKSPACE_IMG_DIR", tmp_path / "ws_img")
    monkeypatch.setattr(paths, "_CHANNEL_IMG_DIR", {"workspace": paths.WORKSPACE_IMG_DIR})

    doc_dir = paths.WORKSPACE_IMG_DIR / "report"
    doc_dir.mkdir(parents=True)
    (doc_dir / "figure1.png").write_bytes(b"\x89PNG\r\n")

    assert paths.image_path("workspace", "report", "figure1.png") is not None
    assert paths.image_path("workspace", "report", "missing.png") is None
    # Traversal attempts collapse to a basename and stay inside the doc dir.
    assert paths.image_path("workspace", "report", "../../secret.png") is None


def test_writer_image_blocks_from_query_and_context(tmp_path, monkeypatch):
    from RAG.agents import writer
    from RAG.services import paths

    monkeypatch.setattr(paths, "WORKSPACE_IMG_DIR", tmp_path / "ws_img")
    monkeypatch.setattr(paths, "_CHANNEL_IMG_DIR", {"workspace": paths.WORKSPACE_IMG_DIR})
    doc_dir = paths.WORKSPACE_IMG_DIR / "lec"
    doc_dir.mkdir(parents=True)
    (doc_dir / "fig.png").write_bytes(b"\x89PNG\r\nfakepng")

    state = {
        "query_images": ["data:image/png;base64,AAAA"],
        "context_images": [{"source": "lec.md", "name": "fig.png", "channel": "workspace"}],
    }
    blocks = writer._image_blocks(state)
    assert len(blocks) == 2
    assert all(b["type"] == "image_url" for b in blocks)
    assert blocks[0]["image_url"]["url"] == "data:image/png;base64,AAAA"
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")


# ── Dual-channel (bi-directional) RAG ──────────────────────────────────────────

def test_channel_configuration():
    from RAG.services.retrieval import (
        CHANNELS,
        COLLECTION_NAME,
        channel_collection,
        channel_corpus_path,
    )

    assert set(CHANNELS) == {"originals", "workspace"}
    assert channel_collection("originals") == "rag_originals"
    assert channel_collection("workspace") == "rag_workspace"
    # Distinct collections and distinct corpus files per channel.
    assert channel_collection("originals") != channel_collection("workspace")
    assert channel_corpus_path("originals") != channel_corpus_path("workspace")
    # Backward-compatible default alias points at the workspace channel.
    assert COLLECTION_NAME == "rag_workspace"


def test_dual_channel_retrievers_are_isolated():
    from RAG.services.retrieval import get_retriever

    originals = get_retriever("originals")
    workspace = get_retriever("workspace")

    assert originals is not workspace
    assert originals.collection_name == "rag_originals"
    assert workspace.collection_name == "rag_workspace"
    # Repeated calls return the same per-channel singleton.
    assert get_retriever("workspace") is workspace


def test_supervisor_routes_update_to_editor():
    import asyncio

    from RAG.agents.supervisor import supervisor_node

    state = {
        "query": "@update Vergi oranı bilgisini %20 olarak güncelle",
        "trace_id": "test-trace-editor",
        "research_results": "",
        "draft_response": "",
        "final_response": "",
        "review_feedback": "",
        "revision_count": 0,
    }
    result = asyncio.run(supervisor_node(state))
    assert result["next_agent"] == "editor"


def test_supervisor_routes_normal_query_to_researcher():
    import asyncio

    from RAG.agents.supervisor import supervisor_node

    state = {
        "query": "Şirketin kuruluş tarihi ve merkezi neresidir?",
        "trace_id": "test-trace-research",
        "research_results": "",
        "draft_response": "",
        "final_response": "",
        "review_feedback": "",
        "revision_count": 0,
    }
    result = asyncio.run(supervisor_node(state))
    assert result["next_agent"] == "researcher"


def test_editor_strip_update_tag_and_clean_markdown():
    from RAG.agents.editor import strip_update_tag, _clean_markdown

    assert strip_update_tag("@update yeni bilgi ekle") == "yeni bilgi ekle"
    assert strip_update_tag("Lütfen @update bunu güncelle") == "Lütfen  bunu güncelle".strip()
    # Accidental code fences from the LLM are stripped.
    assert _clean_markdown("```markdown\n# Başlık\n\nMetin\n```") == "# Başlık\n\nMetin"
    assert _clean_markdown("# Plain") == "# Plain"


def test_converter_rejects_unsupported_type(tmp_path):
    from RAG.services.converter import convert_to_markdown, ConversionError

    bad = tmp_path / "note.rtf"
    bad.write_text("hello", encoding="utf-8")
    with pytest.raises(ConversionError):
        convert_to_markdown(bad)


def test_paths_source_mapping():
    from RAG.services import paths

    source = paths.source_for(paths.sanitize_stem("My Report (v2).pdf"))
    assert source.endswith(".md")
    assert paths.workspace_md_path(source).parent == paths.WORKSPACE_MD_DIR
    assert paths.workspace_pdf_path(source).suffix == ".pdf"
    # Sanitization keeps the stem filesystem/URL friendly.
    assert " " not in source and "(" not in source


def test_pdf_renderer_generates_pdf(tmp_path, monkeypatch):
    pytest.importorskip("markdown")
    # WeasyPrint needs native libs (Pango/Cairo/gobject); importing without them
    # raises OSError, which importorskip does not catch — skip those too.
    try:
        import weasyprint  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"weasyprint native libraries unavailable: {exc}")

    from RAG.services import paths, pdf_renderer

    # Redirect storage into a temp dir so the test never writes into the repo.
    monkeypatch.setattr(paths, "WORKSPACE_MD_DIR", tmp_path / "ws_md")
    monkeypatch.setattr(paths, "WORKSPACE_PDF_DIR", tmp_path / "ws_pdf")
    monkeypatch.setattr(paths, "_ALL_DIRS", (paths.WORKSPACE_MD_DIR, paths.WORKSPACE_PDF_DIR))

    source = "sample.md"
    md_path = paths.workspace_md_path(source)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("# Title\n\nHello **world**.\n", encoding="utf-8")

    out = pdf_renderer.render(source)
    assert out.exists()
    assert out.read_bytes()[:4] == b"%PDF"
 