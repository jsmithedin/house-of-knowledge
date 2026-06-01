from unittest.mock import MagicMock

from app.bedrock import BedrockError, InvokeResult
from app.rag import RagPipeline, SYSTEM_PROMPT


def test_query_no_results():
    store = MagicMock()
    store.query.return_value = {"documents": [[]], "metadatas": [[]]}
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 8
    bedrock = MagicMock()

    pipeline = RagPipeline(
        store=store,
        embedder=embedder,
        bedrock=bedrock,
        wiki_base_url="https://example.com",
        chat_history_window=8,
    )
    answer, sources = pipeline.query("What happened?", arc=None, session=None, tag=None, k=5, history=[])
    assert "No matching notes found" in answer
    bedrock.invoke.assert_not_called()
    assert sources == ""


def test_query_with_results():
    store = MagicMock()
    store.query.return_value = {
        "documents": [["## Section\n\nLore about Ivaran."]],
        "metadatas": [[{
            "source_path": "sessions/note.md",
            "heading": "Section",
            "session": "42",
            "arc": "Test Arc",
            "date": "2025-02-04",
            "tags": "npc/ivaran",
        }]],
    }
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 8
    bedrock = MagicMock()
    bedrock.invoke.return_value = InvokeResult(
        text="Ivaran was revealed as Bergst.", input_tokens=1, output_tokens=1
    )

    pipeline = RagPipeline(
        store=store,
        embedder=embedder,
        bedrock=bedrock,
        wiki_base_url="https://example.com",
        chat_history_window=8,
    )
    answer, sources = pipeline.query("Who is Ivaran?", arc=None, session=None, tag=None, k=5, history=[])
    assert "Ivaran was revealed" in answer
    assert "Sources" in sources
    bedrock.invoke.assert_called_once()


def test_query_logs_usage_on_success():
    store = MagicMock()
    store.query.return_value = {
        "documents": [["Lore."]],
        "metadatas": [[{
            "source_path": "sessions/note.md",
            "heading": "Section",
            "session": "42",
            "arc": "Test Arc",
            "date": "2025-02-04",
            "tags": "",
        }]],
    }
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 8
    bedrock = MagicMock()
    bedrock.model_id = "amazon.nova-lite-v1:0"
    bedrock.invoke.return_value = InvokeResult(text="Answer", input_tokens=10, output_tokens=5)
    usage = MagicMock()

    pipeline = RagPipeline(
        store=store,
        embedder=embedder,
        bedrock=bedrock,
        wiki_base_url="https://example.com",
        chat_history_window=8,
        usage_store=usage,
    )
    pipeline.query("Q?", arc=None, session=None, tag=None, k=5, history=[])
    usage.record_event.assert_called_once_with("amazon.nova-lite-v1:0", 10, 5)


def test_query_skips_usage_on_bedrock_error():
    store = MagicMock()
    store.query.return_value = {
        "documents": [["Lore."]],
        "metadatas": [[{
            "source_path": "sessions/note.md",
            "heading": "Section",
            "session": "42",
            "arc": "Test Arc",
            "date": "2025-02-04",
            "tags": "",
        }]],
    }
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 8
    bedrock = MagicMock()
    bedrock.invoke.side_effect = BedrockError("fail")
    usage = MagicMock()

    pipeline = RagPipeline(
        store=store,
        embedder=embedder,
        bedrock=bedrock,
        wiki_base_url="https://example.com",
        chat_history_window=8,
        usage_store=usage,
    )
    answer, _ = pipeline.query("Q?", arc=None, session=None, tag=None, k=5, history=[])
    assert "Generation failed" in answer
    usage.record_event.assert_not_called()


# ── Agentic-path tests ────────────────────────────────────────────────────────

from app.rag import SEARCH_TOOL  # noqa: E402 — import here after SEARCH_TOOL exists


_META_AGENTIC = {
    "source_path": "sessions/note.md",
    "heading": "Section",
    "session": "42",
    "arc": "Test Arc",
    "date": "2025-02-04",
    "tags": "npc/ivaran",
}

_META_AGENTIC2 = {
    "source_path": "sessions/note2.md",
    "heading": "Other",
    "session": "43",
    "arc": "Test Arc",
    "date": "2025-03-01",
    "tags": "",
}


def _make_agentic_pipeline(store, bedrock, usage_store=None):
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 8
    return RagPipeline(
        store=store,
        embedder=embedder,
        bedrock=bedrock,
        wiki_base_url="https://example.com",
        chat_history_window=8,
        usage_store=usage_store,
    )


def _fake_invoke_with_tools(search_calls):
    """
    Returns a side_effect for bedrock.invoke_with_tools.
    search_calls: list of (query, k) — the searches the 'model' will make.
    """
    def side_effect(system_prompt, initial_messages, tools, tool_executor, max_iterations=5):
        for query, k in search_calls:
            tool_executor("search_knowledge_base", {"query": query, "k": k})
        return InvokeResult(text="Agentic answer.", input_tokens=50, output_tokens=20)
    return side_effect


def test_agentic_query_calls_invoke_with_tools():
    """agentic=True calls invoke_with_tools, not invoke."""
    store = MagicMock()
    store.query.return_value = {"documents": [["Lore."]], "metadatas": [[_META_AGENTIC]]}
    bedrock = MagicMock()
    bedrock.invoke_with_tools.side_effect = _fake_invoke_with_tools([("Ivaran", 5)])

    pipeline = _make_agentic_pipeline(store, bedrock)
    answer, _ = pipeline.query("Who is Ivaran?", arc=None, session=None, tag=None, k=5, history=[], agentic=True)

    assert "Agentic answer." in answer
    bedrock.invoke_with_tools.assert_called_once()
    bedrock.invoke.assert_not_called()


def test_standard_query_does_not_call_invoke_with_tools():
    """agentic=False (default) uses invoke, not invoke_with_tools."""
    store = MagicMock()
    store.query.return_value = {"documents": [["Lore."]], "metadatas": [[_META_AGENTIC]]}
    bedrock = MagicMock()
    bedrock.invoke.return_value = InvokeResult(text="Standard answer.", input_tokens=10, output_tokens=5)

    pipeline = _make_agentic_pipeline(store, bedrock)
    answer, _ = pipeline.query("Who is Ivaran?", arc=None, session=None, tag=None, k=5, history=[])

    assert "Standard answer." in answer
    bedrock.invoke.assert_called_once()
    bedrock.invoke_with_tools.assert_not_called()


def test_agentic_query_passes_search_tool_definition():
    """The search_knowledge_base tool is included in the tools list passed to Bedrock."""
    store = MagicMock()
    store.query.return_value = {"documents": [["Lore."]], "metadatas": [[_META_AGENTIC]]}
    bedrock = MagicMock()
    bedrock.invoke_with_tools.return_value = InvokeResult(text="Answer.", input_tokens=10, output_tokens=5)

    pipeline = _make_agentic_pipeline(store, bedrock)
    pipeline.query("Q?", arc=None, session=None, tag=None, k=5, history=[], agentic=True)

    call_args = bedrock.invoke_with_tools.call_args
    tools = call_args[1].get("tools") or call_args[0][2]
    assert any(t["name"] == "search_knowledge_base" for t in tools)


def test_agentic_query_sources_span_all_tool_calls():
    """Citations include chunks from every search the model made."""
    store = MagicMock()
    store.query.side_effect = [
        {"documents": [["Lore about Ivaran."]], "metadatas": [[_META_AGENTIC]]},
        {"documents": [["Guild conflict."]], "metadatas": [[_META_AGENTIC2]]},
    ]
    bedrock = MagicMock()
    bedrock.invoke_with_tools.side_effect = _fake_invoke_with_tools([
        ("Ivaran", 5),
        ("merchant guild", 5),
    ])

    pipeline = _make_agentic_pipeline(store, bedrock)
    _, sources = pipeline.query("What happened?", arc=None, session=None, tag=None, k=5, history=[], agentic=True)

    assert "Session 42" in sources
    assert "Session 43" in sources


def test_agentic_query_applies_user_filters():
    """Arc/session/tag filters are forwarded to every store.query call."""
    store = MagicMock()
    store.query.return_value = {"documents": [["Lore."]], "metadatas": [[_META_AGENTIC]]}
    bedrock = MagicMock()
    bedrock.invoke_with_tools.side_effect = _fake_invoke_with_tools([("Q", 5)])

    pipeline = _make_agentic_pipeline(store, bedrock)
    pipeline.query("Q?", arc="Arc 1", session=None, tag=None, k=5, history=[], agentic=True)

    _, store_kwargs = store.query.call_args
    where = store_kwargs.get("where")
    assert where is not None
    assert "Arc 1" in str(where)


def test_agentic_query_caps_k_at_10():
    """Model cannot request more than 10 chunks."""
    store = MagicMock()
    store.query.return_value = {"documents": [["Lore."]], "metadatas": [[_META_AGENTIC]]}
    bedrock = MagicMock()

    def side_effect(system_prompt, initial_messages, tools, tool_executor, max_iterations=5):
        tool_executor("search_knowledge_base", {"query": "Q", "k": 99})
        return InvokeResult(text="Answer.", input_tokens=10, output_tokens=5)

    bedrock.invoke_with_tools.side_effect = side_effect

    pipeline = _make_agentic_pipeline(store, bedrock)
    pipeline.query("Q?", arc=None, session=None, tag=None, k=5, history=[], agentic=True)

    _, store_kwargs = store.query.call_args
    assert store_kwargs.get("n_results") <= 10


def test_agentic_query_defaults_k_from_pipeline_when_omitted():
    """If model omits k, the pipeline k value is used."""
    store = MagicMock()
    store.query.return_value = {"documents": [["Lore."]], "metadatas": [[_META_AGENTIC]]}
    bedrock = MagicMock()

    def side_effect(system_prompt, initial_messages, tools, tool_executor, max_iterations=5):
        tool_executor("search_knowledge_base", {"query": "Q"})  # no k
        return InvokeResult(text="Answer.", input_tokens=10, output_tokens=5)

    bedrock.invoke_with_tools.side_effect = side_effect

    pipeline = _make_agentic_pipeline(store, bedrock)
    pipeline.query("Q?", arc=None, session=None, tag=None, k=7, history=[], agentic=True)

    _, store_kwargs = store.query.call_args
    assert store_kwargs.get("n_results") == 7


def test_agentic_query_records_usage():
    """Accumulated token counts are recorded in usage_store."""
    store = MagicMock()
    store.query.return_value = {"documents": [["Lore."]], "metadatas": [[_META_AGENTIC]]}
    bedrock = MagicMock()
    bedrock.model_id = "claude-sonnet-4-6"
    bedrock.invoke_with_tools.side_effect = _fake_invoke_with_tools([("Q", 5)])
    usage = MagicMock()

    pipeline = _make_agentic_pipeline(store, bedrock, usage_store=usage)
    pipeline.query("Q?", arc=None, session=None, tag=None, k=5, history=[], agentic=True)

    usage.record_event.assert_called_once_with("claude-sonnet-4-6", 50, 20)


def test_agentic_query_skips_usage_on_error():
    """BedrockError from invoke_with_tools — usage not recorded."""
    bedrock = MagicMock()
    bedrock.invoke_with_tools.side_effect = BedrockError("fail")
    usage = MagicMock()

    pipeline = _make_agentic_pipeline(MagicMock(), bedrock, usage_store=usage)
    answer, _ = pipeline.query("Q?", arc=None, session=None, tag=None, k=5, history=[], agentic=True)

    assert "Generation failed" in answer
    usage.record_event.assert_not_called()
