from unittest.mock import MagicMock
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
    bedrock.invoke.return_value = "Ivaran was revealed as Bergst."

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
