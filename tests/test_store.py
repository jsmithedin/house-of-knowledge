import tempfile
from app.store import NoteStore


def test_upsert_and_query():
    with tempfile.TemporaryDirectory() as tmp:
        store = NoteStore(chroma_dir=tmp, collection_name="test")
        store.upsert_chunks(
            ids=["note.md#section"],
            documents=["## Section\n\nSome lore about Ivaran."],
            metadatas=[{
                "source_path": "sessions/note.md",
                "heading": "Section",
                "session": "42",
                "arc": "Test Arc",
                "date": "2025-02-04",
                "tags": "npc/ivaran,location/manor",
            }],
            embeddings=[[0.1] * 8],  # dummy embedding
        )
        results = store.query(
            query_embedding=[0.1] * 8,
            n_results=1,
            where={"session": "42"},
        )
        assert results["documents"][0][0].startswith("## Section")
        assert results["metadatas"][0][0]["arc"] == "Test Arc"


def test_distinct_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        store = NoteStore(chroma_dir=tmp, collection_name="test2")
        store.upsert_chunks(
            ids=["a#h", "b#h"],
            documents=["doc a", "doc b"],
            metadatas=[
                {"source_path": "a.md", "heading": "H", "session": "1", "arc": "Arc A", "date": "2025-01-01", "tags": "npc/a"},
                {"source_path": "b.md", "heading": "H", "session": "2", "arc": "Arc B", "date": "2025-02-01", "tags": "npc/b"},
            ],
            embeddings=[[0.1] * 8, [0.2] * 8],
        )
        distinct = store.get_distinct_metadata()
        assert set(distinct["arcs"]) == {"Arc A", "Arc B"}
        assert set(distinct["sessions"]) == {"1", "2"}
        assert set(distinct["tags"]) == {"npc/a", "npc/b"}
