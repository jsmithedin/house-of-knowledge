import tempfile
from pathlib import Path
from scripts.index_notes import index_vault


SAMPLE = """---
session: 42
arc: "Test Arc"
date: 2025-02-04
tags:
  - npc/ivaran
---
# Title

## First Section

Lore content here.
"""


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]


def test_index_vault_creates_chunks():
    with tempfile.TemporaryDirectory() as obs_dir, tempfile.TemporaryDirectory() as chroma_dir:
        note_path = Path(obs_dir) / "sessions" / "test.md"
        note_path.parent.mkdir(parents=True)
        note_path.write_text(SAMPLE)

        stats = index_vault(
            obsidian_dir=obs_dir,
            chroma_dir=chroma_dir,
            embedder=FakeEmbedder(),
            collection_name="test_index",
        )
        assert stats["indexed_chunks"] == 1
        assert stats["skipped_files"] == 0
