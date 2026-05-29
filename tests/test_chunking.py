from pathlib import Path
from app.chunking import chunk_markdown, slugify


def test_slugify():
    assert slugify("Unmasking the Ascension Ritual") == "unmasking-the-ascension-ritual"


def test_chunk_markdown_splits_on_h2_and_h3():
    text = Path("tests/fixtures/sample_note.md").read_text()
    chunks = chunk_markdown(text, source_path="sessions/2025-02-04-test.md")

    assert len(chunks) == 3
    assert chunks[0].heading == "First Section"
    assert "Content of first section" in chunks[0].text
    assert chunks[0].chunk_id == "sessions/2025-02-04-test.md#first-section"

    assert chunks[1].heading == "Subsection A"
    assert chunks[1].chunk_id == "sessions/2025-02-04-test.md#subsection-a"

    assert chunks[2].heading == "Second Section"
    assert "Content of second section" in chunks[2].text
