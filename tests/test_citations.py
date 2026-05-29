from app.citations import build_wiki_url, format_sources_section, resolve_wikilinks


def test_build_wiki_url():
    url = build_wiki_url(
        "https://wordlewarriors.github.io/normal-door-opening",
        "sessions/2025-02-04-unmasking.md",
    )
    assert url == "https://wordlewarriors.github.io/normal-door-opening/sessions/2025-02-04-unmasking"


def test_format_sources_section():
    chunks = [
        {"session": "42", "heading": "Unmasking the Ritual", "source_path": "sessions/2025-02-04-unmasking.md"},
        {"session": "38", "heading": "Exploring the Manor", "source_path": "sessions/2024-12-03-manor.md"},
    ]
    md = format_sources_section(
        "https://wordlewarriors.github.io/normal-door-opening", chunks
    )
    assert "**Sources:**" in md
    assert "Session 42 — Unmasking the Ritual" in md
    assert "sessions/2025-02-04-unmasking" in md
    assert "Session 38 — Exploring the Manor" in md


def test_resolve_wikilinks_simple():
    result = resolve_wikilinks("See [[Thornvale]] for more.", "https://example.com/wiki")
    assert result == "See [Thornvale](https://example.com/wiki/Thornvale) for more."


def test_resolve_wikilinks_with_label():
    result = resolve_wikilinks("Visit [[Thornvale|the village]].", "https://example.com/wiki")
    assert result == "Visit [the village](https://example.com/wiki/Thornvale)."


def test_resolve_wikilinks_no_wikilinks():
    text = "No links here."
    assert resolve_wikilinks(text, "https://example.com/wiki") == text
