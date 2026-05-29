from app.filters import build_where_clause


def test_no_filters_returns_none():
    assert build_where_clause(None, None, None) is None


def test_single_arc_filter():
    assert build_where_clause("Test Arc", None, None) == {"arc": "Test Arc"}


def test_session_and_tag_and_combined():
    result = build_where_clause("Test Arc", "42", "npc/ivaran")
    assert result == {
        "$and": [
            {"arc": "Test Arc"},
            {"session": "42"},
            {"tags": {"$contains": "npc/ivaran"}},
        ]
    }


def test_partial_filters_and_combined():
    result = build_where_clause(None, "42", "npc/ivaran")
    assert result == {
        "$and": [
            {"session": "42"},
            {"tags": {"$contains": "npc/ivaran"}},
        ]
    }
