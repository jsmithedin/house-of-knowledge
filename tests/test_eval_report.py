"""Tests for report aggregation logic."""
from eval.report import _p, scored_answer


def test_p50_of_sorted_values():
    values = [10, 20, 30, 40, 50]
    assert _p(values, 50) == 30


def test_p95_of_short_list():
    values = [100, 200]
    p95 = _p(values, 95)
    assert p95 == 200


def test_p_empty_returns_zero():
    assert _p([], 50) == 0.0


def test_scored_answer_finds_matching_result():
    results = [
        {"query_id": "q01", "model": "nova-lite", "answer": "The answer from Nova."},
        {"query_id": "q01", "model": "haiku-4.5", "answer": "The answer from Haiku."},
    ]
    score = {"query_id": "q01", "model": "nova-lite"}
    assert scored_answer(score, results) == "The answer from Nova."


def test_scored_answer_returns_not_found_when_missing():
    result = scored_answer({"query_id": "q99", "model": "nova-lite"}, [])
    assert "not found" in result
