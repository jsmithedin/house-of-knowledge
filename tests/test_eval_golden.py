"""Tests for golden-set validation logic extracted from eval/runner.py."""
import json
from pathlib import Path

from eval.runner import load_golden_set, validate_golden_set


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_golden_set_returns_all_records(tmp_path):
    records = [
        {"id": "q01", "tier": "lookup", "query": "Q1", "expected_answer": "A1", "source_notes": ["a.md"]},
        {"id": "q02", "tier": "synthesis", "query": "Q2", "expected_answer": "A2", "source_notes": ["b.md"]},
    ]
    p = tmp_path / "golden_set.jsonl"
    _write_jsonl(p, records)
    loaded = load_golden_set(tmp_path)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "q01"
    assert loaded[1]["id"] == "q02"


def test_load_golden_set_skips_blank_lines(tmp_path):
    p = tmp_path / "golden_set.jsonl"
    p.write_text('{"id": "q01", "tier": "lookup", "query": "Q", "expected_answer": "A", "source_notes": ["x.md"]}\n\n\n')
    loaded = load_golden_set(tmp_path)
    assert len(loaded) == 1


def test_validate_warns_on_duplicate_ids(tmp_path, capsys):
    records = [
        {"id": "q01", "tier": "lookup", "query": "Q", "expected_answer": "A", "source_notes": ["a.md"]},
        {"id": "q01", "tier": "lookup", "query": "Q2", "expected_answer": "A2", "source_notes": ["b.md"]},
    ]
    validate_golden_set(records, str(tmp_path))
    captured = capsys.readouterr()
    assert "duplicate" in captured.err.lower()


def test_validate_warns_on_invalid_tier(tmp_path, capsys):
    records = [{"id": "q01", "tier": "INVALID", "query": "Q", "expected_answer": "A", "source_notes": ["a.md"]}]
    validate_golden_set(records, str(tmp_path))
    captured = capsys.readouterr()
    assert "tier" in captured.err.lower()


def test_validate_warns_on_missing_source_note(tmp_path, capsys):
    records = [{"id": "q01", "tier": "lookup", "query": "Q", "expected_answer": "A", "source_notes": ["missing.md"]}]
    validate_golden_set(records, str(tmp_path))
    captured = capsys.readouterr()
    assert "missing.md" in captured.err or "not found" in captured.err.lower()


def test_validate_does_not_warn_when_note_exists(tmp_path, capsys):
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "note.md").write_text("content")
    records = [{"id": "q01", "tier": "lookup", "query": "Q", "expected_answer": "A", "source_notes": ["sessions/note.md"]}]
    validate_golden_set(records, str(tmp_path))
    captured = capsys.readouterr()
    assert "not found" not in captured.err.lower()
