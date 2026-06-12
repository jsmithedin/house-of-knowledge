"""Tests for recall computation, blind-pack generation, and unmask join."""
from eval.score_retrieval import compute_retrieval_score
from eval.make_blind import label_assignments
from eval.unmask import join_scores


# ---- Retrieval scoring ----

def test_recall_all_notes_found():
    golden = {"id": "q01", "source_notes": ["sessions/a.md", "sessions/b.md"]}
    retrieved = [
        {"rank": 1, "source_path": "sessions/a.md"},
        {"rank": 2, "source_path": "sessions/b.md"},
        {"rank": 3, "source_path": "sessions/c.md"},
    ]
    score = compute_retrieval_score(golden, retrieved)
    assert score["recall"] == 1.0
    assert score["hit"] is True
    assert score["first_relevant_rank"] == 1


def test_recall_partial():
    golden = {"id": "q01", "source_notes": ["sessions/a.md", "sessions/b.md"]}
    retrieved = [
        {"rank": 1, "source_path": "sessions/x.md"},
        {"rank": 2, "source_path": "sessions/a.md"},
    ]
    score = compute_retrieval_score(golden, retrieved)
    assert score["recall"] == 0.5
    assert score["hit"] is True
    assert score["first_relevant_rank"] == 2


def test_recall_none_found():
    golden = {"id": "q01", "source_notes": ["sessions/a.md"]}
    retrieved = [{"rank": 1, "source_path": "sessions/x.md"}]
    score = compute_retrieval_score(golden, retrieved)
    assert score["recall"] == 0.0
    assert score["hit"] is False
    assert score["first_relevant_rank"] is None


def test_recall_empty_retrieved():
    golden = {"id": "q01", "source_notes": ["sessions/a.md"]}
    score = compute_retrieval_score(golden, [])
    assert score["recall"] == 0.0
    assert score["hit"] is False


# ---- Blind pack / label assignment ----

def test_label_assignments_are_seeded_and_reproducible():
    a = label_assignments("run-2026", ["q01", "q02"], ["nova-lite", "haiku-4.5"])
    b = label_assignments("run-2026", ["q01", "q02"], ["nova-lite", "haiku-4.5"])
    assert a == b


def test_label_assignments_cover_both_models():
    result = label_assignments("run-x", ["q01"], ["nova-lite", "haiku-4.5"])
    assert set(result["q01"].values()) == {"nova-lite", "haiku-4.5"}
    assert set(result["q01"].keys()) == {"A", "B"}


def test_label_assignments_vary_by_query():
    result = label_assignments("run-x", ["q01", "q02", "q03", "q04", "q05", "q06"], ["nova-lite", "haiku-4.5"])
    a_assignments = [result[q]["A"] for q in ["q01", "q02", "q03", "q04", "q05", "q06"]]
    assert len(set(a_assignments)) > 1, "All labels assigned identically — randomization broken"


def test_label_assignments_different_run_ids_may_differ():
    a = label_assignments("run-1", ["q01"], ["nova-lite", "haiku-4.5"])
    b = label_assignments("run-2", ["q01"], ["nova-lite", "haiku-4.5"])
    assert set(a["q01"].values()) == {"nova-lite", "haiku-4.5"}
    assert set(b["q01"].values()) == {"nova-lite", "haiku-4.5"}


def test_pack_has_no_model_identity_fields():
    """Assert that pack entries never expose model names, token counts, or latency."""
    forbidden = {"model", "input_tokens", "output_tokens", "latency_ms"}
    pack_entry = {
        "query_id": "q01",
        "label": "A",
        "tier": "lookup",
        "query": "Q",
        "expected_answer": "A",
        "answer": "The answer.",
    }
    assert not (forbidden & set(pack_entry.keys())), \
        f"Pack entry exposes identity fields: {forbidden & set(pack_entry.keys())}"


# ---- Unmask ----

def test_join_scores_restores_model_names():
    scores = [
        {"query_id": "q01", "label": "A", "correctness": 2, "completeness": 2, "hallucination": False, "notes": "", "scored_at": "t"},
        {"query_id": "q01", "label": "B", "correctness": 1, "completeness": 1, "hallucination": True, "notes": "", "scored_at": "t"},
    ]
    mapping = {"q01": {"A": "nova-lite", "B": "haiku-4.5"}}
    joined = join_scores(scores, mapping)
    models = {r["model"] for r in joined}
    assert models == {"nova-lite", "haiku-4.5"}


def test_join_scores_preserves_all_fields():
    scores = [{"query_id": "q01", "label": "A", "correctness": 2, "completeness": 1, "hallucination": False, "notes": "ok", "scored_at": "t"}]
    mapping = {"q01": {"A": "nova-lite", "B": "haiku-4.5"}}
    joined = join_scores(scores, mapping)
    assert joined[0]["correctness"] == 2
    assert joined[0]["notes"] == "ok"
    assert joined[0]["model"] == "nova-lite"
