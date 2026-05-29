from pathlib import Path

from app.config import estimate_cost_usd
from app.usage import UsageStore


def test_record_event_inserts_row(tmp_path: Path):
    db = tmp_path / "usage.sqlite"
    store = UsageStore(str(db))
    store.record_event("amazon.nova-lite-v1:0", input_tokens=100, output_tokens=50)

    events = store.get_current_month_events()
    assert len(events) == 1
    assert events[0]["input_tokens"] == 100
    assert events[0]["output_tokens"] == 50
    assert events[0]["model_id"] == "amazon.nova-lite-v1:0"
    assert events[0]["estimated_usd"] == estimate_cost_usd(
        "amazon.nova-lite-v1:0", 100, 50
    )


def test_current_month_summary(tmp_path: Path):
    db = tmp_path / "usage.sqlite"
    store = UsageStore(str(db))
    store.record_event("amazon.nova-lite-v1:0", 200, 100)
    store.record_event("amazon.nova-lite-v1:0", 300, 150)

    summary = store.get_current_month_summary()
    assert summary["request_count"] == 2
    assert summary["input_tokens"] == 500
    assert summary["output_tokens"] == 250


def test_monthly_history_empty(tmp_path: Path):
    store = UsageStore(str(tmp_path / "usage.sqlite"))
    assert store.get_monthly_history() == []


def test_rollup_moves_prior_month_to_monthly_and_deletes_events(tmp_path: Path):
    db = tmp_path / "usage.sqlite"
    store = UsageStore(str(db))

    with store._connect() as conn:
        conn.execute(
            """
            INSERT INTO usage_events
                (created_at, model_id, input_tokens, output_tokens, estimated_usd)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "2020-01-15T12:00:00+00:00",
                "amazon.nova-lite-v1:0",
                1000,
                500,
                0.18,
            ),
        )
        conn.commit()

    store.rollup_stale_months()

    history = store.get_monthly_history()
    assert len(history) == 1
    assert history[0]["year_month"] == "2020-01"
    assert history[0]["request_count"] == 1
    assert history[0]["input_tokens"] == 1000
    assert history[0]["output_tokens"] == 500

    with store._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0]
    assert count == 0
