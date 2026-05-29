import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.config import MODEL_PRICING, estimate_cost_usd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    model_id TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    estimated_usd REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_monthly (
    year_month TEXT PRIMARY KEY,
    request_count INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    estimated_usd REAL NOT NULL
);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _current_year_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _year_month_from_iso(created_at: str) -> str:
    return created_at[:7]


class UsageStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _rollup_stale_months_conn(self, conn: sqlite3.Connection) -> None:
        current_ym = _current_year_month()
        stale = conn.execute(
            """
            SELECT created_at, input_tokens, output_tokens, estimated_usd
            FROM usage_events
            WHERE substr(created_at, 1, 7) < ?
            """,
            (current_ym,),
        ).fetchall()
        if not stale:
            return

        by_month: dict[str, dict] = {}
        for row in stale:
            ym = _year_month_from_iso(row["created_at"])
            bucket = by_month.setdefault(
                ym,
                {"request_count": 0, "input_tokens": 0, "output_tokens": 0, "estimated_usd": 0.0},
            )
            bucket["request_count"] += 1
            bucket["input_tokens"] += row["input_tokens"]
            bucket["output_tokens"] += row["output_tokens"]
            bucket["estimated_usd"] += row["estimated_usd"]

        for ym, totals in by_month.items():
            conn.execute(
                """
                INSERT INTO usage_monthly
                    (year_month, request_count, input_tokens, output_tokens, estimated_usd)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(year_month) DO UPDATE SET
                    request_count = usage_monthly.request_count + excluded.request_count,
                    input_tokens = usage_monthly.input_tokens + excluded.input_tokens,
                    output_tokens = usage_monthly.output_tokens + excluded.output_tokens,
                    estimated_usd = usage_monthly.estimated_usd + excluded.estimated_usd
                """,
                (
                    ym,
                    totals["request_count"],
                    totals["input_tokens"],
                    totals["output_tokens"],
                    totals["estimated_usd"],
                ),
            )

        conn.execute(
            "DELETE FROM usage_events WHERE substr(created_at, 1, 7) < ?",
            (current_ym,),
        )

    def rollup_stale_months(self) -> None:
        with self._connect() as conn:
            self._rollup_stale_months_conn(conn)
            conn.commit()

    def record_event(self, model_id: str, input_tokens: int, output_tokens: int) -> None:
        estimated_usd = estimate_cost_usd(model_id, input_tokens, output_tokens)
        with self._connect() as conn:
            self._rollup_stale_months_conn(conn)
            conn.execute(
                """
                INSERT INTO usage_events
                    (created_at, model_id, input_tokens, output_tokens, estimated_usd)
                VALUES (?, ?, ?, ?, ?)
                """,
                (_utc_now_iso(), model_id, input_tokens, output_tokens, estimated_usd),
            )
            conn.commit()

    def get_current_month_events(self) -> list[dict]:
        ym = _current_year_month()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT created_at, model_id, input_tokens, output_tokens, estimated_usd
                FROM usage_events
                WHERE substr(created_at, 1, 7) = ?
                ORDER BY created_at DESC
                """,
                (ym,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_current_month_summary(self) -> dict:
        ym = _current_year_month()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS request_count,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(estimated_usd), 0.0) AS estimated_usd
                FROM usage_events
                WHERE substr(created_at, 1, 7) = ?
                """,
                (ym,),
            ).fetchone()
        return dict(row)

    def get_monthly_history(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT year_month, request_count, input_tokens, output_tokens, estimated_usd
                FROM usage_monthly
                ORDER BY year_month DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def has_unknown_model_events(self) -> bool:
        events = self.get_current_month_events()
        return any(e["model_id"] not in MODEL_PRICING for e in events)
