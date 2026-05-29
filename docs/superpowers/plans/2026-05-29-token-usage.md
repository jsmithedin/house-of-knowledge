# Token Usage Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log Bedrock input/output tokens to SQLite per chat answer, roll up completed UTC calendar months, and show current-month detail plus historical summaries on a Streamlit page.

**Architecture:** `BedrockClient` returns `InvokeResult` with token counts from the Converse response. `RagPipeline` calls `UsageStore.record_event()` after success (errors swallowed). `UsageStore` handles schema, rollup, and queries. Streamlit multipage adds `app/pages/usage.py` beside existing `app/main.py` chat home.

**Tech Stack:** Python 3.11, sqlite3 (stdlib), Streamlit, boto3, pytest

**Spec:** `docs/superpowers/specs/2026-05-29-token-usage-design.md`

**Implement in:** `.worktrees/dnd-rag/` (dnd-rag app worktree)

---

## File Map

| File | Responsibility |
|------|----------------|
| `app/config.py` | Add `usage_db_path`, `ModelPricing`, `MODEL_PRICING`, `estimate_cost_usd()` |
| `app/usage.py` | `UsageStore` — schema, rollup, record, query helpers |
| `app/bedrock.py` | `InvokeResult` dataclass; parse `usage` from response |
| `app/rag.py` | Accept optional `UsageStore`; log after successful invoke |
| `app/main.py` | Construct `UsageStore`; pass to `RagPipeline` |
| `app/pages/usage.py` | Streamlit usage dashboard |
| `tests/test_config.py` | Pricing / `usage_db_path` defaults |
| `tests/test_usage.py` | Store, rollup, cost estimate |
| `tests/test_bedrock.py` | Update for `InvokeResult` |
| `tests/test_rag.py` | Logging on success / skip on failure |
| `docker-compose.yml` | Persist `./data` so SQLite survives restarts |
| `.env.example` | Document `USAGE_DB_PATH` |

---

### Task 1: Config — pricing table and DB path

**Files:**
- Modify: `.worktrees/dnd-rag/app/config.py`
- Modify: `.worktrees/dnd-rag/tests/test_config.py`
- Modify: `.worktrees/dnd-rag/.env.example`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
from app.config import MODEL_PRICING, Settings, estimate_cost_usd


def test_usage_db_path_default(monkeypatch):
    monkeypatch.delenv("USAGE_DB_PATH", raising=False)
    s = Settings()
    assert s.usage_db_path == "data/usage.sqlite"


def test_usage_db_path_from_env(monkeypatch):
    monkeypatch.setenv("USAGE_DB_PATH", "/tmp/test-usage.sqlite")
    s = Settings()
    assert s.usage_db_path == "/tmp/test-usage.sqlite"


def test_estimate_cost_nova_lite():
    usd = estimate_cost_usd("amazon.nova-lite-v1:0", input_tokens=1_000_000, output_tokens=1_000_000)
    pricing = MODEL_PRICING["amazon.nova-lite-v1:0"]
    expected = pricing.input_per_1m + pricing.output_per_1m
    assert usd == expected


def test_estimate_cost_unknown_model():
    assert estimate_cost_usd("unknown.model", 1000, 1000) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run from worktree root:

```bash
cd .worktrees/dnd-rag && pytest tests/test_config.py::test_usage_db_path_default -v
```

Expected: FAIL (`usage_db_path` or `estimate_cost_usd` not defined)

- [ ] **Step 3: Implement**

Replace `app/config.py` with:

```python
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float
    output_per_1m: float


# USD per 1M tokens — verify against current AWS Bedrock pricing when updating.
MODEL_PRICING: dict[str, ModelPricing] = {
    "amazon.nova-lite-v1:0": ModelPricing(input_per_1m=0.06, output_per_1m=0.24),
    "anthropic.claude-haiku-4-5-20251001-v1:0": ModelPricing(
        input_per_1m=1.00, output_per_1m=5.00
    ),
}


def estimate_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model_id)
    if pricing is None:
        return 0.0
    return (input_tokens / 1_000_000) * pricing.input_per_1m + (
        output_tokens / 1_000_000
    ) * pricing.output_per_1m


@dataclass(frozen=True)
class Settings:
    bedrock_model_id: str = field(
        default_factory=lambda: os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
    )
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "eu-west-2"))
    wiki_base_url: str = field(
        default_factory=lambda: os.getenv(
            "WIKI_BASE_URL", "https://wordlewarriors.github.io/normal-door-opening"
        ).rstrip("/")
    )
    chat_history_window: int = field(
        default_factory=lambda: int(os.getenv("CHAT_HISTORY_WINDOW", "8"))
    )
    obsidian_dir: str = field(default_factory=lambda: os.getenv("OBSIDIAN_DIR", "data/obsidian"))
    chroma_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "data/chromadb"))
    usage_db_path: str = field(
        default_factory=lambda: os.getenv("USAGE_DB_PATH", "data/usage.sqlite")
    )
    collection_name: str = "campaign_notes"
```

Add to `.env.example`:

```bash
USAGE_DB_PATH=data/usage.sqlite
```

- [ ] **Step 4: Run tests**

```bash
cd .worktrees/dnd-rag && pytest tests/test_config.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd .worktrees/dnd-rag && git add app/config.py tests/test_config.py .env.example
git commit -m "feat: add usage DB path and model pricing config"
```

---

### Task 2: UsageStore — schema, record, queries

**Files:**
- Create: `.worktrees/dnd-rag/app/usage.py`
- Create: `.worktrees/dnd-rag/tests/test_usage.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_usage.py`:

```python
from datetime import datetime, timezone
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd .worktrees/dnd-rag && pytest tests/test_usage.py -v
```

Expected: FAIL (`ModuleNotFoundError: app.usage`)

- [ ] **Step 3: Implement UsageStore (without rollup yet)**

Create `app/usage.py`:

```python
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.config import estimate_cost_usd

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

    def record_event(self, model_id: str, input_tokens: int, output_tokens: int) -> None:
        estimated_usd = estimate_cost_usd(model_id, input_tokens, output_tokens)
        with self._connect() as conn:
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
        """True if any current-month event used a model with no pricing."""
        from app.config import MODEL_PRICING

        events = self.get_current_month_events()
        return any(e["model_id"] not in MODEL_PRICING for e in events)
```

- [ ] **Step 4: Run tests**

```bash
cd .worktrees/dnd-rag && pytest tests/test_usage.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd .worktrees/dnd-rag && git add app/usage.py tests/test_usage.py
git commit -m "feat: add UsageStore with schema and current-month queries"
```

---

### Task 3: UsageStore — month rollup

**Files:**
- Modify: `.worktrees/dnd-rag/app/usage.py`
- Modify: `.worktrees/dnd-rag/tests/test_usage.py`

- [ ] **Step 1: Write the failing rollup test**

Add to `tests/test_usage.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd .worktrees/dnd-rag && pytest tests/test_usage.py::test_rollup_moves_prior_month_to_monthly_and_deletes_events -v
```

Expected: FAIL (`AttributeError: rollup_stale_months`)

- [ ] **Step 3: Add rollup to UsageStore**

Add method and call it from `record_event` inside a transaction:

```python
    def rollup_stale_months(self) -> None:
        current_ym = _current_year_month()
        with self._connect() as conn:
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
                    {
                        "request_count": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "estimated_usd": 0.0,
                    },
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
```

Refactor `record_event` to use `_rollup_stale_months_conn` before insert in the same transaction (as shown).

- [ ] **Step 4: Run all usage tests**

```bash
cd .worktrees/dnd-rag && pytest tests/test_usage.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd .worktrees/dnd-rag && git add app/usage.py tests/test_usage.py
git commit -m "feat: roll up stale usage events into monthly summaries"
```

---

### Task 4: Bedrock InvokeResult

**Files:**
- Modify: `.worktrees/dnd-rag/app/bedrock.py`
- Modify: `.worktrees/dnd-rag/tests/test_bedrock.py`

- [ ] **Step 1: Write the failing test**

Replace `tests/test_bedrock.py` with:

```python
import json
from unittest.mock import MagicMock, patch

from app.bedrock import BedrockClient, BedrockError, InvokeResult


def test_invoke_success_with_usage():
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps(
        {
            "output": {"message": {"content": [{"text": "Answer"}]}},
            "usage": {"inputTokens": 120, "outputTokens": 45},
        }
    ).encode()

    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {"body": mock_body}

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        result = bc.invoke("system", "user message")
        assert isinstance(result, InvokeResult)
        assert result.text == "Answer"
        assert result.input_tokens == 120
        assert result.output_tokens == 45


def test_invoke_success_missing_usage_defaults_zero():
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps(
        {"output": {"message": {"content": [{"text": "Answer"}]}}}
    ).encode()

    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {"body": mock_body}

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        result = bc.invoke("system", "user")
        assert result.text == "Answer"
        assert result.input_tokens == 0
        assert result.output_tokens == 0


def test_invoke_failure_raises():
    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = Exception("timeout")

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        try:
            bc.invoke("system", "user")
            assert False, "Should have raised"
        except BedrockError:
            pass
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd .worktrees/dnd-rag && pytest tests/test_bedrock.py -v
```

Expected: FAIL (returns `str` not `InvokeResult`)

- [ ] **Step 3: Implement**

Update `app/bedrock.py`:

```python
import json
import logging
from dataclasses import dataclass

import boto3

log = logging.getLogger(__name__)


class BedrockError(Exception):
    pass


@dataclass(frozen=True)
class InvokeResult:
    text: str
    input_tokens: int
    output_tokens: int


class BedrockClient:
    def __init__(self, model_id: str, region: str):
        self.model_id = model_id
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def invoke(self, system_prompt: str, user_message: str) -> InvokeResult:
        body = {
            "messages": [
                {"role": "user", "content": [{"text": user_message}]},
            ],
            "system": [{"text": system_prompt}],
            "inferenceConfig": {"maxTokens": 2048, "temperature": 0.3},
        }
        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())
            text = result["output"]["message"]["content"][0]["text"]
            usage = result.get("usage") or {}
            if "usage" not in result:
                log.warning("Bedrock response missing usage metadata")
            return InvokeResult(
                text=text,
                input_tokens=int(usage.get("inputTokens", 0)),
                output_tokens=int(usage.get("outputTokens", 0)),
            )
        except Exception as e:
            log.exception("Bedrock invoke failed")
            raise BedrockError(str(e)) from e
```

- [ ] **Step 4: Run tests**

```bash
cd .worktrees/dnd-rag && pytest tests/test_bedrock.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd .worktrees/dnd-rag && git add app/bedrock.py tests/test_bedrock.py
git commit -m "feat: return InvokeResult with token counts from Bedrock"
```

---

### Task 5: RagPipeline — log usage on success

**Files:**
- Modify: `.worktrees/dnd-rag/app/rag.py`
- Modify: `.worktrees/dnd-rag/tests/test_rag.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_rag.py`:

```python
from app.bedrock import BedrockError, InvokeResult


def test_query_logs_usage_on_success():
    store = MagicMock()
    store.query.return_value = {
        "documents": [["Lore."]],
        "metadatas": [[{
            "source_path": "sessions/note.md",
            "heading": "Section",
            "session": "42",
            "arc": "Test Arc",
            "date": "2025-02-04",
            "tags": "",
        }]],
    }
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 8
    bedrock = MagicMock()
    bedrock.model_id = "amazon.nova-lite-v1:0"
    bedrock.invoke.return_value = InvokeResult(text="Answer", input_tokens=10, output_tokens=5)
    usage = MagicMock()

    pipeline = RagPipeline(
        store=store,
        embedder=embedder,
        bedrock=bedrock,
        wiki_base_url="https://example.com",
        chat_history_window=8,
        usage_store=usage,
    )
    pipeline.query("Q?", arc=None, session=None, tag=None, k=5, history=[])
    usage.record_event.assert_called_once_with("amazon.nova-lite-v1:0", 10, 5)


def test_query_skips_usage_on_bedrock_error():
    store = MagicMock()
    store.query.return_value = {
        "documents": [["Lore."]],
        "metadatas": [[{
            "source_path": "sessions/note.md",
            "heading": "Section",
            "session": "42",
            "arc": "Test Arc",
            "date": "2025-02-04",
            "tags": "",
        }]],
    }
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 8
    bedrock = MagicMock()
    bedrock.invoke.side_effect = BedrockError("fail")
    usage = MagicMock()

    pipeline = RagPipeline(
        store=store,
        embedder=embedder,
        bedrock=bedrock,
        wiki_base_url="https://example.com",
        chat_history_window=8,
        usage_store=usage,
    )
    answer, _ = pipeline.query("Q?", arc=None, session=None, tag=None, k=5, history=[])
    assert "Generation failed" in answer
    usage.record_event.assert_not_called()
```

Update existing `test_query_with_results`:

```python
    bedrock.invoke.return_value = InvokeResult(text="Ivaran was revealed as Bergst.", input_tokens=1, output_tokens=1)
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
cd .worktrees/dnd-rag && pytest tests/test_rag.py::test_query_logs_usage_on_success -v
```

Expected: FAIL (`usage_store` unexpected keyword)

- [ ] **Step 3: Implement**

Update `app/rag.py` — add optional `usage_store` to `__init__` and logging in `query`:

```python
import logging

from app.bedrock import BedrockClient, BedrockError
# ... existing imports ...
from app.usage import UsageStore

log = logging.getLogger(__name__)

# ... SYSTEM_PROMPT unchanged ...


class RagPipeline:
    def __init__(
        self,
        store: NoteStore,
        embedder,
        bedrock: BedrockClient,
        wiki_base_url: str,
        chat_history_window: int,
        usage_store: UsageStore | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.bedrock = bedrock
        self.wiki_base_url = wiki_base_url
        self.chat_history_window = chat_history_window
        self.usage_store = usage_store

    def query(
        # ... signature unchanged ...
    ) -> tuple[str, str]:
        # ... retrieve context unchanged until bedrock call ...

        try:
            result = self.bedrock.invoke(SYSTEM_PROMPT, user_message)
        except BedrockError:
            return ("Generation failed — try again.", "")

        if self.usage_store is not None:
            try:
                self.usage_store.record_event(
                    self.bedrock.model_id,
                    result.input_tokens,
                    result.output_tokens,
                )
            except Exception:
                log.exception("Failed to record token usage")

        sources = format_sources_section(self.wiki_base_url, metadatas)
        return (result.text, sources)
```

- [ ] **Step 4: Run all RAG tests**

```bash
cd .worktrees/dnd-rag && pytest tests/test_rag.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd .worktrees/dnd-rag && git add app/rag.py tests/test_rag.py
git commit -m "feat: record Bedrock token usage after successful RAG queries"
```

---

### Task 6: Wire UsageStore in main.py

**Files:**
- Modify: `.worktrees/dnd-rag/app/main.py`

- [ ] **Step 1: Update `_load_pipeline`**

In `app/main.py`, import and construct:

```python
from app.usage import UsageStore

# inside _load_pipeline, after settings = Settings():
    usage_store = UsageStore(settings.usage_db_path)
    usage_store.rollup_stale_months()

# pass usage_store=usage_store to RagPipeline(...)
```

- [ ] **Step 2: Run full test suite**

```bash
cd .worktrees/dnd-rag && pytest -v
```

Expected: all PASS

- [ ] **Step 3: Commit**

```bash
cd .worktrees/dnd-rag && git add app/main.py
git commit -m "feat: wire UsageStore into app startup and RAG pipeline"
```

---

### Task 7: Streamlit usage page

**Files:**
- Create: `.worktrees/dnd-rag/app/pages/1_📊_Token_usage.py`
- Modify: `.worktrees/dnd-rag/Dockerfile` (if pages not copied — `COPY app/ app/` already includes `app/pages/`)

- [ ] **Step 1: Create usage page**

Create `app/pages/1_📊_Token_usage.py`:

```python
import sys
from datetime import datetime, timezone
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from app.config import Settings
from app.usage import UsageStore

st.set_page_config(page_title="Token usage", page_icon="📊", layout="wide")

settings = Settings()
store = UsageStore(settings.usage_db_path)
store.rollup_stale_months()

now = datetime.now(timezone.utc)
month_label = now.strftime("%B %Y")

st.title("📊 Token usage")
st.caption(f"{month_label} (UTC)")

summary = store.get_current_month_summary()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Requests", summary["request_count"])
c2.metric("Input tokens", f"{summary['input_tokens']:,}")
c3.metric("Output tokens", f"{summary['output_tokens']:,}")
c4.metric("Est. cost (USD)", f"${summary['estimated_usd']:.4f}")

if store.has_unknown_model_events():
    st.caption("Cost unknown for one or more models (no pricing configured).")

st.subheader("Request log")
events = store.get_current_month_events()
if events:
    st.dataframe(
        [
            {
                "Time (UTC)": e["created_at"],
                "Model": e["model_id"],
                "Input": e["input_tokens"],
                "Output": e["output_tokens"],
                "Est. USD": f"${e['estimated_usd']:.6f}",
            }
            for e in events
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No requests recorded this month yet.")

st.subheader("Previous months")
history = store.get_monthly_history()
if history:
    st.dataframe(
        [
            {
                "Month": h["year_month"],
                "Requests": h["request_count"],
                "Input": h["input_tokens"],
                "Output": h["output_tokens"],
                "Est. USD": f"${h['estimated_usd']:.4f}",
            }
            for h in history
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No completed months yet.")
```

- [ ] **Step 2: Manual smoke test (optional)**

```bash
cd .worktrees/dnd-rag && streamlit run app/main.py
```

Open sidebar → **Token usage** page; verify metrics render with empty DB.

- [ ] **Step 3: Commit**

```bash
cd .worktrees/dnd-rag && git add app/pages/
git commit -m "feat: add Streamlit token usage dashboard page"
```

---

### Task 8: Docker persistence for SQLite

**Files:**
- Modify: `.worktrees/dnd-rag/docker-compose.yml`

- [ ] **Step 1: Add writable data parent mount**

Replace volume section with:

```yaml
    volumes:
      - ./data:/app/data
      - ./data/obsidian:/app/data/obsidian:ro
      - ./data/chromadb:/app/data/chromadb
```

The `./data:/app/data` mount ensures `data/usage.sqlite` persists on the host. Submounts keep obsidian read-only and chromadb path explicit.

- [ ] **Step 2: Verify `data/` is gitignored**

Confirm `.gitignore` contains `data/` (already present).

- [ ] **Step 3: Commit**

```bash
cd .worktrees/dnd-rag && git add docker-compose.yml
git commit -m "chore: persist usage SQLite via data volume mount"
```

---

### Task 9: Final verification

- [ ] **Step 1: Run full test suite**

```bash
cd .worktrees/dnd-rag && pytest -v
```

Expected: all tests PASS

- [ ] **Step 2: Copy plan to worktree docs (optional sync)**

If worktree maintains its own docs copy:

```bash
cp docs/superpowers/plans/2026-05-29-token-usage.md .worktrees/dnd-rag/docs/superpowers/plans/ 2>/dev/null || true
```

---

## Spec Coverage Checklist

| Spec requirement | Task |
|------------------|------|
| Bedrock-only token logging | Task 4, 5 |
| UTC calendar month | Task 3 (`rollup_stale_months`) |
| Current-month metrics + request log | Task 7 |
| USD estimate from config pricing | Task 1, 2 |
| Previous months summary only | Task 3, 7 |
| Delete per-request rows after month ends | Task 3 |
| SQLite write failure doesn't break chat | Task 5 (`try/except` in rag) |
| Docker persistence | Task 8 |
| All listed tests | Tasks 1–5 |
