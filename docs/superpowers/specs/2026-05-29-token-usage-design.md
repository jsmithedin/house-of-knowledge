# Token Usage Tracking — Design Spec

**Project:** house-of-knowledge / dnd-rag  
**Date:** 2026-05-29  
**Status:** Approved (brainstorming)  
**Parent:** [2026-05-29-dnd-rag-design.md](./2026-05-29-dnd-rag-design.md)

## Goal

Log Bedrock generation token usage to SQLite, roll up completed calendar months, and show current-month detail plus historical summaries on a Streamlit usage page.

## Decisions Summary

| Decision | Choice |
|----------|--------|
| What to count | Bedrock generation only (input + output tokens per successful chat answer) |
| Month boundary | Calendar month in UTC; resets on the 1st |
| Current-month UI | Summary metrics + scrollable per-request log |
| Cost display | Tokens plus estimated USD from a per-model price table in config |
| Past months UI | Summary table only (month → totals); no per-request log |
| Retention | Delete per-request rows after month ends; keep one `usage_monthly` row per month |
| Approach | Thin `app/usage.py` module; log in `RagPipeline` after successful invoke |
| Auth | Same as chat — aggregate usage for all Cloudflare Access users |

## Architecture

```
Streamlit Chat ──► RagPipeline.query()
                        │
                        ├─► BedrockClient.invoke() ──► InvokeResult(text, tokens)
                        │
                        └─► usage.record_event() ──► SQLite (data/usage.sqlite)
                                                      │
Streamlit Usage page ◄── usage queries ◄────────────┘
```

### New / changed components

| Component | Role |
|-----------|------|
| `app/usage.py` | Schema, `record_event`, `rollup_stale_months`, query helpers |
| `app/bedrock.py` | Return `InvokeResult` with token counts from Bedrock response |
| `app/rag.py` | Call `record_event` after successful invoke |
| `app/config.py` | `USAGE_DB_PATH`, model pricing table |
| `pages/usage.py` (or `pages/2_📊_Usage.py`) | Streamlit usage dashboard |
| `docker-compose.yml` | Persist `data/` volume including SQLite file |

## Data Model

### `usage_events` (current month detail only)

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Autoincrement |
| `created_at` | TEXT | ISO 8601 UTC |
| `model_id` | TEXT | Bedrock model at invoke time |
| `input_tokens` | INTEGER | From `usage.inputTokens` |
| `output_tokens` | INTEGER | From `usage.outputTokens` |
| `estimated_usd` | REAL | Computed at insert from price table |

### `usage_monthly` (completed months)

| Column | Type | Notes |
|--------|------|-------|
| `year_month` | TEXT | `YYYY-MM`, primary key |
| `request_count` | INTEGER | |
| `input_tokens` | INTEGER | Sum of events |
| `output_tokens` | INTEGER | Sum of events |
| `estimated_usd` | REAL | Sum of per-event estimates |

### Rollup

On app load and before each `record_event`:

1. Find `usage_events` where `created_at` is before the current UTC calendar month.
2. Aggregate by `YYYY-MM` and `INSERT OR REPLACE` into `usage_monthly`.
3. `DELETE` those events.

All rollup + insert operations run in a single SQLite transaction.

## Bedrock Integration

`BedrockClient.invoke()` returns:

```python
@dataclass(frozen=True)
class InvokeResult:
    text: str
    input_tokens: int
    output_tokens: int
```

Parse from Converse response: `result["usage"]["inputTokens"]`, `result["usage"]["outputTokens"]`.

If `usage` is missing on an otherwise successful response, log a warning and use `0` for both counts.

Failed invokes (`BedrockError`) are not logged.

## Logging Flow

1. `RagPipeline.query()` calls `bedrock.invoke()`.
2. On success, `usage.record_event(model_id, input_tokens, output_tokens)`.
3. `record_event` runs `rollup_stale_months()`, computes `estimated_usd`, inserts row.
4. Return `result.text` to the UI (chat behavior unchanged).

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `USAGE_DB_PATH` | `data/usage.sqlite` | SQLite file path |

Model pricing: hardcoded dict in `config.py` for `amazon.nova-lite-v1:0` and Claude Haiku model IDs used by the project. Unknown model → `estimated_usd = 0.0` with UI footnote.

Cost formula per event:

```
usd = (input_tokens / 1_000_000) * price.input_per_1m
    + (output_tokens / 1_000_000) * price.output_per_1m
```

## Streamlit Usage Page

Multipage app: chat remains home (`main.py`); add `pages/` entry **Token usage**.

Layout (top to bottom):

1. **Header** — current calendar month label, e.g. `May 2026 (UTC)`.
2. **Metrics** — `st.metric` row: requests, input tokens, output tokens, estimated USD.
3. **Request log** — dataframe, newest first: timestamp, model, input, output, est. USD.
4. **Previous months** — table from `usage_monthly`: month, requests, input, output, est. USD.

No month picker. Optional `st.cache_data(ttl=30)` for query helpers.

## Error Handling

| Situation | Behavior |
|-----------|----------|
| Bedrock invoke fails | No DB write |
| Missing `usage` in response | Warn; record with 0 tokens |
| SQLite write fails | Log exception; still return chat answer |
| DB file missing | Create parent dir + schema on first connect |
| Unknown model for pricing | `estimated_usd = 0.0`; footnote on page |
| Rollup fails | Log; do not insert (transaction prevents partial state) |

## Deployment

- Mount `./data:/app/data` in `docker-compose.yml` (or ensure `data/` parent exists for `usage.sqlite`).
- Add `data/usage.sqlite` to `.gitignore` if not already covered by `data/`.

## Testing

| Test | Verifies |
|------|----------|
| `test_bedrock_parses_usage` | Mock response with `usage` → correct `InvokeResult` |
| `test_usage_record_event` | Insert updates current-month totals |
| `test_usage_rollup` | Prior-month events → `usage_monthly` row + events deleted |
| `test_usage_cost_estimate` | Known model → expected USD |
| `test_rag_logs_on_success` | Successful query records one event |
| `test_rag_skips_log_on_failure` | `BedrockError` → no insert |

## Out of Scope

- Per-user attribution
- Local embedding / indexing token counts
- AWS Cost Explorer integration
- Budget alerts or caps
- Rolling 30-day windows
