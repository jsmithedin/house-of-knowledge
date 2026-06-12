# Eval Harness

## 1. What this is

This is an evaluation harness for comparing two Amazon Bedrock models — Nova Lite and Claude Haiku 4.5 — on the House of Knowledge golden set. It runs both models against a fixed set of questions drawn from the D&D session notes, scores the answers, and produces a structured report comparing quality, retrieval recall, cost, and latency. Answers are scored blind: model identity is hidden until after all scoring is complete. This prevents the scorer from unconsciously favouring (or penalising) a model they already have opinions about, ensuring that quality scores reflect only the answer text.

## 2. Prerequisites

- **AWS credentials** with `bedrock:InvokeModel` for both Nova Lite and Haiku 4.5. See the main [README.md](../README.md) for IAM setup — the same credentials used by the app work here.
- **An indexed ChromaDB**: run `python scripts/index_notes.py` before running the harness. The eval runner queries the same ChromaDB collection as the app.
- **Python dependencies**: install eval-specific packages on top of the existing requirements:

```bash
uv sync --extra eval
```

## 3. Writing the golden set

**Location:** `eval/golden_set.jsonl` — one JSON object per line.

**Example record:**

```json
{
  "id": "q01",
  "tier": "lookup",
  "query": "What is the name of the tavern in Ashwick where the party first met?",
  "expected_answer": "The party first met at the Rusty Anchor, a dockside tavern in the harbour district of Ashwick. It is run by a retired sailor named Bram Tully, who is known to the thieves' guild.",
  "source_notes": ["sessions/session-01.md"]
}
```

**Tiers:**

| Tier | Definition |
|------|-----------|
| `lookup` | Single-note factual recall — the answer lives in one session note |
| `synthesis` | Multi-note cross-session reasoning — the answer requires connecting information from two or more notes |
| `temporal` | How something changed over time — the answer tracks state across sessions |

**Finding valid `source_path` values:** paths are relative to `OBSIDIAN_DIR` as stored by the indexer. List available notes with `ls data/obsidian/sessions/`, or query ChromaDB directly to see what `source_path` values are indexed.

**Rule:** write `expected_answer` before running any model. Writing it afterwards risks unconsciously anchoring your expected answer to whatever the model said.

## 4. The workflow

All stages use `--run-id` to identify a run. Use a descriptive ID such as `2026-06-15-baseline`.

| Stage | Command | Output |
|-------|---------|--------|
| 1 (💰 costs money) | `uv run python -m eval.runner --run-id YYYY-MM-DD-name` | `eval/runs/<id>/results.jsonl` |
| 2 (free) | `uv run python -m eval.score_retrieval --run-id YYYY-MM-DD-name` | `eval/runs/<id>/retrieval_scores.jsonl` |
| 3 (free) | `uv run python -m eval.make_blind --run-id YYYY-MM-DD-name` | `eval/runs/<id>/blind/pack.jsonl`, `eval/runs/<id>/blind/private/mapping.json` |
| 4 (free) | `uv run streamlit run eval/scoring_app.py -- --run-id YYYY-MM-DD-name` | `eval/runs/<id>/blind/scores.jsonl` |
| 5 (free) | `uv run python -m eval.unmask --run-id YYYY-MM-DD-name` | `eval/runs/<id>/scored.jsonl` |
| 6 (💰 costs money) | `uv run python -m eval.judge --run-id YYYY-MM-DD-name` | `eval/runs/<id>/judge_scores.jsonl` |
| 7 (free) | `uv run python -m eval.report --run-id YYYY-MM-DD-name` | `eval/runs/<id>/report.md` |

Stages 1 and 6 make Bedrock API calls and incur cost. All other stages are pure Python with no network calls.

## 5. The blind-scoring session

**How to run:**

```bash
uv run streamlit run eval/scoring_app.py -- --run-id <id>
```

Open the URL shown in the terminal. The app presents each answer labelled A or B with no model name visible.

**Correctness rubric:**

| Score | Meaning |
|-------|---------|
| 0 | Factually wrong, or no relevant answer provided |
| 1 | Partially correct with significant gaps |
| 2 | Correct and complete |

**Completeness rubric:**

| Score | Meaning |
|-------|---------|
| 0 | Missing key details |
| 1 | Mostly complete with minor omissions |
| 2 | Covers all key points |

**Hallucination flag:** tick this box when the answer asserts facts that are not present in the session notes — for example, naming an NPC, place, or event that does not appear anywhere in the notes. Do not tick it for hedged or uncertain language ("the party may have...") — only tick it for positive assertions of unsupported facts.

**Resumability:** scoring is resumable. Close the tab at any point, reopen with the same command, and the app picks up where you left off. Do not run Stage 5 (unmask) until all answers are scored.

## 6. Reading the report

**Human scores table:** correctness and completeness are on a 0–2 scale, averaged by model and tier. Hallucinations are a raw count of answers where the hallucination flag was ticked. Higher correctness/completeness is better; lower hallucinations is better.

**Retrieval table:** recall is the fraction of expected source notes where at least one chunk was retrieved in the top-k results. Hit rate is the fraction of queries where recall > 0 (i.e., at least one relevant note was returned). Both are per-tier.

**Cost and latency:** total cost in USD and cost per query for each model, plus p50 and p95 latency in milliseconds. Latency measures the Bedrock API call only, not retrieval.

**Agreement rate:** how often the human hallucination flag matches the judge's faithfulness score falling below the threshold (default 1.0). Disagreements are listed individually for review. Low agreement suggests either the judge prompt or the human rubric needs calibration — both need to be using the same definition of "hallucination".

**Score gap table:** the 3 queries with the largest correctness difference between the two models, with both answers quoted. These are the most interesting cases for a blog post — they illustrate where the models diverge most clearly.

## 7. Resuming and re-running

**Crash recovery:** if the runner crashes mid-run, re-run the same `--run-id` command. Completed (query, model) pairs are detected from the existing `results.jsonl` and skipped automatically.

**Smoke test:** use `--limit 2` to run only the first 2 queries before committing to a full run:

```bash
uv run python -m eval.runner --run-id 2026-06-15-smoke --limit 2
```

**What invalidates a run:** the following changes mean the existing results are no longer comparable and you should start a new run with a new `--run-id`:

- Re-indexing the ChromaDB (changes retrieval)
- Changing `SYSTEM_PROMPT` in `app/rag.py`
- Changing the `k` value in `eval/config.yaml`
- Modifying any record in `eval/golden_set.jsonl`

## 8. Troubleshooting

**Haiku 4.5 not available:** Claude Haiku 4.5 may require completing Anthropic's first-use form in the Bedrock console before it can be invoked. Open the Bedrock console, attempt a test invocation in the Chat playground, and submit the form if prompted. Nova Lite does not require this step.

**Model not available in eu-west-2:** not all models are available in every region. Verify which models are accessible in your account with:

```bash
aws bedrock list-foundation-models --region eu-west-2 \
  --query "modelSummaries[?contains(modelId, 'nova-lite') || contains(modelId, 'haiku-4-5')].modelId"
```

**BGE-M3 first load:** sentence-transformers downloads approximately 1.4 GB on first use. Subsequent loads use the local cache. Ensure you have enough disk space and a stable internet connection for the first run.

**NUC memory (8 GB):** the embedder is loaded once and stays in memory for the duration of the run. The harness never loads the full corpus into memory; ChromaDB streams results from disk. If you see out-of-memory errors, close other processes before starting a run.
