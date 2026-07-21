# Spec: House of Knowledge Evaluation Harness

Implementation spec for Claude Code, written against the actual codebase in `house-of-knowledge/`. Builds the evaluation tooling described in `post_5_notes.md`: a batch runner, retrieval scorer, blind-scoring Streamlit page, LLM judge layer, and analysis report.

## Codebase integration

The harness lives in `house-of-knowledge/eval/` and reuses the existing `app/` modules. The relevant pieces, as they exist today:

- `app.config` — `Settings` (env-driven: `CHROMA_DIR`, `AWS_REGION`, `OBSIDIAN_DIR`, etc.), model ID constants (`NOVA_LITE_MODEL_ID`, `HAIKU_MODEL_ID`), `MODEL_LABELS`, `MODEL_PRICING`, and `estimate_cost_usd()`. **The harness imports all pricing and model identity from here — no duplicated price tables in eval config.** Note `MODEL_PRICING` also carries the `eu.`-prefixed Haiku inference-profile ID; at build time, check which ID the running app actually invokes (`.env` / `BEDROCK_MODEL_ID`) and use the same ones.
- `app.embedder.Embedder` — BGE-M3 singleton, `embed_query(text) -> list[float]`. First instantiation loads the model; do it once at runner start, not per query.
- `app.store.NoteStore` — `query(query_embedding, n_results, where)` returns the ChromaDB dict (`documents`, `metadatas`, `distances`). Chunk metadata fields: `source_path` (note path relative to the vault, e.g. `sessions/session-12.md`), `heading`, `session`, `arc`, `date`, `tags`. Chunk IDs are `{source_path}#{heading-slug}`.
- `app.bedrock.BedrockClient` — `invoke(system_prompt, user_message) -> InvokeResult(text, input_tokens, output_tokens)`. This is the call the harness uses; token counts come back on the result.
- `app.rag.RagPipeline` — **the harness does NOT call `RagPipeline.query()`.** Two reasons: it performs retrieval internally (incompatible with the retrieve-once rule), and it discards token counts from the caller's view (records them to `UsageStore` and returns only `(text, sources)`). Instead the harness composes the same parts itself: embed → store.query → build context → `BedrockClient.invoke(SYSTEM_PROMPT, user_message)`.
- `app.rag.SYSTEM_PROMPT` — imported and used verbatim. The eval must test the production prompt.

### Required refactors (minimal, behaviour-preserving)

1. **Extract context assembly from `RagPipeline._query_standard` into module-level functions in `app/rag.py`:**
   ```python
   def build_context(documents: list[str], metadatas: list[dict]) -> str
   def build_user_message(history_text: str, context: str, message: str) -> str
   ```
   `_query_standard` calls them; the harness imports them. This guarantees the harness sends byte-identical prompts to what production sends, now and after future prompt changes. Existing `tests/test_rag.py` must still pass; add direct tests for the two new functions.

2. **Add an optional `inference_config` parameter to `BedrockClient.invoke`** (default `None` → current behaviour of `{"maxTokens": 2048}`). The harness passes `{"maxTokens": 2048, "temperature": 0.0}` for determinism. The production app currently runs at the model default temperature — this difference gets one line in the blog post, but eval runs need temperature 0. Update `tests/test_bedrock.py` accordingly.

No other changes to `app/` or the Streamlit UI. The harness must not write to `data/usage.sqlite` — that's the production token-usage page's data, and bypassing `RagPipeline` avoids it naturally.

## Design principles

1. **Retrieve once per query.** Each golden query is embedded and retrieved once (`where=None` — no arc/session/tag filters in eval), and the identical context string is sent to both models.
2. **Determinism.** Temperature 0, fixed k, frozen corpus (no reindex mid-run), `history_text=""` (single-turn queries). Run config recorded in every output file.
3. **Append-only JSONL.** Every stage reads the previous stage's file and writes its own; nothing mutates earlier outputs. Everything goes in git except as noted.
4. **The human scores blind.** Model identity is stripped before scoring and restored only by the unmask step. The scoring UI has no code path to the mapping file.

## Layout

```
house-of-knowledge/
  eval/
    README.md             # full user guide — see README requirement below
    config.yaml           # run parameters (NOT prices/model IDs — those come from app.config)
    requirements-eval.txt # ragas + judge deps, pinned; keeps prod requirements.txt untouched
    golden_set.jsonl      # hand-written, version-controlled
    runner.py             # stage 1: batch runs
    score_retrieval.py    # stage 2: deterministic retrieval metrics
    make_blind.py         # stage 3: anonymise for scoring
    scoring_app.py        # stage 4: Streamlit blind-scoring page
    unmask.py             # stage 5: join scores back to models
    judge.py              # stage 6: RAGAS faithfulness via Bedrock
    report.py             # stage 7: analysis + markdown report
    runs/<run_id>/        # all outputs for one run
  tests/
    test_eval_*.py        # pure-logic tests, existing pytest setup (testpaths=tests)
```

Scripts run from the repo root (`python -m eval.runner ...`) so `app` imports resolve via the existing `pythonpath = .` in `pytest.ini` convention. `run_id` format: `YYYY-MM-DD-<slug>`; every script takes `--run-id`, defaulting to the most recent run directory.

## eval/config.yaml

```yaml
models:                 # labels must be keys resolvable via app.config; verify IDs against .env at build time
  - nova-lite           # -> NOVA_LITE_MODEL_ID
  - haiku-4.5           # -> HAIKU_MODEL_ID (or eu. inference profile if that's what the app uses)
k: 5                    # match the production default in the UI
judge_model: <decide at build time; strongest available in eu-west-2; record family-conflict caveat>
faithfulness_threshold: 1.0   # judge score below this = flagged unfaithful, for agreement analysis
```

Region, Chroma path, and vault path come from `app.config.Settings` (env / `.env`), so the harness runs unchanged on the NUC or laptop checkout.

## Data schemas

### golden_set.jsonl (hand-written)
```json
{"id": "q01", "tier": "lookup|synthesis|temporal", "query": "...",
 "expected_answer": "2-3 sentences, written before any model run",
 "source_notes": ["sessions/session-12.md"]}
```
20–30 records, roughly balanced across tiers. **`source_notes` entries are `source_path` values** — paths relative to the vault root, exactly as the indexer stores them. Validation on load: unique ids, valid tiers, non-empty source_notes, and each source_note exists under `Settings.obsidian_dir` (warn, don't fail, since vault and index can briefly diverge).

### runs/<run_id>/results.jsonl (one record per query × model)
```json
{"run_id": "...", "query_id": "q01", "model": "nova-lite",
 "answer": "...",
 "retrieved": [{"rank": 1, "chunk_id": "sessions/session-12.md#the-heist",
                "source_path": "sessions/session-12.md", "heading": "...",
                "session": "12", "distance": 0.31}],
 "context_chars": 0,
 "usage": {"input_tokens": 0, "output_tokens": 0},
 "latency_ms": 0, "retrieval_ms": 0, "timestamp": "...",
 "config": {"k": 5, "temperature": 0.0, "model_id": "amazon.nova-lite-v1:0"}}
```
`retrieved` and `retrieval_ms` are identical across the two records for the same query (retrieve once, copy into both). `latency_ms` is the Converse call only.

### runs/<run_id>/retrieval_scores.jsonl (one record per query)
```json
{"query_id": "q01", "expected_notes": ["..."], "retrieved_notes": ["..."],
 "recall": 0.5, "first_relevant_rank": 2, "hit": true}
```
`recall` = fraction of expected_notes with ≥1 chunk in the top-k (match on `source_path`). `hit` = recall > 0. `first_relevant_rank` = rank of first chunk from any expected note, null if none.

### runs/<run_id>/blind/ — pack.jsonl, private/mapping.json, scores.jsonl
`make_blind.py` labels the two answers per query `A`/`B`, randomised per query, seeded from run_id (reproducible). `pack.jsonl`: query, tier, expected_answer, label, answer — no model names, token counts, or latency (identity leaks). `private/mapping.json`: `{query_id: {"A": model, "B": model}}`. The scoring app never reads `private/`.

`scores.jsonl` (scoring app output):
```json
{"query_id": "q01", "label": "A", "correctness": 0, "completeness": 0,
 "hallucination": false, "notes": "", "scored_at": "..."}
```

### runs/<run_id>/judge_scores.jsonl
```json
{"query_id": "q01", "model": "nova-lite", "faithfulness": 0.0,
 "unsupported_claims": ["..."], "judge_model": "..."}
```

## Component requirements

### runner.py
- CLI: `python -m eval.runner --run-id 2026-06-15-baseline [--limit N] [--query-id q01]`
- Instantiate `Embedder()` once (model load is slow on the NUC; print a "loading embedder" line). One `BedrockClient` per model.
- Per query: embed once, `NoteStore.query(embedding, n_results=k)` once, `build_context` + `build_user_message(history_text="", ...)` once, then `invoke` per model with `SYSTEM_PROMPT` and temperature 0
- Writes results incrementally; skips (query, model) pairs already in results.jsonl so re-running resumes after a crash
- Prints total cost at the end via `app.config.estimate_cost_usd`

### score_retrieval.py
- Pure function of results.jsonl + golden_set.jsonl — no LLM, no network, no embedder
- Prints a per-tier summary table to stdout

### make_blind.py / scoring_app.py / unmask.py
- Scoring app: `streamlit run eval/scoring_app.py -- --run-id ...`
  - One answer at a time, randomised order across all (query, label) pairs — never side-by-side (position-bias mitigation)
  - Shows: query, tier, expected answer, answer under review
  - Inputs: correctness 0–2, completeness 0–2, hallucination toggle, free-text note
  - Saves on submit, shows progress (n of total), resumes from existing scores.jsonl
  - Runs on the existing pinned streamlit version (1.44.1) — no new UI deps
- unmask.py joins scores + mapping → `runs/<run_id>/scored.jsonl` with model names restored. Refuses to run unless every pack entry is scored (`--partial` to override)

### judge.py
- RAGAS faithfulness with the judge model from config via Bedrock, region from `Settings`. Pin the ragas version in requirements-eval.txt (its API moves between releases)
- Input: results.jsonl (answer + retrieved chunk texts — re-fetch chunk documents from the store by chunk_id, or carry document text through results.jsonl; carrying it through is simpler, add a `documents` field to `retrieved` entries if so)
- If RAGAS-on-Bedrock proves awkward, fall back to a hand-rolled judge prompt via `BedrockClient` (extract claims → check each against context → fraction supported) and mark records `"method": "fallback"`
- Judge never sees the expected answer — faithfulness is answer-vs-context only

### report.py
- Output: `runs/<run_id>/report.md`:
  - Per-model, per-tier: mean correctness, completeness, hallucination count
  - Retrieval: recall and hit-rate per tier (model-independent)
  - Cost per query and per run (via `estimate_cost_usd`); latency p50/p95 per model
  - Human–judge agreement: human hallucination flag vs judge faithfulness < threshold as binary labels; agreement rate plus the disagreement cases listed for review
  - The 3 largest per-query score gaps between models, both answers quoted (side-by-side candidates for the blog post)

## README requirement

`eval/README.md` is a deliverable, not an afterthought — a full user guide written for someone who has cloned the repo but hasn't read this spec. It must cover:

- **What this is** — one paragraph: evaluation harness for comparing Bedrock models on the House of Knowledge golden set, and the blind-scoring rationale in two sentences
- **Prerequisites** — AWS credentials with `bedrock:InvokeModel` for both models (link to the IAM section of the main README rather than duplicating it), an indexed ChromaDB (`scripts/index_notes.py` already run), Python deps: `pip install -r eval/requirements-eval.txt` on top of the main requirements
- **Writing the golden set** — the JSONL format with a filled-in example record, the three tiers explained, where to find valid `source_path` values, and the rule: expected answers are written before any model runs
- **The workflow** — all seven stages in order, each with the exact copy-paste command and what files it produces; clearly mark which stages cost money (runner, judge) and which are free
- **The blind-scoring session** — how to run the app, what the rubric scores mean (0/1/2 anchors for correctness and completeness, what counts as a hallucination), that it's resumable, and the rule: don't run unmask until scoring is finished
- **Reading the report** — what each table means, what the agreement rate is telling you
- **Resuming and re-running** — crash recovery, `--limit` smoke tests, why re-runs skip completed pairs, and what invalidates a run (reindexing, prompt changes, config changes)
- **Troubleshooting** — Anthropic first-use form for Haiku, eu-west-2 model availability, BGE-M3 first-load download, memory expectations on the NUC

Also add a one-line pointer to `eval/README.md` in the main repo README.

## Testing

Pure-logic components get pytest coverage under `tests/` following the existing style (no network, no Chroma, no Bedrock in tests): golden-set validation, recall/first-rank computation, blind-pack generation (label randomisation is seeded and identity-free), unmask joining, report aggregation, and the two refactored `app/rag.py` functions. `pytest` must pass cleanly from the repo root.

## Acceptance criteria

- [ ] `python -m eval.runner --limit 2` completes against the live index and produces valid results.jsonl
- [ ] Both models receive byte-identical `user_message` strings for the same query (assert in a test via the extracted builders)
- [ ] Full pipeline runs end-to-end on a 2-query smoke set: run → retrieval scores → blind pack → (manual scores) → unmask → report
- [ ] Scoring app cannot reveal model identity: no path to `private/`, no identity-leaking fields in pack.jsonl
- [ ] Re-running any stage is idempotent or resumable; nothing overwrites raw results
- [ ] No writes to `data/usage.sqlite`; no changes to production behaviour (existing test suite passes, app runs as before)
- [ ] Works on the NUC (8GB RAM): embedder loaded once, corpus never fully loaded into memory
- [ ] `eval/requirements-eval.txt` pinned; main `requirements.txt` and Dockerfile untouched
- [ ] `eval/README.md` complete per the README requirement; main README links to it
- [ ] `pytest` passes from repo root, including new eval tests

## Out of scope

- No changes to the production Streamlit app, indexer, or Docker setup
- No CI integration, no dashboards, no observability stack
- No pairwise judging (single-answer rubric only)
- No MCP server — the harness imports `app/` directly
- Agentic mode (`RagPipeline._query_agentic`) is not evaluated in this run; that's Post 8's harness extension
