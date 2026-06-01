# Bedrock Converse + Multi-Model UI — Design Spec

**Project:** house-of-knowledge  
**Date:** 2026-06-01  
**Status:** Implemented

## Goal

Support **Amazon Nova Lite** and **Claude Haiku 4.5** for all chat generation paths (Standard and Agentic RAG), with a **runtime UI toggle** to switch models without redeploying or editing `.env`. Both models must use a single, maintainable Bedrock integration.

## Decisions Summary

| Decision | Choice |
|----------|--------|
| Bedrock integration | **Converse API** (`bedrock-runtime` `converse`) — one schema for Nova and Anthropic |
| Agentic + Nova | **Full parity** — Agentic tool-calling works on Nova Lite and Haiku (no fallback to Standard) |
| Model selection | UI radio per session; `BEDROCK_MODEL_ID` sets **default** radio only |
| Client lifecycle | Two cached `BedrockClient` instances (one per model ID), selected per query |
| Tool schema in RAG | Keep internal `SEARCH_TOOL` with `input_schema`; convert to Converse `toolConfig` in `BedrockClient` |
| IAM | Unchanged permission: `bedrock:InvokeModel` on each foundation-model ARN (Converse uses same auth) |

### Approaches considered

| Approach | Outcome |
|----------|---------|
| A — Model-family adapters on `invoke_model` | Rejected: two request/response formats to maintain |
| **B — Converse API** | **Selected**: unified messages, tools, and usage across providers |
| C — Separate client classes per family | Rejected: extra abstraction for only two models |

## Architecture

```
Streamlit UI (main.py)
  ├─ Model radio → BedrockClient (nova | haiku)
  ├─ RAG mode radio → RagPipeline.query(agentic=…)
  └─ Filters / k
         │
         ▼
RagPipeline (rag.py)
  ├─ Standard: embed → Chroma → bedrock.invoke(system, user)
  └─ Agentic: bedrock.invoke_with_tools(…, SEARCH_TOOL, execute_tool)
         │
         ▼
BedrockClient (bedrock.py) — Converse only
  client.converse(modelId, system, messages, [toolConfig], inferenceConfig)
         │
         ▼
AWS Bedrock Runtime (eu-west-2)
  amazon.nova-lite-v1:0  |  anthropic.claude-haiku-4-5-20251001-v1:0
```

Embeddings remain **local** (BGE-M3); only generation hits Bedrock.

## Model IDs and configuration

Constants in `app/config.py`:

| Constant | Model ID | UI label |
|----------|----------|----------|
| `NOVA_LITE_MODEL_ID` | `amazon.nova-lite-v1:0` | Nova Lite |
| `HAIKU_MODEL_ID` | `anthropic.claude-haiku-4-5-20251001-v1:0` | Haiku 4.5 |

`MODEL_PRICING` includes both IDs plus `eu.anthropic.claude-haiku-4-5-20251001-v1:0` for usage-page cost estimates if events were recorded under the EU inference profile ID.

| Variable | Default | Purpose |
|----------|---------|---------|
| `BEDROCK_MODEL_ID` | `amazon.nova-lite-v1:0` | Default **Model** radio selection at startup |
| `AWS_REGION` | `eu-west-2` | Bedrock region for both clients |

If `BEDROCK_MODEL_ID` is not one of the two supported IDs, the UI defaults to Nova Lite.

## Bedrock layer (`app/bedrock.py`)

### Public API (unchanged for `RagPipeline`)

- `invoke(system_prompt, user_message) -> InvokeResult`
- `invoke_with_tools(system_prompt, initial_messages, tools, tool_executor, max_iterations=5) -> InvokeResult`
- `InvokeResult`: `text`, `input_tokens`, `output_tokens`
- Failures wrap as `BedrockError`

### Standard invoke

```python
client.converse(
    modelId=self.model_id,
    system=[{"text": system_prompt}],
    messages=[{"role": "user", "content": [{"text": user_message}]}],
    inferenceConfig={"maxTokens": 2048},
)
```

- Expect `stopReason == "end_turn"`.
- Text from `response["output"]["message"]["content"]` (first block with `"text"`).
- Tokens from `usage.inputTokens` / `usage.outputTokens` (default 0 if missing).

### Agentic tool loop

1. `converse` with `toolConfig` built from internal tools (`name`, `description`, `input_schema` → `toolSpec` + `inputSchema.json`).
2. If `stopReason == "tool_use"`: for each `toolUse` block in assistant content, run `tool_executor(name, input)`; append assistant message, then user message with `toolResult` blocks (`toolUseId`, `content: [{text}]`, `status: "success"`).
3. If `stopReason == "end_turn"`: return accumulated text and token totals.
4. Other stop reasons → `BedrockError`.
5. After `max_iterations` without `end_turn` → `BedrockError`.

Message normalization accepts Converse `{text}` blocks and legacy `{type: "text", text: …}` for tests or callers.

### IAM and Anthropic onboarding

- Policy must allow `bedrock:InvokeModel` on **both** ARNs if the UI toggle is used (see `README.md`).
- Haiku requires the one-time Anthropic use-case form in the Bedrock console; Nova does not.

## RAG layer (`app/rag.py`)

No change to retrieval logic. Generation changes:

- `query(..., bedrock: BedrockClient | None = None)` — UI passes the selected client; default is pipeline’s initial client.
- `_record_usage(bedrock, …)` records `bedrock.model_id` for token usage DB.
- Agentic `initial_messages` use Converse format: `content: [{"text": user_content}]`.
- `SEARCH_TOOL` remains Anthropic-style (`input_schema` key); conversion happens in `BedrockClient`.

## UI (`app/main.py`)

Filter bar columns: Arc | Session | Tag | k | **RAG mode** | **Model**.

- **RAG mode:** Standard | Agentic (unchanged behavior).
- **Model:** Nova Lite | Haiku 4.5 — maps to cached `bedrock_clients[model_id]`, passed into `_respond` → `pipeline.query(bedrock=…)`.

`@st.cache_resource` builds store, embedder, both Bedrock clients, and `RagPipeline` once per process.

## Error handling

| Failure | User-visible | Usage recorded |
|---------|----------------|----------------|
| `BedrockError` on invoke / tools | “Generation failed — try again.” | No |
| Empty Chroma results (Standard) | “No matching notes found…” | No |
| Usage DB write failure | Answer still shown | Logged, not raised |

## Testing

| File | Coverage |
|------|----------|
| `tests/test_bedrock.py` | `converse` mocked: invoke, tool loop, multi-tool turn, max iterations, toolConfig mapping, legacy message normalization |
| `tests/test_rag.py` | Standard/agentic paths unchanged (mock `BedrockClient`) |
| `tests/test_config.py` | Defaults and pricing for Nova |

Run: `python -m pytest tests/`

## Out of scope

- Per-user model preference persistence (beyond Streamlit session).
- Streaming responses.
- Additional models (Sonnet, Nova Pro, inference profiles in the UI).
- Changing embedding model or indexer.
- Migrating token-usage page to group by UI label (still uses raw `model_id` from events).

## Success criteria

1. Standard RAG returns answers with Nova and Haiku via UI toggle (no `.env` change).
2. Agentic RAG runs tool loop on **both** models.
3. Token usage records the **actual** `model_id` used for each query.
4. All unit tests pass.
