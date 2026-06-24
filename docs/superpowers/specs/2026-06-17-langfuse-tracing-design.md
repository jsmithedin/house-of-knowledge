# Langfuse Tracing Design

**Date:** 2026-06-17  
**Status:** Approved

## Overview

Add full-pipeline Langfuse tracing to the RAG chatbot. Traces are sent to a self-hosted Langfuse instance at `http://192.168.1.231:3000`. The app must degrade gracefully when Langfuse is unreachable — tracing failures must never surface to the user.

## Config

Four new env vars added to `Settings` and `.env.example`:

```
LANGFUSE_HOST=http://192.168.1.231:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_ENABLED=true
```

Tracing auto-disables if `LANGFUSE_ENABLED=false` or if either key is empty. Keys are required because the self-hosted Langfuse instance still uses project-scoped API keys.

## New Module: `app/tracing.py`

A `Tracer` class and a matching `NoopTracer` sharing the same interface. Call sites use the interface directly — no `if tracer:` guards needed anywhere.

### Interface

```python
tracer.trace(name, input)             # returns a TraceCtx context manager
trace_ctx.span(name, input)           # child span context manager
trace_ctx.generation(name, model, input, usage)  # LLM generation span
trace_ctx.end(output)                 # close the parent trace
```

### Initialisation

`Tracer.__init__` calls `langfuse.auth_check()` with a 3s timeout. On any failure (`ConnectionError`, `TimeoutError`, or any Langfuse SDK exception) it logs:

```
WARNING: Langfuse unreachable at <host> — tracing disabled
```

and the caller falls back to `NoopTracer`.

## Integration Points

### `main.py` (`_load_pipeline`)

Instantiates the tracer after `Settings()` is resolved, passes it to `RagPipeline`. If Langfuse config is absent or startup check fails, `NoopTracer` is used silently.

### `RagPipeline`

`__init__` gains an optional `tracer: Tracer = NoopTracer()` parameter. No other signature changes.

### Standard query span structure

```
Trace: "rag-query"  {query, arc, session, tag, k, agentic=false}
  ├── span: "embed"       {query}
  ├── span: "retrieve"    {k, doc_count, where}
  └── generation: "generate"  {model_id, input_tokens, output_tokens}
```

### Agentic query span structure

```
Trace: "rag-query"  {query, arc, session, tag, k, agentic=true}
  ├── span: "tool-call"   {query, doc_count}   ← one per search_knowledge_base invocation
  └── generation: "generate"  {model_id, total_input_tokens, total_output_tokens}
```

Tool-call spans are added inside the existing `execute_tool` closure in `_query_agentic`. No changes to `BedrockClient`.

## Graceful Degradation

Three layers:

1. **Startup check** — `auth_check()` with 3s timeout. Any failure → `NoopTracer`, warning logged.
2. **Per-call safety** — every span/generation call in the real `Tracer` is wrapped in `try/except Exception`, logging at `DEBUG` and continuing silently.
3. **`NoopTracer` default** — `RagPipeline` defaults to `NoopTracer()`, so tests and uninstrumented code paths work without changes.

## Files Changed

| File | Change |
|------|--------|
| `requirements.txt` | Add `langfuse` |
| `.env.example` | Add four Langfuse vars |
| `app/config.py` | Add four `Settings` fields |
| `app/tracing.py` | New — `Tracer`, `NoopTracer`, `TraceCtx` |
| `app/rag.py` | Accept optional `tracer`, instrument `_query_standard` and `_query_agentic` |
| `app/main.py` | Instantiate tracer in `_load_pipeline`, pass to `RagPipeline` |

## Out of Scope

- Tracing inside `BedrockClient` (individual LLM turns within `invoke_with_tools`)
- Langfuse scores or evaluations
- Token-cost display in Langfuse (token counts captured, cost calculation left to Langfuse UI)
