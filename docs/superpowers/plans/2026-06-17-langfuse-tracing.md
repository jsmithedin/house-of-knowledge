# Langfuse Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full-pipeline Langfuse tracing to the RAG chatbot, sending traces to a self-hosted Langfuse at `http://192.168.1.231:3000`, with silent graceful degradation when the server is unreachable.

**Architecture:** A new `app/tracing.py` module provides a `Tracer`/`NoopTracer` pair sharing the same interface; `RagPipeline` takes an optional `tracer` param (defaults to `NoopTracer`). `main.py` instantiates the real tracer at startup via `create_tracer(settings)`, which does a fast TCP check then auth validation before returning either a live `Tracer` or a `NoopTracer`. Every Langfuse call site is wrapped in `try/except` so a server that goes down mid-session silently drops traces.

**Tech Stack:** Python 3.12, langfuse Python SDK, pytest, unittest.mock

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `requirements.txt` | Modify | Add `langfuse` dependency |
| `.env.example` | Modify | Document four Langfuse env vars |
| `app/config.py` | Modify | Add four `Settings` fields |
| `app/tracing.py` | Create | `Tracer`, `NoopTracer`, `create_tracer`, `_tcp_reachable` |
| `app/rag.py` | Modify | Accept `tracer` param, instrument both query paths |
| `app/main.py` | Modify | Instantiate tracer, pass to `RagPipeline` |
| `tests/test_config.py` | Modify | Tests for new `Settings` fields |
| `tests/test_tracing.py` | Create | Tests for tracer module |
| `tests/test_rag.py` | Modify | Tests for tracing integration in `RagPipeline` |

---

## Task 1: Add Langfuse config to Settings

**Files:**
- Modify: `tests/test_config.py`
- Modify: `app/config.py`
- Modify: `.env.example`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing tests for new Settings fields**

Append to `tests/test_config.py`:

```python
def test_langfuse_defaults(monkeypatch):
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    s = Settings()
    assert s.langfuse_enabled is True
    assert s.langfuse_host == "http://192.168.1.231:3000"
    assert s.langfuse_public_key == ""
    assert s.langfuse_secret_key == ""


def test_langfuse_from_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("LANGFUSE_HOST", "http://my-host:3000")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    s = Settings()
    assert s.langfuse_enabled is False
    assert s.langfuse_host == "http://my-host:3000"
    assert s.langfuse_public_key == "pk-test"
    assert s.langfuse_secret_key == "sk-test"
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_config.py::test_langfuse_defaults tests/test_config.py::test_langfuse_from_env -v
```

Expected: `FAILED` — `Settings` has no `langfuse_enabled` attribute.

- [ ] **Step 3: Add four fields to `Settings` in `app/config.py`**

Add these four fields inside the `Settings` dataclass after `collection_name`:

```python
    langfuse_enabled: bool = field(
        default_factory=lambda: os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
    )
    langfuse_host: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "http://192.168.1.231:3000")
    )
    langfuse_public_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "")
    )
    langfuse_secret_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", "")
    )
```

- [ ] **Step 4: Update `.env.example`**

Append to `.env.example`:

```
# Langfuse tracing (optional — app runs fine without it):
LANGFUSE_HOST=http://192.168.1.231:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_ENABLED=true
```

- [ ] **Step 5: Add langfuse to `requirements.txt`**

Append `langfuse` to `requirements.txt` (no version yet — pin after install):

```
langfuse
```

Then install and pin:

```bash
pip install langfuse && pip show langfuse | grep Version
```

Replace the bare `langfuse` line with the pinned version, e.g. `langfuse==2.60.4`.

- [ ] **Step 6: Run tests to confirm they pass**

```
pytest tests/test_config.py::test_langfuse_defaults tests/test_config.py::test_langfuse_from_env -v
```

Expected: `PASSED`.

- [ ] **Step 7: Confirm existing config tests still pass**

```
pytest tests/test_config.py -v
```

Expected: all `PASSED`.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt .env.example app/config.py tests/test_config.py
git commit -m "feat: add Langfuse config fields to Settings"
```

---

## Task 2: Create `app/tracing.py`

**Files:**
- Create: `tests/test_tracing.py`
- Create: `app/tracing.py`

- [ ] **Step 1: Create `tests/test_tracing.py` with failing tests**

```python
from unittest.mock import MagicMock, patch

import pytest

from app.tracing import NoopTracer, Tracer, _NoopObj, _NoopTrace, _Generation, _Trace, create_tracer


def _settings(enabled=True, host="http://localhost:3000", pub="pk-test", sec="sk-test"):
    s = MagicMock()
    s.langfuse_enabled = enabled
    s.langfuse_host = host
    s.langfuse_public_key = pub
    s.langfuse_secret_key = sec
    return s


# ── NoopTracer ────────────────────────────────────────────────────────────────

def test_noop_tracer_returns_noop_trace():
    assert isinstance(NoopTracer().trace("test", {}), _NoopTrace)


def test_noop_trace_methods_do_not_raise():
    trace = _NoopTrace()
    trace.span("embed", {"query": "x"}).end()
    trace.generation("gen", "model", {"prompt": "x"}).end(output="answer", input_tokens=10, output_tokens=5)
    trace.end("output")


def test_noop_obj_end_accepts_all_signatures():
    obj = _NoopObj()
    obj.end()
    obj.end(output="text")
    obj.end(output="text", input_tokens=1, output_tokens=2)


# ── create_tracer ─────────────────────────────────────────────────────────────

def test_create_tracer_disabled_returns_noop():
    assert isinstance(create_tracer(_settings(enabled=False)), NoopTracer)


def test_create_tracer_empty_public_key_returns_noop():
    assert isinstance(create_tracer(_settings(pub="")), NoopTracer)


def test_create_tracer_empty_secret_key_returns_noop():
    assert isinstance(create_tracer(_settings(sec="")), NoopTracer)


def test_create_tracer_unreachable_host_returns_noop():
    with patch("app.tracing._tcp_reachable", return_value=False):
        tracer = create_tracer(_settings())
    assert isinstance(tracer, NoopTracer)


def test_create_tracer_auth_failure_returns_noop():
    mock_instance = MagicMock()
    mock_instance.auth_check.side_effect = Exception("Unauthorized")
    with patch("app.tracing._tcp_reachable", return_value=True), \
         patch("app.tracing.Langfuse", return_value=mock_instance):
        tracer = create_tracer(_settings())
    assert isinstance(tracer, NoopTracer)


def test_create_tracer_success_returns_tracer():
    mock_instance = MagicMock()
    with patch("app.tracing._tcp_reachable", return_value=True), \
         patch("app.tracing.Langfuse", return_value=mock_instance):
        tracer = create_tracer(_settings())
    assert isinstance(tracer, Tracer)


# ── Tracer error resilience ───────────────────────────────────────────────────

def test_tracer_trace_sdk_failure_returns_noop_trace():
    mock_client = MagicMock()
    mock_client.trace.side_effect = Exception("network error")
    tracer = Tracer(mock_client)
    assert isinstance(tracer.trace("test", {}), _NoopTrace)


def test_trace_span_sdk_failure_returns_noop_obj():
    mock_lf_trace = MagicMock()
    mock_lf_trace.span.side_effect = Exception("error")
    trace = _Trace(mock_lf_trace, MagicMock())
    assert isinstance(trace.span("embed", {}), _NoopObj)


def test_trace_generation_sdk_failure_returns_noop_obj():
    mock_lf_trace = MagicMock()
    mock_lf_trace.generation.side_effect = Exception("error")
    trace = _Trace(mock_lf_trace, MagicMock())
    assert isinstance(trace.generation("gen", "model", {}), _NoopObj)


# ── Generation usage mapping ──────────────────────────────────────────────────

def test_generation_end_passes_usage_to_langfuse():
    mock_gen = MagicMock()
    gen = _Generation(mock_gen)
    gen.end(output="the answer", input_tokens=10, output_tokens=5)
    mock_gen.end.assert_called_once_with(
        output="the answer",
        usage={"input": 10, "output": 5},
    )


def test_generation_end_sdk_failure_does_not_raise():
    mock_gen = MagicMock()
    mock_gen.end.side_effect = Exception("flush error")
    gen = _Generation(mock_gen)
    gen.end(output="answer", input_tokens=1, output_tokens=1)  # must not raise


def test_trace_end_calls_update_and_flush():
    mock_lf_trace = MagicMock()
    mock_client = MagicMock()
    trace = _Trace(mock_lf_trace, mock_client)
    trace.end("final output")
    mock_lf_trace.update.assert_called_once_with(output="final output")
    mock_client.flush.assert_called_once()


def test_trace_end_sdk_failure_does_not_raise():
    mock_lf_trace = MagicMock()
    mock_lf_trace.update.side_effect = Exception("error")
    trace = _Trace(mock_lf_trace, MagicMock())
    trace.end("output")  # must not raise
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_tracing.py -v
```

Expected: `ERROR` — `app.tracing` does not exist.

- [ ] **Step 3: Create `app/tracing.py`**

```python
import logging
import socket
from urllib.parse import urlparse

log = logging.getLogger(__name__)

try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None  # type: ignore[assignment,misc]


def _tcp_reachable(url: str, timeout: float = 2.0) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class _NoopObj:
    def end(self, output=None, input_tokens=0, output_tokens=0):
        pass


class _NoopTrace:
    def span(self, name: str, input: dict | None = None) -> _NoopObj:
        return _NoopObj()

    def generation(self, name: str, model: str, input: dict | None = None) -> _NoopObj:
        return _NoopObj()

    def end(self, output: str | None = None) -> None:
        pass


class NoopTracer:
    def trace(self, name: str, input: dict | None = None) -> _NoopTrace:
        return _NoopTrace()


class _Span:
    def __init__(self, span):
        self._span = span

    def end(self, output=None, **_):
        try:
            self._span.end(output=output)
        except Exception:
            log.debug("Langfuse span.end failed", exc_info=True)


class _Generation:
    def __init__(self, gen):
        self._gen = gen

    def end(self, output: str = "", input_tokens: int = 0, output_tokens: int = 0):
        try:
            self._gen.end(
                output=output,
                usage={"input": input_tokens, "output": output_tokens},
            )
        except Exception:
            log.debug("Langfuse generation.end failed", exc_info=True)


class _Trace:
    def __init__(self, trace, client):
        self._trace = trace
        self._client = client

    def span(self, name: str, input: dict | None = None) -> _Span | _NoopObj:
        try:
            return _Span(self._trace.span(name=name, input=input or {}))
        except Exception:
            log.debug("Langfuse span creation failed", exc_info=True)
            return _NoopObj()

    def generation(self, name: str, model: str, input: dict | None = None) -> _Generation | _NoopObj:
        try:
            return _Generation(self._trace.generation(name=name, model=model, input=input or {}))
        except Exception:
            log.debug("Langfuse generation creation failed", exc_info=True)
            return _NoopObj()

    def end(self, output: str | None = None) -> None:
        try:
            self._trace.update(output=output)
            self._client.flush()
        except Exception:
            log.debug("Langfuse trace.end failed", exc_info=True)


class Tracer:
    def __init__(self, client):
        self._client = client

    def trace(self, name: str, input: dict | None = None) -> _Trace | _NoopTrace:
        try:
            t = self._client.trace(name=name, input=input or {})
            return _Trace(t, self._client)
        except Exception:
            log.debug("Langfuse trace creation failed", exc_info=True)
            return _NoopTrace()


def create_tracer(settings) -> Tracer | NoopTracer:
    if not settings.langfuse_enabled:
        return NoopTracer()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        log.warning("Langfuse keys not configured — tracing disabled")
        return NoopTracer()
    if Langfuse is None:
        log.warning("langfuse package not installed — tracing disabled")
        return NoopTracer()
    if not _tcp_reachable(settings.langfuse_host):
        log.warning("Langfuse unreachable at %s — tracing disabled", settings.langfuse_host)
        return NoopTracer()
    try:
        client = Langfuse(
            host=settings.langfuse_host,
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
        )
        client.auth_check()
        log.info("Langfuse tracing enabled at %s", settings.langfuse_host)
        return Tracer(client)
    except Exception:
        log.warning(
            "Langfuse auth failed at %s — tracing disabled",
            settings.langfuse_host,
            exc_info=True,
        )
        return NoopTracer()
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_tracing.py -v
```

Expected: all `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add app/tracing.py tests/test_tracing.py
git commit -m "feat: add Tracer and NoopTracer with graceful degradation"
```

---

## Task 3: Instrument `RagPipeline`

**Files:**
- Modify: `tests/test_rag.py`
- Modify: `app/rag.py`

- [ ] **Step 1: Add tracing tests to `tests/test_rag.py`**

Add these imports at the top of `tests/test_rag.py` (alongside existing imports):

```python
from unittest.mock import ANY
```

Append these test functions to the bottom of `tests/test_rag.py`:

```python
# ── Tracing integration ───────────────────────────────────────────────────────

_META_TRACE = {
    "source_path": "sessions/note.md",
    "heading": "Section",
    "session": "1",
    "arc": "A",
    "date": "2025-01-01",
    "tags": "",
}


def _pipeline_with_tracer(tracer, store=None, bedrock=None):
    if store is None:
        store = MagicMock()
        store.query.return_value = {
            "documents": [["Lore content."]],
            "metadatas": [[_META_TRACE]],
        }
    if bedrock is None:
        bedrock = MagicMock()
        bedrock.model_id = "amazon.nova-lite-v1:0"
        bedrock.invoke.return_value = InvokeResult(text="Answer.", input_tokens=10, output_tokens=5)
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 8
    return RagPipeline(
        store=store,
        embedder=embedder,
        bedrock=bedrock,
        wiki_base_url="https://example.com",
        chat_history_window=8,
        tracer=tracer,
    )


def test_standard_query_creates_trace():
    tracer = MagicMock()
    mock_trace = MagicMock()
    tracer.trace.return_value = mock_trace

    pipeline = _pipeline_with_tracer(tracer)
    pipeline.query("Who is Ivaran?", arc=None, session=None, tag=None, k=5, history=[])

    tracer.trace.assert_called_once_with(
        "rag-query", {"query": "Who is Ivaran?", "k": 5, "agentic": False}
    )


def test_standard_query_creates_embed_and_retrieve_spans():
    tracer = MagicMock()
    mock_trace = MagicMock()
    tracer.trace.return_value = mock_trace

    pipeline = _pipeline_with_tracer(tracer)
    pipeline.query("Who is Ivaran?", arc=None, session=None, tag=None, k=5, history=[])

    mock_trace.span.assert_any_call("embed", {"query": "Who is Ivaran?"})
    mock_trace.span.assert_any_call("retrieve", ANY)


def test_standard_query_creates_generate_generation_with_model():
    tracer = MagicMock()
    mock_trace = MagicMock()
    tracer.trace.return_value = mock_trace

    pipeline = _pipeline_with_tracer(tracer)
    pipeline.query("Who is Ivaran?", arc=None, session=None, tag=None, k=5, history=[])

    mock_trace.generation.assert_called_once_with("generate", "amazon.nova-lite-v1:0", ANY)


def test_standard_query_ends_trace_with_answer():
    tracer = MagicMock()
    mock_trace = MagicMock()
    tracer.trace.return_value = mock_trace

    pipeline = _pipeline_with_tracer(tracer)
    pipeline.query("Who is Ivaran?", arc=None, session=None, tag=None, k=5, history=[])

    mock_trace.end.assert_called_once_with("Answer.")


def test_standard_query_ends_generation_with_token_counts():
    tracer = MagicMock()
    mock_trace = MagicMock()
    mock_gen = MagicMock()
    tracer.trace.return_value = mock_trace
    mock_trace.generation.return_value = mock_gen

    pipeline = _pipeline_with_tracer(tracer)
    pipeline.query("Who is Ivaran?", arc=None, session=None, tag=None, k=5, history=[])

    mock_gen.end.assert_called_once_with(output="Answer.", input_tokens=10, output_tokens=5)


def test_agentic_query_creates_trace_with_agentic_flag():
    store = MagicMock()
    store.query.return_value = {"documents": [["Lore."]], "metadatas": [[_META_TRACE]]}
    bedrock = MagicMock()
    bedrock.model_id = "amazon.nova-lite-v1:0"
    bedrock.invoke_with_tools.return_value = InvokeResult(text="Agentic answer.", input_tokens=20, output_tokens=10)

    tracer = MagicMock()
    mock_trace = MagicMock()
    tracer.trace.return_value = mock_trace

    pipeline = _pipeline_with_tracer(tracer, store=store, bedrock=bedrock)
    pipeline.query("Who is Ivaran?", arc=None, session=None, tag=None, k=5, history=[], agentic=True)

    tracer.trace.assert_called_once_with(
        "rag-query", {"query": "Who is Ivaran?", "k": 5, "agentic": True}
    )


def test_agentic_query_creates_tool_call_span_per_search():
    store = MagicMock()
    store.query.return_value = {"documents": [["Lore."]], "metadatas": [[_META_TRACE]]}
    bedrock = MagicMock()
    bedrock.model_id = "amazon.nova-lite-v1:0"

    def fake_invoke_with_tools(system_prompt, initial_messages, tools, tool_executor, max_iterations=5):
        tool_executor("search_knowledge_base", {"query": "Ivaran"})
        return InvokeResult(text="Agentic answer.", input_tokens=20, output_tokens=10)

    bedrock.invoke_with_tools.side_effect = fake_invoke_with_tools

    tracer = MagicMock()
    mock_trace = MagicMock()
    tracer.trace.return_value = mock_trace

    pipeline = _pipeline_with_tracer(tracer, store=store, bedrock=bedrock)
    pipeline.query("Who is Ivaran?", arc=None, session=None, tag=None, k=5, history=[], agentic=True)

    mock_trace.span.assert_called_once_with("tool-call", {"query": "Ivaran", "doc_count": 1})


def test_pipeline_works_without_tracer_arg():
    """RagPipeline with no tracer uses NoopTracer and runs without error."""
    store = MagicMock()
    store.query.return_value = {"documents": [["Lore."]], "metadatas": [[_META_TRACE]]}
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 8
    bedrock = MagicMock()
    bedrock.invoke.return_value = InvokeResult(text="Answer.", input_tokens=1, output_tokens=1)

    pipeline = RagPipeline(
        store=store, embedder=embedder, bedrock=bedrock,
        wiki_base_url="https://example.com", chat_history_window=8,
    )
    answer, _ = pipeline.query("Q?", arc=None, session=None, tag=None, k=5, history=[])
    assert "Answer." in answer
```

- [ ] **Step 2: Run new tests to confirm they fail**

```
pytest tests/test_rag.py::test_standard_query_creates_trace -v
```

Expected: `FAILED` — `RagPipeline.__init__` has no `tracer` parameter.

- [ ] **Step 3: Update `app/rag.py`**

Add import at the top of `app/rag.py`:

```python
from app.tracing import NoopTracer
```

Update `RagPipeline.__init__` — add `tracer` parameter and store it:

```python
    def __init__(
        self,
        store: NoteStore,
        embedder,
        bedrock: BedrockClient,
        wiki_base_url: str,
        chat_history_window: int,
        usage_store: UsageStore | None = None,
        tracer=None,
    ):
        self.store = store
        self.embedder = embedder
        self.bedrock = bedrock
        self.wiki_base_url = wiki_base_url
        self.chat_history_window = chat_history_window
        self.usage_store = usage_store
        self.tracer = tracer if tracer is not None else NoopTracer()
```

Replace `_query_standard` with the instrumented version:

```python
    def _query_standard(
        self,
        message: str,
        history_text: str,
        where: dict | None,
        k: int,
        bedrock: BedrockClient,
    ) -> tuple[str, str]:
        trace = self.tracer.trace("rag-query", {"query": message, "k": k, "agentic": False})

        embed_span = trace.span("embed", {"query": message})
        embedding = self.embedder.embed_query(message)
        embed_span.end()

        retrieve_span = trace.span("retrieve", {"k": k, "where": str(where)})
        results = self.store.query(query_embedding=embedding, n_results=k, where=where)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        retrieve_span.end(output={"doc_count": len(documents)})

        if not documents:
            trace.end("no_results")
            return (
                "No matching notes found. Try broadening your filters or rephrasing.",
                "",
            )

        context_parts = []
        for doc, meta in zip(documents, metadatas):
            context_parts.append(
                f"[Session {meta.get('session', '?')} — {meta.get('heading', '?')} "
                f"({meta.get('date', '?')})]\n{doc}"
            )
        context = "\n\n---\n\n".join(context_parts)

        user_message = ""
        if history_text:
            user_message += f"Conversation so far:\n{history_text}\n"
        user_message += f"Retrieved session notes:\n{context}\n\nQuestion: {message}"

        gen = trace.generation("generate", bedrock.model_id, {"user": user_message})
        try:
            result = bedrock.invoke(SYSTEM_PROMPT, user_message)
        except BedrockError:
            gen.end(output="generation_failed")
            trace.end("generation_failed")
            return ("Generation failed — try again.", "")
        gen.end(output=result.text, input_tokens=result.input_tokens, output_tokens=result.output_tokens)

        self._record_usage(bedrock, result.input_tokens, result.output_tokens)
        sources = format_sources_section(self.wiki_base_url, metadatas)
        trace.end(result.text)
        return (result.text, sources)
```

Replace `_query_agentic` with the instrumented version:

```python
    def _query_agentic(
        self,
        message: str,
        history_text: str,
        where: dict | None,
        k: int,
        bedrock: BedrockClient,
    ) -> tuple[str, str]:
        trace = self.tracer.trace("rag-query", {"query": message, "k": k, "agentic": True})
        all_metadatas: list[dict] = []

        def execute_tool(name: str, tool_input: dict) -> str:
            if name != "search_knowledge_base":
                return "Unknown tool."
            query = tool_input["query"]
            n = min(max(int(tool_input.get("k", k)), 1), 10)
            embedding = self.embedder.embed_query(query)
            results = self.store.query(query_embedding=embedding, n_results=n, where=where)
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            all_metadatas.extend(metas)
            tool_span = trace.span("tool-call", {"query": query, "doc_count": len(docs)})
            tool_span.end()
            if not docs:
                return "No results found for that query."
            parts = []
            for doc, meta in zip(docs, metas):
                parts.append(
                    f"[Session {meta.get('session', '?')} — {meta.get('heading', '?')} "
                    f"({meta.get('date', '?')})]\n{doc}"
                )
            return "\n\n---\n\n".join(parts)

        user_content = ""
        if history_text:
            user_content += f"Conversation so far:\n{history_text}\n"
        user_content += f"Question: {message}"

        initial_messages = [
            {"role": "user", "content": [{"text": user_content}]},
        ]

        gen = trace.generation("generate", bedrock.model_id, {"query": message})
        try:
            result = bedrock.invoke_with_tools(
                system_prompt=AGENTIC_SYSTEM_PROMPT,
                initial_messages=initial_messages,
                tools=[SEARCH_TOOL],
                tool_executor=execute_tool,
            )
        except BedrockError:
            gen.end(output="generation_failed")
            trace.end("generation_failed")
            return ("Generation failed — try again.", "")
        gen.end(output=result.text, input_tokens=result.input_tokens, output_tokens=result.output_tokens)

        self._record_usage(bedrock, result.input_tokens, result.output_tokens)
        sources = format_sources_section(self.wiki_base_url, all_metadatas)
        trace.end(result.text)
        return (result.text, sources)
```

- [ ] **Step 4: Run new tracing tests to confirm they pass**

```
pytest tests/test_rag.py -k "trace" -v
```

Expected: all new tracing tests `PASSED`.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```
pytest tests/test_rag.py -v
```

Expected: all tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add app/rag.py tests/test_rag.py
git commit -m "feat: instrument RagPipeline with Langfuse tracing spans"
```

---

## Task 4: Wire tracer into `main.py`

**Files:**
- Modify: `app/main.py`

No automated test for this task — `main.py` is Streamlit plumbing. Verify manually by watching the Langfuse UI after a query.

- [ ] **Step 1: Add import to `app/main.py`**

Add this import at the top of `app/main.py` alongside the other `app.*` imports:

```python
from app.tracing import create_tracer
```

- [ ] **Step 2: Instantiate tracer and pass it to `RagPipeline` in `_load_pipeline`**

In `_load_pipeline`, after `usage_store.rollup_stale_months()`, add:

```python
    tracer = create_tracer(settings)
```

Then pass `tracer=tracer` to the `RagPipeline(...)` constructor call:

```python
    pipeline = RagPipeline(
        store=store,
        embedder=embedder,
        bedrock=default_bedrock,
        wiki_base_url=settings.wiki_base_url,
        chat_history_window=settings.chat_history_window,
        usage_store=usage_store,
        tracer=tracer,
    )
```

- [ ] **Step 3: Run full test suite**

```
pytest -v
```

Expected: all tests `PASSED`.

- [ ] **Step 4: Commit**

```bash
git add app/main.py
git commit -m "feat: wire Langfuse tracer into main pipeline"
```

---

## Manual Verification

After wiring up `main.py`:

1. Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in your `.env` (get these from the Langfuse UI at `http://192.168.1.231:3000` under Settings → API Keys).
2. Start the app: `streamlit run app/main.py`
3. Check the startup log — you should see: `INFO: Langfuse tracing enabled at http://192.168.1.231:3000`
4. Ask a question in the chat UI.
5. Open the Langfuse UI and verify a `rag-query` trace appears with `embed`, `retrieve`, and `generate` child spans.
6. For graceful degradation: stop the Langfuse server and ask another question — the app should respond normally with `WARNING: Langfuse unreachable` in the log.

---

## Self-Review Notes

- All spec requirements covered: config (Task 1), `app/tracing.py` (Task 2), RagPipeline instrumentation (Task 3), `main.py` wiring (Task 4).
- Standard query: `embed` span + `retrieve` span + `generate` generation ✓
- Agentic query: `tool-call` span per `search_knowledge_base` call + `generate` generation ✓
- Graceful degradation: disabled flag, missing keys, TCP unreachable, auth failure, per-call SDK errors all return/use NoopTracer ✓
- Existing `RagPipeline` constructors without `tracer` param still work (`tracer=None` → `NoopTracer()`) ✓
- No changes to `BedrockClient` — agentic tool-call spans are added inside the `execute_tool` closure ✓
