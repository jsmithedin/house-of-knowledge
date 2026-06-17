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


def test_create_tracer_package_not_installed_returns_noop():
    with patch("app.tracing.Langfuse", None), \
         patch("app.tracing._tcp_reachable", return_value=True):
        tracer = create_tracer(_settings())
    assert isinstance(tracer, NoopTracer)


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
