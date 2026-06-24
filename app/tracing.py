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
            if output is not None:
                self._span.update(output=output)
            self._span.end()
        except Exception:
            log.debug("Langfuse span.end failed", exc_info=True)


class _Generation:
    def __init__(self, gen):
        self._gen = gen

    def end(self, output: str = "", input_tokens: int = 0, output_tokens: int = 0):
        try:
            self._gen.update(
                output=output,
                usage_details={"input": input_tokens, "output": output_tokens},
            )
            self._gen.end()
        except Exception:
            log.debug("Langfuse generation.end failed", exc_info=True)


class _Trace:
    def __init__(self, root, client):
        self._root = root
        self._client = client

    def span(self, name: str, input: dict | None = None) -> _Span | _NoopObj:
        try:
            return _Span(self._root.start_observation(name=name, as_type="span", input=input or {}))
        except Exception:
            log.debug("Langfuse span creation failed", exc_info=True)
            return _NoopObj()

    def generation(self, name: str, model: str, input: dict | None = None) -> _Generation | _NoopObj:
        try:
            return _Generation(
                self._root.start_observation(name=name, as_type="generation", model=model, input=input or {})
            )
        except Exception:
            log.debug("Langfuse generation creation failed", exc_info=True)
            return _NoopObj()

    def end(self, output: str | None = None) -> None:
        try:
            self._root.set_trace_io(output=output)
            self._root.end()
            self._client.flush()
        except Exception:
            log.debug("Langfuse trace.end failed", exc_info=True)


class Tracer:
    def __init__(self, client):
        self._client = client

    def trace(self, name: str, input: dict | None = None) -> _Trace | _NoopTrace:
        try:
            root = self._client.start_observation(name=name, as_type="span", input=input or {})
            return _Trace(root, self._client)
        except Exception:
            log.debug("Langfuse trace creation failed", exc_info=True)
            return _NoopTrace()


def create_tracer(settings) -> Tracer | NoopTracer:
    if not settings.langfuse_enabled:
        return NoopTracer()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        log.warning("Langfuse keys not configured — tracing disabled")
        return NoopTracer()
    if not settings.langfuse_host:
        log.warning("Langfuse host not configured — tracing disabled")
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
