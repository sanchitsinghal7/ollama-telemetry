from __future__ import annotations

import functools
import logging
import time
from dataclasses import replace
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from .context import current_context, reset_context, scoped_context, set_context, start_child_span, start_root_context
from .events import TelemetryEvent
from .privacy import sanitize_metadata
from .sqlite_sink import SQLiteSink

logger = logging.getLogger("ollama_telemetry")
F = TypeVar("F", bound=Callable[..., Any])


class _AgentHandle:
    """An agent trace usable as either a decorator or a context manager.

    This makes both supported call styles explicit:

    @telemetry.agent("worker")
    def run(): ...

    with telemetry.agent("worker"):
        ...
    """

    def __init__(
        self,
        telemetry: "Telemetry",
        name: str,
        workflow: str | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        self._telemetry = telemetry
        self._name = name
        self._workflow = workflow
        self._metadata = metadata
        self._scope: Any | None = None

    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            with self._telemetry.agent_scope(
                self._name, workflow=self._workflow, metadata=self._metadata
            ):
                return func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]

    def __enter__(self) -> None:
        self._scope = self._telemetry.agent_scope(
            self._name, workflow=self._workflow, metadata=self._metadata
        )
        return self._scope.__enter__()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool | None:
        assert self._scope is not None
        return self._scope.__exit__(exc_type, exc, traceback)


class Telemetry:
    def __init__(self) -> None:
        self._sink: SQLiteSink | None = None

    def init(self, *, db_path: str | None = None) -> "Telemetry":
        self._sink = SQLiteSink(db_path)
        try:
            self._sink.initialize()
        except Exception:
            logger.debug("Unable to initialize telemetry storage", exc_info=True)
        return self

    @property
    def sink(self) -> SQLiteSink:
        if self._sink is None:
            self.init()
        assert self._sink is not None
        return self._sink

    def emit(self, event: TelemetryEvent) -> None:
        try:
            self.sink.write(event)
        except Exception:
            # Never allow observability to fail the caller.
            logger.debug("Telemetry write failed", exc_info=True)

    def agent(
        self,
        name: str,
        workflow: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> _AgentHandle:
        """Create a root agent trace.

        The returned object supports both ``@telemetry.agent(...)`` and
        ``with telemetry.agent(...):``.
        """
        return _AgentHandle(self, name, workflow, metadata)

    @contextmanager
    def agent_scope(self, name: str, workflow: str | None = None, *, metadata: dict[str, Any] | None = None) -> Iterator[None]:
        root, token = start_root_context(agent_name=name, workflow=workflow, metadata=metadata)
        started = time.perf_counter_ns()
        self.emit(TelemetryEvent(event_type="agent_started", trace_id=root.trace_id, span_id=root.span_id, agent_name=name, workflow=workflow, status="running", metadata=root.metadata))
        try:
            yield
        except Exception as exc:
            self.emit(TelemetryEvent(event_type="agent_failed", trace_id=root.trace_id, span_id=root.span_id, agent_name=name, workflow=workflow, status="error", duration_ms=(time.perf_counter_ns()-started)/1_000_000, error_type=type(exc).__name__, error_message=str(exc)[:512], metadata=root.metadata))
            raise
        else:
            self.emit(TelemetryEvent(event_type="agent_completed", trace_id=root.trace_id, span_id=root.span_id, agent_name=name, workflow=workflow, status="success", duration_ms=(time.perf_counter_ns()-started)/1_000_000, metadata=root.metadata))
        finally:
            reset_context(token)

    def context(self, **kwargs: Any):
        return scoped_context(**kwargs)

    def bind_run_id(self, run_id: str | int | None, *, metadata: dict[str, Any] | None = None) -> None:
        if run_id is None:
            return
        previous = current_context()
        merged = dict(previous.metadata)
        merged.update(sanitize_metadata(metadata))
        set_context(replace(previous, run_id=str(run_id), metadata=merged))

    def capture_ollama_response(self, response: Any, *, model: str | None = None, operation: str = "llm_call", metadata: dict[str, Any] | None = None, started_ns: int | None = None) -> None:
        payload = _as_mapping(response)
        ctx, child_token = start_child_span()
        try:
            prompt_tokens = _int_field(payload, "prompt_eval_count")
            output_tokens = _int_field(payload, "eval_count")
            total_tokens = (prompt_tokens or 0) + (output_tokens or 0) if prompt_tokens is not None or output_tokens is not None else None
            total_ns = _int_field(payload, "total_duration")
            event = TelemetryEvent(
                event_type="llm_call",
                trace_id=ctx.trace_id,
                span_id=ctx.span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                workflow=ctx.workflow,
                entity_type=ctx.entity_type,
                entity_id=ctx.entity_id,
                entity_name=ctx.entity_name,
                run_id=ctx.run_id,
                provider="ollama",
                model=model or _str_field(payload, "model"),
                operation=operation,
                status="success",
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                prompt_eval_duration_ns=_int_field(payload, "prompt_eval_duration"),
                eval_duration_ns=_int_field(payload, "eval_duration"),
                load_duration_ns=_int_field(payload, "load_duration"),
                total_duration_ns=total_ns,
                duration_ms=(total_ns / 1_000_000) if total_ns is not None else ((time.perf_counter_ns()-started_ns)/1_000_000 if started_ns else None),
                metadata={**ctx.metadata, **sanitize_metadata(metadata)},
            )
            self.emit(event)
        finally:
            reset_context(child_token)

    def capture_failure(self, exc: BaseException, *, model: str | None = None, operation: str = "llm_call", metadata: dict[str, Any] | None = None, started_ns: int | None = None) -> None:
        ctx, child_token = start_child_span()
        try:
            self.emit(TelemetryEvent(event_type="llm_call", trace_id=ctx.trace_id, span_id=ctx.span_id, parent_span_id=ctx.parent_span_id, agent_name=ctx.agent_name, workflow=ctx.workflow, entity_type=ctx.entity_type, entity_id=ctx.entity_id, entity_name=ctx.entity_name, run_id=ctx.run_id, provider="ollama", model=model, operation=operation, status="error", duration_ms=((time.perf_counter_ns()-started_ns)/1_000_000 if started_ns else None), error_type=type(exc).__name__, error_message=str(exc)[:512], metadata={**ctx.metadata, **sanitize_metadata(metadata)}))
        finally:
            reset_context(child_token)


def _as_mapping(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(response, "dict"):
        dumped = response.dict()
        return dumped if isinstance(dumped, dict) else {}
    return {key: getattr(response, key) for key in ("model", "prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration", "load_duration", "total_duration") if hasattr(response, key)}


def _int_field(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _str_field(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return str(value) if value is not None else None


def enable(*, db_path: str | None = None, auto_instrument_langchain: bool = False) -> Telemetry:
    telemetry.init(db_path=db_path)
    if auto_instrument_langchain:
        from .integrations.langchain import enable_chatollama_auto_instrumentation
        enable_chatollama_auto_instrumentation(telemetry)
    return telemetry


telemetry = Telemetry()
