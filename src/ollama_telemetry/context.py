from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Iterator
from uuid import uuid4

from .privacy import sanitize_metadata


@dataclass(frozen=True, slots=True)
class TelemetryContext:
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    agent_name: str | None = None
    workflow: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    entity_name: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_current_context: ContextVar[TelemetryContext] = ContextVar("ollama_telemetry_context", default=TelemetryContext())


def current_context() -> TelemetryContext:
    return _current_context.get()


def set_context(value: TelemetryContext):
    return _current_context.set(value)


def reset_context(token) -> None:
    _current_context.reset(token)


@contextmanager
def scoped_context(**kwargs: Any) -> Iterator[TelemetryContext]:
    previous = current_context()
    metadata = dict(previous.metadata)
    metadata.update(sanitize_metadata(kwargs.pop("metadata", None)))
    next_context = replace(previous, metadata=metadata, **{k: v for k, v in kwargs.items() if v is not None})
    token = set_context(next_context)
    try:
        yield next_context
    finally:
        reset_context(token)


def start_root_context(*, agent_name: str, workflow: str | None = None, metadata: dict[str, Any] | None = None) -> tuple[TelemetryContext, object]:
    root = TelemetryContext(
        trace_id=uuid4().hex,
        span_id=uuid4().hex[:16],
        agent_name=agent_name,
        workflow=workflow,
        metadata=sanitize_metadata(metadata),
    )
    return root, set_context(root)


def start_child_span() -> tuple[TelemetryContext, object]:
    previous = current_context()
    trace_id = previous.trace_id or uuid4().hex
    child = replace(
        previous,
        trace_id=trace_id,
        parent_span_id=previous.span_id,
        span_id=uuid4().hex[:16],
    )
    return child, set_context(child)
