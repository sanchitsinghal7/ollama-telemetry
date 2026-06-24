from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class TelemetryEvent:
    event_type: str
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    occurred_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: str = field(default_factory=lambda: uuid4().hex)
    agent_name: str | None = None
    workflow: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    entity_name: str | None = None
    run_id: str | None = None
    provider: str | None = "ollama"
    model: str | None = None
    operation: str | None = None
    status: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_duration_ns: int | None = None
    load_duration_ns: int | None = None
    total_duration_ns: int | None = None
    duration_ms: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
