from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Iterable

from .events import TelemetryEvent
from .privacy import sanitize_metadata

DEFAULT_DB = Path.home() / ".ollama-telemetry" / "telemetry.db"


class SQLiteSink:
    """Small fail-safe SQLite sink. Each operation gets its own connection."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.getenv("OLLAMA_TELEMETRY_DB") or DEFAULT_DB).expanduser()
        self._lock = threading.RLock()
        self._initialized = False

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS telemetry_events (
                        event_id TEXT PRIMARY KEY,
                        occurred_at TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        trace_id TEXT,
                        span_id TEXT,
                        parent_span_id TEXT,
                        agent_name TEXT,
                        workflow TEXT,
                        entity_type TEXT,
                        entity_id TEXT,
                        entity_name TEXT,
                        run_id TEXT,
                        provider TEXT,
                        model TEXT,
                        operation TEXT,
                        status TEXT,
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        total_tokens INTEGER,
                        prompt_eval_duration_ns INTEGER,
                        eval_duration_ns INTEGER,
                        load_duration_ns INTEGER,
                        total_duration_ns INTEGER,
                        duration_ms REAL,
                        error_type TEXT,
                        error_message TEXT,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON telemetry_events(occurred_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_trace_id ON telemetry_events(trace_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_agent_model ON telemetry_events(agent_name, model)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_entity ON telemetry_events(entity_type, entity_id)")
            self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def write(self, event: TelemetryEvent) -> None:
        self.initialize()
        record = event.to_record()
        record["metadata"] = sanitize_metadata(record.get("metadata"))
        columns = [
            "event_id", "occurred_at", "event_type", "trace_id", "span_id", "parent_span_id",
            "agent_name", "workflow", "entity_type", "entity_id", "entity_name", "run_id",
            "provider", "model", "operation", "status", "input_tokens", "output_tokens",
            "total_tokens", "prompt_eval_duration_ns", "eval_duration_ns", "load_duration_ns",
            "total_duration_ns", "duration_ms", "error_type", "error_message", "metadata_json",
        ]
        values = [record.get(column) if column != "metadata_json" else json.dumps(record["metadata"], separators=(",", ":")) for column in columns]
        placeholders = ",".join("?" for _ in columns)
        with self._lock, self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO telemetry_events ({','.join(columns)}) VALUES ({placeholders})",
                values,
            )

    def query(self, sql: str, params: Iterable[object] = ()) -> list[tuple]:
        self.initialize()
        with self._connect() as conn:
            return conn.execute(sql, tuple(params)).fetchall()

    def prune(self, older_than_days: int) -> int:
        self.initialize()
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM telemetry_events WHERE occurred_at < datetime('now', ?)", (f"-{int(older_than_days)} days",))
            return cursor.rowcount

    def vacuum(self) -> None:
        self.initialize()
        with self._lock, self._connect() as conn:
            conn.execute("VACUUM")
