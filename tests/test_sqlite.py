import sqlite3
from ollama_telemetry.events import TelemetryEvent
from ollama_telemetry.sqlite_sink import SQLiteSink


def test_writes_event_without_content(tmp_path):
    db = tmp_path / "telemetry.db"
    sink = SQLiteSink(db)
    sink.write(TelemetryEvent(event_type="llm_call", model="qwen3", metadata={"prompt": "never store", "job_id": "42"}))
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT metadata_json FROM telemetry_events").fetchone()[0]
    assert "never store" not in row
    assert "job_id" in row
