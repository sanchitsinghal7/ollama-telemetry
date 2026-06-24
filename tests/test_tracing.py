import sqlite3
from ollama_telemetry.api import Telemetry


def test_agent_trace_and_response_capture(tmp_path):
    client = Telemetry().init(db_path=str(tmp_path / "telemetry.db"))

    @client.agent("unit_agent", workflow="test")
    def work():
        with client.context(entity_type="item", entity_id="42"):
            client.capture_ollama_response({"model": "qwen3", "prompt_eval_count": 11, "eval_count": 17, "total_duration": 5000000})

    work()
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    rows = conn.execute("SELECT event_type, trace_id, entity_id, total_tokens FROM telemetry_events ORDER BY occurred_at").fetchall()
    assert [r[0] for r in rows] == ["agent_started", "llm_call", "agent_completed"]
    assert rows[1][1] == rows[0][1] == rows[2][1]
    assert rows[1][2] == "42"
    assert rows[1][3] == 28


def test_agent_context_manager_style(tmp_path):
    client = Telemetry().init(db_path=str(tmp_path / "telemetry.db"))

    with client.agent("context_agent", workflow="test"):
        client.capture_ollama_response({"model": "qwen3", "prompt_eval_count": 3, "eval_count": 5})

    conn = sqlite3.connect(tmp_path / "telemetry.db")
    rows = conn.execute("SELECT event_type, trace_id FROM telemetry_events ORDER BY occurred_at").fetchall()
    assert [r[0] for r in rows] == ["agent_started", "llm_call", "agent_completed"]
    assert len({r[1] for r in rows}) == 1


def test_agent_decorator_style_still_works(tmp_path):
    client = Telemetry().init(db_path=str(tmp_path / "telemetry.db"))

    @client.agent("decorator_agent")
    def run():
        client.capture_ollama_response({"model": "qwen3", "prompt_eval_count": 2, "eval_count": 4})

    run()
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    assert conn.execute("SELECT COUNT(*) FROM telemetry_events").fetchone()[0] == 3
