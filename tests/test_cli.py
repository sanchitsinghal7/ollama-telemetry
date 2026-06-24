import pytest

from ollama_telemetry.cli import _parse_window


def test_parse_window_accepts_documented_duration_syntax():
    assert _parse_window("7d") == "-7 days"
    assert _parse_window("24h") == "-24 hours"
    assert _parse_window("30m") == "-30 minutes"
    assert _parse_window("2w") == "-14 days"


def test_parse_window_keeps_bare_day_count_compatible():
    assert _parse_window("1") == "-1 days"


def test_parse_window_rejects_invalid_values():
    with pytest.raises(Exception):
        _parse_window("1month")
