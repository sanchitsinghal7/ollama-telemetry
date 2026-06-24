from ollama_telemetry.privacy import sanitize_metadata


def test_sensitive_metadata_is_removed():
    result = sanitize_metadata({
        "prompt": "secret", "completion": "secret", "messages": ["secret"],
        "safe_id": "abc", "nested": {"response": "secret", "count": 3},
    })
    assert result == {"safe_id": "abc", "nested": {"count": 3}}
