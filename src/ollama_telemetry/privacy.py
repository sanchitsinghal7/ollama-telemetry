from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SENSITIVE_KEY_PARTS = {
    "prompt",
    "completion",
    "message",
    "messages",
    "content",
    "response",
    "raw",
    "payload",
    "body",
    "text",
    "input",
    "output",
}
MAX_METADATA_DEPTH = 4
MAX_STRING_LENGTH = 512
MAX_COLLECTION_ITEMS = 32


def _sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_").replace(" ", "_")
    parts = set(normalized.split("_"))
    return bool(parts & SENSITIVE_KEY_PARTS) or any(token in normalized for token in ("prompt", "completion", "raw_response"))


def sanitize_metadata(value: Mapping[str, Any] | None, *, depth: int = 0) -> dict[str, Any]:
    """Return safe scalar-only metadata without content-like keys or values."""
    if not value or depth > MAX_METADATA_DEPTH:
        return {}
    safe: dict[str, Any] = {}
    for key, raw in list(value.items())[:MAX_COLLECTION_ITEMS]:
        key_str = str(key)
        if _sensitive_key(key_str):
            continue
        cleaned = _sanitize_value(raw, depth + 1)
        if cleaned is not _DROP:
            safe[key_str] = cleaned
    return safe


class _Drop:
    pass


_DROP = _Drop()


def _sanitize_value(value: Any, depth: int) -> Any:
    if depth > MAX_METADATA_DEPTH:
        return _DROP
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:MAX_STRING_LENGTH]
    if isinstance(value, Mapping):
        return sanitize_metadata(value, depth=depth)
    if isinstance(value, (list, tuple, set)):
        cleaned = [_sanitize_value(item, depth + 1) for item in list(value)[:MAX_COLLECTION_ITEMS]]
        return [item for item in cleaned if item is not _DROP]
    # Arbitrary object reprs can contain user content, so omit them.
    return _DROP
