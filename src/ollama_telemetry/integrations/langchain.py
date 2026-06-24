from __future__ import annotations

import time
from typing import Any

from ..api import Telemetry, telemetry


def _response_metadata(output: Any) -> dict[str, Any]:
    if hasattr(output, "response_metadata") and isinstance(output.response_metadata, dict):
        return output.response_metadata
    if isinstance(output, dict):
        return output.get("response_metadata") if isinstance(output.get("response_metadata"), dict) else output
    return {}


class OllamaTelemetryCallback:
    """Optional LangChain callback that records safe Ollama usage metrics."""

    def __init__(self, client: Telemetry | None = None) -> None:
        self._telemetry = client or telemetry
        self._started: dict[str, int] = {}

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: Any, **kwargs: Any) -> None:
        # Prompts are intentionally discarded.
        self._started[str(run_id)] = time.perf_counter_ns()

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        started = self._started.pop(str(run_id), None)
        generations = getattr(response, "generations", None)
        output = None
        if generations and generations[0]:
            output = getattr(generations[0][0], "message", None) or generations[0][0]
        metadata = _response_metadata(output or response)
        self._telemetry.capture_ollama_response(metadata, model=metadata.get("model"), started_ns=started)

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._telemetry.capture_failure(error, started_ns=self._started.pop(str(run_id), None))


def enable_chatollama_auto_instrumentation(client: Telemetry | None = None) -> None:
    """Opt-in convenience patch for ChatOllama.invoke and ainvoke.

    The patch records metrics only; prompt and completion values are not inspected.
    """
    active = client or telemetry
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install 'ollama-telemetry[langchain]'") from exc

    if getattr(ChatOllama, "_ollama_telemetry_patched", False):
        return

    original_invoke = ChatOllama.invoke
    original_ainvoke = ChatOllama.ainvoke

    def invoke(self, *args: Any, **kwargs: Any):
        started = time.perf_counter_ns()
        try:
            result = original_invoke(self, *args, **kwargs)
        except Exception as exc:
            active.capture_failure(exc, model=getattr(self, "model", None), started_ns=started)
            raise
        active.capture_ollama_response(_response_metadata(result), model=getattr(self, "model", None), started_ns=started)
        return result

    async def ainvoke(self, *args: Any, **kwargs: Any):
        started = time.perf_counter_ns()
        try:
            result = await original_ainvoke(self, *args, **kwargs)
        except Exception as exc:
            active.capture_failure(exc, model=getattr(self, "model", None), started_ns=started)
            raise
        active.capture_ollama_response(_response_metadata(result), model=getattr(self, "model", None), started_ns=started)
        return result

    ChatOllama.invoke = invoke
    ChatOllama.ainvoke = ainvoke
    ChatOllama._ollama_telemetry_patched = True
