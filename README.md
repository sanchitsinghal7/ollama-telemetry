# ollama-telemetry

**Privacy-first, SQLite-first observability for local Ollama agents.**

- Stores telemetry locally in SQLite by default.
- Never stores prompts, completions, message bodies, or raw response payloads.
- No cloud account, API key, network service, or proprietary agent integration required.
- Captures Ollama token counts and latency fields when available.
- Supports explicit tracing and optional `ChatOllama` auto-instrumentation.

> Package name availability on PyPI must be checked before publishing.

## Install

```bash
pip install ollama-telemetry
```

For LangChain / `ChatOllama`:

```bash
pip install "ollama-telemetry[langchain]"
```

## Zero-code-ish LangChain setup

```python
from ollama_telemetry import enable

enable(auto_instrument_langchain=True)

# Existing ChatOllama code can run after this point.
```

Auto instrumentation is opt-in and intentionally marked as a convenience mode. The explicit API below is the recommended production approach.

## Explicit tracing

```python
from ollama_telemetry import telemetry
from ollama_telemetry.integrations.langchain import OllamaTelemetryCallback
from langchain_ollama import ChatOllama

telemetry.init()

@telemetry.agent(name="research_agent", workflow="summarization")
def run(document_id: str, messages):
    llm = ChatOllama(
        model="qwen3:8b",
        callbacks=[OllamaTelemetryCallback()],
    )
    with telemetry.context(entity_type="document", entity_id=document_id):
        return llm.invoke(messages)
```

## Direct Ollama response capture

```python
from ollama_telemetry import telemetry
from ollama import chat

telemetry.init()

with telemetry.agent("local_research_agent"):
    response = chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "Summarize this."}],
    )
    telemetry.capture_ollama_response(response, model="qwen3:8b")
```

## Database

Default location:

```text
~/.ollama-telemetry/telemetry.db
```

Override it:

```bash
export OLLAMA_TELEMETRY_DB=/path/to/telemetry.db
```

The SQLite table is `telemetry_events`. It stores identifiers, timing, model, status, safe usage counters, and JSON metadata. It does **not** store content fields.

## CLI

```bash
ollama-telemetry status
ollama-telemetry stats --last 7d
ollama-telemetry models --last 30d
ollama-telemetry agents --last 30d
ollama-telemetry traces --failed
ollama-telemetry prune --older-than 30d
ollama-telemetry vacuum
```

## Privacy guarantees

The package rejects sensitive metadata keys such as `prompt`, `completion`, `messages`, `response`, `content`, and `raw_response`. Safe metadata is limited to scalar values and small lists/dictionaries after recursive filtering.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
python -m build
```

## License

Apache-2.0.
