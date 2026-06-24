"""Privacy-first, SQLite-first observability for local Ollama agents."""
from .api import enable, telemetry
from .events import TelemetryEvent

__all__ = ["TelemetryEvent", "enable", "telemetry"]
__version__ = "0.1.0"
