from __future__ import annotations

import argparse
from pathlib import Path

from .sqlite_sink import SQLiteSink


def _sink_from_args(args: argparse.Namespace) -> SQLiteSink:
    return SQLiteSink(args.db)


def _print_rows(headers: list[str], rows: list[tuple]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value if value is not None else "")))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value if value is not None else "").ljust(widths[i]) for i, value in enumerate(row)))


def command_status(args: argparse.Namespace) -> int:
    sink = _sink_from_args(args)
    sink.initialize()
    total = sink.query("SELECT COUNT(*) FROM telemetry_events")[0][0]
    print(f"Database: {sink.path}")
    print(f"Events: {total}")
    return 0


def command_stats(args: argparse.Namespace) -> int:
    sink = _sink_from_args(args)
    rows = sink.query("""
        SELECT COUNT(*), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0),
               COALESCE(SUM(total_tokens), 0), ROUND(COALESCE(SUM(duration_ms), 0) / 1000.0, 2),
               COALESCE(SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END), 0)
        FROM telemetry_events
        WHERE occurred_at >= datetime('now', ?)
    """, (f"-{args.last} days",))
    _print_rows(["calls", "input_tokens", "output_tokens", "total_tokens", "seconds", "failures"], rows)
    return 0


def command_models(args: argparse.Namespace) -> int:
    sink = _sink_from_args(args)
    rows = sink.query("""
        SELECT COALESCE(model, 'unknown'), COUNT(*), COALESCE(SUM(total_tokens), 0), ROUND(AVG(duration_ms), 2)
        FROM telemetry_events
        WHERE occurred_at >= datetime('now', ?)
        GROUP BY model ORDER BY COUNT(*) DESC
    """, (f"-{args.last} days",))
    _print_rows(["model", "calls", "tokens", "avg_ms"], rows)
    return 0


def command_agents(args: argparse.Namespace) -> int:
    sink = _sink_from_args(args)
    rows = sink.query("""
        SELECT COALESCE(agent_name, 'unattributed'), COUNT(*), COALESCE(SUM(total_tokens), 0), ROUND(AVG(duration_ms), 2)
        FROM telemetry_events
        WHERE occurred_at >= datetime('now', ?)
        GROUP BY agent_name ORDER BY COUNT(*) DESC
    """, (f"-{args.last} days",))
    _print_rows(["agent", "calls", "tokens", "avg_ms"], rows)
    return 0


def command_traces(args: argparse.Namespace) -> int:
    sink = _sink_from_args(args)
    where = "WHERE status = 'error'" if args.failed else ""
    rows = sink.query(f"""
        SELECT COALESCE(trace_id, 'none'), event_type, status, model, error_type, occurred_at
        FROM telemetry_events {where}
        ORDER BY occurred_at DESC LIMIT ?
    """, (args.limit,))
    _print_rows(["trace_id", "event", "status", "model", "error", "occurred_at"], rows)
    return 0


def command_prune(args: argparse.Namespace) -> int:
    count = _sink_from_args(args).prune(args.older_than)
    print(f"Deleted {count} event(s).")
    return 0


def command_vacuum(args: argparse.Namespace) -> int:
    _sink_from_args(args).vacuum()
    print("Vacuum completed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ollama-telemetry", description="Local SQLite telemetry for Ollama agents")
    parser.add_argument("--db", type=Path, help="Telemetry SQLite database path")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status")
    status.set_defaults(func=command_status)

    for name, func in (("stats", command_stats), ("models", command_models), ("agents", command_agents)):
        item = sub.add_parser(name)
        item.add_argument("--last", type=int, default=7, help="Days to include")
        item.set_defaults(func=func)

    traces = sub.add_parser("traces")
    traces.add_argument("--failed", action="store_true")
    traces.add_argument("--limit", type=int, default=25)
    traces.set_defaults(func=command_traces)

    prune = sub.add_parser("prune")
    prune.add_argument("--older-than", type=int, required=True)
    prune.set_defaults(func=command_prune)

    vacuum = sub.add_parser("vacuum")
    vacuum.set_defaults(func=command_vacuum)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))
