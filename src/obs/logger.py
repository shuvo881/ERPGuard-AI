"""Structured JSON observability.

Each request produces one JSON line covering:
  request_id, user_type, intent, tools_called (with args + latency_ms),
  total_latency_ms, prompt_token_estimate.

Logs go to stdout and (append) to logs/trace.jsonl.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "trace.jsonl"


def _approx_tokens(text: str) -> int:
    # Cheap approximation: ~4 chars per token.
    return max(1, len(text) // 4)


class TraceEvent:
    """Accumulates a single request's observability data."""

    def __init__(self, *, user_type: str, user_input: str) -> None:
        self.request_id: str = uuid.uuid4().hex[:12]
        self.user_type: str = user_type
        self.user_input: str = user_input
        self.intent: str | None = None
        self.routed_by: str | None = None
        self.tools: list[dict[str, Any]] = []
        self.prompt_token_estimate: int = 0
        self.output_token_estimate: int = 0
        self.error: str | None = None
        self._t0: float = time.perf_counter()

    def set_intent(self, intent: str, routed_by: str) -> None:
        self.intent = intent
        self.routed_by = routed_by

    def add_tool(self, name: str, args: dict[str, Any], latency_ms: float,
                 result_summary: dict[str, Any] | None = None) -> None:
        self.tools.append({
            "name": name,
            "args": args,
            "latency_ms": round(latency_ms, 2),
            "result_summary": result_summary or {},
        })

    def record_prompt(self, prompt: str) -> None:
        self.prompt_token_estimate += _approx_tokens(prompt)

    def record_output(self, text: str) -> None:
        self.output_token_estimate += _approx_tokens(text)

    def finish(self) -> dict[str, Any]:
        total_ms = (time.perf_counter() - self._t0) * 1000.0
        record = {
            "request_id": self.request_id,
            "user_type": self.user_type,
            "intent": self.intent,
            "routed_by": self.routed_by,
            "user_input": self.user_input,
            "tools_called": self.tools,
            "total_latency_ms": round(total_ms, 2),
            "prompt_token_estimate": self.prompt_token_estimate,
            "output_token_estimate": self.output_token_estimate,
            "error": self.error,
        }
        line = json.dumps(record, ensure_ascii=False)
        # Console (one JSON line — easy to grep).
        print(line, file=sys.stderr)
        try:
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
        return record


@contextmanager
def time_tool(event: TraceEvent, name: str, args: dict[str, Any]):
    """Context manager that times a tool call and registers it on the event."""
    t0 = time.perf_counter()
    holder: dict[str, Any] = {"summary": {}}
    try:
        yield holder
    finally:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        event.add_tool(name, args, latency_ms, holder.get("summary"))


def audit(action: str, *, user_type: str, request_id: str,
          payload: dict[str, Any]) -> None:
    """Append-only audit hook for sensitive tool calls.

    In production this would write to a tamper-evident store
    (e.g., CloudTrail / DB with hash chain). Here we just log.
    """
    line = json.dumps({
        "audit": True,
        "action": action,
        "user_type": user_type,
        "request_id": request_id,
        "payload": payload,
    })
    if os.environ.get("ERPGUARD_AUDIT_STDERR", "1") == "1":
        print(line, file=sys.stderr)
