"""Shared helpers used by every chain node.

Kept separate from `nodes.py` so each file stays small and focused.
"""
from __future__ import annotations

from typing import Any, Callable

from src.ai.model import get_llm, llm_available
from src.obs.logger import TraceEvent, audit, time_tool
from src.schemas import State
from src.security.policy import (
    PermissionDenied,
    assert_allowed,
    redact_payload,
)
from src.session import store as session_store

# TraceEvent is stashed on State under this key so nodes can append to it.
TRACE_KEY = "_trace"


def get_trace(state: State) -> TraceEvent:
    ev = state.get(TRACE_KEY)  # type: ignore[arg-type]
    assert ev is not None, "trace event missing from state"
    return ev  # type: ignore[return-value]


def safe_call(state: State, tool_name: str, args: dict[str, Any],
              fn: Callable[..., Any]) -> Any:
    """Allowlist-check, audit, time, and invoke a deterministic tool."""
    user_type = state["user_type"]
    request_id = state["request_id"]
    try:
        assert_allowed(user_type, tool_name)
    except PermissionDenied as e:
        audit("permission_denied", user_type=user_type,
              request_id=request_id,
              payload={"tool": tool_name, "args": args})
        raise e
    audit("tool_invoked", user_type=user_type, request_id=request_id,
          payload={"tool": tool_name, "args": args})
    with time_tool(get_trace(state), tool_name, args) as h:
        result = fn(**args)
        if isinstance(result, list):
            h["summary"] = {"count": len(result)}
        elif isinstance(result, dict):
            h["summary"] = {"keys": list(result.keys())[:8]}
    return result


def deterministic_format(intent: str, payload: dict[str, Any]) -> str:
    import json as _json
    return f"[{intent}] " + _json.dumps(payload, indent=2, default=str)


def format_with_llm(state: State, system: str,
                    payload: dict[str, Any]) -> str:
    """Use LLM to format the payload if a key is configured; otherwise
    return a deterministic JSON rendering. Payload is PII-redacted first."""
    safe_payload = redact_payload(payload)
    intent = state.get("decision", "DEFAULT")
    if not llm_available():
        return deterministic_format(intent, safe_payload)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        prompt_text = f"{system}\n\nFACTS: {safe_payload}"
        get_trace(state).record_prompt(prompt_text)
        msg = get_llm().invoke([
            SystemMessage(content=system),
            HumanMessage(content=str(safe_payload)),
        ])
        text = getattr(msg, "content", str(msg))
        get_trace(state).record_output(text)
        return text
    except Exception as e:  # pragma: no cover - defensive
        return (deterministic_format(intent, safe_payload)
                + f"\n\n[LLM unavailable: {e!r}]")


def persist_session(state: State, **fields: Any) -> None:
    session_store.update(state["session_id"], **fields)
