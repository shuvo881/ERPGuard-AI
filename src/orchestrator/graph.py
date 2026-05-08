"""LangGraph wiring + a single-shot `run_turn` entry point.

Checkpointing
-------------
The graph is compiled with a checkpointer so each turn for a given
`session_id` (used as `thread_id`) resumes against the previously
persisted State. This gives us:
  * Resumability across crashes (when SQLite backend is enabled).
  * Implicit cross-turn memory in addition to the explicit session_store.
  * The ability to inspect / time-travel a thread via the checkpointer API.

Backends:
  * Default: in-memory `MemorySaver` (lost on process exit).
  * If `ERPGUARD_CHECKPOINT_SQLITE=1`, persists to
    `data/checkpoints.sqlite` via `SqliteSaver` (requires the optional
    `langgraph-checkpoint-sqlite` package). Falls back to MemorySaver
    with a warning if the package is not installed.
"""
from __future__ import annotations

import os
import sys
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.obs.logger import (
    TraceEvent,
    reset_current_trace,
    set_current_trace,
)
from src.router.intent_router import llm_call_router, route_decision
from src.router.nodes import (
    compliance_alternatives_node,
    compliance_filter_node,
    compliance_identify_node,
    default_node,
    format_node,
    kb_search_node,
    ops_stock_node,
    sales_compliance_filter_node,
    sales_hot_picks_node,
    vendor_validate_node,
)
from src.schemas import State, UserType

CHECKPOINT_DB = Path(__file__).resolve().parents[2] / "data" / "checkpoints.sqlite"


@lru_cache(maxsize=1)
def _checkpointer() -> Any:
    """Pick a checkpointer backend once and cache it.

    Note: `SqliteSaver.from_conn_string(...)` returns a context manager,
    not a saver. For an app-lifetime saver we open the sqlite3 connection
    ourselves and pass it to `SqliteSaver(conn)` directly.
    """
    if os.environ.get("ERPGUARD_CHECKPOINT_SQLITE", "1") == "1":
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False so the saver works across threads
            # (LangGraph nodes may execute on a worker thread).
            conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
            return SqliteSaver(conn)
        except ImportError:
            print(
                "[checkpoint] langgraph-checkpoint-sqlite not installed; "
                "falling back to MemorySaver. Install with: "
                "`uv add langgraph-checkpoint-sqlite`",
                file=sys.stderr,
            )
    return MemorySaver()


@lru_cache(maxsize=1)
def build_graph():
    """Tool-per-node LangGraph wiring (mirrors spec §6 chains).

    START -> ROUTER -> {
      SALES_RECO:        sales_hot_picks -> sales_compliance_filter -> format
      COMPLIANCE_CHECK:  compliance_identify -> compliance_filter
                                             -> compliance_alternatives -> format
      VENDOR_ONBOARDING: vendor_validate -> format
      OPS_STOCK:         stock_by_warehouse -> format
      GENERAL_KB:        kb_search -> format
      DEFAULT:           default -> END
    } -> END
    """
    g = StateGraph(State)

    # Router (cheap keyword + optional LLM fallback).
    g.add_node("ROUTER", llm_call_router)

    # Chain A — SALES_RECO.
    g.add_node("sales_hot_picks", sales_hot_picks_node)
    g.add_node("sales_compliance_filter", sales_compliance_filter_node)

    # Chain B — COMPLIANCE_CHECK.
    g.add_node("compliance_identify", compliance_identify_node)
    g.add_node("compliance_filter", compliance_filter_node)
    g.add_node("compliance_alternatives", compliance_alternatives_node)

    # Chain C — VENDOR_ONBOARDING.
    g.add_node("vendor_validate", vendor_validate_node)

    # OPS / KB / DEFAULT.
    g.add_node("ops_stock", ops_stock_node)
    g.add_node("kb_search", kb_search_node)
    g.add_node("default", default_node)

    # Shared format step (per-intent system prompt; reads accumulated state).
    g.add_node("format", format_node)

    # Routing: START -> ROUTER -> first node of the chosen chain.
    g.add_edge(START, "ROUTER")
    g.add_conditional_edges(
        "ROUTER",
        route_decision,
        {
            "SALES_RECO":        "sales_hot_picks",
            "COMPLIANCE_CHECK":  "compliance_identify",
            "VENDOR_ONBOARDING": "vendor_validate",
            "OPS_STOCK":         "ops_stock",
            "GENERAL_KB":        "kb_search",
            "DEFAULT":           "default",
        },
    )

    # Chain A edges.
    g.add_edge("sales_hot_picks", "sales_compliance_filter")
    g.add_edge("sales_compliance_filter", "format")

    # Chain B edges.
    g.add_edge("compliance_identify", "compliance_filter")
    g.add_edge("compliance_filter", "compliance_alternatives")
    g.add_edge("compliance_alternatives", "format")

    # Chain C / OPS / KB edges.
    g.add_edge("vendor_validate", "format")
    g.add_edge("ops_stock", "format")
    g.add_edge("kb_search", "format")

    # DEFAULT node produces its own output and goes straight to END.
    g.add_edge("default", END)
    g.add_edge("format", END)

    return g.compile(checkpointer=_checkpointer())


def _thread_config(session_id: str) -> dict[str, Any]:
    """Map our session_id onto LangGraph's checkpoint thread_id."""
    return {"configurable": {"thread_id": session_id}}


def run_turn(*, user_input: str, user_type: UserType,
             session_id: str | None = None) -> dict[str, Any]:
    """Run one user turn end-to-end against the checkpointed thread.

    The checkpointer persists State between turns keyed by `session_id`.
    Each call passes a fresh `request_id` and `input` — those overwrite
    the prior values; everything else (e.g. last picks, last state_code)
    survives in the thread.

    The active TraceEvent is bound to a ContextVar (NOT to State) so
    LangGraph's checkpoint serializer never sees it.
    """
    sid = session_id or "anon"
    ev = TraceEvent(user_type=user_type, user_input=user_input)

    state: State = {
        "request_id": ev.request_id,
        "session_id": sid,
        "user_type": user_type,
        "input": user_input,
    }

    token = set_current_trace(ev)
    try:
        result = build_graph().invoke(state, config=_thread_config(sid))
        ev.set_intent(result.get("decision", "DEFAULT"),
                      result.get("routed_by", "keyword"))
    except Exception as e:
        ev.error = repr(e)
        record = ev.finish()
        return {"error": repr(e), "trace": record}
    finally:
        reset_current_trace(token)

    record = ev.finish()
    return {
        "request_id": ev.request_id,
        "intent": result.get("decision"),
        "routed_by": result.get("routed_by"),
        "output": result.get("output", ""),
        "tool_results": result.get("tool_results", {}),
        "trace": record,
    }


def get_checkpoint(session_id: str) -> dict[str, Any] | None:
    """Return the latest persisted State for a session, or None.

    Useful for debugging / demoing memory ('what does the graph remember?').
    """
    cp = _checkpointer().get(_thread_config(session_id))
    if cp is None:
        return None
    values = cp.get("channel_values") if isinstance(cp, dict) else getattr(cp, "channel_values", None)
    return dict(values) if values else None


def clear_session(session_id: str) -> None:
    """Best-effort per-thread reset for tests/demos.

    Tries the in-process MemorySaver internals first (deletes only the
    given thread). For other backends (e.g. SqliteSaver), falls back to
    dropping the cached graph + checkpointer (resets ALL threads).
    """
    cp = _checkpointer()
    storage = getattr(cp, "storage", None)
    if isinstance(storage, dict) and session_id in storage:
        del storage[session_id]
        return
    build_graph.cache_clear()
    _checkpointer.cache_clear()


# Helper: brand-new session id (UUID4 first 8 chars).
def new_session_id() -> str:
    return uuid.uuid4().hex[:8]
