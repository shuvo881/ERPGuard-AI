"""LangGraph wiring + a single-shot `run_turn` entry point."""
from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.obs.logger import TraceEvent
from src.router.intent_router import llm_call_router, route_decision
from src.router.nodes import (
    TRACE_KEY,
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

    return g.compile()


def run_turn(*, user_input: str, user_type: UserType,
             session_id: str | None = None) -> dict[str, Any]:
    """Run one user turn end-to-end. Emits one trace event."""
    sid = session_id or "anon"
    ev = TraceEvent(user_type=user_type, user_input=user_input)

    state: State = {
        "request_id": ev.request_id,
        "session_id": sid,
        "user_type": user_type,
        "input": user_input,
        TRACE_KEY: ev,  # type: ignore[typeddict-unknown-key]
    }

    try:
        result = build_graph().invoke(state)
        ev.set_intent(result.get("decision", "DEFAULT"),
                      result.get("routed_by", "keyword"))
    except Exception as e:
        ev.error = repr(e)
        record = ev.finish()
        return {"error": repr(e), "trace": record}

    record = ev.finish()
    return {
        "request_id": ev.request_id,
        "intent": result.get("decision"),
        "routed_by": result.get("routed_by"),
        "output": result.get("output", ""),
        "tool_results": result.get("tool_results", {}),
        "trace": record,
    }


# Helper: brand-new session id (UUID4 first 8 chars).
def new_session_id() -> str:
    return uuid.uuid4().hex[:8]
