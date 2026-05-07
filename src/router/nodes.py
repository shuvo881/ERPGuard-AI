"""Per-tool LangGraph nodes.

Each node performs ONE deterministic tool call (or one preparation/format
step) and writes its slice into the graph state. Chains are wired in
src/router/graph.py — the graph topology mirrors the spec:

  SALES_RECO    : hot_picks -> compliance_filter -> format
  COMPLIANCE    : identify_products -> compliance_filter -> alternatives -> format
  VENDOR_ONB    : vendor_validate -> format
  OPS_STOCK     : stock_by_warehouse -> format
  GENERAL_KB    : kb_search -> format
  DEFAULT       : default -> END (no format step needed)

Rules:
  * Tools decide facts; the LLM (if configured) only formats.
  * Allowlist is enforced and an audit event is emitted per tool call.
  * PII is redacted before anything is sent to the LLM.
"""
from __future__ import annotations

from typing import Any

from src.router.extractors import (
    extract_budget,
    extract_product_ids,
    extract_state,
    extract_vendor_attrs,
    match_products_by_name,
)
from src.router.intent_router import is_basket_followup
from src.router.nodes_common import (
    TRACE_KEY,
    format_with_llm,
    persist_session,
    safe_call,
)
from src.schemas import State
from src.session import store as session_store
from src.tools import (
    compliance_filter,
    hot_picks,
    kb_search,
    stock_by_warehouse,
    vendor_validate,
)

__all__ = [
    "TRACE_KEY",
    # sales chain
    "sales_hot_picks_node", "sales_compliance_filter_node",
    # compliance chain
    "compliance_identify_node", "compliance_filter_node",
    "compliance_alternatives_node",
    # vendor / ops / kb / default
    "vendor_validate_node", "ops_stock_node", "kb_search_node",
    "default_node",
    # shared format
    "format_node",
]


# ---------------------------------------------------------------------------
# Chain A — SALES_RECO: hot_picks -> compliance_filter -> format
# ---------------------------------------------------------------------------

def sales_hot_picks_node(state: State) -> dict:
    """Step 1/2 of Chain A: rank popular products that fit budget+state."""
    sess = session_store.get(state["session_id"])
    st = extract_state(state["input"]) or sess.get("last_state")
    budget = extract_budget(state["input"]) or sess.get("last_budget")

    picks = safe_call(state, "hot_picks",
                      {"state": st, "budget": budget, "limit": 5},
                      hot_picks)
    return {
        "state_code": st,
        "budget": budget,
        "picks": picks,
        "product_ids": [p["product_id"] for p in picks],
    }


def sales_compliance_filter_node(state: State) -> dict:
    """Step 2/2 of Chain A: filter picks down to ALLOWED in `state_code`."""
    st = state.get("state_code")
    picks = state.get("picks", []) or []
    pids = [p["product_id"] for p in picks]

    legality: list[dict[str, Any]] = []
    if st and pids:
        legality = safe_call(state, "compliance_filter",
                             {"state": st, "product_ids": pids},
                             compliance_filter)

    if legality:
        allowed = {r["product_id"] for r in legality if r["status"] == "ALLOWED"}
        final_picks = [p for p in picks if p["product_id"] in allowed]
    else:
        final_picks = picks

    persist_session(state,
                    last_intent="SALES_RECO",
                    last_state=st,
                    last_budget=state.get("budget"),
                    last_product_ids=[p["product_id"] for p in final_picks])

    return {
        "picks": final_picks,
        "legality": legality,
        "product_ids": [p["product_id"] for p in final_picks],
    }


# ---------------------------------------------------------------------------
# Chain B — COMPLIANCE_CHECK:
#   identify_products -> compliance_filter -> alternatives -> format
# ---------------------------------------------------------------------------

def compliance_identify_node(state: State) -> dict:
    """Step 1/3 of Chain B: resolve product_ids by SKU, then name, then memory."""
    sess = session_store.get(state["session_id"])
    text_in = state["input"]
    st = extract_state(text_in) or sess.get("last_state")

    pids = extract_product_ids(text_in)
    matched_by = "sku" if pids else None
    if not pids:
        pids = match_products_by_name(text_in)
        if pids:
            matched_by = "name"
    if not pids:
        pids = list(sess.get("last_product_ids") or [])
        if pids:
            matched_by = "memory"

    return {
        "state_code": st,
        "product_ids": pids,
        "matched_by": matched_by or "none",
    }


def compliance_filter_node(state: State) -> dict:
    """Step 2/3 of Chain B: deterministic legality decision per product."""
    st = state.get("state_code")
    pids = state.get("product_ids") or []
    if not st or not pids:
        return {"legality": []}
    legality = safe_call(state, "compliance_filter",
                         {"state": st, "product_ids": pids},
                         compliance_filter)
    persist_session(state,
                    last_intent="COMPLIANCE_CHECK",
                    last_state=st, last_product_ids=pids)
    return {"legality": legality}


def compliance_alternatives_node(state: State) -> dict:
    """Step 3/3 of Chain B: surface up to 3 ALLOWED alternatives if any
    decision was BLOCKED or REVIEW."""
    st = state.get("state_code")
    legality = state.get("legality") or []
    if not st or not legality:
        return {"alternatives": []}
    if all(r["status"] == "ALLOWED" for r in legality):
        return {"alternatives": []}

    sess = session_store.get(state["session_id"])
    cands = safe_call(state, "hot_picks",
                      {"state": st, "budget": sess.get("last_budget"),
                       "limit": 10}, hot_picks)
    cand_ids = [c["product_id"] for c in cands]
    cand_legality = safe_call(state, "compliance_filter",
                              {"state": st, "product_ids": cand_ids},
                              compliance_filter)
    ok_ids = {r["product_id"] for r in cand_legality
              if r["status"] == "ALLOWED"}
    alternatives = [c for c in cands if c["product_id"] in ok_ids][:3]
    return {"alternatives": alternatives}


# ---------------------------------------------------------------------------
# Chain C — VENDOR_ONBOARDING: vendor_validate -> format
# ---------------------------------------------------------------------------

def vendor_validate_node(state: State) -> dict:
    attrs = extract_vendor_attrs(state["input"]) or state.get("vendor_attrs", {})
    result = safe_call(state, "vendor_validate",
                       {"attributes_json": attrs}, vendor_validate)
    persist_session(state, last_intent="VENDOR_ONBOARDING")
    return {"vendor_attrs": attrs, "vendor_result": result}


# ---------------------------------------------------------------------------
# OPS_STOCK: stock_by_warehouse -> format
# ---------------------------------------------------------------------------

def ops_stock_node(state: State) -> dict:
    sess = session_store.get(state["session_id"])
    pids = extract_product_ids(state["input"]) or list(
        sess.get("last_product_ids") or []
    )
    if not pids:
        return {"stock_rows": [], "product_ids": []}
    rows: list[dict[str, Any]] = []
    for pid in pids:
        rows.append(safe_call(state, "stock_by_warehouse",
                              {"product_id": pid}, stock_by_warehouse))
    persist_session(state, last_intent="OPS_STOCK", last_product_ids=pids)
    return {"stock_rows": rows, "product_ids": pids}


# ---------------------------------------------------------------------------
# GENERAL_KB: kb_search -> format
# ---------------------------------------------------------------------------

_VISIBILITY_BY_USER = {
    "internal_sales": None,       # see everything
    "portal_vendor":  "vendor",
    "portal_customer": "public",
}


def kb_search_node(state: State) -> dict:
    vis = _VISIBILITY_BY_USER.get(state["user_type"])
    snippets = safe_call(state, "kb_search",
                         {"query": state["input"], "top_k": 3,
                          "visibility": vis},
                         kb_search)
    persist_session(state, last_intent="GENERAL_KB")
    return {"kb_snippets": snippets}


# ---------------------------------------------------------------------------
# DEFAULT: memory-based basket follow-up, otherwise punt. Goes straight
# to END (no shared format step needed; produces its own output).
# ---------------------------------------------------------------------------

def default_node(state: State) -> dict:
    if is_basket_followup(state["input"]):
        sess = session_store.get(state["session_id"])
        last_pids = list(sess.get("last_product_ids") or [])
        if last_pids:
            payload = {
                "basket_added": [last_pids[0]],
                "context": {
                    "last_intent": sess.get("last_intent"),
                    "last_state": sess.get("last_state"),
                },
            }
            text = format_with_llm(
                state,
                "You are a sales assistant. Acknowledge that we added the "
                "first prior pick to the basket. Reference the SKU.",
                payload,
            )
            return {"output": text, "tool_results": payload,
                    "product_ids": last_pids[:1]}

    text = ("I couldn't classify that request. Try asking about hot picks, "
            "compliance for a SKU, stock lookup, vendor onboarding, or a "
            "policy question.")
    return {"output": text, "tool_results": {"note": "no_intent_matched"}}


# ---------------------------------------------------------------------------
# Shared FORMAT node — picks the per-intent system prompt and renders the
# accumulated tool outputs from state. The LLM never sees raw seed data:
# only the small, intent-specific payload built here.
# ---------------------------------------------------------------------------

_FORMAT_SYSTEMS: dict[str, str] = {
    "SALES_RECO": (
        "You are an internal sales assistant. Present the picks as a short "
        "ranked list with price and a one-line note. Do not invent products."
    ),
    "COMPLIANCE_CHECK": (
        "You are a compliance assistant. State the decision per SKU using "
        "the reason_code, then list any alternatives. Never override the "
        "deterministic decision."
    ),
    "VENDOR_ONBOARDING": (
        "You are a vendor-onboarding assistant. Tell the vendor exactly "
        "which fields are missing and which documents to upload. Use the "
        "validation result verbatim; do not add or remove items."
    ),
    "OPS_STOCK": (
        "You are an ops assistant. Show the per-warehouse quantities and a "
        "total per SKU. Do not invent warehouses."
    ),
    "GENERAL_KB": (
        "You are a KB assistant. Answer using ONLY the provided snippets. "
        "If the snippets don't cover the question, say so."
    ),
}


def _payload_for(state: State) -> dict[str, Any]:
    intent = state.get("decision", "DEFAULT")
    if intent == "SALES_RECO":
        return {
            "state": state.get("state_code"),
            "budget": state.get("budget"),
            "picks": state.get("picks", []),
            "compliance_summary": state.get("legality", []),
        }
    if intent == "COMPLIANCE_CHECK":
        if not state.get("state_code"):
            return {"error": "missing_state"}
        if not state.get("product_ids"):
            return {"error": "missing_product"}
        return {
            "state": state.get("state_code"),
            "matched_by": state.get("matched_by"),
            "decisions": state.get("legality", []),
            "alternatives": state.get("alternatives", []),
        }
    if intent == "VENDOR_ONBOARDING":
        return {
            "submitted": state.get("vendor_attrs", {}),
            "validation": state.get("vendor_result", {}),
        }
    if intent == "OPS_STOCK":
        if not state.get("product_ids"):
            return {"error": "missing_product"}
        return {"stock": state.get("stock_rows", [])}
    if intent == "GENERAL_KB":
        return {"snippets": state.get("kb_snippets", [])}
    return {"note": "unhandled_intent"}


def format_node(state: State) -> dict:
    intent = state.get("decision", "DEFAULT")
    payload = _payload_for(state)
    if "error" in payload:
        msg = {
            "missing_state": "Need a US state code (e.g. 'CA') to proceed.",
            "missing_product": "Need at least one SKU, product name, or "
                               "product_id to proceed.",
        }.get(payload["error"], "Insufficient information.")
        return {"output": msg, "tool_results": payload}
    system = _FORMAT_SYSTEMS.get(intent, "You are a helpful assistant.")
    text = format_with_llm(state, system, payload)
    return {"output": text, "tool_results": payload}

