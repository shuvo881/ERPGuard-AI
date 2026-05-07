"""Schemas for the orchestrator: graph state, router output, sessions."""
from __future__ import annotations

from typing import Any
from typing_extensions import Literal, TypedDict

from pydantic import BaseModel, Field

# Canonical intents from the spec.
Intent = Literal[
    "SALES_RECO",
    "COMPLIANCE_CHECK",
    "VENDOR_ONBOARDING",
    "OPS_STOCK",
    "GENERAL_KB",
    "DEFAULT",
]

UserType = Literal["internal_sales", "portal_vendor", "portal_customer"]


class Route(BaseModel):
    """LLM-fallback router structured output."""

    step: Intent = Field(
        "DEFAULT", description="The next step in the routing process."
    )


class Session(TypedDict, total=False):
    """Per-session memory (carried across turns)."""

    session_id: str
    last_intent: Intent
    last_state: str  # e.g. "CA"
    last_budget: float
    last_product_ids: list[int]


class State(TypedDict, total=False):
    """LangGraph state for a single turn."""

    # Inputs
    request_id: str
    session_id: str
    user_type: UserType
    input: str
    # Routing
    decision: Intent
    routed_by: str  # "keyword" | "llm" | "memory"
    # Extracted slots
    state_code: str
    budget: float
    product_ids: list[int]
    vendor_attrs: dict[str, Any]
    matched_by: str  # "sku" | "name" | "memory" (compliance chain)
    # Per-step intermediate tool outputs (each chain-node writes its slice;
    # nothing here is ever dumped wholesale into a prompt).
    picks: list[dict[str, Any]]
    legality: list[dict[str, Any]]
    alternatives: list[dict[str, Any]]
    vendor_result: dict[str, Any]
    stock_rows: list[dict[str, Any]]
    kb_snippets: list[dict[str, Any]]
    # Aggregated payload + final response.
    tool_results: dict[str, Any]
    output: str
    # Observability handle — TraceEvent object passed through every node.
    # Declared here so LangGraph preserves it across state updates.
    _trace: Any
