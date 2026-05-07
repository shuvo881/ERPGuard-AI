"""compliance_filter(state, product_ids) — deterministic legality decision.

Returns one record per product_id with status in {ALLOWED, BLOCKED, REVIEW}
and a machine-readable reason_code. The LLM may explain these results but
must NOT override or invent them.
"""
from __future__ import annotations

from typing import Any

from src.data.loader import product_by_id

# State -> set of restricted product flags (PoC policy table).
# In production this is sourced from a policy DB and versioned.
STATE_FLAG_RESTRICTIONS: dict[str, set[str]] = {
    "UT": {"thc", "kratom", "mushroom"},
    "ID": {"thc", "kratom", "mushroom"},
    "MA": {"nicotine"},
    "WI": {"thc"},
    "NY": {"mushroom"},
    "CA": set(),
}


def _decide(state: str, product: dict[str, Any]) -> dict[str, Any]:
    state_u = state.upper()
    pid = product["product_id"]
    sku = product["sku"]

    if state_u in {s.upper() for s in product["blocked_states"]}:
        return {
            "product_id": pid, "sku": sku,
            "status": "BLOCKED",
            "reason_code": "STATE_BLOCKED",
            "detail": f"{sku} is on the blocked-states list for {state_u}.",
        }

    restricted_flags = STATE_FLAG_RESTRICTIONS.get(state_u, set())
    triggered = [f for f, on in product["flags"].items()
                 if on and f in restricted_flags]
    if triggered:
        return {
            "product_id": pid, "sku": sku,
            "status": "BLOCKED",
            "reason_code": "STATE_FLAG_RESTRICTED",
            "detail": f"{sku} contains {','.join(triggered)} which is "
                      f"restricted in {state_u}.",
        }

    if product["lab_report_required"]:
        return {
            "product_id": pid, "sku": sku,
            "status": "REVIEW",
            "reason_code": "LAB_REPORT_REQUIRED",
            "detail": f"{sku} requires a current COA/lab report on file "
                      f"before shipping to {state_u}.",
        }

    return {
        "product_id": pid, "sku": sku,
        "status": "ALLOWED",
        "reason_code": "OK",
        "detail": f"{sku} has no restrictions for {state_u}.",
    }


def compliance_filter(state: str,
                      product_ids: list[int]) -> list[dict[str, Any]]:
    """Decide ALLOWED/BLOCKED/REVIEW for each product in `state`."""
    results: list[dict[str, Any]] = []
    for pid in product_ids:
        prod = product_by_id(int(pid))
        if prod is None:
            results.append({
                "product_id": pid, "sku": None,
                "status": "REVIEW",
                "reason_code": "UNKNOWN_PRODUCT",
                "detail": f"product_id={pid} not found in catalog.",
            })
            continue
        results.append(_decide(state, prod))
    return results
