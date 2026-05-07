"""hot_picks(state, budget, limit) — popularity-ranked products that fit budget.

Deterministic. No LLM involvement.
"""
from __future__ import annotations

from typing import Any

from src.data.loader import products


def hot_picks(state: str | None, budget: float | None,
              limit: int = 5) -> list[dict[str, Any]]:
    """Return top `limit` products by popularity_score that the requester
    can plausibly afford and that aren't outright blocked in `state`.

    Notes:
      * Affordability heuristic = unit price <= budget. (PoC; real engine
        would model qty * price + tax + shipping.)
      * State filter only excludes products whose `blocked_states` lists
        the state. Final legality is decided by `compliance_filter`.
    """
    rows = products()
    out: list[dict[str, Any]] = []
    for p in rows:
        if budget is not None and p["price"] > float(budget):
            continue
        if state is not None and state.upper() in {s.upper() for s in p["blocked_states"]}:
            continue
        out.append({
            "product_id": p["product_id"],
            "sku": p["sku"],
            "name": p["name"],
            "category": p["category"],
            "price": p["price"],
            "popularity_score": p["popularity_score"],
        })
    out.sort(key=lambda r: r["popularity_score"], reverse=True)
    return out[: max(1, int(limit))]
