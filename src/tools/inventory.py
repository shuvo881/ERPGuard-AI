"""stock_by_warehouse(product_id) — qty per warehouse for a product."""
from __future__ import annotations

from typing import Any

from src.data.loader import inventory, product_by_id


def stock_by_warehouse(product_id: int) -> dict[str, Any]:
    pid = int(product_id)
    prod = product_by_id(pid)
    rows = [
        {"warehouse": r["warehouse"], "qty": r["qty"]}
        for r in inventory()
        if r["product_id"] == pid
    ]
    rows.sort(key=lambda r: r["qty"], reverse=True)
    return {
        "product_id": pid,
        "sku": prod["sku"] if prod else None,
        "name": prod["name"] if prod else None,
        "total_qty": sum(r["qty"] for r in rows),
        "warehouses": rows,
    }
