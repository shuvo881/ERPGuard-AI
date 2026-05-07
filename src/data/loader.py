"""Single source of truth for seed data. Loaded once and cached."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "seed_data.json"


@lru_cache(maxsize=1)
def load_seed() -> dict[str, Any]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def products() -> list[dict[str, Any]]:
    return load_seed()["products"]


def inventory() -> list[dict[str, Any]]:
    return load_seed()["inventory"]


def vendors() -> list[dict[str, Any]]:
    return load_seed()["vendors"]


def customers() -> list[dict[str, Any]]:
    return load_seed()["customers"]


def kb_docs() -> list[dict[str, Any]]:
    return load_seed()["kb_docs"]


def product_by_id(pid: int) -> dict[str, Any] | None:
    for p in products():
        if p["product_id"] == pid:
            return p
    return None


def product_by_sku(sku: str) -> dict[str, Any] | None:
    sku_n = sku.strip().upper()
    for p in products():
        if p["sku"].upper() == sku_n:
            return p
    return None
