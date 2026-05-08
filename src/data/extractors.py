"""Cheap deterministic slot extraction from the user's text.

Used by the router and the chain nodes so we don't burn LLM tokens
just to pull out a state code, a budget, or a SKU.
"""
from __future__ import annotations

import json
import re
from typing import Any

US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY",
}

_STATE_RE = re.compile(r"\b([A-Z]{2})\b")
_BUDGET_RE = re.compile(
    r"(?:under|below|<=?|less than|max(?:imum)?|budget(?: of)?)\s*\$?\s*"
    r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(k|K)?",
)
_DOLLAR_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(k|K)?")
_SKU_RE = re.compile(r"\bSKU[-_ ]?(\d{3,6})\b", re.IGNORECASE)
_PID_RE = re.compile(r"\bproduct[_ ]?id\s*[:=]?\s*(\d{3,6})\b", re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_state(text: str) -> str | None:
    for m in _STATE_RE.finditer(text or ""):
        code = m.group(1)
        if code in US_STATES:
            return code
    return None


def _to_float(num: str, k_suffix: str | None) -> float:
    val = float(num.replace(",", ""))
    if k_suffix:
        val *= 1000.0
    return val


def extract_budget(text: str) -> float | None:
    if not text:
        return None
    m = _BUDGET_RE.search(text)
    if m:
        return _to_float(m.group(1), m.group(2))
    m = _DOLLAR_RE.search(text)
    if m:
        return _to_float(m.group(1), m.group(2))
    return None


def extract_product_ids(text: str) -> list[int]:
    """SKU / product_id matches only (cheap regex). For name matches use
    `match_products_by_name`."""
    ids: list[int] = []
    for m in _SKU_RE.finditer(text or ""):
        ids.append(int(m.group(1)))
    for m in _PID_RE.finditer(text or ""):
        v = int(m.group(1))
        if v not in ids:
            ids.append(v)
    return ids


def match_products_by_name(text: str, limit: int = 5) -> list[int]:
    """Naive name match against the catalog.

    A product matches if any non-stopword token from its name appears as a
    word in the user text. Ranked by number of overlapping tokens.
    Imported lazily to avoid a circular import at module load time.
    """
    if not text:
        return []
    from src.data.loader import products  # local import (data layer)

    user_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)
                   if len(t) >= 3}
    if not user_tokens:
        return []

    # Tokens that are too generic to count as a name match on their own.
    generic = {
        "the", "and", "for", "with", "vape", "beverage", "tincture",
        "gummies", "accessories", "product", "products", "kratom",
        "mushroom", "nicotine", "thc", "cbd",
    }
    discriminating = user_tokens - generic

    scored: list[tuple[int, int]] = []
    for p in products():
        name_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", p["name"])
                       if len(t) >= 3}
        # Require at least one *discriminating* overlap (e.g. "Cloud", "90"),
        # not just a generic category word.
        overlap = name_tokens & discriminating
        if not overlap:
            continue
        scored.append((len(overlap), p["product_id"]))

    scored.sort(reverse=True)
    return [pid for _, pid in scored[:limit]]


def extract_vendor_attrs(text: str) -> dict[str, Any]:
    """Pull a JSON object out of the text if present, else infer from prose.

    The PoC accepts either:
      * an inline JSON blob, OR
      * a sentence like "missing Net Wt and no lab report" (best effort).
    """
    if not text:
        return {}
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    lower = text.lower()
    attrs: dict[str, Any] = {
        "vendor_id": "VEND-DEMO",
        "product_name": "Demo Product",
        "category": "THC Beverage",
        "net_wt_oz": 8.0,
        "net_vol_ml": 250.0,
        "nicotine_content": 0.0,
        "images": ["img1.jpg"],
        "lab_report_url": "https://example.com/coa.pdf",
    }
    if "net wt" in lower or "net weight" in lower:
        attrs["net_wt_oz"] = None
    if "net vol" in lower or "net volume" in lower:
        attrs["net_vol_ml"] = None
    if "lab report" in lower or "coa" in lower:
        attrs["lab_report_url"] = None
    if "image" in lower and "no image" in lower:
        attrs["images"] = []
    if "nicotine content" in lower and "missing" in lower:
        attrs["nicotine_content"] = None
    return attrs
