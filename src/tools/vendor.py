"""vendor_validate(attributes_json) — PASS/REVIEW/FAIL on vendor product upload."""
from __future__ import annotations

from typing import Any

# Required for every product upload.
REQUIRED_FIELDS: list[str] = [
    "vendor_id",
    "product_name",
    "category",
    "net_wt_oz",
    "net_vol_ml",
    "nicotine_content",
    "images",
]

# Categories whose uploads MUST include a current lab report (COA).
LAB_REPORT_CATEGORIES: set[str] = {
    "THC Beverage", "CBD Tincture", "Mushroom Gummies", "Kratom",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def vendor_validate(attributes_json: dict[str, Any]) -> dict[str, Any]:
    """Validate a vendor product-upload payload.

    Returns:
      {
        "status": "PASS" | "REVIEW" | "FAIL",
        "missing_fields": [...],
        "required_documents": [...],
        "checklist": [{field, ok, note}, ...],
      }
    """
    attrs = attributes_json or {}
    missing: list[str] = [f for f in REQUIRED_FIELDS if _is_missing(attrs.get(f))]

    required_docs: list[str] = []
    category = (attrs.get("category") or "").strip()
    if category in LAB_REPORT_CATEGORIES:
        if _is_missing(attrs.get("lab_report_url")):
            required_docs.append("LAB_REPORT_COA")

    # State restrictions optional but if present must be a list.
    if "state_restrictions" in attrs and not isinstance(
        attrs["state_restrictions"], list
    ):
        missing.append("state_restrictions(list)")

    checklist = [
        {"field": f, "ok": f not in missing,
         "note": "missing" if f in missing else "ok"}
        for f in REQUIRED_FIELDS
    ]
    if category in LAB_REPORT_CATEGORIES:
        ok = "LAB_REPORT_COA" not in required_docs
        checklist.append({
            "field": "lab_report_url", "ok": ok,
            "note": "required for this category" if not ok else "ok",
        })

    if missing or required_docs:
        # FAIL when core identity fields are missing; REVIEW for doc gaps.
        hard_missing = {"vendor_id", "product_name", "category"} & set(missing)
        status = "FAIL" if hard_missing else "REVIEW"
    else:
        status = "PASS"

    return {
        "status": status,
        "missing_fields": missing,
        "required_documents": required_docs,
        "checklist": checklist,
    }
