"""Tool allowlists per user_type and a PII redaction stub.

The orchestrator MUST call `assert_allowed(user_type, tool_name)` before
invoking any tool. PII redaction is applied to anything we hand to the LLM.
"""
from __future__ import annotations

import re
from typing import Any

from src.schemas import UserType

# Tool name -> set of user_types allowed to invoke it.
TOOL_ALLOWLIST: dict[str, set[str]] = {
    "hot_picks":          {"internal_sales", "portal_customer"},
    "compliance_filter":  {"internal_sales", "portal_customer", "portal_vendor"},
    "stock_by_warehouse": {"internal_sales"},
    "vendor_validate":    {"internal_sales", "portal_vendor"},
    "kb_search":          {"internal_sales", "portal_customer", "portal_vendor"},
}


class PermissionDenied(Exception):
    pass


def is_allowed(user_type: UserType, tool_name: str) -> bool:
    allowed = TOOL_ALLOWLIST.get(tool_name, set())
    return user_type in allowed


def assert_allowed(user_type: UserType, tool_name: str) -> None:
    if not is_allowed(user_type, tool_name):
        raise PermissionDenied(
            f"user_type={user_type!r} is not allowed to call tool={tool_name!r}"
        )


# --- PII redaction --------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b")
# Naive credit-card-ish 13-19 digit run.
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def redact_pii(text: str) -> str:
    """Replace common PII before sending text to an LLM.

    NOTE: This is a deliberately small stub. Production should use a
    dedicated DLP pipeline (Presidio, regex+ML, named-entity scrubbing).
    """
    if not text:
        return text
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _PHONE_RE.sub("<PHONE>", text)
    text = _CARD_RE.sub("<CARD>", text)
    return text


def redact_payload(payload: Any) -> Any:
    """Recursively redact PII inside dict/list payloads handed to an LLM."""
    if isinstance(payload, str):
        return redact_pii(payload)
    if isinstance(payload, dict):
        return {k: redact_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [redact_payload(v) for v in payload]
    return payload
