"""Cheap keyword-first router with optional LLM fallback.

Per spec: classify intent cheaply WITHOUT calling an expensive model just
to route. We score each intent by keyword hits; if the winning score is
0 or there's a tie, we optionally fall back to the LLM (only if a key is
configured).
"""
from __future__ import annotations

import re

from src.ai.model import get_llm, llm_available
from src.schemas import Intent, State
from src.schemas.model import Route

# Keyword cues per intent. Lowercase. Order doesn't matter; we sum hits.
_KEYWORDS: dict[Intent, list[str]] = {
    "SALES_RECO": [
        "hot pick", "hot picks", "best seller", "recommend", "recommendation",
        "top product", "popular", "under $", "budget", "show me product",
    ],
    "COMPLIANCE_CHECK": [
        "legal", "allowed", "blocked", "ban", "compliance", "restricted",
        "why is", "alternative", "alternatives", "permitted", "lab report",
        "coa",
    ],
    "VENDOR_ONBOARDING": [
        "vendor", "onboard", "onboarding", "upload", "checklist",
        "missing field", "net wt", "net weight", "net vol", "fix",
    ],
    "OPS_STOCK": [
        "stock", "inventory", "warehouse", "in stock", "how much",
        "qty", "quantity", "available",
    ],
    "GENERAL_KB": [
        "policy", "policies", "sop", "shipping", "return", "returns",
        "guide", "how do", "what is the",
    ],
}

_BASKET_RE = re.compile(
    r"\b(?:add|put|order|buy)\b.*\b(?:basket|cart|order)\b", re.IGNORECASE
)


def keyword_route(text: str) -> tuple[Intent, int]:
    """Score-based classification. Returns (intent, score)."""
    t = (text or "").lower()
    scores: dict[Intent, int] = {k: 0 for k in _KEYWORDS}
    for intent, cues in _KEYWORDS.items():
        for cue in cues:
            if cue in t:
                scores[intent] += 1
    best = max(scores, key=lambda k: scores[k])
    return best, scores[best]


def is_basket_followup(text: str) -> bool:
    """Heuristic: 'add 2 of the first one to the basket' style follow-ups."""
    return bool(_BASKET_RE.search(text or ""))


def llm_call_router(state: State) -> dict:
    """Cheap router (no LLM). Falls back to LLM only on ambiguity AND
    only if ANTHROPIC_API_KEY is set."""
    text = state.get("input", "")

    intent, score = keyword_route(text)
    if score > 0:
        return {"decision": intent, "routed_by": "keyword"}

    # Cheap memory-based follow-up (e.g. "add 2 of the first one to the basket")
    # is handled by the default node — no LLM needed.
    if is_basket_followup(text):
        return {"decision": "DEFAULT", "routed_by": "memory"}

    if llm_available():
        try:
            router = get_llm().with_structured_output(Route)
            decision: Route = router.invoke(
                "Classify the user's request into one of: "
                "SALES_RECO, COMPLIANCE_CHECK, VENDOR_ONBOARDING, "
                "OPS_STOCK, GENERAL_KB, DEFAULT.\n\n"
                f"User: {text}"
            )
            return {"decision": decision.step, "routed_by": "llm"}
        except Exception:  # pragma: no cover - defensive
            pass

    return {"decision": "DEFAULT", "routed_by": "keyword"}


def route_decision(state: State) -> str:
    """LangGraph conditional-edge selector."""
    return state.get("decision", "DEFAULT")
