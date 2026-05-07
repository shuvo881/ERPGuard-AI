"""Lazy LLM wrapper.

We only construct the Anthropic client when something actually needs it.
If ANTHROPIC_API_KEY is unset, callers should use the deterministic
formatters in src/router/nodes.py instead.
"""
from __future__ import annotations

import os
from functools import lru_cache
from langchain_openai import ChatOpenAI
from typing import Any

MODEL_NAME = os.environ.get("ERPGUARD_LLM_MODEL", "gpt-5")


def llm_available() -> bool:
    # return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return bool(os.environ.get("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def get_llm() -> Any:
    """Return a configured ChatOpenAI. Raises if no API key is set."""
    if not llm_available():
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it to enable LLM formatting/"
            "fallback routing, or run in deterministic mode."
        )
    
    return ChatOpenAI(model=MODEL_NAME, temperature=0)
