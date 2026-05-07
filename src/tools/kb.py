"""kb_search(query) — naive keyword search over kb_docs.

Per spec: NOT used for live inventory or compliance decisions.
"""
from __future__ import annotations

import re
from typing import Any

from src.data.loader import kb_docs

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are",
    "for", "on", "with", "what", "how", "do", "does", "i", "we",
}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")
            if t.lower() not in _STOPWORDS]


def kb_search(query: str, top_k: int = 3,
              visibility: str | None = None) -> list[dict[str, Any]]:
    """Return up to top_k snippets ranked by keyword overlap.

    visibility: optional filter ("public" | "internal" | "vendor").
    """
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return []

    scored: list[tuple[int, dict[str, Any]]] = []
    for doc in kb_docs():
        if visibility is not None and doc.get("visibility") != visibility:
            continue
        body = f"{doc.get('title', '')} {doc.get('text', '')}"
        d_tokens = set(_tokenize(body))
        score = len(q_tokens & d_tokens)
        if score == 0:
            continue
        scored.append((score, {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "snippet": doc["text"],
            "score": score,
        }))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[: max(1, int(top_k))]]
