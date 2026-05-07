"""Minimal per-session memory.

In-memory dict keyed by session_id. Acceptable for the PoC per spec;
swap for Redis with the same interface in production.
"""
from __future__ import annotations

from threading import Lock
from typing import Any

from src.schemas import Session

_LOCK = Lock()
_STORE: dict[str, Session] = {}


def get(session_id: str) -> Session:
    with _LOCK:
        sess = _STORE.get(session_id)
        if sess is None:
            sess = Session(session_id=session_id)
            _STORE[session_id] = sess
        # Return a shallow copy so callers can't mutate the store directly.
        return dict(sess)  # type: ignore[return-value]


def update(session_id: str, **fields: Any) -> Session:
    with _LOCK:
        sess = _STORE.setdefault(session_id, Session(session_id=session_id))
        for k, v in fields.items():
            if v is not None:
                sess[k] = v  # type: ignore[literal-required]
        return dict(sess)  # type: ignore[return-value]


def clear(session_id: str) -> None:
    with _LOCK:
        _STORE.pop(session_id, None)
