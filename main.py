"""Interactive CLI + scripted demo for ERPGuard-AI PoC.

Usage:
    # Interactive REPL (prompts for user_type + session id)
    python main.py

    # Scripted run of the 5 demo prompts from the spec (no LLM key needed)
    python main.py --demo

Env:
    ANTHROPIC_API_KEY  optional; if set, the LLM formats the final response.
                       Without it, a deterministic JSON formatter is used.
"""
from __future__ import annotations

import argparse
import sys
from typing import get_args

from src.orchestrator.graph import new_session_id, run_turn
from src.schemas import UserType

USER_TYPES = list(get_args(UserType))


DEMO_SCRIPT = [
    ("internal_sales", "Give me hot picks for CA under $5000"),
    ("internal_sales", "Why is SKU-1006 not available in UT? Suggest alternatives."),
    ("internal_sales", "How much stock does SKU-1002 have and where?"),
    ("portal_vendor",  "I'm uploading a product missing Net Wt and no lab report — what do I fix?"),
    ("internal_sales", "Ok add 2 of the first one to the basket"),
]


def _print_result(prompt: str, user_type: str, result: dict) -> None:
    print()
    print("=" * 72)
    print(f"USER ({user_type}): {prompt}")
    print("-" * 72)
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return
    print(f"intent     : {result.get('intent')} (routed_by={result.get('routed_by')})")
    print(f"request_id : {result['request_id']}")
    print(f"output     : {result.get('output')}")


def run_demo() -> int:
    sid = new_session_id()
    print(f"[demo] session_id={sid}")
    for user_type, prompt in DEMO_SCRIPT:
        result = run_turn(user_input=prompt, user_type=user_type, session_id=sid)
        _print_result(prompt, user_type, result)
    print()
    print("=" * 72)
    print("Structured traces appended to logs/trace.jsonl")
    return 0


def run_repl() -> int:
    print("ERPGuard-AI REPL. Type 'quit' to exit.")
    print(f"Available user_types: {', '.join(USER_TYPES)}")
    user_type = input("user_type [internal_sales]: ").strip() or "internal_sales"
    if user_type not in USER_TYPES:
        print(f"unknown user_type {user_type!r}; using internal_sales")
        user_type = "internal_sales"
    sid = input("session_id [auto]: ").strip() or new_session_id()
    print(f"session_id={sid}")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text.lower() in {"quit", "exit", ":q"}:
            return 0
        result = run_turn(user_input=text, user_type=user_type, session_id=sid)
        _print_result(text, user_type, result)


def main() -> int:
    parser = argparse.ArgumentParser(description="ERPGuard-AI PoC")
    parser.add_argument("--demo", action="store_true",
                        help="Run the scripted demo prompts and exit.")
    args = parser.parse_args()
    return run_demo() if args.demo else run_repl()


if __name__ == "__main__":
    sys.exit(main())
