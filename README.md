# ERPGuard-AI

A small, runnable PoC chat service that classifies a user's intent cheaply,
runs canonical tool chains (sales / compliance / vendor onboarding / ops /
KB), keeps per-session memory, and emits structured observability traces.
All facts come from deterministic tools that read `data/seed_data.json` —
the LLM only formats the final response.

---

## How to run

### Prerequisites
- Python **3.13+**
- [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- An OpenAI API key (optional — see [Without an API key](#without-an-api-key))

### One-time setup

```bash
uv sync                       # install dependencies into .venv
cp .env.example .env          # create your local env file (gitignored)
$EDITOR .env                  # set OPENAI_API_KEY=sk-...
```

### Run the scripted demo (one command)

Replays the 5 spec prompts through a single session and writes JSON
traces to `logs/trace.jsonl`:

```bash
uv run python main.py --demo
```

> **Skipping `.env`?** Inline the key instead:
> `OPENAI_API_KEY=sk-... uv run python main.py --demo`

### Run the interactive REPL

```bash
uv run python main.py
```

The REPL prompts for:
- **`user_type`** — `internal_sales` (default), `portal_vendor`, or `portal_customer`
- **`session_id`** — leave blank to auto-generate, or paste a prior id to resume

Then type one prompt per line. Turns under the same `session_id` share
memory via the LangGraph checkpointer, so follow-ups like *"add 2 of
the first one to the basket"* resolve against the prior turn's
products. Type `quit` (or Ctrl-D) to exit.

```text
$ uv run python main.py
ERPGuard-AI REPL. Type 'quit' to exit.
Available user_types: internal_sales, portal_vendor, portal_customer
user_type [internal_sales]:
session_id [auto]:
session_id=0e35ac1a

> Give me hot picks for CA under $500
========================================================================
USER (internal_sales): Give me hot picks for CA under $500
------------------------------------------------------------------------
intent     : SALES_RECO (routed_by=keyword)
request_id : 7f3c...
output     : Here are the top picks for CA under $500: ...

> Ok add 2 of the first one to the basket
...
```

### Resuming a session after restart

When `ERPGUARD_CHECKPOINT_SQLITE=1` (the default), graph state is
persisted to `data/checkpoints.sqlite`. Re-enter the same
`session_id` at the REPL prompt and the prior `state_code`, `budget`,
`product_ids`, etc. are re-hydrated automatically.

### Without an API key

The service still runs end-to-end:
- Deterministic tools execute normally against `seed_data.json`.
- A fallback formatter renders results as JSON instead of prose.
- The intent router uses keyword matching only (the LLM-fallback branch is skipped).

### Inspecting output

```bash
tail -f logs/trace.jsonl | jq               # one structured event per turn
sqlite3 data/checkpoints.sqlite '.tables'   # persisted graph state
```

### Environment variables

All can be set via `.env` (recommended) or inline. See
[`.env.example`](.env.example) for the template.

| Var | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | unset | If set, the LLM formats final responses and provides router fallback. Otherwise a deterministic JSON formatter is used. |
| `ERPGUARD_LLM_MODEL` | `gpt-5` | Override the OpenAI chat model. |
| `ERPGUARD_CHECKPOINT_SQLITE` | `1` | `1` persists graph state to `data/checkpoints.sqlite` (resumable across restarts). `0` uses an in-memory checkpointer instead. |

---

## Architecture overview (1 page)

![High Level Architecture](docs/High%20Level%20Architecture.png)

**Per turn:**

1. `run_turn` creates a `TraceEvent`, binds it to a `ContextVar`, and
   invokes the LangGraph with `thread_id = session_id`.
2. The **router** classifies intent with a cheap keyword scorer; falls
   back to a structured-output LLM call only when the keyword score is
   zero AND the input isn't a memory follow-up.
3. The chosen **chain** runs as one node per tool. Every tool call is
   gated by an allowlist, audited, and timed.
4. A shared **format** node renders the small, intent-specific payload
   (PII-redacted) — never the raw seed data.
5. Graph state is **checkpointed** by LangGraph keyed on `session_id`,
   so the next turn resumes with prior `state_code`, `budget`,
   `product_ids`, etc. already in scope.

**Design rules** (matches the spec disqualifiers):

- Tools are deterministic and own all live facts (stock, prices,
  legality). The LLM never decides; it only formats.
- Compliance decisions live in `src/tools/compliance.py` as a static
  policy table — the LLM cannot override them.
- Only the per-intent payload (5–20 lines) is sent to the LLM; the seed
  dataset is never dumped into a prompt.
- Tool allowlists are enforced before every call; PII is redacted
  before any LLM hand-off; an audit event is emitted per call.

---

## Where things live

### Routing
- `src/router/intent_router.py` — cheap keyword router (`keyword_route`),
  basket-follow-up detector, and the LLM-fallback structured-output call.
  Exposes `llm_call_router(state)` and `route_decision(state)`.
- `src/orchestrator/graph.py` — LangGraph wiring: nodes, edges,
  checkpointer selection (Memory or SQLite), `run_turn(...)` entry point,
  `get_checkpoint(session_id)`, `clear_session(session_id)`.
- `src/router/nodes.py` — one node per deterministic tool plus a shared
  `format_node`. Chains:
  - **SALES_RECO**: `sales_hot_picks_node → sales_compliance_filter_node → format_node`
  - **COMPLIANCE_CHECK**: `compliance_identify_node → compliance_filter_node → compliance_alternatives_node → format_node`
  - **VENDOR_ONBOARDING**: `vendor_validate_node → format_node`
  - **OPS_STOCK**: `ops_stock_node → format_node`
  - **GENERAL_KB**: `kb_search_node → format_node`
  - **DEFAULT**: `default_node` (memory-based basket follow-up) → END
- `src/router/nodes_common.py` — `safe_call` (allowlist + audit + timing
  wrapper), `format_with_llm` (PII-redact + LLM-or-deterministic
  formatter), `persist_session`.
- `src/data/extractors.py` — cheap deterministic slot extraction
  (`extract_state`, `extract_budget`, `extract_product_ids`,
  `match_products_by_name`, `extract_vendor_attrs`).

### Tools (deterministic; read `seed_data.json`)
- `src/tools/hot_picks.py` — `hot_picks(state, budget, limit)`
- `src/tools/compliance.py` — `compliance_filter(state, product_ids)`
  (static state-flag policy table — LLM cannot override)
- `src/tools/inventory.py` — `stock_by_warehouse(product_id)`
- `src/tools/vendor.py` — `vendor_validate(attributes_json)`
- `src/tools/kb.py` — `kb_search(query, top_k, visibility)`
- `src/data/loader.py` — single cached loader for `seed_data.json`

### State / memory
- `src/schemas/model.py` — `State` (LangGraph TypedDict), `Session`
  (per-session memory shape), `Intent`, `UserType`, `Route`.
- `src/session/store.py` — in-process per-session dict holding
  `last_intent`, `last_state`, `last_budget`, `last_product_ids`.
- **LangGraph checkpointer** — persists the entire graph `State` per
  `thread_id = session_id`. Backend selected in
  `src/orchestrator/graph.py::_checkpointer()`:
  - default `InMemorySaver`
  - `SqliteSaver` writing to `data/checkpoints.sqlite` when
    `ERPGUARD_CHECKPOINT_SQLITE=1` (default).

### Security
- `src/security/policy.py` — `TOOL_ALLOWLIST` per `user_type`
  (`internal_sales` / `portal_vendor` / `portal_customer`),
  `assert_allowed()`, `redact_pii()` / `redact_payload()` (applied
  before anything is sent to the LLM).

### Observability
- `src/obs/logger.py` — `TraceEvent` accumulator (one per request);
  `time_tool()` context manager; `audit()` for sensitive tool calls.
  Each request emits one JSON line to **stderr** and appends to
  **`logs/trace.jsonl`** with: `request_id`, `user_type`, `intent`,
  `routed_by`, `tools_called` (name + args + `latency_ms` +
  `result_summary`), `total_latency_ms`, `prompt_token_estimate`,
  `output_token_estimate`, `error`. The active `TraceEvent` is bound
  to a `ContextVar` (not to graph State) so the checkpointer never
  serializes it.

### Entry point
- `main.py` — interactive REPL + `--demo` mode.

---

## Demo prompts (run by `--demo`)

| # | user_type | Prompt | Expected chain |
|---|---|---|---|
| 1 | internal_sales | "Give me hot picks for CA under $5000" | SALES_RECO |
| 2 | internal_sales | "Why is SKU-1006 not available in UT? Suggest alternatives." | COMPLIANCE_CHECK |
| 3 | internal_sales | "How much stock does SKU-1002 have and where?" | OPS_STOCK |
| 4 | portal_vendor  | "I'm uploading a product missing Net Wt and no lab report — what do I fix?" | VENDOR_ONBOARDING |
| 5 | internal_sales | "Ok add 2 of the first one to the basket" | DEFAULT (memory follow-up) |

Inspect traces: `tail -f logs/trace.jsonl | jq`
Inspect persisted graph state: open `data/checkpoints.sqlite`.
