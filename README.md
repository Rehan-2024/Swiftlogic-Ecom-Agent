---
title: Swiftlogic CommerceOps v2
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# Swiftlogic CommerceOps v2

> **Autonomous AI Startup Operator** — a dynamic, config-driven e-commerce world for the Meta × PyTorch × OpenEnv × Scaler Hackathon Grand Finale.

## Overview

CommerceOps v2 simulates a full digital storefront an AI agent must operate autonomously across a **50-day** business cycle. Every step the agent chooses to restock inventory, resolve a customer ticket, allocate ad spend, or wait — and the world responds with stochastic Poisson demand, revenue, ticket spawning, and cash-flow mechanics.

The environment is **config-driven**: the same `WorldEngine` runs fashion, pharmacy, or SaaS businesses by swapping a JSON file. Swap the config live via `POST /config` and the entire world — products, prices, demand, tickets, rewards, graders — changes in under 100 ms with no server restart.

- Runtime: FastAPI + Uvicorn on port `7860`
- Validator: OpenEnv v0.2.3 compliant (`/reset`, `/step`, `/state`, `/tasks`, `/grader`)
- Framework extras: `POST /config` live business hot-swap
- Determinism: seeded on `POST /reset`, reproducible across episodes

---

## Observation Space (`EcomObservation`)

| Field | Type | Description |
|---|---|---|
| `current_day` | `int` | Current simulation day (1–50) |
| `current_week` | `int` | `current_day // 7` |
| `step_count` | `int` | Steps taken this episode |
| `bank_balance` | `float` | Cash on hand |
| `inventory` | `Dict[str, int]` | SKU → units in stock |
| `pending_orders` | `Dict[str, int]` | SKU → units awaiting fulfillment (aggregate in-flight counter) |
| `pending_orders_schedule` | `Dict[str, List[List[int]]]` | SKU → list of `[delivery_day, quantity]` pairs so the policy can distinguish one big shipment from several small ones |
| `active_tickets` | `List[Ticket]` | Open + resolved support tickets (with `urgency`, `created_day`) |
| `daily_sales` | `Dict[str, int]` | SKU → units sold today (Poisson realization) |
| `active_ad_spend` | `Dict[str, float]` | SKU → ad budget active this step |
| `prices` | `Dict[str, float]` | SKU → current sell price |
| `competitor_prices` | `Dict[str, float]` | SKU → benchmark competitor price (static for the whole episode by design; a future revision may add slow drift) |
| `cumulative_revenue` | `float` | Total revenue booked this episode |
| `supplier_quotes` | `Dict[str, float]` | SKU → standing supplier unit-price quote |
| `supplier_quote_expiry` | `Dict[str, int]` | SKU → `step_count` at which the paired quote goes stale |
| `daily_revenue` | `float` | Gross revenue booked on the most recent simulated day (matches `sum(daily_sales * prices)`) |
| `reward` / `done` | `float` / `bool` | Last step reward / episode termination flag |

## Action Space (`EcomAction`)

Pydantic `RootModel` discriminated union. One of:

| Action | Fields | Effect |
|---|---|---|
| `RestockAction` | `sku: str, quantity: int` | Deducts `unit_price * quantity` from `bank_balance`, schedules delivery via `restock_lead_days` (see below). If a live supplier quote exists for the SKU, the quoted price is used; otherwise the list cost is charged with a `supplier.spot_premium` surcharge so negotiation still pays off. Rejected if unaffordable. |
| `RefundAction` | `ticket_id: str` | Resolves the ticket and deducts a payout drawn from `tickets.refund_amount_range` (now a required config key). Rejected with no balance change if funds are insufficient. |
| `AdSpendAction` | `sku: str, budget: float` | Deducts `budget` from `bank_balance`, boosts next-step Poisson demand via `ad_elasticity`. The campaign is active for the *current* step's `_simulate_day`, then zeroed at the top of the following `step` so observations reflect what just ran. |
| `NegotiateAction` | `sku: str, quantity: int` | Requests a supplier unit-price quote. No cash moves. The quote is stored in `supplier_quotes[sku]` and consumed by the next successful `RestockAction` on that SKU. When `quantity <= supplier.volume_free_units` the quote is further discounted by `supplier.volume_discount`. |
| `SetPriceAction` | `sku: str, price: float` | **v2.3 — new.** Overrides `prices[sku]` with the provided value. Rejected (with `invalid_action`) if the price falls outside `[competitor_price * actions.price_min_mult_competitor, competitor_price * actions.price_max_mult_competitor]`. No cash moves — the policy pays through next step's demand response. |
| `WaitAction` | — | No-op. Business day still advances. |

Invalid actions (unknown SKU, non-positive quantity, missing ticket, insufficient funds, action not allowed by the active config) return the `invalid_action` reward penalty.

### Supplier Quote Lifecycle

1. **Create / overwrite.** `NegotiateAction` computes a unit price from the rule-based `SupplierAgent`, using a **3-day rolling mean of `daily_sales`** as the demand signal (not a single noisy day). A repeat negotiate on the same SKU overwrites the prior quote.
2. **Hard price ceiling.** Quotes are clamped at `base_price * price_cap_multiplier` (default `2.5x`) so bulk-order or hot-demand premiums cannot blow up.
3. **TTL expiry.** Each quote is stamped with `expiry_step = step_count + supplier.quote_expiry_steps` (default `3`). If the next restock happens after expiry, the quote is silently evicted and list cost applies.
4. **One-shot consumption.** A successful restock consumes the quote; insufficient funds leaves it in place so the agent can retry.

## Reward Model (`EcomReward`)

Dense reward per step (see `env/reward_engine.py`):

```
reward = base_reward(action)
       + revenue_multiplier        * daily_revenue        (linear | log | cap)
       + solvency_per_step           (if bank_balance >= solvency_threshold)
       + stockout_penalty            (per SKU that hit 0 this step; skipped
                                       when ``stockout_transition_grace`` is on
                                       AND a restock is already in-flight)
       + urgent_ticket_per_step    * (# aging urgent tickets)
       + critical_ticket_per_step  * (# aging critical tickets)
       + ad_roi_positive             (per SKU where ``ad_spend_applied`` produced sales)
       + bankruptcy_terminal         (if bank_balance <= bankruptcy_threshold)
       + bank_balance_delta_weight * (Δbank_balance - daily_revenue)
       + inventory_target_bonus      (if state.inventory[target_sku] >= target_units)   [v2.3]
```

All coefficients come from the active business config under `rewards.*`. The
delta term subtracts `daily_revenue` so `revenue_multiplier` and
`bank_balance_delta_weight` don't double-count the same inflow — the delta
now reflects only non-revenue cash flows (restocks, ad spend, refunds).
`critical_ticket_per_step` defaults to `1.5x urgent_ticket_per_step` when
not configured.

**Revenue modes** (`rewards.revenue_mode`):

- `"linear"` — legacy behaviour, `revenue_multiplier * daily_revenue` (default when the key is absent).
- `"log"` — `revenue_multiplier * log1p(daily_revenue)`. Keeps the signal monotone in revenue but squashes the tail so one enterprise sale cannot drown out the shaping terms. All three shipped configs use `"log"`.
- `"cap"` — `min(revenue_multiplier * daily_revenue, rewards.revenue_cap_per_step)`.

**Ad demand multiplier** — the daily Poisson lambda is scaled by
`1 + log1p(ad_spend / 100) * ad_elasticity`, hard-capped at
`MAX_AD_MULTIPLIER = 5.0`. This prevents the revenue term from exploding on
high-elasticity SKUs paired with maxed-out ad budgets.

**Observability hook** — every `/step` returns `info["reward_breakdown"]`
with per-term floats (`base`, `revenue`, `solvency`, `stockout`,
`ticket_aging`, `ad_roi`, `bankruptcy`, `delta`, `inventory_target_bonus`,
and the `daily_revenue` metadata scalar) so GRPO loops can log reward
dynamics without changing the scalar reward. As of v2.3 the per-term
entries are rounded *once* and `total == sum(breakdown_terms)` exactly at
4 dp, so downstream reconciliation no longer needs a tolerance. The same
`daily_revenue` scalar is also mirrored onto the observation so callers
that never read `info` can still watch realised per-tick revenue.

**Unit-cost fallback alarm** — `env.actions.do_restock` logs a
`commerceops.actions` WARNING (`unit_cost_fallback_used`) if an SKU is
missing from the `unit_costs` map at restock time. The config validator
already prevents this state, so the log is purely a tripwire for future
regressions — it must never fire in a healthy run.

### Config field status

All config keys listed below are active in v2.3. Legacy keys that were
silently ignored in v2.2 (`financials.solvency_bonus_threshold`,
`products[*].demand.demand_model`, `rewards.bankruptcy_threshold`) now emit a
`commerceops.world_engine` deprecation WARNING and will be removed in the
next major bump. Unknown keys under any section also emit a
`config_unknown_key` WARNING so hand-editing typos (e.g. `rewards.stockot_penalty`)
surface in CI logs without breaking config loading.

- `products[*].restock_lead_days` — **now live.** Restock quantities land in
  `state["pending_deliveries"]` and are drained back into `inventory` at the
  top of the simulated day whose index reaches `order_day + lead_days`.
  `observation.pending_orders` therefore tracks in-flight units rather than
  always reading zero.
- `tickets.resolved_retention_days` (default `7`) — resolved tickets older
  than this are pruned each tick so long horizons don't balloon
  `active_tickets`.
- `supplier.volume_discount` + `supplier.spot_premium` — see the
  `NegotiateAction` / `RestockAction` rows above.
- `actions.price_min_mult_competitor` / `price_max_mult_competitor` — bounds
  used by `SetPriceAction`.
- `rewards.inventory_target_bonus` — optional dense bonus that fires each
  step the inventory-target SKU's stock is at or above
  `graders.inventory_task.target_units`. Keeps shaping aligned with the
  medium-difficulty grader.
- `rewards.stockout_penalty` is deliberately `0.0` on `configs/stackbase_saas.json`
  because SaaS plans have infinite stock (`initial_stock: 9999`); the other
  configs keep a negative penalty. See the config's top-level `_notes`.
- `tickets.max_active` (optional, default unbounded) — hard cap on the
  number of *open* tickets `spawn_daily_tickets` will create per call.
  Useful for long-horizon episodes where a high `spawn_rate_per_day`
  could otherwise balloon the queue beyond anything a policy can triage.
- `financials.bankruptcy_threshold` — single source of truth for the
  terminal bankruptcy guardrail. The legacy `rewards.bankruptcy_threshold`
  mirror still loads for backward compatibility but emits a deprecation
  WARNING; if both are present the validator enforces that they are equal.

### Grader context — explicit `context=` kwarg

`grade_inventory_task` and `grade_profit_task` accept an explicit
`context=env.grader_context` kwarg. Calling them without it falls back to
a module-level mirror (`_GRADER_CONTEXT`) and now emits a
`DeprecationWarning`. Callers running more than one `EcomEnv` in the same
process must pass the kwarg to avoid racing on the shared mirror; the
mirror itself will be removed in v2.4.

## Episode Lifecycle

- `reset(seed)` draws 3–5 tickets (urgency-weighted), seeds `random` + `numpy.random`, and zeroes `daily_sales` / `active_ad_spend`.
- `step(action)` processes the action, runs the daily demand simulation (Poisson draws per SKU, modulated by ad spend + seasonality + competitor-price ratio), books revenue, depletes inventory, decays ad spend, and stochastically spawns new tickets.
- Termination: `step_count >= 50` or `bank_balance <= 0` (bankruptcy).

## Tasks

Each grader returns a score strictly within `(0.01, 0.99)` (OpenEnv Phase 2 Deep Validation):

- **Easy — Ticket Triage.** Ratio of resolved tickets to total tickets (`grade_triage_task`).
- **Medium — Inventory Health.** Ratio of the target SKU's stock vs its configured target units (`grade_inventory_task`). Target SKU is read from the active business config.
- **Hard — Profit Maximization.** Bank-balance growth normalized around break-even (0.5). The normalizer is taken from `graders.profit_task.normalizer` in the active config (e.g. 20 000 INR for Siyaani, 4 000 USD for Stackbase), with a capital-scaled fallback for backwards compatibility (`grade_profit_task`).

---

## Quick Start

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t commerce-ops-v2 .
docker run -p 7860:7860 commerce-ops-v2
```

### Inference loop (LLM agent)

```bash
API_BASE_URL=<openai-compatible-endpoint> MODEL_NAME=<id> python inference.py
```

Runs the full 50-step episode across all three tasks and prints structured `[START] / [STEP] / [END]` logs.

### GRPO training

Open `swiftlogic_grpo_training.ipynb` in Colab. Cells install `trl/transformers`, hook the env over HTTP (`ENV_URL` env var), run a 200-episode GRPO training loop with the `Thought + Action` explainability prompt, and emit `reward_curve.png`, `before_after_comparison.json`, and `thought_logs.json`.

---

## HTTP API

| Method | Path | Body | Response |
|---|---|---|---|
| `GET`  | `/` | — | Service metadata, active business, endpoint list |
| `GET`  | `/health` | — | `{"status": "ok"}` |
| `POST` | `/reset` | `{"seed": 42}` (optional) | `{observation, reward, done}` |
| `POST` | `/step` | flat **or** wrapped action JSON | `{observation, reward, done, info}` |
| `GET`  | `/state` | — | `{observation}` |
| `GET`  | `/tasks` | — | Task descriptors + graders |
| `POST` | `/grader` | — | `{scores: [{task_id, score, grader}]}`. Returns **409** when called before `/reset` (no baseline snapshot). |
| `POST` | `/config` | `{"business_id": "siyaani_fashion"\|"medplus_pharmacy"\|"stackbase_saas"}` | Hot-swaps the world |
| `GET`  | `/debug/last_step_info` | — | Returns the last `/step` info dict. Only enabled when the server process was started with `COMMERCEOPS_DEBUG=1`; otherwise returns 404. |

All write endpoints enforce a **64 KiB body-size limit** and reject larger
payloads with `413 Payload Too Large`. Internal 500 responses never leak
exception strings — they log the traceback server-side and return a stable
generic `detail`.

### `/step` accepts both formats

```bash
# Flat
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "wait"}'

# Wrapped
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "wait"}}'
```

### Live business hot-swap

```bash
curl -X POST http://localhost:7860/config \
  -H "Content-Type: application/json" \
  -d '{"business_id": "medplus_pharmacy"}'
```

---

## Business Configs

Same `WorldEngine`, three worlds:

| Config | Products | Initial Balance | Unique Twist |
|---|---|---|---|
| `siyaani_fashion` (default) | silk_kurta, cotton_set, linen_dupatta, silk_saree | ₹50 000 | Weekend seasonality up to 2.0× |
| `medplus_pharmacy` | paracetamol, vitamin_d3, bp_monitor | ₹80 000 | Higher urgent-ticket penalty, steeper stockout cost |
| `stackbase_saas` | starter_plan, pro_plan, enterprise_plan | $10 000 | Near-zero unit cost, high ad elasticity, no stockout |

Configs live in [`configs/`](configs). Add a new business by dropping in another JSON file — no code changes required.

---

## Repository Layout

```
commerce-ops-v2/
├── configs/                     # business config JSON (siyaani/medplus/stackbase)
├── env/                         # WorldEngine + helper modules
│   ├── __init__.py
│   ├── world_engine.py          # simulation core + dispatcher
│   ├── actions.py               # pure action handlers (restock/refund/...)
│   ├── constants.py             # shared fallback constants
│   ├── demand_model.py
│   ├── ticket_system.py
│   ├── supplier_agent.py
│   └── reward_engine.py         # dense shaping with per-term helpers
├── server/
│   └── app.py                   # FastAPI (adds POST /config)
├── ecom_env.py                  # Pydantic models + EcomEnv adapter + graders
├── inference.py                 # LLM inference loop (50 steps per task)
├── scripts/
│   └── smoke_env.py             # live-server smoke check (manual only)
├── swiftlogic_grpo_training.ipynb
├── openenv.yaml                 # name: commerce-ops-v2
├── Dockerfile
├── pyproject.toml / requirements.txt
└── README.md / PROJECT_REPORT.md
```

## Determinism

`POST /reset` seeds both `random` and `numpy.random`, so two runs with the same seed produce byte-identical observations, rewards, and grader outputs — including the stochastic Poisson demand draws.

## Operations Notes

- **Single-session benchmark mode.** The FastAPI server owns one `EcomEnv`
  singleton per process. A `threading.Lock` guards all state-mutating routes
  (`/reset`, `/step`, `/config`, `/grader`) so concurrent requests can't
  corrupt the world. This is sufficient for the OpenEnv validator and the
  single-client GRPO training loop.
- **Multi-client serving caveat.** Because the env is a singleton, multiple
  clients share one world. For production-style concurrent benchmarking,
  run one container per evaluator (or extend to a per-session env pool).
- **Input safety.** `/config` only accepts `business_id` values matching
  `^[a-z0-9][a-z0-9_\-]{0,63}$` and backed by a real file under
  [`configs/`](configs). Malformed or unknown ids return 4xx without touching
  the filesystem. Malformed `/step` / `/reset` payloads also return stable
  `400`s instead of uncaught `500`s.
- **Regression tests.** `pytest tests/` runs contract, simulation-invariant,
  supplier-flow, reward-engine, demand-model, and grader-bound sweeps. All
  65 tests should pass before submitting. The smoke script that hits a
  live Hugging Face Space lives under `scripts/smoke_env.py` so it is never
  picked up by pytest collection.

## Baseline Performance (zero-shot Qwen2.5-0.5B-Instruct)

| Task | Difficulty | Zero-Shot Score | Notes |
|---|---|---|---|
| Ticket Triage | Easy | ~0.80 | Often resolves every seeded ticket |
| Inventory Health | Medium | ~0.40 | Balance between restocks and cash |
| Profit Maximization | Hard | ~0.15 | Undershoots revenue; frequent bankruptcy |

Post-GRPO (target): profit task ≥ 0.50, bankruptcy rate < 20%.
