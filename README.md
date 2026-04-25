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

> **Autonomous AI Startup Operator** — a dynamic, config-driven e-commerce world with a lightweight AI-CEO + department structure, full explainability, and OpenEnv compliance. Built for the Meta × PyTorch × OpenEnv × Scaler Hackathon Grand Finale.

## Problem statement

Small and mid-size e-commerce operators lose money because decisions that look
locally correct (discounting, delaying restocks, skipping refunds, over-spending
on ads) create delayed systemic failures: stockouts, cash crunch, ticket backlog,
customer-satisfaction collapse, and eventual bankruptcy. Existing benchmarks
usually score only one objective (for example, short-term profit), so they fail
to measure whether an agent can run a business end-to-end under uncertainty.

This project solves that gap by exposing a deterministic OpenEnv environment
where one policy must jointly optimize service quality, inventory health,
profitability, competitor response, and crisis recovery over a 50-day horizon,
with reproducible grading and auditable step-level explanations.

## At a glance

| Status | Detail |
|---|---|
| Env | frozen at `release/env-frozen-v2.3` (additive-only after this point) |
| Training | GRPO + Unsloth, Qwen2.5-0.5B-Instruct, 4-bit QLoRA, 3-stage curriculum |
| HF Space | [Swiftlogic / E-commerce-agent](https://huggingface.co/spaces/Swiftlogic/E-commerce-agent) — landing page with **Run Demo** button |
| Tests | 218 + 30 new (landing, training, evaluation-only graders, info-keys) — all green |

**Composite score (per [`artifacts/composite_score.json`](artifacts/composite_score.json)): `0.61 -> 0.66 (+9%)`**
*(provenance: `heuristic_fallback` — these numbers are produced by `scripts/run_full_pipeline.py` so the repo is self-contained for judges; the Colab notebook overwrites them with `provenance: trained_adapter` after a real GRPO run on T4. The headline string lives in the JSON, not the README, so a fresh training run automatically updates this section via `scripts/refresh_readme_headline.py`.)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Swiftlogic/CommerceOps-v2/blob/feature/training/swiftlogic_grpo_training.ipynb) — full GRPO + curriculum training, ≈ 90 min on free T4.

### How to reproduce

```bash
git clone https://github.com/Swiftlogic/CommerceOps-v2.git && cd CommerceOps-v2
pip install -r requirements.txt
pytest -q                                   # 218+ tests green
python scripts/run_full_pipeline.py --fast-mode   # ~2 min, regenerates every artifact under artifacts/
```

For the actual GRPO training, open the notebook above in Colab — it runs the
exact same `training/*` modules the local pipeline uses, just on a real GPU.

### Theme alignment

This project is a Theme-2/3/4 entry with Theme-1 support:

* **Theme 2 — Real-world environment.** A 50-day Indian e-commerce SMB with stochastic Poisson demand, reactive competitors, market shocks, refund flow, supplier lead-time / capacity / partial-fill, satisfaction drift, holding cost, and bankruptcy.
* **Theme 3 — Curriculum + multi-task RL.** A 3-stage curriculum (`siyaani_fashion_easy.json` → `siyaani_fashion.json` → `siyaani_fashion_demo.json`) and 6 graders (3 training + 3 evaluation-only) drive a single GRPO policy across difficulty and surface.
* **Theme 4 — Generalization + behavior evolution.** The same trained adapter is evaluated unchanged on `medplus_pharmacy` + `stackbase_saas`; behavior signatures and exploration entropy are tracked across checkpoints in [`artifacts/policy_signature.json`](artifacts/policy_signature.json) + [`artifacts/exploration_curve.png`](artifacts/exploration_curve.png).
* **Theme 1 — OpenEnv contract.** Every endpoint (`/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/config`, `/health`) is contract-tested at the Pydantic and HTTP-wire layers (`tests/test_openenv_contract_http.py`).

### RL learning proof — six checked-in artifacts

| # | Claim | Artifact |
|---|---|---|
| 1 | Reward goes up | [`artifacts/reward_curve.png`](artifacts/reward_curve.png) + [`training_log.txt`](artifacts/training_log.txt) |
| 2 | Behavior changes | [`artifacts/policy_signature.json`](artifacts/policy_signature.json) + [`policy_evolution.png`](artifacts/policy_evolution.png) |
| 3 | Exploration drops | [`artifacts/exploration_curve.png`](artifacts/exploration_curve.png) |
| 4 | Generalises to unseen configs | [`artifacts/generalization.json`](artifacts/generalization.json) + [`generalization.png`](artifacts/generalization.png) |
| 5 | Self-improves on hard seeds | [`artifacts/hard_seed_retraining.json`](artifacts/hard_seed_retraining.json) |
| 6 | Beats every baseline | [`artifacts/before_metrics.json`](artifacts/before_metrics.json) + [`after_metrics.json`](artifacts/after_metrics.json) + [`before_after_comparison.png`](artifacts/before_after_comparison.png) + [`composite_score.json`](artifacts/composite_score.json) |

Bonus storytelling artifact (Part B+.7): [`artifacts/failure_vs_recovery.png`](artifacts/failure_vs_recovery.png) — same seed, baseline goes bankrupt, the trained agent recovers.

## Overview

CommerceOps v2 simulates a full digital storefront an AI agent must operate autonomously across a **50-day** business cycle. Every step the agent chooses to restock inventory, resolve a customer ticket, allocate ad spend, negotiate with a supplier, set a price, or wait — and the world responds with stochastic Poisson demand, revenue, ticket spawning, cash-flow mechanics, reactive competitors, market shocks, and customer satisfaction drift.

The environment is **config-driven**: the same `WorldEngine` runs fashion, pharmacy, SaaS, or a fully-loaded demo business by swapping a JSON file. Swap the config live via `POST /config` and the entire world — products, prices, demand, tickets, rewards, graders — changes in under 100 ms with no server restart.

- Runtime: FastAPI + Uvicorn on port `7860`
- Validator: **OpenEnv v0.2.3** compliant (`/reset`, `/step`, `/state`, `/tasks`, `/grader`)
- Framework extras: `POST /config` live business hot-swap, `GET /debug/last_step_info` behind a debug flag
- Determinism: seeded on `POST /reset`, byte-identical replays under the same seed
- Test suite: **218 passing tests** across 11 files (contract, invariants, adversarial, regression matrix)
- Explainability: every `/step` emits 20+ additive `info` keys (intent, trend, KPIs, departments, anomalies, confidence, episode summary)

---

## Architecture at a glance

```
Client / LLM / GRPO trainer
        │   HTTP
        ▼
┌─────────────────────────────────────────┐
│  FastAPI server  (server/app.py)        │
│   • thread-lock around stateful routes  │
│   • 64 KiB body cap, stable 4xx/5xx     │
│   • multi-worker warning guard          │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  EcomEnv adapter  (ecom_env.py)         │
│   • Pydantic observation/action schemas │
│   • 3 clamped graders (0.01, 0.99)      │
│   • resolves demo vs prod config        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  WorldEngine  (env/world_engine.py)     │
│   ├── action dispatch  → env/actions.py │
│   ├── demand model     → env/demand_model.py
│   ├── ticket system    → env/ticket_system.py
│   ├── supplier agent   → env/supplier_agent.py
│   ├── reward engine    → env/reward_engine.py
│   ├── invariants       → env/invariants.py
│   └── validators       → env/validators.py
│                                          │
│   produces additive info payload:       │
│   reward_breakdown · action_effect ·    │
│   demand_factors · kpis · trend ·       │
│   intent · why_failed · anomalies ·     │
│   policy_stability · confidence ·       │
│   competitor_reaction · episode_summary │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Inference / CEO layer  (inference.py)  │
│   • state_summary, reasoning, causal    │
│   • department suggestions (inv/mkt/sup)│
│   • decision_context ← CEO intent       │
│   • demo printer + training plots       │
└─────────────────────────────────────────┘
```

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
| `pending_orders_schedule` | `Dict[str, List[List[int]]]` | SKU → list of `[delivery_day, quantity]` pairs |
| `active_tickets` | `List[Ticket]` | Open + resolved support tickets (with `urgency`, `created_day`) |
| `daily_sales` | `Dict[str, int]` | SKU → units sold today (Poisson realization) |
| `active_ad_spend` | `Dict[str, float]` | SKU → ad budget active this step |
| `prices` | `Dict[str, float]` | SKU → current sell price (rounded to 2 dp for display consistency) |
| `competitor_prices` | `Dict[str, float]` | SKU → competitor anchor (rounded to 2 dp; walks daily when `products[*].competitor_price_volatility` is set) |
| `cumulative_revenue` | `float` | Total revenue booked this episode |
| `supplier_quotes` | `Dict[str, float]` | SKU → standing supplier unit-price quote |
| `supplier_quote_expiry` | `Dict[str, int]` | SKU → `step_count` at which the paired quote goes stale |
| `daily_revenue` | `float` | Gross revenue booked on the most recent simulated day |
| `reward` / `done` | `float` / `bool` | Last step reward / episode termination flag |

## Action Space (`EcomAction`)

Pydantic `RootModel` discriminated union over **6 actions**:

| Action | Fields | Effect |
|---|---|---|
| `RestockAction` | `sku: str, quantity: int` | Deducts `unit_price * quantity` from `bank_balance`, schedules delivery via `restock_lead_days`. Uses a live supplier quote when present; otherwise charges list cost plus `supplier.spot_premium`. Supports partial fills when `supplier.capacity_per_sku` is configured (`info.restock` reports `requested_quantity`, `filled_qty`, `unfilled_qty`). Rejected if unaffordable. |
| `RefundAction` | `ticket_id: str` | Resolves the ticket and deducts a payout drawn from `tickets.refund_amount_range`. When `tickets.allow_partial_refund` is enabled and funds fall short, the engine issues the partial amount available, keeps the ticket open, and tracks `partial_refund_paid` / `partial_refund_due` / `partial_refund_count` on the ticket. Without the flag, insufficient funds reject the action. |
| `AdSpendAction` | `sku: str, budget: float` | Deducts `budget`, boosts current-step Poisson demand via `1 + log1p(budget/100) * ad_elasticity` (hard-capped at `MAX_AD_MULTIPLIER=5.0`). Campaign auto-zeroed at the top of the next step. |
| `NegotiateAction` | `sku: str, quantity: int` | Requests a supplier unit-price quote. No cash moves. Quantity is clamped to `supplier.capacity_per_sku` before quoting (`info.negotiate` reports `requested_quantity`, `capacity_capped`). Small orders within `supplier.volume_free_units` get `supplier.volume_discount`. |
| `SetPriceAction` | `sku: str, price: float` | Overrides `prices[sku]`. Rejected with `invalid_action` if outside `[competitor_price * price_min_mult_competitor, competitor_price * price_max_mult_competitor]`. No cash moves — the policy pays through next step's demand response. Can trigger reactive competitor move when `competitor.reactive_enabled=true`. |
| `WaitAction` | — | No-op. Business day still advances. |

Invalid actions (unknown SKU, non-positive quantity, missing ticket, insufficient funds, action not allowed by the active config) return the `invalid_action` reward penalty and an `info.error` string.

### Supplier Quote Lifecycle

1. **Create / overwrite.** `NegotiateAction` computes a unit price from the rule-based `SupplierAgent` using a **3-day rolling mean of `daily_sales`** as the demand signal. Capacity-capped before quoting.
2. **Hard price ceiling.** Quotes are clamped at `base_price * price_cap_multiplier` (default `2.5×`).
3. **TTL expiry.** Each quote is stamped with `expiry_step = step_count + supplier.quote_expiry_steps` (default `3`). Expired quotes are silently evicted.
4. **One-shot consumption.** A successful restock consumes the quote; insufficient funds leaves it in place so the agent can retry.

---

## Reward Model (`EcomReward`)

Dense reward per step (`env/reward_engine.py`). Eight shaping terms plus an action base term:

```
reward = base_reward(action)
       + revenue_multiplier       * daily_revenue           (linear | log | cap)
       + solvency_per_step          (if bank_balance >= solvency_threshold
                                      AND (base_reward > 0 OR productive non-revenue
                                           action with non-negative bank delta))
       + stockout_penalty           (per SKU that hit 0 this step; skipped when
                                      stockout_transition_grace is on AND a
                                      restock is already in-flight)
       + urgent_ticket_per_step   * (# aging urgent tickets)
       + critical_ticket_per_step * (# aging critical tickets)
       + ad_roi_positive            (ROI-aware when rewards.ad_roi_scaled=true)
       + bankruptcy_terminal        (if bank_balance <= bankruptcy_threshold)
       + bank_balance_delta_weight * (Δbank_balance - daily_revenue
                                      + restock_cost_amortised)
       + inventory_target_bonus     (if state.inventory[target_sku] >= target_units
                                      AND attribution gate fires)
```

All coefficients live under `rewards.*` in the active config. Key properties:

- **No double counting.** The delta term subtracts `daily_revenue` so `revenue_multiplier` and `bank_balance_delta_weight` don't stack on the same inflow.
- **Amortised vs punitive split.** Only `restock_cost_amortised` (quote-backed) is added back to the delta term; punitive spot/overflow stays in the delta path so bad negotiation remains penalised (A2-14).
- **Revenue modes** (`rewards.revenue_mode`): `"linear"`, `"log"` (`revenue_multiplier * log1p(daily_revenue)`), `"cap"`. All three shipped configs use `"log"`.
- **Native-precision scalar.** `compute_step_reward` does not apply an internal aggregate `round(..., 4)`; HTTP consumers may round for display. Breakdown term sum remains consistent with the scalar (A2-43).
- **Critical fallback.** `critical_ticket_per_step` defaults to `1.5 × urgent_ticket_per_step` when unset.
- **Per-term observability.** Every `/step` returns `info.reward_breakdown` with `base, revenue, solvency, stockout, ticket_aging, ad_roi, bankruptcy, delta, inventory_target_bonus, daily_revenue, scale_hint`.

### Config field status

All config keys listed below are active in the current code. Deprecated keys emit a `commerceops.world_engine` deprecation WARNING; unknown keys (at any nesting level, including `products.*` and `graders.*` sub-keys) emit `config_unknown_key` / `config_unknown_nested_key`.

- `products[*].restock_lead_days` — delivery delay; in-flight units tracked in `pending_orders` + `pending_orders_schedule`.
- `tickets.resolved_retention_days` (default `7`) — resolved-ticket pruning horizon.
- `tickets.allow_partial_refund` + `tickets.partial_refund_min_fraction` — enable partial refund engine path.
- `supplier.volume_discount` / `supplier.spot_premium` / `supplier.capacity_per_sku` — economics + partial-fill cap.
- `actions.price_min_mult_competitor` / `price_max_mult_competitor` — `SetPriceAction` bounds.
- `rewards.inventory_target_bonus` — dense bonus while target SKU stock ≥ `graders.inventory_task.target_units`.
- `competitor.reactive_enabled` (+ tuning multipliers) — rule-based competitor reaction to our `set_price`, evaluated before demand generation.
- `market.shock_*` — optional bounded seeded demand multipliers with configurable probability/duration.
- `financials.inventory_holding_cost_per_unit_per_day` — daily inventory carrying cost debited from bank balance.
- `customer.satisfaction_*` — bounded `[min, max]` scalar that modulates demand and integrates ticket decay.
- `financials.bankruptcy_threshold` — single source of truth (legacy `rewards.bankruptcy_threshold` still loads with a deprecation warning; validator enforces equality when both present).
- `tickets.max_active` (optional) — cap on open tickets created per tick (advisory for high-spawn long-horizon episodes).

### Grader context — explicit `context=` kwarg

`grade_inventory_task` and `grade_profit_task` accept an explicit `context=env.grader_context` kwarg. Omitting it falls back to a module-level mirror (`_GRADER_CONTEXT`) and emits a `DeprecationWarning`. Set `COMMERCEOPS_STRICT_GRADER_CONTEXT=1` to escalate the deprecated path to `RuntimeError`. The server and `EcomEnv.graders()` always bind `context=env.grader_context`; the `POST /grader` scoring runs inside the same thread lock as `/reset` and `/step` (A2-1).

---

## Explainability Contract (`info`)

Every `/step` returns an additive `info` payload. Schema and observation are untouched — all explainability, CEO logic, and anomaly detection live here.

### Base diagnostics

| Key | Shape | Meaning |
|---|---|---|
| `reward_breakdown` | dict | Per-term reward floats + `daily_revenue` passthrough + `scale_hint` |
| `bank_balance_delta` | float | Convenience mirror of `action_effect.bank_change` |
| `customer_satisfaction` | float | End-of-day (post-mutation) satisfaction scalar |
| `market_shock` | dict | `remaining_days`, `sku_multipliers` |
| `inventory_holding_cost` | float | Dollars debited this tick for carrying cost |

### Causal trace

| Key | Shape | Meaning |
|---|---|---|
| `competitor_reaction` | `{triggered, reason, magnitude, sku, our_price, competitor_before, competitor_after}` | **Only** emits `triggered=true` for genuine reactive moves (undercut/follow/deadzone). Random-walk deltas live in `competitor_walk` and never fake causation. |
| `competitor_walk` | `{sku: Δ}` | Relative competitor price change from the daily volatility walk. |
| `action_effect` | dict | Pre/post totals for inventory, bank, tickets. Includes `tickets_spawned` and `tickets_resolved` split, plus `daily_revenue` attributed to the step. |
| `demand_factors` | `{sku: {...}}` | Per-SKU decomposition: `base, ad_multiplier, price_ratio, shock, satisfaction, seasonality, season_combined, external, effective_lambda, ad_spend, price, competitor_price, raw_demand, on_hand, sold`. |
| `market_event_multipliers` | `{sku: float}` | Pure shock multiplier snapshot (excludes satisfaction). |
| `satisfaction_for_demand` | float | Satisfaction scalar used for *this* tick's demand draw (distinct from end-of-day `customer_satisfaction`). |

### KPIs and CEO layer

| Key | Shape | Meaning |
|---|---|---|
| `kpis` | dict | `profit_margin, stockout_rate, inventory_turnover, cost_of_goods_sold, gross_profit, units_sold, inventory_on_hand, sku_stockouts, daily_revenue, revenue_trend ("up"/"down"/"flat"), revenue_change_pct` |
| `trend` | dict | `up`/`down`/`flat` over the rolling history window for: `revenue, bank_balance, inventory, sales_total, open_tickets, customer_satisfaction, reward, demand` |
| `intent` | str | CEO intent ladder: `avoid_stockout` → `clear_tickets` → `increase_profit` → **`maintain_balance`** (healthy fallback) |
| `intent_signals` | dict | `stockout_rate, aged_tickets, open_tickets, profit_margin, revenue_trend` that drove the ladder |

### Quality, stability, anomalies

| Key | Shape | Meaning |
|---|---|---|
| `why_failed` | `List[str]` | Aggregated reasons: action rejection, negative reward terms, stockout, low satisfaction, unfilled restock |
| `policy_stability` | `{score, distribution, last_action, window}` | 1 − entropy-of-recent-actions (higher = more consistent policy) |
| `anomalies` | `List[str]` | Detectors: `demand_spike`, `demand_collapse`, `loss_despite_sales`, `stockout_cliff`, `cash_slide` |
| `confidence` | float in `[0, 1]` | See formula below |
| `confidence_breakdown` | `{score, concentration, baseline, causal_bonus, term_sum, formula}` | Full components for auditors |

### Confidence formula (documented)

```
T             = {revenue, solvency, stockout, ticket_aging, ad_roi,
                 bankruptcy, delta, inventory_target_bonus}
term_sum      = Σ |reward_breakdown[t]|  for t ∈ T
concentration = max(|reward_breakdown[t]|) / term_sum     (∈ [0, 1])
              = 1.0  when term_sum ≤ 1e-9 (no reward to explain)

causal_bonus  = 0.10 if info.competitor_reaction.triggered
              + 0.05 if market_shock.remaining_days > 0
              + 0.10 if info.error is a structured string      (≤ 0.25 total)

confidence    = min(1.0, 0.60 * concentration + 0.40 + causal_bonus)
```

Interpretation: 1.00 = single dominant term + one causal handle; 0.80 = single term dominates with no extra handle; 0.50 = five roughly-equal terms; 0.40 = fully diffuse reward with zero causal handles (floor).

### Terminal-step summary

| Key | Shape | Meaning |
|---|---|---|
| `episode_summary` | `{total_profit, strategy, termination_reason, mistakes, final_bank, intent_distribution}` | Emitted only on `done=True` so clients get a one-shot closeout |

---

## CEO + Department Layer (inference-side)

`inference.py` derives a lightweight company structure **without altering the engine**:

- **`build_department_suggestions(obs, info, state_summary)`** — derives `inventory`, `marketing`, `support` suggestions + urgency levels from observation + engine KPIs.
- **`build_decision_context(action, departments, info)`** — maps the chosen action to consulted departments and attaches the engine-side `info.intent`. When intent is `maintain_balance` and the action is `wait`, the context is tagged `steady_state_policy` for clean narration.
- **`build_step_trace(...)`** — combines all of the above with `build_state_summary`, `build_reasoning`, `build_market_reaction`, `build_outcome`, `build_reward_summary`, `build_causal_chain`, `build_why_it_worked`.

All builders are deterministic and run as pure functions over `(obs_before, obs_after, action, info, reward)`.

---

## Episode Lifecycle

- `reset(seed)` draws 3–5 tickets (urgency-weighted), seeds isolated `random.Random` + `numpy.random.Generator` instances, zeroes `daily_sales` / `active_ad_spend`, snapshots baseline for graders.
- `step(action)`:
  1. Dispatch action to its handler (mutates bank/inventory/tickets/pricing).
  2. `_simulate_day`: drain due deliveries, walk competitor prices (A2-15: **before** demand), optionally run reactive competitor, generate Poisson demand with decomposed factor capture, book revenue, debit holding cost, update satisfaction (exactly once per day — audit MEDIUM #3), spawn new tickets.
  3. Compute scalar reward + breakdown.
  4. Assemble additive `info` (all 20+ keys).
  5. Assert state invariants when `COMMERCEOPS_ASSERT_INVARIANTS=1` (enabled automatically in the test suite).
- Termination: `step_count >= episode.max_steps`, or `bank_balance <= financials.bankruptcy_threshold`, or stall-guard when `rewards.stall_terminate_steps` is set. Terminal step carries `info.episode_summary`.

## Tasks

Each grader returns a score strictly within `(0.01, 0.99)` (OpenEnv Phase 2 Deep Validation):

| Task ID | Difficulty | Grader | Formula |
|---|---|---|---|
| `triage_task` | Easy | `grade_triage_task` | `#resolved / #total_tickets`, +`triage_sku_match_bonus` (≤0.1) when resolved tickets carry an SKU tag. Neutral 0.5 for zero-ticket episodes. |
| `inventory_task` | Medium | `grade_inventory_task` | `stock[target_sku] / target_units`, clamped. |
| `profit_task` | Hard | `grade_profit_task` | `0.5 + (Δbank / normalizer)`, normalizer = `max(config.normalizer, 0.4·initial_bank, DEFAULT_PROFIT_NORMALIZER)`. |

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

### Demo mode (all realism features on)

```bash
COMMERCEOPS_DEMO_MODE=1 \
API_BASE_URL=<openai-compatible-endpoint> MODEL_NAME=<id> \
python inference.py
```

Auto-loads `configs/siyaani_fashion_demo.json` (reactive competitor + market shocks + satisfaction + supplier capacity + holding cost) and prints the full CEO-view step trace.

### Full demo with explainability + training proof + plots

```bash
COMMERCEOPS_DEMO_MODE=1 \
COMMERCEOPS_CEO_TRACE=1 \
COMMERCEOPS_CEO_TRACE_PATH=ceo_decision_traces.json \
COMMERCEOPS_TRAINING_PROOF=1 \
COMMERCEOPS_TRAINING_PROOF_PATH=training_proof.json \
COMMERCEOPS_REWARD_CURVE_PATH=reward_curve_inference.png \
COMMERCEOPS_PROFIT_CURVE_PATH=profit_curve_inference.png \
COMMERCEOPS_INVENTORY_CURVE_PATH=inventory_curve_inference.png \
API_BASE_URL=<openai-compatible-endpoint> MODEL_NAME=<id> \
python inference.py
```

Writes deterministic per-step explainability records (`state_summary, decision, reasoning, market_reaction, outcome, reward_summary, causal_chain, why_it_worked, departments, decision_context, intent, trend, kpis, policy_stability, anomalies, confidence`) and generates three training plots on episode end.

Matplotlib and OpenAI are lazy-imported inside `inference.py` so unit tests and non-LLM consumers never pay the import cost.

Final-polish formatting (inference layer only):
- `reasoning`, `causal_chain`, `why_it_worked`: 2–3 items each, ≤12 words/item.
- `reward_summary.top_drivers`: top 2 real contributors by absolute value from `info.reward_breakdown`.
- Deterministic tie-breaks (component name ascending when magnitudes tie).

### GRPO training

Open `swiftlogic_grpo_training.ipynb` in Colab. Cells install `trl/transformers`, hook the env over HTTP (`ENV_URL`), run a 200-episode GRPO training loop with the `Thought + Action` explainability prompt, and emit `reward_curve.png`, `before_after_comparison.json`, and `thought_logs.json`.

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
| `POST` | `/config` | `{"business_id": "siyaani_fashion"\|"medplus_pharmacy"\|"stackbase_saas"\|"siyaani_fashion_demo"}` | Hot-swaps the world |
| `GET`  | `/debug/last_step_info` | — | Returns the last `/step` info dict. Only enabled when `COMMERCEOPS_DEBUG=1`; otherwise returns 404. |

All write endpoints enforce a **64 KiB body-size limit** and reject larger payloads with `413 Payload Too Large`. Internal 500 responses never leak exception strings — they log the traceback server-side and return a stable generic `detail`. The server also prints a prominent warning when launched under multi-worker Uvicorn (`UVICORN_WORKERS`, `WEB_CONCURRENCY`, or `--workers N>1`) because a shared singleton env would defeat deterministic grading.

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
  -d '{"business_id": "siyaani_fashion_demo"}'
```

---

## Business Configs

Same `WorldEngine`, four worlds:

| Config | Products | Initial Balance | Horizon | Unique Twist |
|---|---|---|---|---|
| `siyaani_fashion` (default) | silk_kurta, cotton_set, linen_dupatta, silk_saree | ₹50 000 | 50 | Weekend seasonality up to 2.0× |
| `medplus_pharmacy` | paracetamol, vitamin_d3, bp_monitor | ₹80 000 | 50 | Higher urgent-ticket penalty, steeper stockout cost |
| `stackbase_saas` | starter_plan, pro_plan, enterprise_plan | $10 000 | 50 | Near-zero unit cost, high ad elasticity, no stockout |
| `siyaani_fashion_demo` | same as fashion | ₹50 000 | 50 | **All Tier-1 realism on**: reactive competitor + market shocks + satisfaction + supplier capacity + holding cost |

Configs live in [`configs/`](configs). Add a new business by dropping in another JSON file — no code changes required.

Config resolution order (first hit wins):
1. `EcomEnv(config_path=...)` explicit argument.
2. `COMMERCEOPS_CONFIG` environment variable.
3. `COMMERCEOPS_DEMO_MODE=1` → `configs/siyaani_fashion_demo.json`.
4. Default: `configs/siyaani_fashion.json`.

---

## Repository Layout

```
commerce-ops-v2/
├── configs/                     # siyaani_fashion, medplus_pharmacy, stackbase_saas, siyaani_fashion_demo
├── env/                         # simulation core
│   ├── world_engine.py          # action dispatch, day simulation, info assembly, CEO signals
│   ├── actions.py               # pure action handlers (restock/refund/ad_spend/negotiate/set_price/wait)
│   ├── demand_model.py          # Poisson demand with factor decomposition
│   ├── supplier_agent.py        # rule-based quote model
│   ├── ticket_system.py         # seed + daily spawn
│   ├── reward_engine.py         # 8 shaping terms + breakdown
│   ├── invariants.py            # opt-in state invariant assertions
│   ├── validators.py            # config schema + cross-key checks
│   └── constants.py             # canonical fallback constants + history window
├── server/
│   └── app.py                   # FastAPI endpoints, thread lock, multi-worker guard
├── ecom_env.py                  # Pydantic schemas + EcomEnv adapter + 3 graders
├── inference.py                 # LLM loop + explainability builders + CEO + department layer + demo printer + plots
├── scripts/
│   └── smoke_env.py             # live-server smoke check (manual only)
├── swiftlogic_grpo_training.ipynb
├── openenv.yaml                 # name: commerce-ops-v2
├── Dockerfile
├── requirements.txt / pyproject.toml
├── tests/                       # 218 tests across 11 files
└── README.md / PROJECT_REPORT.md / AUDIT_REPORT.md
```

---

## Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `COMMERCEOPS_CONFIG` | unset | Absolute/relative path to business config JSON |
| `COMMERCEOPS_DEMO_MODE` | `0` | When `1`, `EcomEnv` auto-loads the demo config and `inference.py` prints the CEO-view demo step |
| `COMMERCEOPS_ASSERT_INVARIANTS` | unset | When `1`, `WorldEngine.step` asserts state invariants (auto-enabled in `tests/conftest.py`) |
| `COMMERCEOPS_DEBUG` | unset | Enables `GET /debug/last_step_info` |
| `COMMERCEOPS_CEO_TRACE` / `COMMERCEOPS_CEO_TRACE_PATH` | `0` / `ceo_decision_traces.json` | Dump per-step explainability JSON from inference |
| `COMMERCEOPS_TRAINING_PROOF` / `COMMERCEOPS_TRAINING_PROOF_PATH` | `0` / `training_proof.json` | Dump aggregate training artefacts |
| `COMMERCEOPS_REWARD_CURVE_PATH` / `COMMERCEOPS_PROFIT_CURVE_PATH` / `COMMERCEOPS_INVENTORY_CURVE_PATH` | `*_inference.png` | Output paths for the three Matplotlib plots |
| `COMMERCEOPS_REORDER_THRESHOLD` | `3` | Inference-side reorder threshold fallback |
| `COMMERCEOPS_STRICT_GRADER_CONTEXT` | unset | When `1`, bare-grader-no-context escalates from `DeprecationWarning` to `RuntimeError` |

---

## Determinism

`POST /reset` seeds env-local `random.Random` + `numpy.random.Generator` instances (no global RNG pollution), so two runs with the same seed produce byte-identical observations, rewards, grader outputs, demand draws, competitor walks, ticket spawns, and `info` payloads. Verified by `test_simulation_invariants.py::test_reset_isolated_rng_streams` and end-to-end replays.

## Operations Notes

- **Single-session benchmark mode.** The FastAPI server owns one `EcomEnv` singleton per process. A `threading.Lock` guards all state-mutating routes (`/reset`, `/step`, `/config`, `/grader`) so concurrent requests cannot corrupt the world. Grader scoring runs inside that same lock (A2-1).
- **Multi-client serving caveat.** Because the env is a singleton, multiple clients share one world. Running Uvicorn with `--workers N>1` logs a prominent warning at startup because each worker would hold an independent env and deterministic grading would break. For production-style concurrent benchmarking, run one container per evaluator.
- **Input safety.** `/config` only accepts `business_id` values matching `^[a-z0-9][a-z0-9_\-]{0,63}$` and backed by a real file under [`configs/`](configs). Malformed or unknown ids return 4xx without touching the filesystem. Malformed `/step` / `/reset` payloads also return stable `400`s instead of uncaught `500`s.
- **Regression tests.** `pytest tests/` runs contract, simulation-invariant, supplier-flow, reward-engine, demand-model, grader-bound, config-validation, post-audit regression matrix, explainability-integration, and API-adversarial sweeps. **All 218 tests must pass** before submitting. The smoke script that hits a live Hugging Face Space lives under `scripts/smoke_env.py` so it is never picked up by pytest collection.
- **Async wrappers.** `EcomEnv.reset_async` / `step_async` are `async def` convenience shims over the synchronous core; they do not release the GIL. For true multi-env concurrency in asyncio code, wrap with `asyncio.to_thread`.

## Baseline Performance (zero-shot Qwen2.5-0.5B-Instruct)

| Task | Difficulty | Zero-Shot Score | Notes |
|---|---|---|---|
| Ticket Triage | Easy | ~0.80 | Often resolves every seeded ticket |
| Inventory Health | Medium | ~0.40 | Balance between restocks and cash |
| Profit Maximization | Hard | ~0.15 | Undershoots revenue; frequent bankruptcy |

Post-GRPO (target): profit task ≥ 0.50, bankruptcy rate < 20%.
