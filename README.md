---
title: E-commerce Agent Env
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# AI Autonomous Startup Operator Environment

> **commerce_ops_v1** — A benchmark environment for autonomous digital storefront operations.

## Overview

This OpenEnv environment simulates a real-world **e-commerce storefront** that an AI agent must operate autonomously. The agent manages inventory levels, handles customer support tickets, and optimises ad spend — all through a discrete action space evaluated over a fixed time horizon.

Built for the **Scalerxmeta Hackathon**, the environment is intentionally lightweight (runs within 2 vCPU / 8 GB RAM) and fully deterministic when seeded.

### Motivation & Scenario
As the AI Operator for a digital storefront (e.g., ethnic wear brand *Siyaani*), the agent must balance operational efficiency with financial health. This environment tests an agent's ability to prioritize:
- **Reputation:** Resolving customer tickets via `RefundAction`.
- **Supply Chain:** Maintaining stock via `RestockAction`.
- **Fiscal Responsibility:** Managing a limited `bank_balance` to avoid bankruptcy.

---

## Observation Space (`EcomObservation`)

| Field             | Type                | Description                            |
|-------------------|---------------------|----------------------------------------|
| `current_day`     | `int`               | Current simulation day                 |
| `step_count`      | `int`               | Number of actions taken so far         |
| `bank_balance`    | `float`             | Cash on hand                           |
| `inventory`       | `Dict[str, int]`    | SKU → units in stock                   |
| `pending_orders`  | `Dict[str, int]`    | SKU → units awaiting fulfilment        |
| `active_tickets`  | `List[Ticket]`      | Open/resolved support tickets          |
| `daily_sales`     | `Dict[str, int]`    | SKU → units sold today                 |
| `active_ad_spend` | `Dict[str, float]`  | SKU → current daily ad budget          |

## Action Space (`EcomAction`)

The agent selects **one** of three actions per step:

| Action          | Fields                          | Effect                          |
|-----------------|----------------------------------|---------------------------------|
| `RestockAction` | `sku: str`, `quantity: int`      | Adds units to inventory         |
| `RefundAction`  | `ticket_id: str`                 | Resolves a customer ticket      |
| `WaitAction`    | *(none)*                         | Do nothing this step            |

Invalid actions (unknown SKU, non-positive quantity, missing ticket) incur a **−0.2 reward penalty**.

## Reward Space (`EcomReward`)

A single float `value` returned after each step:

| Outcome                | Reward  |
|------------------------|---------|
| Successful restock     | +0.1    |
| Successful refund      | +0.3    |
| Wait                   | 0.0     |
| Invalid action         | −0.2    |

---

## Environment Dynamics
Every `step` represents a strategic decision point in a business day.
- **Financials:** The `bank_balance` updates after every action (costs for restocking vs. losses from refunds).
- **Episode Length:** Simulations run for a 10-step "business cycle."
- **Validation:** Utilizes Pydantic v2 `RootModels` to enforce strict schema validation, preventing hallucinated actions from crashing the simulation.

## Tasks

### Easy — Ticket Triage
Resolve any one open support ticket. Graded by `grade_triage_task`.

### Medium — Inventory Health
Ensure `cotton_set` stock remains above zero **and** the bank balance stays positive. Graded by `grade_inventory_task`.

### Hard — Profit Maximisation
Grow the bank balance beyond the initial $1 000 seed capital. Normalised profit is mapped to `[0.0, 1.0]` via `grade_profit_task`.

---

## Deterministic Seeding

> The environment supports deterministic seeding (`env.reset(seed=42)`) to ensure strict reproducibility for RL agent evaluation, aligning with DeepMind and Meta research standards.

---

## Quick Start

```bash
pip install -r requirements.txt
python inference.py
```

### Docker

```bash
docker build -t commerce-ops .
docker run -p 7860:7860 commerce-ops
```

The container exposes the OpenEnv HTTP API on port **7860** (required by the Hugging Face validator).
