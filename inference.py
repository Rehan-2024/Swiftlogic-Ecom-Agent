"""
inference.py — Swiftlogic CommerceOps v2 inference loop for OpenEnv evaluation.
Prints strictly formatted [START], [STEP], and [END] lines to stdout.

Runs the full 50-step business cycle per task. Supports all five action types:
restock, refund, ad_spend, negotiate, wait. Falls back to WaitAction on any
parsing error. Also emits a per-task diagnostic line summarising negotiate
usage and quote-to-restock conversion to aid policy debugging.
"""

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

# Audit MINOR #9 — ``openai`` and ``matplotlib`` are heavyweight imports
# that previously loaded at module import time. They are not needed for
# unit tests that import the explainability builders (e.g.
# ``build_step_trace``), so we defer them to the call sites that
# actually use them. ``main()`` imports ``OpenAI`` lazily, and
# ``_save_training_proof`` imports ``matplotlib.pyplot`` lazily. This
# keeps test collection fast and removes the hard runtime dependency
# on a network-capable openai client for offline/deterministic tests.


logger = logging.getLogger("commerceops.inference")

from ecom_env import (
    EcomEnv,
    WaitAction,
    RestockAction,
    RefundAction,
    AdSpendAction,
    NegotiateAction,
    SetPriceAction,
    grade_triage_task,
    grade_inventory_task,
    grade_profit_task,
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float], graders_str: str = "") -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    graders_part = f" graders={graders_str}" if graders_str else ""
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}{graders_part}",
        flush=True,
    )


def log_diagnostics(task: str, negotiate_count: int, restock_count: int,
                    negotiated_restock_count: int, total_steps: int) -> None:
    """Emit a single structured diagnostics line per task.

    Metrics:
      * negotiate_rate       - fraction of steps that used NegotiateAction
      * quote_conversion     - fraction of restocks that consumed a quote
    """
    neg_rate = (negotiate_count / total_steps) if total_steps else 0.0
    conv_rate = (negotiated_restock_count / restock_count) if restock_count else 0.0
    print(
        f"[DIAG] task={task} negotiate_count={negotiate_count} restock_count={restock_count} "
        f"negotiated_restock_count={negotiated_restock_count} "
        f"negotiate_rate={neg_rate:.3f} quote_conversion={conv_rate:.3f}",
        flush=True,
    )


def _obs_summary(obs) -> dict:
    inv = getattr(obs, "inventory", {}) or {}
    tickets = getattr(obs, "active_tickets", []) or []
    return {
        "day": int(getattr(obs, "current_day", 0)),
        "bank_balance": float(getattr(obs, "bank_balance", 0.0)),
        "low_stock_skus": [k for k, v in inv.items() if int(v) <= 3],
        "open_tickets": sum(1 for t in tickets if getattr(t, "status", None) == "open"),
        "customer_satisfaction": float(getattr(obs, "customer_satisfaction", 1.0)),
    }


def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if isinstance(obs, dict):
        return dict(obs)
    return {}


def _word_limit(text: str, limit: int = 12) -> str:
    words = str(text).split()
    return " ".join(words[:limit])


def _cap_items(items: List[str], min_items: int = 2, max_items: int = 3) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        short = _word_limit(item.strip(), 12)
        if not short:
            continue
        key = short.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(short)
        if len(out) >= max_items:
            break
    if not out:
        out = [_word_limit("No strong state signal", 12)]
    if len(out) < min_items:
        out.append(_word_limit("State remained broadly stable", 12))
    return out[:max_items]


def _extract_action_params(action: Any) -> Dict[str, Any]:
    if hasattr(action, "model_dump"):
        raw = action.model_dump()
    elif isinstance(action, dict):
        raw = dict(action)
    else:
        raw = {"action_type": getattr(action, "action_type", "wait")}
    raw.pop("action_type", None)
    return raw


def build_state_summary(
    obs_before: Any,
    obs_after: Any,
    action: Any,
    reorder_threshold: int | None = None,
) -> Dict[str, Any]:
    b = _obs_to_dict(obs_before)
    a = _obs_to_dict(obs_after)
    inventory = a.get("inventory", {}) or {}
    prices = a.get("prices", {}) or {}
    competitors = a.get("competitor_prices", {}) or {}
    tickets = a.get("active_tickets", []) or []
    params = _extract_action_params(action)
    focus_sku = params.get("sku")
    if not isinstance(focus_sku, str) or focus_sku not in inventory:
        focus_sku = sorted(inventory.keys())[0] if inventory else ""
    # Audit MINOR #11 — honour an explicit caller-supplied reorder
    # threshold; fall back to the ``COMMERCEOPS_REORDER_THRESHOLD``
    # environment variable for deployment tuning, then to the legacy
    # literal (3). Guarded against negative values so a mis-config
    # cannot flip every SKU to "healthy".
    if reorder_threshold is None:
        try:
            reorder_threshold = int(
                os.getenv("COMMERCEOPS_REORDER_THRESHOLD", "3") or 3
            )
        except (TypeError, ValueError):
            reorder_threshold = 3
    reorder_threshold = max(0, int(reorder_threshold))
    inv_qty = int(inventory.get(focus_sku, 0)) if focus_sku else 0
    # Audit MINOR #10 — previously ``obs_before`` was accepted but only
    # the "daily_sales" delta used it. We also use it to surface the
    # focus SKU's per-step inventory delta so downstream consumers see
    # whether our own action changed stock on the SKU we care about.
    inv_before = b.get("inventory", {}) or {}
    focus_inv_delta = int(inventory.get(focus_sku, 0)) - int(inv_before.get(focus_sku, 0)) if focus_sku else 0
    inventory_status = "low" if inv_qty < reorder_threshold else "healthy"
    our_price = float(prices.get(focus_sku, 0.0)) if focus_sku else 0.0
    comp_price = float(competitors.get(focus_sku, 0.0)) if focus_sku else 0.0
    if our_price <= 0 or comp_price <= 0:
        price_position = "price signal unavailable"
    elif our_price < comp_price:
        price_position = "undercutting competitor"
    elif our_price > comp_price:
        price_position = "overpriced vs competitor"
    else:
        price_position = "price parity with competitor"
    prev_demand = sum(int(v) for v in (b.get("daily_sales", {}) or {}).values())
    curr_demand = sum(int(v) for v in (a.get("daily_sales", {}) or {}).values())
    if curr_demand > prev_demand:
        demand_trend = "increasing"
    elif curr_demand < prev_demand:
        demand_trend = "decreasing"
    else:
        demand_trend = "flat"
    urgent_open = 0
    for t in tickets:
        t_dict = t if isinstance(t, dict) else getattr(t, "model_dump", lambda: {})()
        if str(t_dict.get("status", "")) == "open" and str(t_dict.get("urgency", "")).lower() in {"urgent", "critical"}:
            urgent_open += 1
    ticket_pressure = "high" if urgent_open > 2 else "normal"
    return {
        "focus_sku": focus_sku,
        "inventory_status": inventory_status,
        "price_position": price_position,
        "demand_trend": demand_trend,
        "ticket_pressure": ticket_pressure,
        "urgent_open_tickets": urgent_open,
        "bank_balance": float(a.get("bank_balance", 0.0)),
        "focus_inventory_delta": int(focus_inv_delta),
        "reorder_threshold": int(reorder_threshold),
    }


def build_decision(action: Any) -> Dict[str, Any]:
    return {
        "action": getattr(action, "action_type", "wait"),
        "parameters": _extract_action_params(action),
    }


def build_market_reaction(obs_before: Any, obs_after: Any, info: Dict[str, Any], action: Any) -> Dict[str, Any]:
    b = _obs_to_dict(obs_before)
    a = _obs_to_dict(obs_after)
    params = _extract_action_params(action)
    sku = params.get("sku")
    before_map = b.get("competitor_prices", {}) or {}
    after_map = a.get("competitor_prices", {}) or {}
    if not isinstance(sku, str) or sku not in after_map:
        sku = sorted(after_map.keys())[0] if after_map else ""
    before_price = float(before_map.get(sku, 0.0)) if sku else 0.0
    after_price = float(after_map.get(sku, 0.0)) if sku else 0.0
    # Wave 1 — explainability truth. The competitor narration must only
    # say "undercut"/"follow" when the engine actually fired a reactive
    # rule (``info.competitor_reaction.triggered == True``). Walk-only
    # moves (baseline stochastic drift) are labelled ``random_walk``.
    # A zero delta with no reactive trigger is ``hold``. This prevents
    # the explainer from inventing causal stories out of pure noise.
    reaction = info.get("competitor_reaction") if isinstance(info, dict) else None
    reaction = reaction if isinstance(reaction, dict) else {}
    triggered = bool(reaction.get("triggered", False))
    reaction_reason = str(reaction.get("reason", "none"))
    reaction_sku = reaction.get("sku") if reaction.get("sku") else sku
    reaction_magnitude = reaction.get("magnitude", 0.0)
    try:
        reaction_magnitude = float(reaction_magnitude or 0.0)
    except (TypeError, ValueError):
        reaction_magnitude = 0.0

    walk = info.get("competitor_walk") if isinstance(info, dict) else None
    walk_delta = 0.0
    if isinstance(walk, dict) and sku in walk:
        try:
            walk_delta = float(walk.get(sku) or 0.0)
        except (TypeError, ValueError):
            walk_delta = 0.0

    if triggered and reaction_reason in {"undercut", "follow"}:
        competitor_action = reaction_reason
    elif abs(walk_delta) > 1e-9:
        competitor_action = "random_walk"
    elif after_price < before_price:
        competitor_action = "drift_down"
    elif after_price > before_price:
        competitor_action = "drift_up"
    else:
        competitor_action = "hold"
    event_active = False
    event_type = "not_available"
    multiplier = None
    if isinstance(info.get("market_shock"), dict):
        ms = info.get("market_shock") or {}
        multipliers = ms.get("sku_multipliers", {}) or {}
        if sku and sku in multipliers:
            multiplier = float(multipliers.get(sku))
            event_active = bool(multiplier and abs(multiplier - 1.0) > 1e-12)
            event_type = "market_shock" if event_active else "none"
        else:
            event_type = "market_shock_unknown_sku"
    elif "event_active" in info:
        event_active = bool(info.get("event_active"))
        event_type = str(info.get("event_type", "unknown"))
        if info.get("multiplier") is not None:
            multiplier = float(info.get("multiplier"))
    return {
        "sku": sku,
        "competitor_action": competitor_action,
        "price_before": before_price,
        "price_after": after_price,
        "event_active": event_active,
        "event_type": event_type,
        "multiplier": multiplier,
        "demand_generated": int(sum(int(v) for v in (a.get("daily_sales", {}) or {}).values())),
        "reaction_triggered": bool(triggered),
        "reaction_reason": reaction_reason,
        "reaction_sku": reaction_sku,
        "reaction_magnitude": round(float(reaction_magnitude), 6),
    }


def build_outcome(obs_before: Any, obs_after: Any) -> Dict[str, Any]:
    b = _obs_to_dict(obs_before)
    a = _obs_to_dict(obs_after)
    inv_before = b.get("inventory", {}) or {}
    inv_after = a.get("inventory", {}) or {}
    inv_delta = {
        sku: int(inv_after.get(sku, 0)) - int(inv_before.get(sku, 0))
        for sku in sorted(set(inv_before) | set(inv_after))
    }
    return {
        "sales": dict(a.get("daily_sales", {}) or {}),
        "inventory_change": inv_delta,
        "tickets_change": len(a.get("active_tickets", []) or []) - len(b.get("active_tickets", []) or []),
        "bank_balance_delta": float(a.get("bank_balance", 0.0)) - float(b.get("bank_balance", 0.0)),
    }


def build_reward_summary(total_reward: float, info: Dict[str, Any]) -> Dict[str, Any]:
    bd = (info or {}).get("reward_breakdown", {}) or {}
    non_terms = {"daily_revenue", "scale_hint"}
    items: List[tuple[str, float]] = []
    for k, v in bd.items():
        if k in non_terms:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        items.append((str(k), fv))
    items.sort(key=lambda kv: (-abs(kv[1]), kv[0]))
    top_drivers = [k for k, _ in items[:2]]
    return {"total": float(total_reward), "top_drivers": top_drivers}


def build_reasoning(state_summary: Dict[str, Any], decision: Dict[str, Any], market_reaction: Dict[str, Any]) -> List[str]:
    rules: List[str] = []
    if state_summary.get("inventory_status") == "low":
        rules.append("Inventory below safety level")
    if state_summary.get("price_position") == "overpriced vs competitor":
        rules.append("Our price above competitor")
    if state_summary.get("demand_trend") == "increasing":
        rules.append("Demand increasing")
    if state_summary.get("ticket_pressure") == "high":
        rules.append("Urgent ticket pressure high")
    if market_reaction.get("competitor_action") == "undercut":
        rules.append("Competitor undercut pricing")
    if decision.get("action") == "restock":
        rules.append("Restock chosen to protect inventory")
    return _cap_items(rules, min_items=2, max_items=3)


def build_causal_chain(decision: Dict[str, Any], market_reaction: Dict[str, Any], outcome: Dict[str, Any]) -> List[str]:
    chain: List[str] = []
    action = decision.get("action")
    sales_total = sum(int(v) for v in (outcome.get("sales", {}) or {}).values())
    bank_delta = float(outcome.get("bank_balance_delta", 0.0))
    inv_delta = outcome.get("inventory_change", {}) or {}
    if action == "set_price" and sales_total > 0:
        chain.append("Price move influenced demand")
    if market_reaction.get("competitor_action") == "undercut":
        chain.append("Competitor undercut shifted price pressure")
    if action == "restock":
        chain.append("Restock increased available stock")
    if any(v < 0 for v in inv_delta.values()):
        chain.append("Demand reduced inventory")
    if bank_delta > 0:
        chain.append("Sales improved bank balance")
    return _cap_items(chain, min_items=2, max_items=3)


def build_why_it_worked(decision: Dict[str, Any], state_summary: Dict[str, Any], outcome: Dict[str, Any], reward_summary: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    action = decision.get("action")
    sales_total = sum(int(v) for v in (outcome.get("sales", {}) or {}).values())
    if action == "restock" and state_summary.get("inventory_status") == "low":
        out.append("Restocking reduced stockout risk")
    if action == "set_price" and sales_total > 0:
        out.append("Pricing decision aligned with demand")
    if float(outcome.get("bank_balance_delta", 0.0)) > 0:
        out.append("Revenue increased cash position")
    drivers = reward_summary.get("top_drivers", []) or []
    if drivers:
        out.append(f"Reward led by {drivers[0]}")
    return _cap_items(out, min_items=2, max_items=3)


def build_department_suggestions(
    obs_after: Any,
    info: Dict[str, Any],
    state_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Wave 4 — lightweight per-department advisor, inference-only.

    Each department returns a *suggestion* string plus a tiny structured
    payload (so downstream agents can consume it programmatically). The
    logic is purely derivative of observable state + the ``info`` fields
    the engine already emits; it does NOT mutate the env. Determinism
    is preserved: given the same obs/info we return the same dict.
    """
    a = _obs_to_dict(obs_after)
    inventory = a.get("inventory", {}) or {}
    prices = a.get("prices", {}) or {}
    competitors = a.get("competitor_prices", {}) or {}
    tickets = a.get("active_tickets", []) or []

    reorder_threshold = int(state_summary.get("reorder_threshold", 3) or 3)
    low_skus = sorted(
        sku for sku, qty in inventory.items() if int(qty or 0) < reorder_threshold
    )
    if low_skus:
        focus = low_skus[0]
        short = int(max(1, reorder_threshold * 3 - int(inventory.get(focus, 0) or 0)))
        inv_suggestion = f"restock {focus} ~{short} units"
        inv_urgency = "high" if int(inventory.get(focus, 0) or 0) <= 0 else "medium"
    else:
        focus = ""
        inv_suggestion = "hold restocks - inventory healthy"
        inv_urgency = "low"

    cheaper_skus: List[tuple[str, float, float]] = []
    for sku, our_price in prices.items():
        try:
            op = float(our_price or 0.0)
            cp = float(competitors.get(sku, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if op > 0.0 and cp > 0.0 and op > cp:
            cheaper_skus.append((sku, op, cp))
    cheaper_skus.sort(key=lambda row: (row[1] - row[2]) / max(1e-9, row[2]), reverse=True)
    if cheaper_skus:
        mk_sku, our_p, comp_p = cheaper_skus[0]
        gap_pct = (our_p - comp_p) / max(1e-9, comp_p) * 100.0
        marketing_suggestion = (
            f"reduce price on {mk_sku} (~{gap_pct:.1f}% above competitor)"
        )
        marketing_urgency = "high" if gap_pct > 15.0 else "medium"
    elif state_summary.get("demand_trend") == "decreasing":
        marketing_suggestion = "increase ad spend - demand trending down"
        marketing_urgency = "medium"
    else:
        marketing_suggestion = "hold pricing - competitive"
        marketing_urgency = "low"

    open_urgent = 0
    open_total = 0
    for t in tickets:
        t_dict = t if isinstance(t, dict) else getattr(t, "model_dump", lambda: {})()
        if str(t_dict.get("status", "")) != "open":
            continue
        open_total += 1
        if str(t_dict.get("urgency", "")).lower() in {"urgent", "critical"}:
            open_urgent += 1
    if open_urgent >= 3:
        support_suggestion = f"prioritize refunds - {open_urgent} urgent tickets open"
        support_urgency = "high"
    elif open_urgent >= 1:
        support_suggestion = f"resolve {open_urgent} urgent ticket(s)"
        support_urgency = "medium"
    elif open_total > 0:
        support_suggestion = f"work through {open_total} open ticket(s)"
        support_urgency = "low"
    else:
        support_suggestion = "support queue clear"
        support_urgency = "low"

    return {
        "inventory": {
            "suggestion": inv_suggestion,
            "urgency": inv_urgency,
            "focus_sku": focus,
            "low_skus": low_skus,
        },
        "marketing": {
            "suggestion": marketing_suggestion,
            "urgency": marketing_urgency,
            "overpriced_skus": [row[0] for row in cheaper_skus[:3]],
        },
        "support": {
            "suggestion": support_suggestion,
            "urgency": support_urgency,
            "open_urgent": int(open_urgent),
            "open_total": int(open_total),
        },
    }


def build_decision_context(
    decision: Dict[str, Any],
    state_summary: Dict[str, Any],
    departments: Dict[str, Any],
    info: Dict[str, Any],
) -> Dict[str, Any]:
    """Wave 4 — map the chosen action to the departments whose signals
    most likely motivated it. Deterministic and purely additive.
    """
    action = str(decision.get("action", "wait"))
    based_on: List[str] = []
    # Inventory-motivated actions.
    if action in {"restock", "negotiate"}:
        based_on.append("inventory")
        if state_summary.get("inventory_status") == "low":
            based_on.append("inventory_low_signal")
    # Marketing / pricing.
    if action in {"set_price", "ad_spend"}:
        based_on.append("marketing")
        if state_summary.get("price_position") == "overpriced vs competitor":
            based_on.append("competitor_pricing_signal")
        if state_summary.get("demand_trend") == "decreasing":
            based_on.append("demand_decline_signal")
    # Support.
    if action == "refund":
        based_on.append("support")
        if state_summary.get("ticket_pressure") == "high":
            based_on.append("urgent_ticket_signal")
    # Intent from engine (Wave 3).
    intent = info.get("intent") if isinstance(info, dict) else None
    if intent:
        based_on.append(f"ceo_intent:{intent}")
    # Gap 4 — when the engine's CEO intent is ``maintain_balance``
    # (no pressing signal) and the agent picked ``wait``, that's a
    # deliberate "steady-state" choice, not a null decision. Tag it
    # so the narrative reads correctly.
    if intent == "maintain_balance" and action == "wait":
        based_on.append("steady_state_policy")
    # Dedup while preserving order.
    seen: set = set()
    deduped: List[str] = []
    for item in based_on:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return {
        "chosen_action": action,
        "based_on": deduped if deduped else ["default_policy"],
        "departments_consulted": sorted(departments.keys()) if isinstance(departments, dict) else [],
        "intent": intent,
    }


def validate_trace_schema(trace: Dict[str, Any]) -> None:
    required = {
        "state_summary",
        "decision",
        "reasoning",
        "market_reaction",
        "outcome",
        "reward_summary",
        "causal_chain",
        "why_it_worked",
    }
    missing = required - set(trace.keys())
    if missing:
        raise ValueError(f"trace missing keys: {sorted(missing)}")
    for key in ("reasoning", "causal_chain", "why_it_worked"):
        arr = trace.get(key) or []
        if not isinstance(arr, list):
            raise ValueError(f"{key} must be list")
        if not (2 <= len(arr) <= 3):
            raise ValueError(f"{key} must contain 2-3 items")
        for item in arr:
            if len(str(item).split()) > 12:
                raise ValueError(f"{key} item exceeds 12 words: {item!r}")


def build_step_trace(obs_before: Any, obs_after: Any, action: Any, info: Dict[str, Any], reward_val: float) -> Dict[str, Any]:
    state_summary = build_state_summary(obs_before, obs_after, action)
    decision = build_decision(action)
    market_reaction = build_market_reaction(obs_before, obs_after, info, action)
    outcome = build_outcome(obs_before, obs_after)
    reward_summary = build_reward_summary(reward_val, info)
    reasoning = build_reasoning(state_summary, decision, market_reaction)
    causal_chain = build_causal_chain(decision, market_reaction, outcome)
    why_it_worked = build_why_it_worked(decision, state_summary, outcome, reward_summary)
    departments = build_department_suggestions(obs_after, info or {}, state_summary)
    decision_context = build_decision_context(decision, state_summary, departments, info or {})
    trace = {
        "state_summary": state_summary,
        "decision": decision,
        "reasoning": reasoning,
        "market_reaction": market_reaction,
        "outcome": outcome,
        "reward_summary": reward_summary,
        "causal_chain": causal_chain,
        "why_it_worked": why_it_worked,
        "departments": departments,
        "decision_context": decision_context,
        "intent": info.get("intent") if isinstance(info, dict) else None,
        "trend": info.get("trend") if isinstance(info, dict) else None,
        "kpis": info.get("kpis") if isinstance(info, dict) else None,
        "why_failed": info.get("why_failed") if isinstance(info, dict) else None,
        "confidence": info.get("confidence") if isinstance(info, dict) else None,
        "policy_stability": info.get("policy_stability") if isinstance(info, dict) else None,
        "anomalies": info.get("anomalies") if isinstance(info, dict) else None,
    }
    validate_trace_schema(trace)
    return trace


def _build_action(action_data: dict):
    a_type = action_data.get("action_type")
    if a_type == "restock":
        return RestockAction(**action_data)
    if a_type == "refund":
        return RefundAction(**action_data)
    if a_type == "ad_spend":
        return AdSpendAction(**action_data)
    if a_type == "negotiate":
        return NegotiateAction(**action_data)
    if a_type == "set_price":
        return SetPriceAction(**action_data)
    return WaitAction()


def _fmt_num(value: Any, fmt: str = ".2f", default: str = "n/a") -> str:
    """Null-safe float formatter used by demo output."""
    try:
        if value is None:
            return default
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return default


def _print_demo_step(step: int, trace: Dict[str, Any], reward_val: float) -> None:
    decision = trace.get("decision", {}) or {}
    market = trace.get("market_reaction", {}) or {}
    outcome = trace.get("outcome", {}) or {}
    why = trace.get("why_it_worked", []) or []
    departments = trace.get("departments", {}) or {}
    decision_context = trace.get("decision_context", {}) or {}
    kpis = trace.get("kpis", {}) or {}
    intent = trace.get("intent")
    trend = trace.get("trend") or {}
    why_failed = trace.get("why_failed") or []
    confidence = trace.get("confidence")
    policy_stability = trace.get("policy_stability") or {}
    anomalies = trace.get("anomalies") or []

    action = decision.get("action", "wait")
    params = decision.get("parameters", {}) or {}
    print(f"Day {step}:", flush=True)
    print(f"- CEO intent: {intent or 'n/a'}", flush=True)
    print(f"- Action: {action} {params}", flush=True)
    print(f"- Reason: {', '.join(trace.get('reasoning', [])[:2])}", flush=True)
    mk_action = market.get("competitor_action", "n/a")
    mk_sku = market.get("sku", "n/a")
    mk_before = _fmt_num(market.get("price_before"), ".2f")
    mk_after = _fmt_num(market.get("price_after"), ".2f")
    triggered = "yes" if market.get("reaction_triggered") else "no"
    print(
        f"- Competitor: {mk_action} [{mk_sku}] {mk_before}->{mk_after} "
        f"(engine_triggered={triggered}, reason={market.get('reaction_reason', 'none')})",
        flush=True,
    )
    print(
        f"- Outcome: bank_delta={_fmt_num(outcome.get('bank_balance_delta'), '.2f')} "
        f"tickets_delta={outcome.get('tickets_change', 0)} "
        f"sales={sum(int(v or 0) for v in (outcome.get('sales', {}) or {}).values())}",
        flush=True,
    )
    pm = _fmt_num(kpis.get("profit_margin"), ".3f")
    sr = _fmt_num(kpis.get("stockout_rate"), ".3f")
    it = _fmt_num(kpis.get("inventory_turnover"), ".3f")
    print(f"- KPIs: margin={pm} stockout_rate={sr} turnover={it}", flush=True)
    if trend:
        print(
            f"- Trend: rev={trend.get('revenue', 'n/a')} inv={trend.get('inventory', 'n/a')} "
            f"demand={trend.get('demand', 'n/a')} cash={trend.get('bank_balance', 'n/a')}",
            flush=True,
        )
    inv_dept = departments.get("inventory", {}) or {}
    mk_dept = departments.get("marketing", {}) or {}
    sp_dept = departments.get("support", {}) or {}
    print(
        f"- Departments:\n"
        f"    inventory: {inv_dept.get('suggestion', 'n/a')} ({inv_dept.get('urgency', 'low')})\n"
        f"    marketing: {mk_dept.get('suggestion', 'n/a')} ({mk_dept.get('urgency', 'low')})\n"
        f"    support:   {sp_dept.get('suggestion', 'n/a')} ({sp_dept.get('urgency', 'low')})",
        flush=True,
    )
    if decision_context:
        print(
            f"- Decision context: chosen={decision_context.get('chosen_action', 'n/a')} "
            f"based_on={decision_context.get('based_on', [])}",
            flush=True,
        )
    print(f"- Reward: {_fmt_num(reward_val, '.3f')} (confidence={_fmt_num(confidence, '.2f')})", flush=True)
    if policy_stability:
        print(
            f"- Policy stability: {_fmt_num(policy_stability.get('score'), '.2f')} "
            f"(window={policy_stability.get('window', 0)}, "
            f"last={policy_stability.get('last_action', 'n/a')})",
            flush=True,
        )
    if anomalies:
        top_anom = ", ".join(str(a.get("type", "unknown")) for a in anomalies[:3] if isinstance(a, dict))
        print(f"- Anomalies: {top_anom}", flush=True)
    if why_failed:
        print(f"- Why failed: {', '.join(str(r) for r in why_failed[:3])}", flush=True)
    print(f"- Why it worked: {', '.join(why[:2])}", flush=True)


def _stockout_rate_from_obs(obs: Any) -> float:
    inv = (_obs_to_dict(obs).get("inventory", {}) or {})
    if not inv:
        return 0.0
    zeros = sum(1 for _, v in inv.items() if int(v) <= 0)
    return float(zeros) / float(len(inv))


def _save_training_proof(
    before: Dict[str, float],
    after: Dict[str, float],
    reward_curve: List[float],
    profit_curve: List[float] | None = None,
    inventory_curve: List[float] | None = None,
) -> None:
    """Write training proof artefacts (metrics JSON + curve PNGs).

    ``matplotlib`` is imported lazily so that unit tests that never
    call this function do not pay the import cost or require a display
    backend. The ``Agg`` backend is forced when the import is made
    headless so CI and sandboxed servers can still emit PNGs.
    """
    out_path = Path(os.getenv("COMMERCEOPS_TRAINING_PROOF_PATH", "training_proof.json"))
    payload = {"before_training": before, "after_training": after}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        # Matplotlib missing/misconfigured is not fatal — the JSON proof
        # is the primary artefact and is already on disk. Log and move
        # on so the demo/training run never crashes for a plotting
        # backend issue.
        logger.warning("training_proof_plot_skipped exc=%s", exc.__class__.__name__)
        print(f"[PROOF] metrics={out_path} curves=skipped", flush=True)
        return

    curve_path = Path(os.getenv("COMMERCEOPS_REWARD_CURVE_PATH", "reward_curve_inference.png"))
    profit_path = Path(os.getenv("COMMERCEOPS_PROFIT_CURVE_PATH", "profit_curve_inference.png"))
    inv_path = Path(os.getenv("COMMERCEOPS_INVENTORY_CURVE_PATH", "inventory_curve_inference.png"))

    try:
        plt.figure(figsize=(8, 3))
        plt.plot(reward_curve)
        plt.title("Episode Reward Curve")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.tight_layout()
        plt.savefig(curve_path, dpi=120)
        plt.close()
    except Exception as exc:
        logger.warning("reward_curve_failed exc=%s", exc.__class__.__name__)

    if profit_curve:
        try:
            plt.figure(figsize=(8, 3))
            plt.plot(profit_curve)
            plt.title("Profit (bank balance) over Time")
            plt.xlabel("Step")
            plt.ylabel("Bank balance")
            plt.tight_layout()
            plt.savefig(profit_path, dpi=120)
            plt.close()
        except Exception as exc:
            logger.warning("profit_curve_failed exc=%s", exc.__class__.__name__)

    if inventory_curve:
        try:
            plt.figure(figsize=(8, 3))
            plt.plot(inventory_curve)
            plt.title("Total Inventory over Time")
            plt.xlabel("Step")
            plt.ylabel("Units on hand")
            plt.tight_layout()
            plt.savefig(inv_path, dpi=120)
            plt.close()
        except Exception as exc:
            logger.warning("inventory_curve_failed exc=%s", exc.__class__.__name__)

    print(
        f"[PROOF] metrics={out_path} reward_curve={curve_path} "
        f"profit_curve={profit_path} inventory_curve={inv_path}",
        flush=True,
    )


def main():
    # Audit MINOR #9 — the OpenAI client is only needed when we are
    # actually calling out to a live LLM. Importing it inside ``main()``
    # lets offline tests (which import ``build_step_trace`` etc.) run
    # without a working ``openai`` install.
    from openai import OpenAI

    benchmark = "commerce_ops_v2"
    api_base = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME", "default-model")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

    client = OpenAI(base_url=api_base, api_key=api_key, timeout=20)
    trace_enabled = os.getenv("COMMERCEOPS_CEO_TRACE", "0").strip().lower() in {"1", "true", "yes"}
    trace_path = Path(os.getenv("COMMERCEOPS_CEO_TRACE_PATH", "ceo_decision_traces.json"))
    demo_mode = os.getenv("COMMERCEOPS_DEMO_MODE", "0").strip().lower() in {"1", "true", "yes"}
    training_proof = os.getenv("COMMERCEOPS_TRAINING_PROOF", "0").strip().lower() in {"1", "true", "yes"}
    ceo_traces: dict[str, list[dict]] = {}
    after_metrics: list[Dict[str, float]] = []
    before_metrics: list[Dict[str, float]] = []
    # Wave 6 — auxiliary curves for the training proof plot. Populated
    # during the most recent task run; passed to ``_save_training_proof``.
    profit_curve: list[float] = []
    inventory_curve: list[float] = []

    env = EcomEnv()

    system_prompt = """You are an autonomous digital storefront operator for an Indian ethnic wear brand (Siyaani).
Your goal is to manage inventory, resolve customer support tickets, allocate ad spend,
negotiate supplier prices, and maximize profit over a 50-day business cycle without going bankrupt.

You must return ONLY raw JSON matching exactly one of these Action schemas:
1. {"action_type": "restock", "sku": "<string>", "quantity": <int>}
2. {"action_type": "refund", "ticket_id": "<string>"}
3. {"action_type": "ad_spend", "sku": "<string>", "budget": <float>}
4. {"action_type": "negotiate", "sku": "<string>", "quantity": <int>}
5. {"action_type": "wait"}
6. {"action_type": "set_price", "sku": "<string>", "price": <float>}

Negotiate requests a supplier unit-price quote for a future restock on the same
SKU. Quotes expire after 3 steps if not consumed by a restock. Small-volume
negotiated orders unlock a supplier volume discount; un-negotiated restocks
pay a spot-market premium over list cost.

set_price directly mutates the sell price for a SKU. It must stay within the
configured [price_min_mult_competitor, price_max_mult_competitor] band vs the
competitor's price; out-of-band prices are rejected with invalid_action.

The observation exposes ``pending_orders`` (aggregate in-flight quantity per
SKU) and ``pending_orders_schedule`` (per-SKU list of [delivery_day, qty]
pairs). Use the schedule to avoid over-restocking when a previous order is
already in transit.

Do not output any markdown formatting or explanations, just the JSON object.
"""

    tasks = ["triage_task", "inventory_task", "profit_task"]

    for task_name in tasks:
        env.reset(seed=42)
        initial_state = env.state().model_copy(deep=True)

        log_start(task_name, benchmark, model_name)

        rewards: list[float] = []
        # Reset Wave-6 auxiliary curves for each task so the saved
        # proof reflects the most recent task run.
        profit_curve = []
        inventory_curve = []
        total_steps = 50
        done = False
        step_num = 0

        negotiate_count = 0
        restock_count = 0
        negotiated_restock_count = 0
        task_traces: list[dict] = []

        for step_num in range(1, total_steps + 1):
            obs_state = env.state()
            obs_before_obj = obs_state

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Current Observation:\n{obs_state.model_dump_json()}"},
                    ],
                    response_format={"type": "json_object"},
                )
                raw_text = response.choices[0].message.content
                action_data = json.loads(raw_text)
                action = _build_action(action_data)
            except Exception as exc:
                # Phase G.2 — surface parse failures so a run of all-``wait``
                # steps is distinguishable from an offline policy. We log at
                # WARNING with the step number and exception class, but do
                # NOT emit the raw model output because it can be large and
                # may contain PII in some eval harnesses.
                logger.warning(
                    "action_parse_failed step=%s task=%s exc=%s",
                    step_num,
                    task_name,
                    exc.__class__.__name__,
                )
                action = WaitAction()

            obs, reward_obj, done, info = env.step(action)

            reward_val = reward_obj.value
            rewards.append(reward_val)
            # Wave 6 — deterministic aggregates for the proof plots.
            try:
                profit_curve.append(float(getattr(obs, "bank_balance", 0.0) or 0.0))
                inv_map = getattr(obs, "inventory", {}) or {}
                inventory_curve.append(float(sum(int(v or 0) for v in inv_map.values())))
            except Exception:
                pass

            # Diagnostics bookkeeping -------------------------------------------------
            if action.action_type == "negotiate" and not info.get("error"):
                negotiate_count += 1
            if action.action_type == "restock" and not info.get("error"):
                restock_count += 1
                if info.get("restock", {}).get("negotiated"):
                    negotiated_restock_count += 1

            error_str = info.get("error")
            log_step(step_num, action.action_type, reward_val, done, error_str)
            step_trace = build_step_trace(obs_before_obj, obs, action, info, reward_val)
            if trace_enabled:
                task_traces.append({"step": int(step_num), **step_trace})
            if demo_mode:
                _print_demo_step(step_num, step_trace, reward_val)

            if done:
                break

        success = not done or step_num == total_steps
        raw_score = sum(rewards)
        score = max(0.01, min(0.99, raw_score))

        final_state = env.state()

        # Post-audit m-6 — pass the env-local grader context explicitly so
        # we bypass the module mirror (and its DeprecationWarning).
        grader_ctx = getattr(env, "grader_context", None)
        if task_name == "triage_task":
            grader_score = grade_triage_task(initial_state, final_state)
        elif task_name == "inventory_task":
            grader_score = grade_inventory_task(
                initial_state, final_state, context=grader_ctx
            )
        elif task_name == "profit_task":
            grader_score = grade_profit_task(
                initial_state, final_state, context=grader_ctx
            )
        else:
            grader_score = 0.0

        graders_str = f"{task_name}:{grader_score:.2f}"

        log_end(success, len(rewards), score, rewards, graders_str)
        log_diagnostics(
            task=task_name,
            negotiate_count=negotiate_count,
            restock_count=restock_count,
            negotiated_restock_count=negotiated_restock_count,
            total_steps=len(rewards) or 1,
        )
        after_metrics.append(
            {
                "avg_reward": float(sum(rewards) / max(1, len(rewards))),
                "avg_profit": float(final_state.bank_balance - initial_state.bank_balance),
                "stockout_rate": _stockout_rate_from_obs(final_state),
            }
        )
        if trace_enabled:
            ceo_traces[task_name] = task_traces

        if training_proof:
            # Deterministic baseline: wait-only policy on same task/seed.
            env.reset(seed=42)
            b_initial = env.state().model_copy(deep=True)
            b_rewards: List[float] = []
            b_done = False
            for _ in range(50):
                b_obs, b_reward, b_done, _b_info = env.step(WaitAction())
                b_rewards.append(float(b_reward.value))
                if b_done:
                    break
            before_metrics.append(
                {
                    "avg_reward": float(sum(b_rewards) / max(1, len(b_rewards))),
                    "avg_profit": float(b_obs.bank_balance - b_initial.bank_balance),
                    "stockout_rate": _stockout_rate_from_obs(b_obs),
                }
            )

    if trace_enabled:
        trace_path.write_text(json.dumps(ceo_traces, indent=2), encoding="utf-8")
        print(f"[TRACE] wrote={trace_path}", flush=True)
    if training_proof and after_metrics and before_metrics:
        def _avg(rows: List[Dict[str, float]], key: str) -> float:
            return float(sum(float(r.get(key, 0.0)) for r in rows) / max(1, len(rows)))
        before = {k: _avg(before_metrics, k) for k in ("avg_reward", "avg_profit", "stockout_rate")}
        after = {k: _avg(after_metrics, k) for k in ("avg_reward", "avg_profit", "stockout_rate")}
        # Reward curve from the most recent task run is deterministic under fixed seed/model output stream.
        _save_training_proof(before, after, rewards, profit_curve, inventory_curve)


if __name__ == "__main__":
    main()
