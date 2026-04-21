"""
reward_engine.py — Dense reward shaping for CommerceOps v2.

The engine combines an action's base reward with seven shaping terms:
    * revenue signal (linear / log / capped)
    * solvency bonus above a configurable threshold
    * stockout penalty on SKUs that dropped to zero this step
    * urgent- and critical-ticket aging penalty
    * ad ROI bonus per SKU with spend + sales
    * bankruptcy terminal penalty
    * bank-balance delta alignment term

Each term is isolated in a small private helper so individual behaviours can
be unit-tested and re-tuned without touching the aggregator. All coefficients
come from the ``rewards`` section of the active business config.

Revenue modes (``rewards.revenue_mode``):
    * ``"linear"`` (default) — legacy behaviour, ``rev_mult * daily_revenue``.
    * ``"log"`` — ``rev_mult * log1p(daily_revenue)`` so one giant-ticket sale
      doesn't dwarf the shaping signal that keeps the policy multi-objective.
    * ``"cap"`` — ``min(rev_mult * daily_revenue, revenue_cap_per_step)``.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple


logger = logging.getLogger("commerceops.reward")


def _as_list(tickets) -> List[dict]:
    """Coerce mixed ticket sequences (dicts or Pydantic) into a list of dicts.

    v2.3 Phase 5.3 — previously a malformed/non-serialisable entry was
    silently replaced with ``{}``, which hid bad ticket data behind
    "no-urgent-ticket" reward readings. We now log a WARNING so the
    upstream producer surfaces in a real run.
    """
    out: List[dict] = []
    for t in tickets or []:
        if isinstance(t, dict):
            out.append(t)
        else:
            try:
                out.append(t.model_dump())
            except Exception as exc:
                logger.warning(
                    "ticket_model_dump_failed type=%s exc=%s",
                    type(t).__name__,
                    exc.__class__.__name__,
                )
                out.append({})
    return out


def _daily_revenue(state_after: Dict) -> float:
    sales = state_after.get("daily_sales", {}) or {}
    prices = state_after.get("prices", {}) or {}
    return sum(
        float(sales.get(sku, 0)) * float(prices.get(sku, 0.0)) for sku in sales
    )


# ---------------------------------------------------------------------------
# Per-term helpers. Each returns a float that simply gets summed by
# ``compute_step_reward`` -- no shared mutable state.
# ---------------------------------------------------------------------------

def _revenue_term(state_after: Dict, cfg: Dict, daily_revenue: float) -> float:
    rev_mult = float(cfg.get("revenue_multiplier", 0.001))
    mode = str(cfg.get("revenue_mode", "linear")).lower()
    if mode == "log":
        return rev_mult * math.log1p(max(0.0, daily_revenue))
    if mode == "cap":
        cap = float(cfg.get("revenue_cap_per_step", 1.0))
        return min(rev_mult * daily_revenue, cap)
    return rev_mult * daily_revenue


def _solvency_term(state_after: Dict, cfg: Dict) -> float:
    threshold = float(cfg.get("solvency_threshold", 500.0))
    bonus = float(cfg.get("solvency_per_step", 0.05))
    return bonus if state_after.get("bank_balance", 0.0) >= threshold else 0.0


def _stockout_term(state_before: Dict, state_after: Dict, cfg: Dict) -> float:
    """Penalise SKUs that transitioned from >0 to 0 units this step.

    v2.3 optimisation — when ``rewards.stockout_transition_grace`` is
    truthy (default **off** for backward compat), SKUs with an in-flight
    restock (``pending_orders[sku] > 0``) are **exempt** from the
    penalty. Rationale: the config's ``restock_lead_days`` can easily
    push a valid refill past the moment inventory hits zero, and the
    policy has already taken the correct corrective action — punishing
    it there penalises *doing the right thing too late*, not a real
    operational failure. A sustained stockout (next tick still zero,
    still no pending delivery) is already captured by the "continues to
    sell nothing" signal via ``revenue_multiplier``.
    """
    penalty = float(cfg.get("stockout_penalty", -0.2))
    if penalty == 0.0:
        return 0.0
    inv_before = state_before.get("inventory", {}) or {}
    inv_after = state_after.get("inventory", {}) or {}
    grace = bool(cfg.get("stockout_transition_grace", False))
    pending = state_after.get("pending_orders", {}) or {} if grace else {}
    transitions = 0
    for sku, qty_after in inv_after.items():
        if int(qty_after) != 0 or int(inv_before.get(sku, 0)) <= 0:
            continue
        if grace and int(pending.get(sku, 0)) > 0:
            continue
        transitions += 1
    return penalty * transitions


def _ticket_aging_term(state_after: Dict, cfg: Dict) -> float:
    urgent_penalty = float(cfg.get("urgent_ticket_per_step", -0.1))
    critical_penalty = float(
        cfg.get("critical_ticket_per_step", urgent_penalty * 1.5)
    )
    age_threshold = int(cfg.get("urgent_ticket_age_days", 3))
    current_day = int(state_after.get("current_day", 1))
    aging = [
        t
        for t in _as_list(state_after.get("active_tickets"))
        if t.get("status") == "open"
        and current_day - int(t.get("created_day", 1)) >= age_threshold
    ]
    urgent = sum(1 for t in aging if t.get("urgency") == "urgent")
    critical = sum(1 for t in aging if t.get("urgency") == "critical")
    return urgent_penalty * urgent + critical_penalty * critical


def _ad_roi_term(
    state_before: Dict,
    state_after: Dict,
    cfg: Dict,
    action_result: Dict,
) -> float:
    """Credit ``ad_roi_positive`` per SKU where active ad spend produced sales.

    v2.3 Phase 1.1 — the canonical source of "ad budget active this tick" is
    now ``action_result["ad_spend_applied"]``; the WorldEngine populates it
    from ``do_ad_spend``'s info on the same step. We keep the legacy
    ``state_before.active_ad_spend`` fallback so hand-crafted reward-engine
    unit tests (which don't go through WorldEngine) still work.
    """
    bonus = float(cfg.get("ad_roi_positive", 0.0))
    if not bonus:
        return 0.0
    applied = action_result.get("ad_spend_applied") or {}
    prior_ads = applied or (state_before.get("active_ad_spend", {}) or {})
    daily_sales = state_after.get("daily_sales", {}) or {}
    reward = 0.0
    for sku, spend in prior_ads.items():
        try:
            spend_val = float(spend)
        except (TypeError, ValueError):
            continue
        if spend_val <= 0:
            continue
        if int(daily_sales.get(sku, 0)) > 0:
            reward += bonus
    return reward


def _bankruptcy_term(state_after: Dict, cfg: Dict) -> float:
    threshold = float(cfg.get("bankruptcy_threshold", 0.0))
    terminal = float(cfg.get("bankruptcy_terminal", -1.0))
    return terminal if state_after.get("bank_balance", 0.0) <= threshold else 0.0


def _delta_term(
    state_before: Dict,
    state_after: Dict,
    cfg: Dict,
    daily_revenue: float,
) -> float:
    # ``revenue_multiplier`` already credits ``daily_revenue``, so this term
    # subtracts it and only captures non-revenue cash flows (restock cost,
    # ad spend, refund payouts). Both coefficients stay useful independently.
    weight = float(cfg.get("bank_balance_delta_weight", 0.01))
    delta = float(state_after.get("bank_balance", 0.0)) - float(
        state_before.get("bank_balance", 0.0)
    )
    return weight * (delta - daily_revenue)


def _inventory_target_term(
    state_after: Dict,
    cfg: Dict,
    grader_ctx: Dict,
) -> float:
    """Dense bonus that aligns the reward with the ``inventory_task`` grader.

    v2.3 Phase 6.2 — previously the inventory grader's target was only
    probed at episode end via ``/grader`` and never appeared in any
    per-step shaping signal, so a policy had no in-band way to learn
    "keep the target SKU stocked". This term credits
    ``rewards.inventory_target_bonus`` per step whenever the target SKU's
    stock is at or above the configured target. Defaults to 0.0 so
    configs that don't enable it see no behaviour change.
    """
    bonus = float(cfg.get("inventory_target_bonus", 0.0))
    if not bonus:
        return 0.0
    if not isinstance(grader_ctx, dict):
        return 0.0
    target_sku = grader_ctx.get("inventory_target_sku")
    try:
        target_units = float(grader_ctx.get("inventory_target_units", 0))
    except (TypeError, ValueError):
        return 0.0
    if not target_sku or target_units <= 0:
        return 0.0
    inv = state_after.get("inventory", {}) or {}
    try:
        stock = float(inv.get(target_sku, 0))
    except (TypeError, ValueError):
        return 0.0
    return bonus if stock >= target_units else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_step_reward(
    action_result: Dict,
    state_before: Dict,
    state_after: Dict,
    rewards_config: Dict,
    *,
    return_breakdown: bool = False,
    grader_context: Dict | None = None,
) -> "float | Tuple[float, Dict[str, float]]":
    """Return the dense step reward shaped by ``rewards_config`` coefficients.

    When ``return_breakdown=True`` returns ``(total, breakdown_dict)`` so the
    WorldEngine can emit a per-term trace in ``info`` for observability
    without changing the scalar reward that RL algorithms see.

    The engine prefers the pre-computed ``daily_revenue`` on ``action_result``
    when present (WorldEngine threads it through to avoid a double
    recomputation). Legacy callers that pass only ``base_reward`` still work.

    ``grader_context`` (v2.3 Phase 6.2) is the per-env ``EcomEnv.grader_context``
    dict. It is read only by the optional ``inventory_target_bonus`` term; all
    other shaping stays untouched so policies trained on older configs keep
    their behaviour.
    """
    base = float(action_result.get("base_reward", 0.0))
    daily_revenue = float(
        action_result.get("daily_revenue", _daily_revenue(state_after))
    )

    revenue = _revenue_term(state_after, rewards_config, daily_revenue)
    solvency = _solvency_term(state_after, rewards_config)
    stockout = _stockout_term(state_before, state_after, rewards_config)
    aging = _ticket_aging_term(state_after, rewards_config)
    ad_roi = _ad_roi_term(state_before, state_after, rewards_config, action_result)
    bankruptcy = _bankruptcy_term(state_after, rewards_config)
    delta = _delta_term(state_before, state_after, rewards_config, daily_revenue)
    inv_bonus = _inventory_target_term(
        state_after, rewards_config, grader_context or {}
    )

    # v2.3 Phase 5.6 — round per-term values *once*, then sum. Previously
    # both the per-term entries AND ``total`` were independently rounded,
    # so ``sum(breakdown.values())`` did not always equal ``total`` to the
    # last decimal. Now ``total`` is the exact sum of the rounded terms
    # so downstream loggers can reconcile the two without tolerances.
    if not return_breakdown:
        total = round(
            base + revenue + solvency + stockout + aging + ad_roi
            + bankruptcy + delta + inv_bonus,
            4,
        )
        return total

    breakdown_terms = {
        "base": round(base, 4),
        "revenue": round(revenue, 4),
        "solvency": round(solvency, 4),
        "stockout": round(stockout, 4),
        "ticket_aging": round(aging, 4),
        "ad_roi": round(ad_roi, 4),
        "bankruptcy": round(bankruptcy, 4),
        "delta": round(delta, 4),
        "inventory_target_bonus": round(inv_bonus, 4),
    }
    total = round(sum(breakdown_terms.values()), 4)
    # Post-audit D.3 — defense-in-depth invariant: ``total`` must equal
    # the rounded sum of the per-term values. Holds trivially today
    # because that's exactly how ``total`` is built, but guards against a
    # future refactor where someone inlines a separate aggregate path and
    # forgets to keep the two in sync. We never *raise* here — RL
    # rollouts treat this as a soft warning so a reward anomaly doesn't
    # crash the training loop; CI catches regressions via the dedicated
    # invariant test.
    recomputed = round(sum(breakdown_terms.values()), 4)
    if abs(recomputed - total) > 1e-6:
        logger.warning(
            "reward_breakdown_sum_mismatch total=%s sum=%s terms=%s",
            total,
            recomputed,
            breakdown_terms,
        )
    breakdown = dict(breakdown_terms)
    breakdown["daily_revenue"] = round(daily_revenue, 2)
    return total, breakdown
