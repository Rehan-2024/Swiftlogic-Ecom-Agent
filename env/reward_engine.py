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
    """Reward a SKU's realised daily revenue.

    Modes:
        * ``"linear"`` (default) — ``rev_mult * daily_revenue``; optionally
          soft-capped at ``revenue_cap_per_step`` when that key is present
          and positive. Post-audit R-4 — previously ``cap`` was only
          honoured in ``"cap"`` mode, so a config with a linear multiplier
          and a huge one-time sale could spike the reward arbitrarily. The
          cap now applies uniformly whenever it is set, making the reward
          bounded-per-step by default.
        * ``"log"`` — ``rev_mult * log1p(daily_revenue)``.
        * ``"cap"`` — ``min(rev_mult * daily_revenue, revenue_cap_per_step)``.
    """
    rev_mult = float(cfg.get("revenue_multiplier", 0.001))
    mode = str(cfg.get("revenue_mode", "linear")).lower()
    if mode == "log":
        value = rev_mult * math.log1p(max(0.0, daily_revenue))
    else:
        value = rev_mult * daily_revenue
    # Apply an optional per-step soft cap uniformly across linear/cap modes.
    # ``0`` or a negative cap is treated as "disabled" so pre-audit configs
    # that omitted the key keep their behaviour.
    if "revenue_cap_per_step" in cfg:
        try:
            cap = float(cfg.get("revenue_cap_per_step"))
        except (TypeError, ValueError):
            cap = 0.0
        if cap > 0.0:
            value = min(value, cap)
    return value


def _solvency_term(
    state_before: Dict,
    state_after: Dict,
    cfg: Dict,
    action_result: Dict | None = None,
    daily_revenue: float | None = None,
) -> float:
    """Per-step solvency bonus.

    Post-audit H-1 / round-2 A2-10 — the bonus is now tightly gated so a
    passive ``wait`` policy CANNOT farm it just because daily revenue
    accrued while the agent did nothing:

        * ``bank_after >= solvency_threshold`` — unchanged.
        * ``non_revenue_delta = (bank_after - bank_before) - daily_revenue``
          must be ``>= 0`` (the agent cannot be burning cash).
        * ``base_reward > 0`` (an agent-initiated productive action must
          have taken place this step).

    The ``action_result`` and ``daily_revenue`` kwargs default to ``None``
    so legacy unit tests that call ``_solvency_term`` with the historical
    three-arg signature keep working (they effectively fall back to the
    old "threshold + growth" check). New callers thread both through.
    """
    threshold = float(cfg.get("solvency_threshold", 500.0))
    bonus = float(cfg.get("solvency_per_step", 0.05))
    if bonus == 0.0:
        return 0.0
    try:
        bank_after = float(state_after.get("bank_balance", 0.0))
        bank_before = float(state_before.get("bank_balance", 0.0))
    except (TypeError, ValueError):
        return 0.0
    if bank_after < threshold:
        return 0.0
    if bank_after <= bank_before:
        # No growth = no bonus; a stationary solvent policy earns 0 here.
        return 0.0
    # Tighter gate when the caller supplies action context (the
    # WorldEngine always does). Legacy test callers skip this branch
    # because they omit ``action_result`` / ``daily_revenue``.
    if action_result is not None and daily_revenue is not None:
        non_revenue_delta = (bank_after - bank_before) - float(daily_revenue)
        try:
            base_reward = float(action_result.get("base_reward", 0.0))
        except (TypeError, ValueError):
            base_reward = 0.0
        if non_revenue_delta < 0.0:
            # Agent is burning cash faster than it earns revenue.
            return 0.0
        # Audit MEDIUM #6 — the original gate (``base_reward > 0``)
        # only credited ``restock`` / ``refund`` because those are the
        # only actions with a non-zero ``rewards.*`` in shipped
        # configs. A genuinely productive ``set_price`` / ``ad_spend``
        # / ``negotiate`` action got zero solvency credit, even when
        # it was the *correct* move. The anti-farming guardrails are
        # (a) ``wait`` never qualifies, (b) the handler didn't emit an
        # ``error`` (so ``invalid_action`` is rejected), and (c)
        # ``non_revenue_delta >= 0`` (we're not burning cash).
        action_type = str(action_result.get("action_type", "") or "")
        action_error = action_result.get("action_error")
        is_productive_zero_base = (
            action_type in {"set_price", "ad_spend", "negotiate"}
            and action_error is None
        )
        if base_reward > 0.0:
            return bonus
        if is_productive_zero_base:
            return bonus
        # Everything else (wait, failed actions, negative base) falls
        # through to zero.
        return 0.0
    return bonus


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
    """Aggregate penalty for aged open tickets.

    Post-audit R-6 — the unbounded sum over aged tickets meant a single
    high-spawn-rate run could pile on tens of aged criticals and swamp
    every other shaping term. The new ``ticket_aging_penalty_cap`` (int)
    caps the *total count of aged tickets* that contribute to the
    penalty. Criticals count first (they have the larger coefficient) so
    the cap always saturates with the worst offenders. ``0`` or missing
    key disables the cap and preserves the pre-audit behaviour.

    Post-audit round-2 (A2-29) — when ``rewards.urgency_penalty_map`` is
    set, the per-urgency coefficient comes from that table (keyed by the
    lower-cased urgency label). The legacy ``urgent_ticket_per_step`` /
    ``critical_ticket_per_step`` scalars remain the fallback when the
    map is absent so existing configs are unchanged.

    Post-audit round-2 (A2-30) — urgency comparison is case-insensitive
    so a ticket with ``urgency="Critical"`` is still penalised.
    """
    # Resolve per-urgency coefficients. The explicit map wins; otherwise
    # fall back to the legacy urgent/critical scalars.
    urgent_penalty = float(cfg.get("urgent_ticket_per_step", -0.1))
    # Magic ``1.5`` — pinned as a named default; see env.constants.
    try:
        from . import constants as _const
        default_crit_mult = float(_const.DEFAULT_CRITICAL_MULTIPLIER)
    except Exception:  # pragma: no cover — defensive fallback
        default_crit_mult = 1.5
    critical_penalty = float(
        cfg.get("critical_ticket_per_step", urgent_penalty * default_crit_mult)
    )
    urgency_map_raw = cfg.get("urgency_penalty_map") or {}
    urgency_map: Dict[str, float] = {}
    if isinstance(urgency_map_raw, dict):
        for lbl, coef in urgency_map_raw.items():
            try:
                urgency_map[str(lbl).lower()] = float(coef)
            except (TypeError, ValueError):
                continue

    age_threshold = int(cfg.get("urgent_ticket_age_days", 3))
    current_day = int(state_after.get("current_day", 1))
    aging = [
        t
        for t in _as_list(state_after.get("active_tickets"))
        if t.get("status") == "open"
        and current_day - int(t.get("created_day", 1)) >= age_threshold
    ]

    # Tally tickets by lower-cased urgency label.
    tallies: Dict[str, int] = {}
    for t in aging:
        label = str(t.get("urgency", "")).lower()
        tallies[label] = tallies.get(label, 0) + 1
    urgent = tallies.get("urgent", 0)
    critical = tallies.get("critical", 0)

    try:
        cap = int(cfg.get("ticket_aging_penalty_cap", 0) or 0)
    except (TypeError, ValueError):
        cap = 0

    # Coefficient lookup per label. When ``urgency_penalty_map`` is set
    # it overrides both scalars; otherwise fall back to the legacy
    # urgent/critical coefficients and ignore other labels.
    def _coef(label: str) -> float:
        if urgency_map:
            return urgency_map.get(label, 0.0)
        if label == "urgent":
            return urgent_penalty
        if label == "critical":
            return critical_penalty
        return 0.0

    if cap > 0:
        # Sort labels by absolute coefficient (largest penalty first) so
        # the cap always saturates with the most expensive offenders.
        labels_ranked = sorted(
            tallies.keys(), key=lambda lbl: abs(_coef(lbl)), reverse=True
        )
        total = 0.0
        remaining = cap
        for lbl in labels_ranked:
            take = min(tallies[lbl], remaining)
            if take <= 0:
                continue
            total += _coef(lbl) * take
            remaining -= take
            if remaining <= 0:
                break
        return total

    if urgency_map:
        # Full (uncapped) per-label penalty using the map.
        return sum(_coef(lbl) * n for lbl, n in tallies.items())
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

    Post-audit round-2 (A2-31) — when ``rewards.ad_roi_scaled`` is truthy,
    the bonus is scaled by an ROI-ratio proxy instead of being a binary
    "any sale? full bonus" flag::

        roi_ratio = (sold_units * unit_price) / max(spend, 1.0)
        reward   += bonus * min(1.0, max(0.0, roi_ratio - 1.0))

    This means a penny-farm ad spend that produces a single sale earns
    close to zero, while a genuinely positive-ROI campaign gets the full
    bonus (once revenue reaches 2× spend). Gated on the flag so legacy
    configs keep the simple behaviour.
    """
    bonus = float(cfg.get("ad_roi_positive", 0.0))
    if not bonus:
        return 0.0
    applied = action_result.get("ad_spend_applied") or {}
    prior_ads = applied or (state_before.get("active_ad_spend", {}) or {})
    daily_sales = state_after.get("daily_sales", {}) or {}
    prices = state_after.get("prices", {}) or {}
    scaled = bool(cfg.get("ad_roi_scaled", False))
    reward = 0.0
    for sku, spend in prior_ads.items():
        try:
            spend_val = float(spend)
        except (TypeError, ValueError):
            continue
        if spend_val <= 0:
            continue
        sold = int(daily_sales.get(sku, 0))
        if sold <= 0:
            continue
        if not scaled:
            reward += bonus
            continue
        unit_price = float(prices.get(sku, 0.0))
        revenue = sold * unit_price
        roi_ratio = revenue / max(spend_val, 1.0)
        scale = min(1.0, max(0.0, roi_ratio - 1.0))
        reward += bonus * scale
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
    action_result: Dict,
) -> float:
    """Bank-balance delta shaping, with revenue and restock cost excluded.

    Post-audit M-2 / P-1 — without this fix, a successful restock that
    burned $800 of cash with delivery arriving ``restock_lead_days``
    later would ding the delta term by ``-8`` (with the default weight),
    which dwarfed the ``+0.1`` base restock success reward and trained
    the policy to avoid lead-time purchases. Restock spend is *capital
    allocation*, not operating burn — it will return as revenue once the
    goods land.

    Post-audit round-2 (A2-14) — only the **amortisable** portion of
    the restock cost (the part covered by a live quote) is excluded.
    Any overflow paid at the spot premium is *punitive* and stays in
    the delta so the policy learns to right-size its negotiated
    quantity. Callers that don't thread the new ``restock_cost_amortised``
    key fall back to the historical ``restock_cost`` value so legacy
    unit tests keep working.

    Post-audit round-2 (A2-32) — an optional
    ``rewards.refund_payout_delta_cap`` caps the refund contribution
    into the delta so a single catastrophic refund doesn't dominate.
    When the action result reports a ``refund_payout`` (the cash the
    refund handler drew from the bank), any amount above the cap is
    added back into the delta so the penalty saturates gracefully.
    """
    weight = float(cfg.get("bank_balance_delta_weight", 0.01))
    delta = float(state_after.get("bank_balance", 0.0)) - float(
        state_before.get("bank_balance", 0.0)
    )
    # Prefer the precise "amortised" component; fall back to the raw
    # restock cost if an older caller doesn't supply the split.
    try:
        restock_cost_amort = float(
            action_result.get(
                "restock_cost_amortised",
                action_result.get("restock_cost", 0.0),
            )
            or 0.0
        )
    except (TypeError, ValueError):
        restock_cost_amort = 0.0
    # Compute refund-payout saturation correction.
    refund_payout_correction = 0.0
    try:
        refund_payout = float(action_result.get("refund_payout", 0.0) or 0.0)
    except (TypeError, ValueError):
        refund_payout = 0.0
    if refund_payout > 0.0 and "refund_payout_delta_cap" in cfg:
        try:
            cap = float(cfg.get("refund_payout_delta_cap", 0.0) or 0.0)
        except (TypeError, ValueError):
            cap = 0.0
        if cap > 0.0 and refund_payout > cap:
            # The raw delta already carries ``-refund_payout``; adding
            # ``(refund_payout - cap)`` back reduces the absolute impact
            # to ``-cap``.
            refund_payout_correction = refund_payout - cap
    # ``delta = bank_after - bank_before = daily_revenue - restock_cost - ad_spend - refunds``
    # so subtracting ``daily_revenue`` and ADDING BACK the amortisable
    # restock cost gives us the "non-revenue, non-investment" cash flow
    # only. The refund-cap correction is added on top.
    adjusted = delta - daily_revenue + restock_cost_amort + refund_payout_correction
    return weight * adjusted


def _inventory_target_term(
    state_before: Dict,
    state_after: Dict,
    cfg: Dict,
    grader_ctx: Dict,
    action_result: Dict | None = None,
) -> float:
    """Dense bonus that aligns the reward with the ``inventory_task`` grader.

    v2.3 Phase 6.2 introduced this term. Post-audit H-1 / round-2 A2-11
    tightens the semantics so it cannot be farmed by passive growth
    driven by scheduled deliveries alone:

        * ``stock_after >= target`` — unchanged.
        * ``stock_after > stock_before`` — Δ-condition: the bonus only
          fires on steps where the target SKU's inventory actively went
          up.
        * If ``action_result`` is supplied, the bonus additionally
          requires either a restock issued on the target SKU this step
          OR a positive ``target_sku_net_landed_units`` figure the
          engine computed for the target SKU (covering the case where a
          quoted delivery arrived and was caused by an earlier agent
          action). Legacy callers that omit ``action_result`` fall back
          to the previous Δ-only gate so existing unit tests keep
          working.

    Defaults to 0.0 bonus so configs that don't enable it see no
    behaviour change.
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
    inv_after = state_after.get("inventory", {}) or {}
    inv_before = state_before.get("inventory", {}) or {}
    try:
        stock_after = float(inv_after.get(target_sku, 0))
        stock_before = float(inv_before.get(target_sku, 0))
    except (TypeError, ValueError):
        return 0.0
    if stock_after < target_units:
        return 0.0
    if stock_after <= stock_before:
        # Passive — inventory did not grow this step. Withhold the bonus
        # so a "wait-forever-at-full-stock" policy can't farm it.
        return 0.0
    if action_result is not None:
        # Attribute the growth to an agent action either issued this
        # step on the target SKU OR a matured quoted delivery the engine
        # flagged for this SKU (``target_sku_net_landed_units > 0``).
        # Accept the restock SKU at either of the two wire locations
        # the engine may thread it through: a nested ``restock`` dict
        # (original shape) or a flat ``restock_sku`` field (post-audit
        # round-2 WorldEngine.step plumbing).
        restock_info = action_result.get("restock") or {}
        restock_sku = ""
        if isinstance(restock_info, dict):
            restock_sku = str(restock_info.get("sku", ""))
        if not restock_sku:
            flat = action_result.get("restock_sku", "")
            if isinstance(flat, str):
                restock_sku = flat
        try:
            landed = float(action_result.get("target_sku_net_landed_units", 0.0) or 0.0)
        except (TypeError, ValueError):
            landed = 0.0
        if restock_sku != str(target_sku) and landed <= 0.0:
            return 0.0
    return bonus


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
    solvency = _solvency_term(
        state_before, state_after, rewards_config, action_result, daily_revenue
    )
    stockout = _stockout_term(state_before, state_after, rewards_config)
    aging = _ticket_aging_term(state_after, rewards_config)
    ad_roi = _ad_roi_term(state_before, state_after, rewards_config, action_result)
    bankruptcy = _bankruptcy_term(state_after, rewards_config)
    delta = _delta_term(
        state_before, state_after, rewards_config, daily_revenue, action_result
    )
    inv_bonus = _inventory_target_term(
        state_before, state_after, rewards_config, grader_context or {},
        action_result,
    )

    # Post-audit round-2 (A2-43) — no 4dp rounding on the internal reward
    # scalar. Rounding the aggregate (or each term) injects training jitter
    # and can desynchronise the breakdown sum from the returned total.
    # Callers that need fixed-width logs should round at the I/O edge.
    if not return_breakdown:
        return (
            base
            + revenue
            + solvency
            + stockout
            + aging
            + ad_roi
            + bankruptcy
            + delta
            + inv_bonus
        )

    breakdown_terms = {
        "base": base,
        "revenue": revenue,
        "solvency": solvency,
        "stockout": stockout,
        "ticket_aging": aging,
        "ad_roi": ad_roi,
        "bankruptcy": bankruptcy,
        "delta": delta,
        "inventory_target_bonus": inv_bonus,
    }
    total = sum(breakdown_terms.values())
    # Post-audit D.3 — defense-in-depth invariant: ``total`` must equal
    # the rounded sum of the per-term values. Holds trivially today
    # because that's exactly how ``total`` is built, but guards against a
    # future refactor where someone inlines a separate aggregate path and
    # forgets to keep the two in sync. We never *raise* here — RL
    # rollouts treat this as a soft warning so a reward anomaly doesn't
    # crash the training loop; CI catches regressions via the dedicated
    # invariant test.
    recomputed = sum(breakdown_terms.values())
    if abs(recomputed - total) > 1e-9:
        logger.warning(
            "reward_breakdown_sum_mismatch total=%s sum=%s terms=%s",
            total,
            recomputed,
            breakdown_terms,
        )
    breakdown = dict(breakdown_terms)
    breakdown["daily_revenue"] = round(daily_revenue, 2)
    # Post-audit C.5 (v2.3.x) — additive, non-mutating diagnostic. The
    # ``scale_hint`` bucket lets training harnesses eyeball whether the
    # reward is currently dominated by small shaping terms (``small``),
    # mid-range shaping + revenue (``medium``), or a terminal event such
    # as a bankruptcy penalty or a large revenue booking (``large``).
    # Reward math is NOT altered; this field is strictly observational
    # and callers that pop it for assertions will still see the correct
    # per-term totals.
    abs_total = abs(float(total))
    if abs_total < 1.0:
        scale_hint = "small"
    elif abs_total < 10.0:
        scale_hint = "medium"
    else:
        scale_hint = "large"
    breakdown["scale_hint"] = scale_hint
    return total, breakdown


def compute_step_reward_with_breakdown(
    action_result: Dict,
    state_before: Dict,
    state_after: Dict,
    rewards_config: Dict,
    *,
    grader_context: Dict | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Typed-return shim over :func:`compute_step_reward` (A2-64).

    Always returns ``(total, breakdown)``. Exists so callers don't have
    to reason about the overloaded ``compute_step_reward`` return type —
    when you want the breakdown, call this; when you just want the
    scalar, call ``compute_step_reward(..., return_breakdown=False)``.
    """
    result = compute_step_reward(
        action_result=action_result,
        state_before=state_before,
        state_after=state_after,
        rewards_config=rewards_config,
        return_breakdown=True,
        grader_context=grader_context,
    )
    if isinstance(result, tuple):
        return result  # type: ignore[return-value]
    # Defensive: should never happen, but keeps the signature stable
    # under future refactors.
    return float(result), {}  # pragma: no cover
