"""
actions.py — Action handlers for CommerceOps v2.

Each ``do_*`` handler takes the live :class:`WorldEngine` and a raw action
dict, returns ``(reward, info)``, and directly mutates ``engine.state`` /
``engine.supplier_agent`` where appropriate. This module exists so the
engine's ``step`` / ``_process_action`` stay thin dispatchers and each
action can be unit-tested or re-tuned independently.

Behaviour is byte-for-byte identical to the legacy ``_do_*`` methods that
previously lived on :class:`WorldEngine`; only the physical location moved.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, Tuple

from . import constants


logger = logging.getLogger("commerceops.actions")


# ---------------------------------------------------------------------------
# Restock
# ---------------------------------------------------------------------------

def _consume_expired_quote(quotes: Dict[str, float], expiry: Dict[str, int],
                           sku: str, current_step: int):
    """Pop stale quotes and return the live quote_price (or ``None``).

    Kept as a small private helper so ``do_restock`` stays a linear
    five-step read: resolve → fund-check → debit → schedule → report.
    """
    if sku not in quotes:
        return None
    expiry_step = int(expiry.get(sku, -1))
    if expiry_step >= 0 and current_step > expiry_step:
        quotes.pop(sku, None)
        expiry.pop(sku, None)
        return None
    return float(quotes[sku])


def _resolve_unit_price(engine, sku: str, quote_price):
    """Return the per-unit price to charge for a restock.

    Applies the ``spot_premium`` surcharge on un-negotiated orders and
    emits the defensive WARNING if the SKU is somehow missing from
    ``unit_costs`` (a config-validator invariant violation). Returns a
    ``(unit_cost, negotiated_flag)`` tuple.
    """
    if quote_price is not None:
        # A locked quote already factors in the supplier's volume discount /
        # demand premium; no spot premium applies on negotiated orders.
        return float(quote_price), True
    if sku in engine.unit_costs:
        base_cost = float(engine.unit_costs[sku])
        spot_premium = float(getattr(engine, "_spot_premium", 0.0))
        # Clamp defensively — 100% premium (2x list price) is the hard
        # ceiling even if a config asks for more, mirroring the supplier's
        # own ``price_cap_multiplier`` behaviour.
        spot_premium = max(0.0, min(1.0, spot_premium))
        return base_cost * (1.0 + spot_premium), False
    logger.warning(
        "unit_cost_fallback_used sku=%s fallback=%s "
        "(config invariant violated: sku missing from unit_costs)",
        sku,
        constants.FALLBACK_UNIT_COST,
    )
    return float(constants.FALLBACK_UNIT_COST), False


def _schedule_delivery(engine, sku: str, quantity: int) -> Tuple[int, int]:
    """Land the restock into inventory now or queue it for later delivery.

    Returns ``(lead_days, delivery_day)``. ``pending_orders`` is maintained
    as a per-SKU counter tracking the in-flight quantity so the observation
    can surface it without re-deriving from ``pending_deliveries``.
    """
    lead_days = int(getattr(engine, "_get_lead_days", lambda _s: 0)(sku))
    if lead_days <= 0:
        engine.state["inventory"][sku] = (
            int(engine.state["inventory"].get(sku, 0)) + quantity
        )
        delivery_day = int(engine.state.get("current_day", 0))
        return lead_days, delivery_day
    # v2.3 Phase 4.1 — schedule the delivery for current_day + lead_days.
    # _simulate_day drains matured deliveries at the top of each tick.
    delivery_day = int(engine.state.get("current_day", 0)) + lead_days
    schedule = engine.state.setdefault("pending_deliveries", {}).setdefault(sku, [])
    schedule.append((delivery_day, int(quantity)))
    engine.state.setdefault("pending_orders", {})[sku] = (
        int(engine.state.get("pending_orders", {}).get(sku, 0)) + quantity
    )
    return lead_days, delivery_day


def do_restock(engine, action: Dict) -> Tuple[float, Dict]:
    """Five-step flow: validate → price → fund-check → debit+schedule → report."""
    rewards_cfg = engine.config.get("rewards", {})
    sku = action.get("sku")
    quantity = action.get("quantity", 0)
    try:
        quantity = int(quantity)
    except (TypeError, ValueError):
        quantity = 0
    if sku not in engine.state["inventory"] or quantity <= 0:
        return (
            float(rewards_cfg.get("invalid_action", -0.2)),
            {"error": "invalid_restock"},
        )

    # Step 1: resolve pricing (consume expired quotes, apply spot_premium).
    quotes: Dict[str, float] = engine.state.setdefault("supplier_quotes", {})
    expiry: Dict[str, int] = engine.state.setdefault("supplier_quote_expiry", {})
    current_step = int(engine.state.get("step_count", 0))
    quote_price = _consume_expired_quote(quotes, expiry, sku, current_step)
    unit_cost, negotiated = _resolve_unit_price(engine, sku, quote_price)
    cost = unit_cost * quantity

    # Step 2: fund check — refuse if the restock would overdraw cash.
    if engine.state["bank_balance"] < cost:
        return (
            float(rewards_cfg.get("invalid_action", -0.2)),
            {
                "error": "insufficient_funds",
                "required": cost,
                "negotiated": negotiated,
                "unit_price_paid": unit_cost,  # post-audit m-1 — stable key
            },
        )

    # Step 3: debit bank, land / queue inventory.
    engine.state["bank_balance"] -= cost
    lead_days, delivery_day = _schedule_delivery(engine, sku, quantity)

    # Step 4: consume the negotiated quote so it can't be reused.
    if negotiated:
        quotes.pop(sku, None)
        expiry.pop(sku, None)

    # Step 5: emit the info dict and the positive reward.
    return (
        float(rewards_cfg.get("restock_success", 0.1)),
        {
            "restock": {
                "sku": sku,
                "quantity": quantity,
                "cost": cost,
                "unit_price_paid": unit_cost,
                "negotiated": negotiated,
                "lead_days": lead_days,
                "delivery_day": delivery_day,
            }
        },
    )


# ---------------------------------------------------------------------------
# Negotiate
# ---------------------------------------------------------------------------

def do_negotiate(engine, action: Dict) -> Tuple[float, Dict]:
    """Ask the rule-based SupplierAgent for a unit-price quote on a SKU.

    The quote is stored on ``state.supplier_quotes[sku]`` and is consumed by
    the next successful ``RestockAction`` on the same SKU. No money changes
    hands, so graders stay within their (0.01, 0.99) envelope.
    """
    rewards_cfg = engine.config.get("rewards", {})
    sku = action.get("sku")
    quantity = action.get("quantity", 0)
    try:
        quantity = int(quantity)
    except (TypeError, ValueError):
        quantity = 0
    if sku not in engine.state["inventory"] or quantity <= 0:
        return (
            float(rewards_cfg.get("invalid_action", -0.2)),
            {"error": "invalid_negotiate"},
        )

    demand_signal = engine._recent_sales_signal(sku)
    quote = float(engine.supplier_agent.quote_price(sku, quantity, demand_signal))

    engine.state.setdefault("supplier_quotes", {})[sku] = quote
    ttl = int(
        engine.config.get("supplier", {}).get(
            "quote_expiry_steps", constants.DEFAULT_QUOTE_TTL_STEPS
        )
    )
    engine.state.setdefault("supplier_quote_expiry", {})[sku] = (
        int(engine.state.get("step_count", 0)) + max(0, ttl)
    )

    # Default the per-step reward to the (neutral) 'wait' reward so existing
    # configs without an explicit 'negotiate' key keep working unchanged.
    reward_val = float(rewards_cfg.get("negotiate", rewards_cfg.get("wait", 0.0)))
    return (
        reward_val,
        {
            "negotiate": {
                "sku": sku,
                "quantity": quantity,
                "unit_price": quote,
                "demand_signal": round(demand_signal, 3),
                "list_price": engine.supplier_agent.list_price(sku),
            }
        },
    )


# ---------------------------------------------------------------------------
# Refund
# ---------------------------------------------------------------------------

def do_refund(engine, action: Dict) -> Tuple[float, Dict]:
    rewards_cfg = engine.config.get("rewards", {})
    ticket_id = action.get("ticket_id")
    ticket = _find_ticket(engine.state["active_tickets"], ticket_id)
    if ticket is None:
        return (
            float(rewards_cfg.get("invalid_action", -0.2)),
            {"error": "ticket_not_found"},
        )
    if ticket.get("status") == "resolved":
        return (
            float(rewards_cfg.get("invalid_action", -0.2)),
            {"error": "ticket_already_resolved"},
        )
    # v2.3 Phase 5.4 — pure refactor extracting ticket resolution into a
    # dedicated helper so the refund payout path has a single failure mode
    # (``insufficient_funds``) and is trivially unit-testable.
    return _resolve_ticket(ticket, engine, rewards_cfg)


def _find_ticket(active_tickets, ticket_id):
    """Return the first ticket matching ``ticket_id`` or ``None``."""
    for ticket in active_tickets:
        if ticket.get("ticket_id") == ticket_id:
            return ticket
    return None


def _resolve_ticket(ticket, engine, rewards_cfg) -> Tuple[float, Dict]:
    """Resolve a ticket, draw a refund payout, and deduct it from the bank.

    v2.3 Phase 2.2 — previously an under-funded refund silently clamped the
    bank balance to 0 and still awarded ``refund_success``, which was a free
    reward pump. We now mirror ``do_restock`` semantics: reject the action
    with ``insufficient_funds`` and leave the ticket open.

    ``_validate_config`` already guarantees ``refund_amount_range`` exists,
    is a valid ``[lo, hi]`` pair of non-negatives, and ``lo <= hi``.
    """
    ticket_id = ticket.get("ticket_id")
    rr = engine.config.get("tickets", {}).get("refund_amount_range")
    # Validator guarantees shape; this is a safety net only.
    if not (isinstance(rr, (list, tuple)) and len(rr) == 2):
        ticket["status"] = "resolved"
        return (
            float(rewards_cfg.get("refund_success", 0.3)),
            {"refund": {"ticket_id": ticket_id, "resolved": True}},
        )

    lo, hi = float(rr[0]), float(rr[1])
    # v2.3 Phase 5.1 — prefer the engine's per-env RNG so two envs in the
    # same process don't synchronise on the global ``random`` stream.
    rng = getattr(engine, "_py_rng", None) or random
    payout = rng.uniform(lo, hi)
    if engine.state["bank_balance"] < payout:
        return (
            float(rewards_cfg.get("invalid_action", -0.2)),
            {
                "error": "insufficient_funds",
                "required": round(payout, 2),
                "available": round(float(engine.state["bank_balance"]), 2),
            },
        )
    ticket["status"] = "resolved"
    engine.state["bank_balance"] = float(engine.state["bank_balance"]) - payout
    return (
        float(rewards_cfg.get("refund_success", 0.3)),
        {"refund": {"ticket_id": ticket_id, "resolved": True, "payout": round(payout, 2)}},
    )


# ---------------------------------------------------------------------------
# Ad spend
# ---------------------------------------------------------------------------

def do_ad_spend(engine, action: Dict) -> Tuple[float, Dict]:
    rewards_cfg = engine.config.get("rewards", {})
    sku = action.get("sku")
    try:
        budget = float(action.get("budget", 0.0))
    except (TypeError, ValueError):
        budget = 0.0
    max_per_step = float(
        engine.config.get("actions", {}).get("ad_spend_max_per_step", float("inf"))
    )
    if sku not in engine.state["inventory"] or budget <= 0 or budget > max_per_step:
        return (
            float(rewards_cfg.get("invalid_action", -0.2)),
            {"error": "invalid_ad_spend"},
        )
    if engine.state["bank_balance"] < budget:
        return (
            float(rewards_cfg.get("invalid_action", -0.2)),
            {"error": "insufficient_funds"},
        )
    engine.state["bank_balance"] -= budget
    engine.state["active_ad_spend"][sku] = budget
    # v2.3 Phase 1.1 — surface the realised budget so the reward engine can
    # credit ``ad_roi_positive`` regardless of where it appears in the step
    # ordering. Previously ``_ad_roi_term`` read from ``state_before`` which
    # was snapshotted *before* this handler ran, so the term never fired.
    return (
        float(rewards_cfg.get("wait", 0.0)),
        {
            "ad_spend": {"sku": sku, "budget": budget},
            "ad_spend_applied": {sku: budget},
        },
    )


# ---------------------------------------------------------------------------
# Wait
# ---------------------------------------------------------------------------

def do_wait(engine, action: Dict) -> Tuple[float, Dict]:
    rewards_cfg = engine.config.get("rewards", {})
    return float(rewards_cfg.get("wait", 0.0)), {}


# ---------------------------------------------------------------------------
# Set price (v2.3 Phase 2.4)
# ---------------------------------------------------------------------------

# Engine-wide safety band. These mirror the hard clamps inside the demand
# model's competitor-ratio shaping. Configs can narrow the band via
# ``actions.price_min_mult_competitor`` / ``actions.price_max_mult_competitor``
# but can never loosen it past these hard floors/ceilings.
_PRICE_MIN_MULT_HARD_FLOOR = 0.1
_PRICE_MAX_MULT_HARD_CEILING = 10.0
_PRICE_MIN_MULT_DEFAULT = 0.25
_PRICE_MAX_MULT_DEFAULT = 4.0


def do_set_price(engine, action: Dict) -> Tuple[float, Dict]:
    """Mutate ``engine.state["prices"][sku]`` within competitor-ratio bounds.

    Rejects unknown SKUs, non-positive prices, and any price that falls
    outside ``[min_mult * competitor, max_mult * competitor]``. No money
    changes hands on this action. Policies pay the "cost" of a bad price
    through the resulting demand response in the next simulate_day tick.
    """
    rewards_cfg = engine.config.get("rewards", {})
    sku = action.get("sku")
    try:
        new_price = float(action.get("price"))
    except (TypeError, ValueError):
        return (
            float(rewards_cfg.get("invalid_action", -0.2)),
            {"error": "invalid_set_price", "reason": "non_numeric_price"},
        )
    if sku not in engine.state.get("prices", {}):
        return (
            float(rewards_cfg.get("invalid_action", -0.2)),
            {"error": "invalid_set_price", "reason": "unknown_sku"},
        )
    if new_price <= 0:
        return (
            float(rewards_cfg.get("invalid_action", -0.2)),
            {"error": "invalid_set_price", "reason": "non_positive_price"},
        )

    actions_cfg = engine.config.get("actions", {}) or {}
    min_mult = float(
        actions_cfg.get("price_min_mult_competitor", _PRICE_MIN_MULT_DEFAULT)
    )
    max_mult = float(
        actions_cfg.get("price_max_mult_competitor", _PRICE_MAX_MULT_DEFAULT)
    )
    min_mult = max(_PRICE_MIN_MULT_HARD_FLOOR, min_mult)
    max_mult = min(_PRICE_MAX_MULT_HARD_CEILING, max_mult)

    competitor = float(
        engine.state.get("competitor_prices", {}).get(
            sku, float(engine.competitor_prices.get(sku, new_price))
        )
    )
    if competitor <= 0:
        competitor = new_price  # defensive: can't build a meaningful band
    lo = competitor * min_mult
    hi = competitor * max_mult
    if new_price < lo or new_price > hi:
        return (
            float(rewards_cfg.get("invalid_action", -0.2)),
            {
                "error": "invalid_set_price",
                "reason": "out_of_bounds",
                "sku": sku,
                "price": round(new_price, 2),
                "allowed_range": [round(lo, 2), round(hi, 2)],
                "competitor_price": round(competitor, 2),
            },
        )

    old_price = float(engine.state["prices"].get(sku, 0.0))
    engine.state["prices"][sku] = round(new_price, 2)
    return (
        float(rewards_cfg.get("set_price", rewards_cfg.get("wait", 0.0))),
        {
            "set_price": {
                "sku": sku,
                "old_price": round(old_price, 2),
                "new_price": round(new_price, 2),
                "competitor_price": round(competitor, 2),
            }
        },
    )


# ---------------------------------------------------------------------------
# Dispatch table — used by ``WorldEngine._process_action``.
# ---------------------------------------------------------------------------

ACTION_HANDLERS = {
    "restock": do_restock,
    "refund": do_refund,
    "ad_spend": do_ad_spend,
    "negotiate": do_negotiate,
    "wait": do_wait,
    "set_price": do_set_price,
}


__all__ = [
    "ACTION_HANDLERS",
    "do_restock",
    "do_refund",
    "do_ad_spend",
    "do_negotiate",
    "do_wait",
    "do_set_price",
]
