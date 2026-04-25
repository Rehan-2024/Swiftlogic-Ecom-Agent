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

    Post-audit M-3 — the quote branch is no longer consulted here
    because ``do_restock`` handles the (quote, spot overflow) blended
    pricing directly. Kept for API stability in case external callers
    invoke this helper with ``quote_price=None``.
    """
    if quote_price is not None:
        return float(quote_price), True
    return _spot_unit_cost(engine, sku), False


def _spot_unit_cost(engine, sku: str) -> float:
    """Return the *spot* (un-negotiated) unit cost for ``sku``.

    Post-audit M-3 — shared between ``_resolve_unit_price`` and the
    quantity-overflow path in ``do_restock`` so both callers pull the
    same clamped ``spot_premium`` and the same fallback-log wording.
    """
    if sku in engine.unit_costs:
        base_cost = float(engine.unit_costs[sku])
        spot_premium = float(getattr(engine, "_spot_premium", 0.0))
        spot_premium = max(0.0, min(1.0, spot_premium))
        return base_cost * (1.0 + spot_premium)
    logger.warning(
        "unit_cost_fallback_used sku=%s fallback=%s "
        "(config invariant violated: sku missing from unit_costs)",
        sku,
        constants.FALLBACK_UNIT_COST,
    )
    return float(constants.FALLBACK_UNIT_COST)


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
    # Post-audit round-2 (A2-13) — per-step restock quantity cap. Reject
    # the action outright when the cap is configured and exceeded;
    # callers get a stable ``invalid_action`` reward and a structured
    # ``info`` payload they can inspect.
    actions_cfg = engine.config.get("actions", {}) or {}
    qty_cap = actions_cfg.get("restock_max_qty_per_step")
    if qty_cap is not None:
        try:
            qty_cap_i = int(qty_cap)
        except (TypeError, ValueError):
            qty_cap_i = None
        if qty_cap_i is not None and quantity > qty_cap_i:
            return (
                float(rewards_cfg.get("invalid_action", -0.2)),
                {
                    "error": "invalid_restock",
                    "reason": "qty_over_cap",
                    "requested": quantity,
                    "cap": qty_cap_i,
                },
            )

    # Optional supplier capacity (per-SKU, per-step cap). 0 or missing means
    # unlimited (legacy behavior). We allow partial fulfillment and surface it
    # in ``info`` without changing the action schema.
    req_quantity = int(quantity)
    cap_map = getattr(engine, "_supplier_capacity_per_sku", {}) or {}
    cap_val = cap_map.get(sku)
    if cap_val is not None:
        try:
            cap_i = int(cap_val)
        except (TypeError, ValueError):
            cap_i = 0
        if cap_i > 0:
            quantity = min(quantity, cap_i)
    filled_qty = int(quantity)
    unfilled_qty = max(0, req_quantity - filled_qty)
    if filled_qty <= 0:
        return (
            float(rewards_cfg.get("invalid_action", -0.2)),
            {
                "error": "invalid_restock",
                "reason": "supplier_capacity_exhausted",
                "requested_quantity": req_quantity,
                "filled_qty": 0,
                "unfilled_qty": req_quantity,
            },
        )

    # Step 1: resolve pricing (consume expired quotes, apply spot_premium).
    #
    # Post-audit M-3 — a negotiated quote is now *quantity-bound*. The
    # quote covers at most ``supplier_quoted_qty[sku]`` units; any
    # overflow is charged at the spot rate (``list * (1 + spot_premium)``)
    # so a negotiator cannot lock a discount on a tiny qty and then
    # exploit it with a giant restock. ``supplier_quoted_qty`` is
    # populated by ``do_negotiate`` and defaults to ``quantity`` when a
    # legacy caller or test only set ``supplier_quotes`` without the
    # companion qty map (backward compatibility).
    quotes: Dict[str, float] = engine.state.setdefault("supplier_quotes", {})
    expiry: Dict[str, int] = engine.state.setdefault("supplier_quote_expiry", {})
    quoted_qtys: Dict[str, int] = engine.state.setdefault("supplier_quoted_qty", {})
    current_step = int(engine.state.get("step_count", 0))
    quote_price = _consume_expired_quote(quotes, expiry, sku, current_step)

    negotiated: object = False  # False | True | "partial"
    covered_qty = 0
    overflow_qty = 0
    quote_unit_price = None
    spot_unit_price = None
    if quote_price is not None:
        # Legacy fallback: when quoted_qty is absent, treat the quote as
        # covering the full requested quantity so v2.2 clients keep
        # working. New do_negotiate always writes the qty so modern runs
        # take the bind-to-qty path.
        quoted_qty = int(quoted_qtys.get(sku, quantity))
        quoted_qty = max(0, quoted_qty)
        covered_qty = min(filled_qty, quoted_qty)
        overflow_qty = max(0, filled_qty - covered_qty)
        quote_unit_price = float(quote_price)
        if overflow_qty > 0:
            spot_unit_price = _spot_unit_cost(engine, sku)
            cost = covered_qty * quote_unit_price + overflow_qty * spot_unit_price
            unit_cost = cost / max(1, filled_qty)
            negotiated = "partial"
        else:
            cost = covered_qty * quote_unit_price
            unit_cost = quote_unit_price
            negotiated = True
    else:
        unit_cost, negotiated = _resolve_unit_price(engine, sku, quote_price)
        cost = unit_cost * filled_qty
        spot_unit_price = unit_cost

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
    lead_days, delivery_day = _schedule_delivery(engine, sku, filled_qty)

    # Step 4: consume the negotiated quote so it can't be reused.
    if quote_price is not None:
        quotes.pop(sku, None)
        expiry.pop(sku, None)
        quoted_qtys.pop(sku, None)

    # Step 5: emit the info dict and the positive reward.
    #
    # Post-audit round-2 (A2-14) — split the cost into the amortisable
    # portion (the part covered by a live quote = capital allocation
    # that will return as revenue) and the punitive spot-overflow
    # portion that stays in the delta term so the policy learns to
    # right-size its negotiated quantity.
    quote_cost_component = 0.0
    spot_cost_component = 0.0
    if quote_price is not None:
        quote_cost_component = covered_qty * (quote_unit_price or 0.0)
        if overflow_qty > 0 and spot_unit_price is not None:
            spot_cost_component = overflow_qty * spot_unit_price
    else:
        spot_cost_component = float(cost)
    restock_cost_amortised = float(quote_cost_component)
    restock_cost_punitive = float(spot_cost_component)
    return (
        float(rewards_cfg.get("restock_success", 0.1)),
        {
            "restock": {
                "sku": sku,
                "quantity": filled_qty,
                "requested_quantity": req_quantity,
                "filled_qty": filled_qty,
                "unfilled_qty": unfilled_qty,
                "cost": cost,
                "unit_price_paid": unit_cost,
                "negotiated": negotiated,
                "covered_qty": covered_qty,
                "overflow_qty": overflow_qty,
                "quote_unit_price": (
                    round(quote_unit_price, 4)
                    if quote_unit_price is not None
                    else None
                ),
                "spot_unit_price": (
                    round(spot_unit_price, 4)
                    if spot_unit_price is not None
                    else None
                ),
                "lead_days": lead_days,
                "delivery_day": delivery_day,
            },
            "restock_cost": float(cost),
            "restock_cost_amortised": restock_cost_amortised,
            "restock_cost_punitive": restock_cost_punitive,
            "restock_unfilled_qty": int(unfilled_qty),
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

    # Audit MINOR #14 — honour ``supplier.capacity_per_sku`` when sizing
    # the quoted qty. A quote for 1000 units against a 10-unit capacity
    # is a misleading signal for the policy; the subsequent restock
    # will only consume the first 10 units at the quoted price anyway.
    # We clamp the requested qty to capacity BEFORE pricing so the
    # volume-discount / demand-signal math uses the realistic quantity
    # the supplier can actually deliver. Unknown / None capacity falls
    # through to the historical unlimited behaviour.
    capped_flag = False
    requested_quantity = int(quantity)
    cap_map = getattr(engine, "_supplier_capacity_per_sku", {}) or {}
    cap_val = cap_map.get(sku)
    if cap_val is not None:
        try:
            cap_int = int(cap_val)
        except (TypeError, ValueError):
            cap_int = None
        if cap_int is not None and cap_int >= 0 and quantity > cap_int:
            quantity = max(0, cap_int)
            capped_flag = True
            if quantity <= 0:
                return (
                    float(rewards_cfg.get("invalid_action", -0.2)),
                    {
                        "error": "invalid_negotiate",
                        "reason": "supplier_capacity_exhausted",
                        "requested_quantity": requested_quantity,
                        "capacity": int(cap_int),
                    },
                )

    demand_signal = engine._recent_sales_signal(sku)
    quote = float(engine.supplier_agent.quote_price(sku, quantity, demand_signal))

    quotes_store = engine.state.setdefault("supplier_quotes", {})
    quoted_qty_store = engine.state.setdefault("supplier_quoted_qty", {})
    expiry_store = engine.state.setdefault("supplier_quote_expiry", {})
    current_step = int(engine.state.get("step_count", 0))

    # Post-audit round-2 (A2-20) — keep the best live quote. When a
    # previous quote for this SKU exists and has not expired, compare
    # the new price and the requested quantity:
    #   * If the new unit price is strictly lower, replace the stored
    #     quote.
    #   * If the new quote covers a larger qty while matching or beating
    #     the previous unit price, replace it.
    #   * Otherwise, keep the existing quote. The ``expiry`` timer is
    #     always refreshed (this represents the supplier reconfirming
    #     the existing offer).
    existing_price = quotes_store.get(sku)
    existing_qty = quoted_qty_store.get(sku)
    existing_expiry = expiry_store.get(sku, -1)
    is_live = (
        existing_price is not None
        and int(existing_expiry) >= current_step
    )
    new_is_better = True
    if is_live:
        try:
            ex_p = float(existing_price)
            ex_q = int(existing_qty) if existing_qty is not None else 0
        except (TypeError, ValueError):
            ex_p = float("inf")
            ex_q = 0
        if quote > ex_p and quantity <= ex_q:
            # Strictly worse on price without broader qty coverage.
            new_is_better = False
    if new_is_better:
        quotes_store[sku] = quote
        quoted_qty_store[sku] = int(quantity)
    ttl = int(
        engine.config.get("supplier", {}).get(
            "quote_expiry_steps", constants.DEFAULT_QUOTE_TTL_STEPS
        )
    )
    expiry_store[sku] = current_step + max(0, ttl)

    # Default the per-step reward to the (neutral) 'wait' reward so existing
    # configs without an explicit 'negotiate' key keep working unchanged.
    reward_val = float(rewards_cfg.get("negotiate", rewards_cfg.get("wait", 0.0)))
    negotiate_info = {
        "sku": sku,
        "quantity": quantity,
        "unit_price": quote,
        "demand_signal": round(demand_signal, 3),
        "list_price": engine.supplier_agent.list_price(sku),
    }
    if capped_flag:
        negotiate_info["requested_quantity"] = requested_quantity
        negotiate_info["capacity_capped"] = True
    return (
        reward_val,
        {"negotiate": negotiate_info},
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
    bank_available = float(engine.state["bank_balance"])
    if bank_available < payout:
        # Audit MEDIUM #8 — optional engine-side partial refund. When
        # ``tickets.allow_partial_refund`` is truthy the bank pays
        # whatever it can (``bank_available``), the ticket remains
        # ``open`` so a follow-up refund finishes it, and the action
        # earns a *proportional* slice of ``refund_success``. No schema
        # change: the config knob is additive, the info payload
        # preserves ``refund`` / ``refund_payout`` plus a new
        # ``partial`` marker.
        tickets_cfg = engine.config.get("tickets", {}) or {}
        allow_partial = bool(tickets_cfg.get("allow_partial_refund", False))
        min_partial = float(tickets_cfg.get("partial_refund_min_fraction", 0.1) or 0.0)
        fraction = (bank_available / payout) if payout > 0 else 0.0
        if allow_partial and bank_available > 0 and fraction >= min_partial:
            base_reward = float(rewards_cfg.get("refund_success", 0.3))
            scaled_reward = base_reward * max(0.0, min(1.0, fraction))
            engine.state["bank_balance"] = 0.0
            # Track partial progress on the ticket so follow-up
            # refunds can settle the remainder without accidentally
            # double-counting ``refund_success``.
            ticket["partial_refund_paid"] = round(
                float(ticket.get("partial_refund_paid", 0.0)) + bank_available, 2
            )
            ticket["partial_refund_due"] = round(
                max(0.0, float(payout) - bank_available), 2
            )
            ticket["partial_refund_count"] = int(
                ticket.get("partial_refund_count", 0)
            ) + 1
            return (
                scaled_reward,
                {
                    "refund": {
                        "ticket_id": ticket_id,
                        "resolved": False,
                        "partial": True,
                        "payout": round(bank_available, 2),
                        "required": round(payout, 2),
                        "fraction": round(fraction, 4),
                    },
                    "refund_payout": float(bank_available),
                },
            )
        return (
            float(rewards_cfg.get("invalid_action", -0.2)),
            {
                "error": "insufficient_funds",
                "required": round(payout, 2),
                "available": round(bank_available, 2),
            },
        )
    ticket["status"] = "resolved"
    engine.state["bank_balance"] = bank_available - payout
    return (
        float(rewards_cfg.get("refund_success", 0.3)),
        {
            "refund": {
                "ticket_id": ticket_id,
                "resolved": True,
                "payout": round(payout, 2),
            },
            # Post-audit round-2 (A2-32) — surface the cash payout the
            # refund drew from the bank so ``_delta_term`` can honour
            # ``rewards.refund_payout_delta_cap`` when set.
            "refund_payout": float(payout),
        },
    )


# ---------------------------------------------------------------------------
# Ad spend
# ---------------------------------------------------------------------------

def do_ad_spend(engine, action: Dict) -> Tuple[float, Dict]:
    """Allocate a per-SKU ad budget for the current step.

    Post-audit H-2 / R-1 — enforces a configurable minimum spend
    (``actions.ad_spend_min_per_step``, default ``0.0``) so a trivial
    "spend one cent, make a sale, claim ``ad_roi_positive``" farming
    loop becomes impossible. Shipped configs set the floor well above
    zero; legacy configs without the key keep the pre-audit behaviour.
    """
    rewards_cfg = engine.config.get("rewards", {})
    sku = action.get("sku")
    try:
        budget = float(action.get("budget", 0.0))
    except (TypeError, ValueError):
        budget = 0.0
    actions_cfg = engine.config.get("actions", {}) or {}
    max_per_step = float(actions_cfg.get("ad_spend_max_per_step", float("inf")))
    try:
        min_per_step = float(
            actions_cfg.get(
                "ad_spend_min_per_step", constants.DEFAULT_AD_SPEND_MIN_PER_STEP
            )
        )
    except (TypeError, ValueError):
        min_per_step = float(constants.DEFAULT_AD_SPEND_MIN_PER_STEP)
    min_per_step = max(0.0, min_per_step)
    if (
        sku not in engine.state["inventory"]
        or budget <= 0
        or budget > max_per_step
        or budget < min_per_step
    ):
        return (
            float(rewards_cfg.get("invalid_action", -0.2)),
            {
                "error": "invalid_ad_spend",
                "budget": budget,
                "min": min_per_step,
                "max": max_per_step if max_per_step != float("inf") else None,
            },
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
