"""Phase 6.3 — supplier negotiation & quote lifecycle tests.

Covers:
  * quote creation on negotiate
  * quote overwrite on repeated negotiate
  * quote consumption on successful restock
  * quote persistence on insufficient funds
  * TTL expiry after configured number of steps
  * fallback to list cost when no quote is active
  * hard price cap at base_price * price_cap_multiplier
  * rolling 3-day demand signal stabilises pricing
"""

from __future__ import annotations

import pytest


def _first_sku(world):
    return next(iter(world.state["inventory"].keys()))


def test_quote_creation_and_reward_neutrality(world):
    sku = _first_sku(world)
    before_bank = world.state["bank_balance"]
    _s, reward, _d, info = world.step({
        "action_type": "negotiate", "sku": sku, "quantity": 10,
    })
    assert "negotiate" in info
    assert sku in world.state["supplier_quotes"]
    # negotiate must not debit cash (reward stays bounded; no cash moves).
    assert world.state["bank_balance"] == before_bank + _daily_rev(world)


def test_quote_overwrite_on_repeated_negotiate(world):
    sku = _first_sku(world)
    world.step({"action_type": "negotiate", "sku": sku, "quantity": 5})
    first = world.state["supplier_quotes"][sku]
    world.step({"action_type": "negotiate", "sku": sku, "quantity": 500})  # big volume premium
    second = world.state["supplier_quotes"][sku]
    assert second >= first  # higher quantity -> higher (or capped) price


def test_quote_consumed_on_successful_restock(world):
    sku = _first_sku(world)
    world.step({"action_type": "negotiate", "sku": sku, "quantity": 5})
    assert sku in world.state["supplier_quotes"]
    _s, _r, _d, info = world.step({"action_type": "restock", "sku": sku, "quantity": 3})
    assert info["restock"]["negotiated"] is True
    assert sku not in world.state["supplier_quotes"]
    assert sku not in world.state["supplier_quote_expiry"]


def test_quote_persists_on_insufficient_funds(world):
    sku = _first_sku(world)
    # Drain bank so restock can't afford anything.
    world.state["bank_balance"] = 1.0
    world.step({"action_type": "negotiate", "sku": sku, "quantity": 5})
    _s, _r, _d, info = world.step({"action_type": "restock", "sku": sku, "quantity": 100})
    assert info.get("error") == "insufficient_funds"
    assert sku in world.state["supplier_quotes"], "quote should persist for retry"


def test_quote_expires_after_ttl(world):
    sku = _first_sku(world)
    world.step({"action_type": "negotiate", "sku": sku, "quantity": 10})
    # TTL defaults to 3, so 4 waits guarantee expiry.
    for _ in range(4):
        world.step({"action_type": "wait"})
    _s, _r, _d, info = world.step({"action_type": "restock", "sku": sku, "quantity": 1})
    assert info["restock"]["negotiated"] is False
    assert sku not in world.state["supplier_quotes"]


def test_fallback_to_list_cost_without_quote(world):
    """v2.3 Phase 1.2 — un-negotiated restocks now pay ``list_cost * (1 + spot_premium)``.

    Previously the fallback paid the bare list cost, which made
    ``negotiate`` economically dominated (it only saved money via the
    volume discount, which a plain restock also got). Adding a spot
    premium gives negotiation a meaningful edge.
    """
    sku = _first_sku(world)
    _s, _r, _d, info = world.step({"action_type": "restock", "sku": sku, "quantity": 2})
    assert info["restock"]["negotiated"] is False
    list_cost = world.unit_costs[sku]
    spot_premium = float(
        (world.config.get("supplier", {}) or {}).get("spot_premium", 0.0)
    )
    expected = round(list_cost * (1.0 + spot_premium), 2)
    assert info["restock"]["unit_price_paid"] == pytest.approx(expected, abs=1e-6)


def test_supplier_price_cap_enforced(world):
    sku = _first_sku(world)
    base = world.unit_costs[sku]
    # Massive qty + saturated demand signal should still be bounded by cap.
    world.state["daily_sales_history"][sku] = [99, 99, 99]
    world.step({"action_type": "negotiate", "sku": sku, "quantity": 100000})
    quote = world.state["supplier_quotes"][sku]
    cap = base * world.supplier_agent.price_cap_multiplier
    assert quote <= cap + 1e-6, (quote, cap)


def test_rolling_signal_smoother_than_single_day(world):
    sku = _first_sku(world)
    # Inject artificial history vs single-day spike.
    world.state["daily_sales_history"][sku] = [0, 0, 0]
    world.state["daily_sales"][sku] = 50  # huge single-day spike
    smooth = world._recent_sales_signal(sku)
    # Wipe history -> only the spike remains.
    world.state["daily_sales_history"][sku] = []
    spike = world._recent_sales_signal(sku)
    assert smooth < spike, (smooth, spike)


def test_quote_expiry_surfaced_on_observation(world):
    """Phase D.4 - supplier_quote_expiry must be observable end-to-end.

    Exposing the TTL on the observation lets a policy plan its
    negotiate -> restock window without remembering the step it negotiated.
    """
    from ecom_env import EcomEnv

    env = EcomEnv("configs/siyaani_fashion.json")
    env.reset(seed=7)
    sku = next(iter(env.state().inventory.keys()))
    obs, _r, _d, _i = env.step({"action_type": "negotiate", "sku": sku, "quantity": 5})
    assert sku in obs.supplier_quote_expiry, obs.supplier_quote_expiry
    # Consuming the quote via a successful restock clears the expiry entry.
    env.state().bank_balance  # sanity touch to keep obs model alive
    obs2, _r, _d, info = env.step({"action_type": "restock", "sku": sku, "quantity": 1})
    if info.get("restock", {}).get("negotiated"):
        assert sku not in obs2.supplier_quote_expiry


def _daily_rev(world):
    sales = world.state["daily_sales"]
    prices = world.state["prices"]
    return sum(int(sales.get(k, 0)) * float(prices.get(k, 0.0)) for k in sales)


def test_unit_cost_fallback_emits_warning(world, caplog):
    """If an SKU is missing from ``unit_costs`` the fallback constant kicks
    in and must log a WARNING so silent drift can't hide in production.

    ``_validate_config`` normally prevents this state, so we simulate the
    invariant break by popping the cost out from under the restock call.
    """
    import logging

    sku = _first_sku(world)
    world.unit_costs.pop(sku, None)
    world.state["bank_balance"] = 10_000.0  # make sure we can afford fallback

    with caplog.at_level(logging.WARNING, logger="commerceops.actions"):
        world.step({"action_type": "restock", "sku": sku, "quantity": 1})

    msgs = [r.getMessage() for r in caplog.records if r.name == "commerceops.actions"]
    assert any("unit_cost_fallback_used" in m and sku in m for m in msgs), msgs


def test_daily_revenue_exposed_on_state_and_observation(world):
    """``daily_revenue`` must appear on both the engine state dict and on
    the ``EcomObservation``, matching sum(daily_sales * prices).
    """
    from ecom_env import EcomEnv

    world.step({"action_type": "wait"})
    assert "daily_revenue" in world.state
    assert world.state["daily_revenue"] == _daily_rev(world)

    env = EcomEnv("configs/siyaani_fashion.json")
    env.reset(seed=11)
    obs, _r, _d, _i = env.step({"action_type": "wait"})
    expected = sum(int(obs.daily_sales.get(k, 0)) * float(obs.prices.get(k, 0.0)) for k in obs.daily_sales)
    assert obs.daily_revenue == expected


def test_daily_revenue_zero_after_reset(world):
    """Right after ``reset`` no day has elapsed, so ``daily_revenue`` must
    be ``0.0`` — guards against stale-value leakage across episodes.
    """
    world.reset(seed=99)
    assert world.state["daily_revenue"] == 0.0


# ---------------------------------------------------------------------------
# v2.3 Phase 2.4 — SetPriceAction
# ---------------------------------------------------------------------------

def test_set_price_within_bounds_updates_prices(world):
    sku = _first_sku(world)
    competitor = float(world.config["products"][0].get("competitor_price", 2000))
    target = round(competitor * 1.1, 2)
    state, _r, _d, info = world.step(
        {"action_type": "set_price", "sku": sku, "price": target}
    )
    assert state["prices"][sku] == pytest.approx(target, abs=1e-6)
    set_price_info = info.get("set_price") or {}
    assert set_price_info.get("sku") == sku
    assert set_price_info.get("new_price") == pytest.approx(target, abs=1e-2)


def test_set_price_outside_bounds_is_rejected(world):
    sku = _first_sku(world)
    competitor = float(world.config["products"][0].get("competitor_price", 2000))
    price_before = world.state["prices"][sku]
    # 10x competitor exceeds the upper multiplier (4.0) -> the handler
    # must leave prices untouched and surface an ``invalid_set_price``
    # error rather than silently clamping.
    _state, reward, _d, info = world.step(
        {"action_type": "set_price", "sku": sku, "price": competitor * 10}
    )
    assert world.state["prices"][sku] == pytest.approx(price_before, abs=1e-6)
    assert info.get("error") == "invalid_set_price"
    assert info.get("reason") == "out_of_bounds"
    expected_penalty = float(world.config["rewards"].get("invalid_action", -0.2))
    # ``step`` applies shaping on top, so we only assert the sign.
    assert reward <= expected_penalty + 0.2


# ---------------------------------------------------------------------------
# v2.3 Phase 4.1 — restock_lead_days (pending deliveries)
# ---------------------------------------------------------------------------

def test_restock_lead_days_delays_delivery(world):
    """A product with ``restock_lead_days`` >= 1 must not land inventory on
    the same tick. The units show up on the step whose ``current_day`` is
    ``>= order_day + lead_days``. ``pending_orders`` is a per-SKU counter
    that tracks the in-flight quantity.
    """
    sku = "silk_kurta"  # lead_days=2 in siyaani config
    world.state["bank_balance"] = 100_000.0
    qty = 5
    world.step({"action_type": "restock", "sku": sku, "quantity": qty})

    # Incoming units are held in ``pending_deliveries`` / ``pending_orders``
    # rather than credited to ``inventory`` on the same tick.
    pending = world.state.get("pending_orders", {}).get(sku, 0)
    assert pending >= qty, pending

    for _ in range(5):
        world.step({"action_type": "wait"})

    # After enough days elapse, the pending counter for this SKU must drain.
    assert world.state["pending_orders"].get(sku, 0) == 0


# ---------------------------------------------------------------------------
# Post-audit m-5 — SupplierAgent fallback WARNINGs on unknown SKUs
# ---------------------------------------------------------------------------

def test_list_price_unknown_sku_logs_warning(caplog):
    """When :class:`SupplierAgent.list_price` hits an SKU that wasn't in
    the base-price table it silently returns the module-level fallback.
    That's invisible in logs today — post-audit we emit a WARNING so the
    missing wiring shows up during regression tests.
    """
    import logging

    from env.supplier_agent import SupplierAgent

    agent = SupplierAgent(base_prices={"known_sku": 100.0})
    with caplog.at_level(logging.WARNING, logger="commerceops.supplier"):
        value = agent.list_price("ghost_sku")
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "supplier_list_price_fallback" in m and "ghost_sku" in m for m in messages
    ), messages
    # Post-audit round-2 (A2-21): the fallback is now derived from the
    # mean of configured base_prices when any are present, not the
    # module constant. With only ``known_sku=100.0`` in the map we
    # expect 100.0 back instead of the ``FALLBACK_BASE_PRICE`` default.
    assert value == 100.0


def test_quote_price_unknown_sku_logs_warning(caplog):
    """Same guarantee for ``quote_price`` — the hot path that drives the
    supplier_quotes state — so negotiating on an unknown SKU is at least
    visible in logs even though the caller still gets a quote.
    """
    import logging

    from env.supplier_agent import SupplierAgent

    agent = SupplierAgent(base_prices={"known_sku": 100.0})
    with caplog.at_level(logging.WARNING, logger="commerceops.supplier"):
        agent.quote_price("ghost_sku", quantity_requested=1, demand_signal=0.0)
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "supplier_quote_price_fallback" in m and "ghost_sku" in m for m in messages
    ), messages
