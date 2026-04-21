"""Phase D.2 - unit tests for the reward engine.

These exercise ``compute_step_reward`` with hand-crafted state dicts so we
assert each term fires (or does not fire) independently. The world engine
integration is covered separately by ``test_simulation_invariants`` and
``test_supplier_flow``.
"""

from __future__ import annotations

import math

import pytest

from env.reward_engine import compute_step_reward


def _base_state(**overrides):
    """Return a minimal state dict that the reward engine understands."""
    state = {
        "bank_balance": 1000.0,
        "current_day": 1,
        "inventory": {"sku_a": 5},
        "daily_sales": {"sku_a": 0},
        "prices": {"sku_a": 100.0},
        "active_tickets": [],
        "active_ad_spend": {"sku_a": 0.0},
    }
    state.update(overrides)
    return state


def _quiet_cfg(**overrides):
    """Reward config with every shaping term silenced by default.

    Individual tests override just the term they want to verify, so the
    assertion is isolated from any defaults the engine would otherwise
    apply (solvency bonus, stockout, ticket aging, ad ROI, etc.).
    """
    cfg = {
        "revenue_multiplier": 0.0,
        "solvency_per_step": 0.0,
        "stockout_penalty": 0.0,
        "urgent_ticket_per_step": 0.0,
        "critical_ticket_per_step": 0.0,
        "ad_roi_positive": 0.0,
        "bankruptcy_terminal": 0.0,
        "bank_balance_delta_weight": 0.0,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Revenue term — three modes
# ---------------------------------------------------------------------------

def test_revenue_linear_mode_default():
    before = _base_state()
    after = _base_state(daily_sales={"sku_a": 10})  # 10 * 100 = 1000 revenue
    cfg = _quiet_cfg(revenue_multiplier=0.001)
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    assert r == pytest.approx(1.0, abs=1e-6)


def test_revenue_log_mode_squashes_big_revenue():
    before = _base_state()
    after = _base_state(daily_sales={"sku_a": 1000})  # revenue = 100_000
    cfg = _quiet_cfg(revenue_multiplier=0.001, revenue_mode="log")
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    # compute_step_reward rounds to 4dp, so relax tolerance accordingly.
    assert r == pytest.approx(0.001 * math.log1p(100_000), abs=1e-3)


def test_revenue_cap_mode_hard_bounds_reward():
    before = _base_state()
    after = _base_state(daily_sales={"sku_a": 1000})  # revenue = 100_000
    cfg = _quiet_cfg(
        revenue_multiplier=0.001,
        revenue_mode="cap",
        revenue_cap_per_step=5.0,
    )
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    assert r == pytest.approx(5.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Stockout term — only on transition 1 -> 0
# ---------------------------------------------------------------------------

def test_stockout_penalty_fires_on_transition():
    before = _base_state(inventory={"sku_a": 3})
    after = _base_state(inventory={"sku_a": 0}, daily_sales={"sku_a": 3})
    cfg = _quiet_cfg(stockout_penalty=-0.5)
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    assert r == pytest.approx(-0.5, abs=1e-6)


def test_stockout_penalty_does_not_fire_when_already_zero():
    before = _base_state(inventory={"sku_a": 0})
    after = _base_state(inventory={"sku_a": 0})
    cfg = _quiet_cfg(stockout_penalty=-0.5)
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    assert r == pytest.approx(0.0, abs=1e-6)


def test_stockout_grace_skips_penalty_when_delivery_pending():
    """v2.3 — when ``stockout_transition_grace`` is on AND there's a
    pending restock for the SKU, the transition penalty is suppressed.
    The policy has already issued the right corrective action.
    """
    before = _base_state(inventory={"sku_a": 2})
    after = _base_state(
        inventory={"sku_a": 0},
        daily_sales={"sku_a": 2},
        pending_orders={"sku_a": 5},
    )
    cfg = _quiet_cfg(stockout_penalty=-0.5, stockout_transition_grace=True)
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    assert r == pytest.approx(0.0, abs=1e-6)


def test_stockout_grace_off_still_penalises_pending_restock():
    """Backward compat: without the grace flag, even an in-flight
    delivery does not exempt the transition from the penalty.
    """
    before = _base_state(inventory={"sku_a": 2})
    after = _base_state(
        inventory={"sku_a": 0},
        daily_sales={"sku_a": 2},
        pending_orders={"sku_a": 5},
    )
    cfg = _quiet_cfg(stockout_penalty=-0.5)  # grace not set => defaults False
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    assert r == pytest.approx(-0.5, abs=1e-6)


def test_stockout_grace_still_fires_when_no_pending_order():
    """Grace is *conditional* — a stockout with no pending restock still
    triggers the penalty, so the signal isn't fully disabled.
    """
    before = _base_state(inventory={"sku_a": 2})
    after = _base_state(
        inventory={"sku_a": 0},
        daily_sales={"sku_a": 2},
        pending_orders={"sku_a": 0},
    )
    cfg = _quiet_cfg(stockout_penalty=-0.5, stockout_transition_grace=True)
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    assert r == pytest.approx(-0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Ticket aging penalty — urgent vs critical
# ---------------------------------------------------------------------------

def test_urgent_and_critical_penalties_applied_separately():
    before = _base_state()
    tickets = [
        {"ticket_id": "T1", "urgency": "urgent", "status": "open", "created_day": 1},
        {"ticket_id": "T2", "urgency": "critical", "status": "open", "created_day": 1},
        {"ticket_id": "T3", "urgency": "normal", "status": "open", "created_day": 1},
    ]
    after = _base_state(current_day=5, active_tickets=tickets)
    cfg = _quiet_cfg(
        urgent_ticket_per_step=-0.1,
        critical_ticket_per_step=-0.2,
        urgent_ticket_age_days=3,
    )
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    assert r == pytest.approx(-0.3, abs=1e-6)


def test_critical_default_is_1_5x_urgent():
    before = _base_state()
    tickets = [
        {"ticket_id": "T1", "urgency": "critical", "status": "open", "created_day": 1},
    ]
    after = _base_state(current_day=5, active_tickets=tickets)
    # Deliberately omit critical_ticket_per_step so the 1.5x default kicks in.
    cfg = _quiet_cfg(urgent_ticket_per_step=-0.1, urgent_ticket_age_days=3)
    del cfg["critical_ticket_per_step"]
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    assert r == pytest.approx(-0.15, abs=1e-6)


# ---------------------------------------------------------------------------
# Ad ROI — only pays when prior_ads > 0 AND sales > 0
# ---------------------------------------------------------------------------

def test_ad_roi_fires_only_with_spend_and_sales():
    before = _base_state(active_ad_spend={"sku_a": 100.0, "sku_b": 100.0})
    after = _base_state(
        daily_sales={"sku_a": 5, "sku_b": 0},
        prices={"sku_a": 10.0, "sku_b": 10.0},
    )
    cfg = _quiet_cfg(ad_roi_positive=0.5)
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    assert r == pytest.approx(0.5, abs=1e-6)


def test_ad_roi_silent_when_coefficient_zero():
    before = _base_state(active_ad_spend={"sku_a": 100.0})
    after = _base_state(daily_sales={"sku_a": 5}, prices={"sku_a": 10.0})
    cfg = _quiet_cfg(ad_roi_positive=0.0)
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    assert r == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Bank balance delta term — must subtract daily_revenue (no double-count)
# ---------------------------------------------------------------------------

def test_bank_delta_subtracts_daily_revenue():
    before = _base_state(bank_balance=1000.0)
    after = _base_state(
        bank_balance=1500.0,
        daily_sales={"sku_a": 6},
        prices={"sku_a": 100.0},
    )
    cfg = _quiet_cfg(bank_balance_delta_weight=0.01)
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    # delta = 500, daily_revenue = 600, non_revenue_delta = -100, term = -1.0
    assert r == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Bankruptcy terminal
# ---------------------------------------------------------------------------

def test_bankruptcy_terminal_fires_at_threshold():
    before = _base_state(bank_balance=100.0)
    after = _base_state(bank_balance=0.0)
    cfg = _quiet_cfg(bankruptcy_threshold=0.0, bankruptcy_terminal=-1.0)
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    assert r == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Solvency bonus
# ---------------------------------------------------------------------------

def test_solvency_bonus_applied_above_threshold():
    before = _base_state()
    after = _base_state(bank_balance=10_000.0)
    cfg = _quiet_cfg(solvency_threshold=1000.0, solvency_per_step=0.05)
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    assert r == pytest.approx(0.05, abs=1e-6)


def test_solvency_bonus_not_applied_below_threshold():
    before = _base_state()
    after = _base_state(bank_balance=500.0)
    cfg = _quiet_cfg(solvency_threshold=1000.0, solvency_per_step=0.05)
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    assert r == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Base reward is honoured
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# v2.3 Phase 6.2 — inventory_target_bonus term (wired via grader_context)
# ---------------------------------------------------------------------------

def test_inventory_target_bonus_fires_when_stock_at_or_above_target():
    before = _base_state()
    after = _base_state(inventory={"sku_a": 12})
    cfg = _quiet_cfg(inventory_target_bonus=0.05)
    ctx = {"inventory_target_sku": "sku_a", "inventory_target_units": 10}
    _, breakdown = compute_step_reward(
        {"base_reward": 0.0}, before, after, cfg,
        return_breakdown=True, grader_context=ctx,
    )
    assert breakdown["inventory_target_bonus"] == pytest.approx(0.05, abs=1e-6)


def test_inventory_target_bonus_silent_when_below_target():
    before = _base_state()
    after = _base_state(inventory={"sku_a": 3})
    cfg = _quiet_cfg(inventory_target_bonus=0.05)
    ctx = {"inventory_target_sku": "sku_a", "inventory_target_units": 10}
    _, breakdown = compute_step_reward(
        {"base_reward": 0.0}, before, after, cfg,
        return_breakdown=True, grader_context=ctx,
    )
    assert breakdown["inventory_target_bonus"] == pytest.approx(0.0, abs=1e-6)


def test_inventory_target_bonus_zero_coefficient_is_noop():
    before = _base_state()
    after = _base_state(inventory={"sku_a": 12})
    cfg = _quiet_cfg(inventory_target_bonus=0.0)
    ctx = {"inventory_target_sku": "sku_a", "inventory_target_units": 10}
    r = compute_step_reward(
        {"base_reward": 0.0}, before, after, cfg, grader_context=ctx,
    )
    assert r == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# v2.3 Phase 5.6 — breakdown must sum exactly to total at 4dp
# ---------------------------------------------------------------------------

def test_breakdown_sum_equals_total_exactly():
    before = _base_state(bank_balance=1000.0)
    after = _base_state(
        bank_balance=2100.0,
        daily_sales={"sku_a": 7},
        prices={"sku_a": 100.0},
    )
    cfg = _quiet_cfg(
        revenue_multiplier=0.001,
        solvency_per_step=0.02,
        solvency_threshold=500,
        bank_balance_delta_weight=0.01,
    )
    total, bd = compute_step_reward(
        {"base_reward": 0.1}, before, after, cfg, return_breakdown=True,
    )
    # daily_revenue is metadata, not a reward term.
    terms = {k: v for k, v in bd.items() if k != "daily_revenue"}
    assert round(sum(terms.values()), 4) == total


def test_base_reward_pass_through():
    before = _base_state()
    after = _base_state()
    cfg = _quiet_cfg()
    r = compute_step_reward({"base_reward": 0.42}, before, after, cfg)
    assert r == pytest.approx(0.42, abs=1e-6)
