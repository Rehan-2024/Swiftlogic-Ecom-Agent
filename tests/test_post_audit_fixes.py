"""Post-audit regression tests.

Every test in this file corresponds to a specific issue flagged by the
senior-engineer audit report (``PROJECT_REPORT.md``). New fixes land with
a test here so future regressions are caught immediately.

Sections map to audit bug IDs:
    * H-1 — Inflated training baselines (solvency + inventory-target
      bonuses require Δ-growth).
    * H-2 / R-1 — Ad-ROI farming blocked by ``ad_spend_min_per_step``.
    * M-1 — ``step_async`` returns the full 4-tuple.
    * M-2 / P-1 — Bank-delta term amortises restock cost.
    * M-3 — Supplier quote is quantity-bound (overflow pays spot premium).
    * B-1 — Stepping after ``done=True`` is rejected.
    * D-1 — Demand model rejects ``price <= 0``.
    * D-3 — Ad-multiplier cap can be tuned via ``actions.max_ad_multiplier``.
    * R-4 — Linear revenue mode honours ``revenue_cap_per_step``.
    * R-6 — Ticket-aging penalty cap.
    * P-4 — Product numeric validation (restock_lead_days, ad_elasticity).
    * S-4 / L-3 — Supplier numeric validation.
    * L-1 — Grader ``target_units`` must be > 0.
    * L-2 — ``rewards.set_price`` whitelisted (no unknown-key warning).
    * L-14 — ``EcomObservation`` / ``Ticket`` ignore unknown fields.
    * Realism — competitor price walk + stockout churn.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from env.reward_engine import compute_step_reward
from env.world_engine import ConfigValidationError, WorldEngine


# ---------------------------------------------------------------------------
# H-1 — Solvency bonus requires bank growth (not just a high balance).
# ---------------------------------------------------------------------------

def _base_state(**overrides):
    state = {
        "current_day": 1,
        "step_count": 0,
        "bank_balance": 1000.0,
        "inventory": {"sku_a": 5},
        "pending_orders": {"sku_a": 0},
        "active_tickets": [],
        "daily_sales": {"sku_a": 0},
        "daily_sales_history": {"sku_a": []},
        "active_ad_spend": {"sku_a": 0.0},
        "prices": {"sku_a": 100.0},
        "competitor_prices": {"sku_a": 110.0},
        "supplier_quotes": {},
        "supplier_quote_expiry": {},
    }
    state.update(overrides)
    return state


def _quiet_cfg(**overrides):
    cfg = {
        "solvency_per_step": 0.0,
        "solvency_threshold": 0.0,
        "revenue_multiplier": 0.0,
        "stockout_penalty": 0.0,
        "urgent_ticket_per_step": 0.0,
        "critical_ticket_per_step": 0.0,
        "ad_roi_positive": 0.0,
        "bank_balance_delta_weight": 0.0,
        "bankruptcy_terminal": 0.0,
        "bankruptcy_threshold": -1.0,
        "inventory_target_bonus": 0.0,
    }
    cfg.update(overrides)
    return cfg


def test_solvency_requires_bank_growth():
    """H-1 — bank_after >= threshold alone is not enough."""
    before = _base_state(bank_balance=10_000.0)
    after = _base_state(bank_balance=10_000.0)  # no change
    cfg = _quiet_cfg(solvency_per_step=0.05, solvency_threshold=1000.0)
    assert compute_step_reward({"base_reward": 0.0}, before, after, cfg) == pytest.approx(0.0)


def test_solvency_fires_only_on_positive_growth():
    # Post-audit round-2 (A2-10): bonus requires an agent-initiated
    # productive action (``base_reward > 0``) in addition to bank growth
    # above the threshold. The legacy behaviour of rewarding pure growth
    # regardless of the action let a ``wait`` policy farm solvency.
    before = _base_state(bank_balance=10_000.0)
    after = _base_state(bank_balance=10_500.0)
    cfg = _quiet_cfg(solvency_per_step=0.05, solvency_threshold=1000.0)
    # With base_reward=0 (wait-like) the bonus is gated off.
    assert compute_step_reward({"base_reward": 0.0}, before, after, cfg) == pytest.approx(0.0)
    # With base_reward>0 (productive action), the bonus fires.
    reward = compute_step_reward(
        {"base_reward": 0.1, "daily_revenue": 0.0}, before, after, cfg
    )
    assert reward == pytest.approx(0.1 + 0.05)


def test_solvency_no_bonus_when_bank_shrinks():
    before = _base_state(bank_balance=10_000.0)
    after = _base_state(bank_balance=9_500.0)
    cfg = _quiet_cfg(solvency_per_step=0.05, solvency_threshold=1000.0)
    assert compute_step_reward({"base_reward": 0.0}, before, after, cfg) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# H-1 — inventory_target_bonus requires Δ-growth on the target SKU.
# ---------------------------------------------------------------------------

def test_inventory_target_bonus_requires_stock_growth():
    """Passive high-stock state should NOT farm the bonus."""
    before = _base_state(inventory={"sku_a": 20})
    after = _base_state(inventory={"sku_a": 20})  # no change
    cfg = _quiet_cfg(inventory_target_bonus=0.05)
    ctx = {"inventory_target_sku": "sku_a", "inventory_target_units": 10}
    reward = compute_step_reward(
        {"base_reward": 0.0}, before, after, cfg, grader_context=ctx
    )
    assert reward == pytest.approx(0.0)


def test_inventory_target_bonus_fires_on_growth_above_target():
    # Post-audit round-2 (A2-11): bonus requires the growth to be
    # attributable to the current step (either an explicit restock on the
    # target SKU or a flagged landed delivery). Without attribution the
    # bonus stays off even if the stock level crosses the threshold.
    before = _base_state(inventory={"sku_a": 5})
    after = _base_state(inventory={"sku_a": 20})
    cfg = _quiet_cfg(inventory_target_bonus=0.05)
    ctx = {"inventory_target_sku": "sku_a", "inventory_target_units": 10}
    # No attribution → no bonus.
    reward = compute_step_reward(
        {"base_reward": 0.0}, before, after, cfg, grader_context=ctx
    )
    assert reward == pytest.approx(0.0)
    # With an explicit landed delivery on the target SKU → bonus fires.
    reward = compute_step_reward(
        {"base_reward": 0.0, "target_sku_net_landed_units": 15},
        before,
        after,
        cfg,
        grader_context=ctx,
    )
    assert reward == pytest.approx(0.05)


def test_inventory_target_bonus_silent_when_stock_drops():
    before = _base_state(inventory={"sku_a": 20})
    after = _base_state(inventory={"sku_a": 15})
    cfg = _quiet_cfg(inventory_target_bonus=0.05)
    ctx = {"inventory_target_sku": "sku_a", "inventory_target_units": 10}
    reward = compute_step_reward(
        {"base_reward": 0.0}, before, after, cfg, grader_context=ctx
    )
    assert reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# H-2 / R-1 — ad_spend_min_per_step blocks penny-farming.
# ---------------------------------------------------------------------------

def test_ad_spend_below_min_is_rejected(world):
    world.config["actions"]["ad_spend_min_per_step"] = 50.0
    sku = world.config["products"][0]["sku"]
    bank_before = world.state["bank_balance"]
    _s, _r, _d, info = world.step(
        {"action_type": "ad_spend", "sku": sku, "budget": 1.0}
    )
    assert info.get("error") == "invalid_ad_spend"
    # Rejected handler must not debit the bank for the refused budget.
    # ``bank_balance`` may still increase from the day's demand draw, but
    # it must never drop by ``$1`` (the would-be ad spend).
    assert world.state["bank_balance"] >= bank_before - 0.5


def test_ad_spend_at_or_above_min_is_accepted(world):
    world.config["actions"]["ad_spend_min_per_step"] = 50.0
    sku = world.config["products"][0]["sku"]
    _s, _r, _d, info = world.step(
        {"action_type": "ad_spend", "sku": sku, "budget": 75.0}
    )
    assert "error" not in info
    assert info.get("ad_spend", {}).get("budget") == 75.0


# ---------------------------------------------------------------------------
# M-1 — step_async returns (obs, reward, done, info).
# ---------------------------------------------------------------------------

def test_step_async_returns_full_tuple():
    from ecom_env import EcomEnv, WaitAction

    env = EcomEnv("configs/siyaani_fashion.json")
    env.reset(seed=7)
    result = asyncio.run(env.step_async(WaitAction()))
    assert isinstance(result, tuple) and len(result) == 4
    obs, reward, done, info = result
    assert hasattr(obs, "bank_balance")
    assert hasattr(reward, "value")
    assert isinstance(done, bool)
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# M-2 / P-1 — delta term amortises restock cost.
# ---------------------------------------------------------------------------

def test_delta_term_excludes_restock_cost():
    before = _base_state(bank_balance=10_000.0)
    after = _base_state(bank_balance=9_200.0, daily_sales={"sku_a": 0})
    cfg = _quiet_cfg(bank_balance_delta_weight=0.01)
    # Without threading restock_cost through, the delta term would charge
    # (9200 - 10000) * 0.01 = -8. With amortisation it reads the $800
    # restock_cost from the action_result and cancels it out, leaving 0.
    r_no = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    r_with = compute_step_reward(
        {"base_reward": 0.0, "restock_cost": 800.0}, before, after, cfg
    )
    assert r_no == pytest.approx(-8.0, abs=1e-4)
    assert r_with == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# M-3 — supplier quote is quantity-bound; overflow pays spot premium.
# ---------------------------------------------------------------------------

def test_quote_binds_to_quantity_overflow_pays_spot(world):
    sku = world.config["products"][0]["sku"]
    world.step({"action_type": "negotiate", "sku": sku, "quantity": 5})
    quote_price = world.state["supplier_quotes"][sku]
    # Restock more than the quoted qty. Blended cost = 5 * quote + 5 * spot.
    _s, _r, _d, info = world.step(
        {"action_type": "restock", "sku": sku, "quantity": 10}
    )
    # If an error came back, negotiate or restock was rejected — surface it.
    assert "error" not in info, info
    restock = info["restock"]
    assert restock["covered_qty"] == 5
    assert restock["overflow_qty"] == 5
    assert restock["negotiated"] == "partial"
    # Spot price is strictly higher than the quote (quote includes volume
    # discount, spot applies premium).
    assert restock["spot_unit_price"] > quote_price
    # Blended unit price falls between quote and spot.
    assert quote_price < restock["unit_price_paid"] <= restock["spot_unit_price"]


def test_quote_fully_consumed_when_restock_within_quoted_qty(world):
    sku = world.config["products"][0]["sku"]
    world.step({"action_type": "negotiate", "sku": sku, "quantity": 10})
    _s, _r, _d, info = world.step(
        {"action_type": "restock", "sku": sku, "quantity": 5}
    )
    restock = info["restock"]
    assert restock["covered_qty"] == 5
    assert restock["overflow_qty"] == 0
    assert restock["negotiated"] is True
    # Quote is consumed regardless of whether the full qty was used.
    assert sku not in world.state["supplier_quotes"]
    assert sku not in world.state["supplier_quoted_qty"]


# ---------------------------------------------------------------------------
# B-1 — stepping after done=True is rejected.
# ---------------------------------------------------------------------------

def test_step_after_done_returns_no_op(world):
    max_steps = int(world.config["episode"]["max_steps"])
    for _ in range(max_steps):
        _s, _r, done, _i = world.step({"action_type": "wait"})
    assert done is True
    snapshot_before = dict(world.state)
    obs, reward, done_after, info = world.step({"action_type": "wait"})
    assert done_after is True
    assert reward == 0.0
    assert info.get("error") == "episode_terminated"
    # Engine state must not advance past termination.
    assert world.state["step_count"] == snapshot_before["step_count"]


# ---------------------------------------------------------------------------
# D-1 / D-3 — demand model guards.
# ---------------------------------------------------------------------------

def test_demand_model_rejects_non_positive_price():
    from env.demand_model import generate_demand

    import numpy as np

    rng = np.random.default_rng(0)
    for bad in (0.0, -5.0, -1e-9):
        got = generate_demand(
            sku="x",
            ad_spend=0.0,
            price=bad,
            competitor_price=100.0,
            base=100.0,
            rng=rng,
        )
        assert got == 0


def test_max_ad_multiplier_is_config_driven():
    from env.demand_model import generate_demand

    import numpy as np

    # With the default cap of 5.0 a huge budget saturates around base * 5.
    # Tightening the cap to 2.0 must keep the mean demand near ``base * 2``.
    rng = np.random.default_rng(42)
    samples_tight = [
        generate_demand(
            sku="x",
            ad_spend=100_000.0,
            price=100.0,
            competitor_price=100.0,
            base=10.0,
            ad_elasticity=5.0,
            rng=rng,
            max_ad_multiplier=2.0,
        )
        for _ in range(200)
    ]
    # Mean under a 2x cap must sit well below the 5x default cap's ceiling.
    mean_tight = sum(samples_tight) / len(samples_tight)
    assert mean_tight < 30.0  # generously below 10 * 5 = 50


# ---------------------------------------------------------------------------
# R-4 — linear revenue mode honours revenue_cap_per_step.
# ---------------------------------------------------------------------------

def test_linear_revenue_mode_respects_cap():
    before = _base_state()
    after = _base_state(daily_sales={"sku_a": 1000})  # 1000 * 100 = 100k revenue
    cfg = _quiet_cfg(revenue_multiplier=0.001, revenue_mode="linear", revenue_cap_per_step=3.0)
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    # Without the cap this would be 0.001 * 100000 = 100. Cap clamps to 3.
    assert r == pytest.approx(3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# R-6 — ticket aging penalty cap.
# ---------------------------------------------------------------------------

def test_ticket_aging_penalty_cap_saturates_criticals_first():
    aged_tickets = [
        {"status": "open", "urgency": "critical", "created_day": 0} for _ in range(10)
    ] + [
        {"status": "open", "urgency": "urgent", "created_day": 0} for _ in range(10)
    ]
    before = _base_state()
    after = _base_state(active_tickets=aged_tickets, current_day=10)
    cfg = _quiet_cfg(
        urgent_ticket_per_step=-0.1,
        critical_ticket_per_step=-0.15,
        urgent_ticket_age_days=3,
        ticket_aging_penalty_cap=6,
    )
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    # 6 criticals at -0.15 each, no urgents contributing (cap saturated with criticals).
    assert r == pytest.approx(-0.90, abs=1e-4)


def test_ticket_aging_penalty_uncapped_when_cap_is_zero():
    aged_tickets = [
        {"status": "open", "urgency": "critical", "created_day": 0} for _ in range(10)
    ]
    before = _base_state()
    after = _base_state(active_tickets=aged_tickets, current_day=10)
    cfg = _quiet_cfg(
        urgent_ticket_per_step=-0.1,
        critical_ticket_per_step=-0.15,
        urgent_ticket_age_days=3,
        ticket_aging_penalty_cap=0,
    )
    r = compute_step_reward({"base_reward": 0.0}, before, after, cfg)
    # 10 criticals at -0.15 = -1.5 (no cap).
    assert r == pytest.approx(-1.5, abs=1e-4)


# ---------------------------------------------------------------------------
# P-4 — product numeric validation.
# ---------------------------------------------------------------------------

def _write_config(tmp_path: Path, overrides: dict) -> Path:
    """Build a minimal valid config, layer in overrides, and write to disk."""
    import copy
    import json

    base = {
        "business_id": "tc",
        "financials": {"initial_bank_balance": 1000, "bankruptcy_threshold": 0},
        "episode": {"max_steps": 50, "steps_per_day": 1},
        "products": [
            {
                "sku": "a",
                "unit_cost": 10,
                "sell_price": 20,
                "initial_stock": 5,
                "demand": {"base_units_per_day": 2, "ad_elasticity": 1.0},
            }
        ],
        "tickets": {
            "min_initial": 1,
            "refund_amount_range": [1, 5],
            "spawn_rate_per_day": 0.0,
        },
        "actions": {"allowed": ["wait"]},
        "rewards": {"wait": 0.0},
        "graders": {
            "inventory_task": {"target_sku": "a", "target_units": 1},
            "profit_task": {"normalizer": 100},
        },
    }
    cfg = copy.deepcopy(base)
    _deep_update(cfg, overrides)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    return path


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def test_negative_restock_lead_days_rejected(tmp_path):
    path = _write_config(
        tmp_path, {"products": [{"sku": "a", "unit_cost": 10, "sell_price": 20,
                                 "initial_stock": 5, "restock_lead_days": -1,
                                 "demand": {"base_units_per_day": 2}}]}
    )
    with pytest.raises(ConfigValidationError, match="restock_lead_days"):
        WorldEngine(str(path))


def test_non_numeric_ad_elasticity_rejected(tmp_path):
    path = _write_config(
        tmp_path, {"products": [{"sku": "a", "unit_cost": 10, "sell_price": 20,
                                 "initial_stock": 5,
                                 "demand": {"base_units_per_day": 2, "ad_elasticity": "oops"}}]}
    )
    with pytest.raises(ConfigValidationError, match="ad_elasticity"):
        WorldEngine(str(path))


def test_duplicate_sku_rejected(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "products": [
                {"sku": "a", "unit_cost": 10, "sell_price": 20,
                 "initial_stock": 5, "demand": {"base_units_per_day": 2}},
                {"sku": "a", "unit_cost": 10, "sell_price": 20,
                 "initial_stock": 5, "demand": {"base_units_per_day": 2}},
            ]
        },
    )
    with pytest.raises(ConfigValidationError, match="Duplicate product sku"):
        WorldEngine(str(path))


# ---------------------------------------------------------------------------
# S-4 / L-3 — supplier numeric validation.
# ---------------------------------------------------------------------------

def test_supplier_volume_discount_out_of_range_rejected(tmp_path):
    path = _write_config(tmp_path, {"supplier": {"volume_discount": 0.9}})
    with pytest.raises(ConfigValidationError, match="volume_discount"):
        WorldEngine(str(path))


def test_supplier_spot_premium_negative_rejected(tmp_path):
    path = _write_config(tmp_path, {"supplier": {"spot_premium": -0.1}})
    with pytest.raises(ConfigValidationError, match="spot_premium"):
        WorldEngine(str(path))


def test_supplier_price_cap_below_one_rejected(tmp_path):
    path = _write_config(tmp_path, {"supplier": {"price_cap_multiplier": 0.5}})
    with pytest.raises(ConfigValidationError, match="price_cap_multiplier"):
        WorldEngine(str(path))


# ---------------------------------------------------------------------------
# L-1 — grader target_units must be > 0.
# ---------------------------------------------------------------------------

def test_inventory_grader_target_units_zero_rejected(tmp_path):
    path = _write_config(
        tmp_path, {"graders": {"inventory_task": {"target_sku": "a", "target_units": 0}}}
    )
    with pytest.raises(ConfigValidationError, match="target_units"):
        WorldEngine(str(path))


def test_inventory_grader_target_units_negative_rejected(tmp_path):
    path = _write_config(
        tmp_path, {"graders": {"inventory_task": {"target_sku": "a", "target_units": -5}}}
    )
    with pytest.raises(ConfigValidationError, match="target_units"):
        WorldEngine(str(path))


# ---------------------------------------------------------------------------
# L-2 — rewards.set_price whitelisted (no unknown-key warning).
# ---------------------------------------------------------------------------

def test_set_price_reward_does_not_trigger_unknown_key_warning(tmp_path, caplog):
    path = _write_config(tmp_path, {"rewards": {"wait": 0.0, "set_price": 0.05}})
    with caplog.at_level(logging.WARNING, logger="commerceops.world_engine"):
        WorldEngine(str(path))
    joined = "\n".join(r.getMessage() for r in caplog.records)
    # No unknown-key warning for ``set_price``.
    assert "config_unknown_key section=rewards key=set_price" not in joined


def test_ticket_aging_penalty_cap_reward_whitelisted(tmp_path, caplog):
    path = _write_config(
        tmp_path, {"rewards": {"wait": 0.0, "ticket_aging_penalty_cap": 5}}
    )
    with caplog.at_level(logging.WARNING, logger="commerceops.world_engine"):
        WorldEngine(str(path))
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "config_unknown_key section=rewards key=ticket_aging_penalty_cap" not in joined


# ---------------------------------------------------------------------------
# L-14 — EcomObservation / Ticket ignore unknown fields.
# ---------------------------------------------------------------------------

def test_ecom_observation_ignores_unknown_fields():
    from ecom_env import EcomObservation

    obs = EcomObservation(
        current_day=1,
        step_count=0,
        bank_balance=1000.0,
        inventory={"a": 5},
        pending_orders={"a": 0},
        active_tickets=[],
        daily_sales={"a": 0},
        active_ad_spend={"a": 0.0},
        mystery_field="should_be_dropped",
        another_new_key=42,
    )
    assert not hasattr(obs, "mystery_field")
    assert obs.model_dump().get("mystery_field") is None


def test_ticket_ignores_unknown_fields():
    from ecom_env import Ticket

    t = Ticket(
        ticket_id="TKT-001",
        issue_type="refund",
        status="open",
        urgency="urgent",
        created_day=1,
        internal_note="fork field",
        priority_color="red",
    )
    dumped = t.model_dump()
    assert "internal_note" not in dumped
    assert "priority_color" not in dumped


# ---------------------------------------------------------------------------
# Realism — competitor price walks within the configured band.
# ---------------------------------------------------------------------------

def test_competitor_price_random_walk_stays_within_band(world):
    sku = world.config["products"][0]["sku"]
    base = float(world.competitor_prices[sku])
    # Run 50 waits; competitor_price should drift but stay inside [0.5x, 2x] base.
    seen = [world.state["competitor_prices"][sku]]
    for _ in range(world.config["episode"]["max_steps"]):
        _s, _r, done, _i = world.step({"action_type": "wait"})
        seen.append(world.state["competitor_prices"][sku])
        if done:
            break
    lo = 0.5 * base
    hi = 2.0 * base
    assert all(lo - 1e-6 <= p <= hi + 1e-6 for p in seen)
    # With volatility > 0 the price should actually move at least once.
    if float(world.config["products"][0].get("competitor_price_volatility", 0.0)) > 0:
        assert len(set(round(p, 2) for p in seen)) > 1


def test_competitor_price_static_when_volatility_zero(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "products": [
                {
                    "sku": "a",
                    "unit_cost": 10,
                    "sell_price": 20,
                    "competitor_price": 25,
                    "competitor_price_volatility": 0.0,
                    "initial_stock": 5,
                    "demand": {"base_units_per_day": 2},
                }
            ]
        },
    )
    w = WorldEngine(str(path))
    w.reset(seed=1)
    for _ in range(10):
        w.step({"action_type": "wait"})
    assert w.state["competitor_prices"]["a"] == pytest.approx(25.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Realism — stockout churn scales up spawn rate.
# ---------------------------------------------------------------------------

def test_stockout_churn_multiplier_increases_spawn_rate(tmp_path):
    import numpy as np

    # Config: single SKU with 1 unit and guaranteed demand >= 1 per day.
    path = _write_config(
        tmp_path,
        {
            "products": [
                {
                    "sku": "a",
                    "unit_cost": 1,
                    "sell_price": 10,
                    "initial_stock": 1,
                    "demand": {"base_units_per_day": 10},
                }
            ],
            "tickets": {
                "min_initial": 0,
                "max_initial": 0,
                "initial_count": 0,
                "spawn_rate_per_day": 0.5,
                "stockout_churn_multiplier": 2.0,
                "refund_amount_range": [1, 1],
            },
        },
    )
    w = WorldEngine(str(path))
    w.reset(seed=7)
    # Step once — stock drops 1→0, triggering churn scaling on spawn rate.
    w.step({"action_type": "wait"})
    # Stockout flagged; with churn=2 the effective spawn this tick is
    # 0.5 * (1 + 2*1) = 1.5, so we expect at least one ticket this step.
    # Deterministic under seed 7 — guaranteed >= 1.
    assert len(w.state["active_tickets"]) >= 1
