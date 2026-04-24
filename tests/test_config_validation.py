"""v2.3 Phase 7 — regression tests for ``WorldEngine._validate_config``.

The validator grew several new rules during v2.3 remediation
(refund range required, cross-key consistency, set_price in the
allowlist, zero-ticket guard). These tests exercise the error paths
directly so a regression in the validator is caught without needing
a full ``/step`` / ``/grader`` round-trip.

The tests build a config in-memory, write it to a temp JSON file, and
feed it to :class:`env.world_engine.WorldEngine`. That way we cover the
real ``load_config`` pipeline rather than patching the internal dict.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from env.world_engine import ConfigValidationError, WorldEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG = {
    "business_id": "unit_test_shop",
    "display_name": "Unit Test Shop",
    "currency": "INR",
    "financials": {"initial_bank_balance": 1000, "bankruptcy_threshold": 0},
    "episode": {"max_steps": 10, "steps_per_day": 1},
    "products": [
        {
            "sku": "widget",
            "display_name": "Widget",
            "unit_cost": 10,
            "sell_price": 20,
            "competitor_price": 22,
            "initial_stock": 50,
            "restock_lead_days": 0,
            "demand": {
                "base_units_per_day": 2,
                "ad_elasticity": 1.0,
                "seasonality_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
        }
    ],
    "tickets": {
        "initial_count": None,
        "min_initial": 2,
        "max_initial": 3,
        "spawn_rate_per_day": 0.3,
        "issue_types": ["refund"],
        "urgency_levels": ["normal", "urgent", "critical"],
        "urgency_weights": [0.5, 0.3, 0.2],
        "urgency_age_threshold_days": 2,
        "refund_amount_range": [10, 50],
    },
    "actions": {
        "allowed": ["restock", "refund", "ad_spend", "negotiate", "wait", "set_price"],
        "ad_spend_max_per_step": 100,
        "price_min_mult_competitor": 0.5,
        "price_max_mult_competitor": 2.0,
    },
    "supplier": {
        "volume_free_units": 10,
        "volume_rate": 0.01,
        "demand_rate": 0.1,
        "price_cap_multiplier": 2.0,
        "volume_discount": 0.05,
        "spot_premium": 0.05,
        "quote_expiry_steps": 3,
    },
    "rewards": {
        "invalid_action": -0.2,
        "bankruptcy_terminal": -1.0,
        "solvency_per_step": 0.01,
        "solvency_threshold": 500,
        "revenue_multiplier": 0.001,
        "urgent_ticket_per_step": -0.1,
        "critical_ticket_per_step": -0.15,
        "urgent_ticket_age_days": 2,
        "stockout_penalty": -0.1,
        "bankruptcy_threshold": 0,
        "bank_balance_delta_weight": 0.01,
        "revenue_mode": "linear",
    },
    "graders": {
        "triage_task": {"difficulty": "easy"},
        "inventory_task": {
            "difficulty": "medium",
            "target_sku": "widget",
            "target_units": 10,
        },
        "profit_task": {"difficulty": "hard", "normalizer": 5000},
    },
}


def _write_cfg(tmp_path: Path, overrides: dict | None = None) -> str:
    """Return a JSON path with a deep-copied minimal config plus overrides."""
    cfg = copy.deepcopy(_MINIMAL_CONFIG)
    if overrides:
        for dotted, value in overrides.items():
            cur = cfg
            parts = dotted.split(".")
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            if value is _DELETE:
                cur.pop(parts[-1], None)
            else:
                cur[parts[-1]] = value
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    return str(path)


_DELETE = object()  # sentinel: delete a key instead of setting it


# ---------------------------------------------------------------------------
# Baseline — the minimal config must load cleanly so later negative tests
# only exercise the specific failure they target.
# ---------------------------------------------------------------------------

def test_minimal_config_loads_cleanly(tmp_path):
    path = _write_cfg(tmp_path)
    w = WorldEngine(path)
    assert w.config["business_id"] == "unit_test_shop"


# ---------------------------------------------------------------------------
# refund_amount_range (Phase 2.2)
# ---------------------------------------------------------------------------

def test_refund_range_required(tmp_path):
    path = _write_cfg(tmp_path, {"tickets.refund_amount_range": _DELETE})
    with pytest.raises(ConfigValidationError, match="refund_amount_range"):
        WorldEngine(path)


def test_refund_range_rejects_inverted(tmp_path):
    path = _write_cfg(tmp_path, {"tickets.refund_amount_range": [100, 10]})
    with pytest.raises(ConfigValidationError, match="lo <= hi"):
        WorldEngine(path)


def test_refund_range_rejects_negative(tmp_path):
    path = _write_cfg(tmp_path, {"tickets.refund_amount_range": [-5, 10]})
    with pytest.raises(ConfigValidationError, match=">= 0"):
        WorldEngine(path)


# ---------------------------------------------------------------------------
# set_price allowlist (Phase 2.4)
# ---------------------------------------------------------------------------

def test_set_price_is_accepted_in_allowlist(tmp_path):
    # Redundant with the baseline test but makes the Phase 2.4 coverage explicit.
    path = _write_cfg(tmp_path, {"actions.allowed": ["wait", "set_price"]})
    w = WorldEngine(path)
    assert "set_price" in w.config["actions"]["allowed"]


def test_unknown_action_is_rejected(tmp_path):
    path = _write_cfg(tmp_path, {"actions.allowed": ["wait", "launch_rocket"]})
    with pytest.raises(ConfigValidationError, match="unknown action types"):
        WorldEngine(path)


# ---------------------------------------------------------------------------
# Cross-key reward consistency (Phase 4.3)
# ---------------------------------------------------------------------------

def test_revenue_cap_requires_cap_value(tmp_path):
    path = _write_cfg(
        tmp_path,
        {"rewards.revenue_mode": "cap", "rewards.revenue_cap_per_step": _DELETE},
    )
    with pytest.raises(ConfigValidationError, match="revenue_cap_per_step"):
        WorldEngine(path)


def test_solvency_below_bankruptcy_is_rejected(tmp_path):
    path = _write_cfg(
        tmp_path,
        {"rewards.solvency_threshold": -10, "rewards.bankruptcy_threshold": 0},
    )
    with pytest.raises(ConfigValidationError, match="solvency_threshold"):
        WorldEngine(path)


# ---------------------------------------------------------------------------
# Zero-ticket guard (Phase 4.4)
# ---------------------------------------------------------------------------

def test_zero_ticket_episode_is_rejected(tmp_path):
    path = _write_cfg(
        tmp_path,
        {
            "tickets.min_initial": 0,
            "tickets.spawn_rate_per_day": 0.0,
            "tickets.initial_count": 0,
        },
    )
    with pytest.raises(ConfigValidationError, match="zero tickets"):
        WorldEngine(path)


# ---------------------------------------------------------------------------
# revenue_mode enum (pre-existing, but phase 4.3 tightened it)
# ---------------------------------------------------------------------------

def test_revenue_mode_rejects_garbage(tmp_path):
    path = _write_cfg(tmp_path, {"rewards.revenue_mode": "rainbow"})
    with pytest.raises(ConfigValidationError, match="revenue_mode"):
        WorldEngine(path)


# ---------------------------------------------------------------------------
# Deprecated-key warnings (Phase 4.2)
# ---------------------------------------------------------------------------

def test_deprecated_financials_key_logs_warning(tmp_path, caplog):
    """``financials.solvency_bonus_threshold`` was dropped in v2.3 — loading
    a config that still sets it must succeed (for rolling upgrades) but
    emit a WARNING so maintainers notice.
    """
    import logging

    path = _write_cfg(tmp_path, {"financials.solvency_bonus_threshold": 999})
    with caplog.at_level(logging.WARNING, logger="commerceops.world_engine"):
        WorldEngine(path)
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "config_deprecated" in m and "solvency_bonus_threshold" in m for m in messages
    ), messages


def test_deprecated_demand_model_key_logs_warning(tmp_path, caplog):
    """``products[*].demand.demand_model`` is a dead Poisson-selector knob
    as of v2.3 — demand_model.py only runs Poisson. Loading must still
    succeed but log a WARNING.
    """
    import logging

    path = _write_cfg(tmp_path, {"products": [
        {
            **_MINIMAL_CONFIG["products"][0],
            "demand": {
                **_MINIMAL_CONFIG["products"][0]["demand"],
                "demand_model": "poisson",
            },
        }
    ]})
    with caplog.at_level(logging.WARNING, logger="commerceops.world_engine"):
        WorldEngine(path)
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "config_deprecated" in m and "demand.demand_model" in m for m in messages
    ), messages


# ---------------------------------------------------------------------------
# Numeric sanity (pre-existing but worth pinning)
# ---------------------------------------------------------------------------

def test_rewards_must_be_numeric(tmp_path):
    path = _write_cfg(tmp_path, {"rewards.revenue_multiplier": "not-a-number"})
    with pytest.raises(ConfigValidationError, match="revenue_multiplier"):
        WorldEngine(path)


def test_inventory_target_sku_must_exist(tmp_path):
    path = _write_cfg(tmp_path, {"graders.inventory_task.target_sku": "ghost_sku"})
    with pytest.raises(ConfigValidationError, match="target_sku"):
        WorldEngine(path)


# ---------------------------------------------------------------------------
# Post-audit m-2 — unknown config keys emit WARNINGs (never raise)
# ---------------------------------------------------------------------------

def test_unknown_rewards_key_logs_warning(tmp_path, caplog):
    """Typos like ``rewards.stockot_penalty`` must load cleanly (back-compat
    with forks that extend configs) but log a WARNING so a CI grep can
    surface them.
    """
    import logging

    path = _write_cfg(tmp_path, {"rewards.stockot_penalty": -0.1})
    with caplog.at_level(logging.WARNING, logger="commerceops.world_engine"):
        WorldEngine(path)
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "config_unknown_key" in m and "section=rewards" in m and "stockot_penalty" in m
        for m in messages
    ), messages


def test_unknown_top_level_key_logs_warning(tmp_path, caplog):
    import logging

    path = _write_cfg(tmp_path, {"mystery_section": {"x": 1}})
    with caplog.at_level(logging.WARNING, logger="commerceops.world_engine"):
        WorldEngine(path)
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "config_unknown_key" in m and "mystery_section" in m for m in messages
    ), messages


def test_known_keys_do_not_warn(tmp_path, caplog):
    """The shipped minimal config must NOT produce any unknown-key WARNINGs."""
    import logging

    path = _write_cfg(tmp_path)
    with caplog.at_level(logging.WARNING, logger="commerceops.world_engine"):
        WorldEngine(path)
    unknown_warns = [
        r for r in caplog.records if "config_unknown_key" in r.getMessage()
    ]
    assert not unknown_warns, [r.getMessage() for r in unknown_warns]


# ---------------------------------------------------------------------------
# Post-audit m-3 — malformed JSON surfaces as ConfigValidationError
# ---------------------------------------------------------------------------

def test_malformed_json_raises_config_validation_error(tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text("{ this is not json ")
    with pytest.raises(ConfigValidationError, match="not valid UTF-8 JSON"):
        WorldEngine(str(bad))


def test_non_utf8_config_raises_config_validation_error(tmp_path):
    bad = tmp_path / "latin1.json"
    bad.write_bytes(b"{\"business_id\": \"caf\xe9_store\"}")
    with pytest.raises(ConfigValidationError, match="not valid UTF-8 JSON"):
        WorldEngine(str(bad))


# ---------------------------------------------------------------------------
# Post-audit C.2 — bankruptcy_threshold mirror consistency
# ---------------------------------------------------------------------------

def test_bankruptcy_threshold_mirrors_must_agree_when_both_set(tmp_path):
    """If a config carries both ``financials.bankruptcy_threshold`` and the
    deprecated ``rewards.bankruptcy_threshold`` mirror, the two MUST be
    equal. Otherwise the hard-stop check in ``step`` and the reward-engine
    penalty trigger could disagree.
    """
    path = _write_cfg(
        tmp_path,
        {
            "financials.bankruptcy_threshold": 100,
            "rewards.bankruptcy_threshold": 0,
        },
    )
    with pytest.raises(ConfigValidationError, match="bankruptcy_threshold"):
        WorldEngine(path)


def test_bankruptcy_threshold_single_source_financials_only(tmp_path):
    """Removing ``rewards.bankruptcy_threshold`` entirely must still load;
    the reward engine reads the financials mirror as a fallback.
    """
    path = _write_cfg(
        tmp_path,
        {"rewards.bankruptcy_threshold": _DELETE},
    )
    w = WorldEngine(path)
    assert "bankruptcy_threshold" not in w.config["rewards"]


# ---------------------------------------------------------------------------
# Post-audit B.1 (v2.3.x) — price bound consistency
# ---------------------------------------------------------------------------

def test_inverted_price_bounds_raise(tmp_path):
    """``price_min_mult_competitor > price_max_mult_competitor`` is a
    config-time bug (the feasible SetPrice interval becomes empty). The
    validator must raise so operators notice before a training run starts.
    """
    path = _write_cfg(
        tmp_path,
        {
            "actions.price_min_mult_competitor": 2.5,
            "actions.price_max_mult_competitor": 1.0,
        },
    )
    with pytest.raises(ConfigValidationError, match="price_min_mult_competitor"):
        WorldEngine(path)


def test_non_positive_price_bounds_raise(tmp_path):
    path = _write_cfg(
        tmp_path,
        {"actions.price_min_mult_competitor": 0},
    )
    with pytest.raises(ConfigValidationError, match="price_min_mult_competitor"):
        WorldEngine(path)


def test_non_numeric_price_bounds_raise(tmp_path):
    path = _write_cfg(
        tmp_path,
        {"actions.price_min_mult_competitor": "cheap"},
    )
    with pytest.raises(ConfigValidationError, match="numeric"):
        WorldEngine(path)


def test_equal_price_bounds_load(tmp_path):
    """``min == max`` is a degenerate but valid fixed-price config; it
    must load so operators can intentionally pin the policy to a single
    price multiplier for A/B testing.
    """
    path = _write_cfg(
        tmp_path,
        {
            "actions.price_min_mult_competitor": 1.0,
            "actions.price_max_mult_competitor": 1.0,
        },
    )
    w = WorldEngine(path)
    actions_cfg = w.config.get("actions", {})
    assert actions_cfg["price_min_mult_competitor"] == 1.0
    assert actions_cfg["price_max_mult_competitor"] == 1.0


def test_competitor_reactive_enabled_must_be_boolean(tmp_path):
    path = _write_cfg(tmp_path, {"competitor.reactive_enabled": "yes"})
    with pytest.raises(ConfigValidationError, match="reactive_enabled"):
        WorldEngine(path)


def test_supplier_capacity_map_requires_non_negative_ints(tmp_path):
    path = _write_cfg(tmp_path, {"supplier.capacity_per_sku": {"widget": -1}})
    with pytest.raises(ConfigValidationError, match="capacity_per_sku"):
        WorldEngine(path)


def test_market_shock_probability_in_range(tmp_path):
    path = _write_cfg(tmp_path, {"market.shock_probability": 1.5})
    with pytest.raises(ConfigValidationError, match="shock_probability"):
        WorldEngine(path)


def test_customer_satisfaction_bounds_valid(tmp_path):
    path = _write_cfg(
        tmp_path,
        {"customer.satisfaction_min": 0.9, "customer.satisfaction_max": 0.5},
    )
    with pytest.raises(ConfigValidationError, match="satisfaction bounds"):
        WorldEngine(path)


# ---------------------------------------------------------------------------
# Post-audit B.2 (v2.3.x) — financials numeric validation
# ---------------------------------------------------------------------------

def test_non_numeric_bankruptcy_threshold_raises(tmp_path):
    path = _write_cfg(
        tmp_path,
        {
            "financials.bankruptcy_threshold": "broke",
            # Drop the rewards mirror so the cross-key equality check
            # doesn't short-circuit before we reach _validate_financials.
            "rewards.bankruptcy_threshold": _DELETE,
        },
    )
    with pytest.raises(ConfigValidationError, match="bankruptcy_threshold"):
        WorldEngine(path)


def test_initial_below_bankruptcy_threshold_warns_but_loads(tmp_path, caplog):
    """Soft warning: the engine must not reject a config where the agent
    starts below the bankruptcy floor — it's a legitimate (if exotic)
    stress-test configuration. Loud WARNING + successful load is the
    contract.
    """
    import logging

    path = _write_cfg(
        tmp_path,
        {
            "financials.initial_bank_balance": 50,
            "financials.bankruptcy_threshold": 100,
            "rewards.bankruptcy_threshold": 100,
        },
    )
    with caplog.at_level(logging.WARNING, logger="commerceops.world_engine"):
        w = WorldEngine(path)
    messages = [r.getMessage() for r in caplog.records]
    assert any("config_soft_warn" in m for m in messages), messages
    assert w.config["financials"]["initial_bank_balance"] == 50
