"""Shared test fixtures and helpers (post-audit round-2 A2-66).

Centralises the minimal CommerceOps v2 config and the
``_write_cfg`` / ``_deep_update`` helpers that several test modules
used to redeclare locally. Importing from a shared module:

* Avoids drift between copies (e.g. one file updates a key and the
  others silently go stale).
* Makes new regression tests easier to write — they import one dict
  and an ``_write_cfg`` helper instead of hand-rolling a config.

The dict is deep-copied before every mutation so tests that modify
sections never leak state into each other.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


_DELETE = object()
"""Sentinel used by ``_write_cfg`` overrides to remove a key."""


MINIMAL_CONFIG: Dict[str, Any] = {
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
                "seasonality_weights": [1.0] * 7,
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


def deep_update(dst: Dict[str, Any], src: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursive in-place dict merge. Returns ``dst`` for chaining."""
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def write_cfg(
    tmp_path: Path,
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    base: Optional[Mapping[str, Any]] = None,
) -> str:
    """Write a minimal-plus-overrides config to ``tmp_path/cfg.json``.

    ``overrides`` uses dotted keys (e.g. ``"tickets.spawn_rate_per_day"``)
    so tests can tweak deep fields in one line. A value of ``_DELETE``
    removes the key instead of setting it.
    """
    cfg = copy.deepcopy(base or MINIMAL_CONFIG)
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


__all__ = ["MINIMAL_CONFIG", "deep_update", "write_cfg", "_DELETE"]
