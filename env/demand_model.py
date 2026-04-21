"""
demand_model.py — Stochastic demand simulator for CommerceOps v2.

Demand per SKU per step is drawn from a Poisson distribution whose rate is
modulated by ad spend (elasticity-weighted), relative price vs competitor,
and an optional day-of-week seasonality profile.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from . import constants


# Re-exported for backwards compatibility — any caller that imports
# ``MAX_AD_MULTIPLIER`` from ``env.demand_model`` keeps working, but the
# canonical definition lives in ``env.constants``.
MAX_AD_MULTIPLIER = constants.MAX_AD_MULTIPLIER


def generate_demand(
    sku: str,
    ad_spend: float,
    price: float,
    competitor_price: float,
    base: float,
    ad_elasticity: float = 1.0,
    seasonality_multiplier: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Return a Poisson sample of demand for one SKU given the market state.

    v2.3 Phase 5.1 — the optional ``rng`` arg lets each ``WorldEngine``
    instance own its own ``numpy.random.Generator`` so two envs running
    in the same process don't cross-contaminate through the numpy global
    seed. Legacy callers that omit ``rng`` transparently fall back to
    ``np.random.poisson`` (the module-level RNG).
    """
    if base <= 0:
        return 0
    # Ads increase effective lambda with sub-linear (log1p) scaling so the
    # marginal return on spend diminishes past ~$100. The result is hard-capped
    # at MAX_AD_MULTIPLIER so pathological budgets cannot explode demand.
    raw_ad_mult = 1.0 + math.log1p(max(0.0, ad_spend) / 100.0) * max(ad_elasticity, 0.0)
    ad_multiplier = min(raw_ad_mult, MAX_AD_MULTIPLIER)
    # Cheaper than competitor -> higher demand. Clamp to prevent runaway values.
    price_ratio = competitor_price / max(price, 1.0)
    price_ratio = max(0.25, min(4.0, price_ratio))
    effective_lambda = max(0.0, base * ad_multiplier * price_ratio * seasonality_multiplier)
    if rng is not None:
        return int(rng.poisson(effective_lambda))
    return int(np.random.poisson(effective_lambda))


def generate_all_demand(
    inventory: Dict[str, int],
    active_ad_spend: Dict[str, float],
    prices: Dict[str, float],
    competitor_prices: Dict[str, float],
    demand_configs: Dict[str, dict],
    current_day: int,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, int]:
    """Compute today's realized sales per SKU (capped by inventory on hand)."""
    sales: Dict[str, int] = {}
    day_of_week = current_day % 7
    for sku in inventory:
        dcfg = demand_configs.get(sku, {})
        base = float(dcfg.get("base_units_per_day", 0))
        ad_elasticity = float(dcfg.get("ad_elasticity", 1.0))
        raw_weights = dcfg.get("seasonality_weights", [1.0] * 7)
        # Robust fallback: coerce to numeric list, drop NaN/invalid entries, and
        # use flat weights when the config is missing/empty/malformed so the
        # demand model never throws on bad data.
        seasonality: list = []
        if isinstance(raw_weights, (list, tuple)):
            for w in raw_weights:
                try:
                    wf = float(w)
                except (TypeError, ValueError):
                    continue
                # NaN coerces cleanly through float() but poisons downstream
                # math (np.random.poisson raises on NaN lambda), so drop it.
                if math.isnan(wf) or math.isinf(wf):
                    continue
                seasonality.append(wf)
        if not seasonality:
            seasonality = [1.0] * 7
        season_mult = float(seasonality[day_of_week % len(seasonality)])
        raw = generate_demand(
            sku=sku,
            ad_spend=float(active_ad_spend.get(sku, 0.0)),
            price=float(prices.get(sku, 100.0)),
            competitor_price=float(competitor_prices.get(sku, 100.0)),
            base=base,
            ad_elasticity=ad_elasticity,
            seasonality_multiplier=season_mult,
            rng=rng,
        )
        on_hand = int(inventory.get(sku, 0))
        sales[sku] = max(0, min(raw, on_hand))
    return sales
