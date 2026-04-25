"""
demand_model.py — Stochastic demand simulator for CommerceOps v2.

Demand per SKU per step is drawn from a Poisson distribution whose rate is
modulated by ad spend (elasticity-weighted), relative price vs competitor,
and an optional day-of-week seasonality profile.

v2.3.x post-audit — the ad-multiplier cap is now a per-call parameter
so ``WorldEngine`` can honour a config-level ``actions.max_ad_multiplier``
override without mutating the module-level constant. The legacy default
(``MAX_AD_MULTIPLIER=5.0``) is preserved for callers that don't pass the
new kwarg.
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
    max_ad_multiplier: Optional[float] = None,
    price_ratio_bounds: Optional[tuple] = None,
    record_factors: Optional[Dict[str, float]] = None,
) -> int:
    """Return a Poisson sample of demand for one SKU given the market state.

    v2.3 Phase 5.1 — the optional ``rng`` arg lets each ``WorldEngine``
    instance own its own ``numpy.random.Generator`` so two envs running
    in the same process don't cross-contaminate through the numpy global
    seed. Legacy callers that omit ``rng`` transparently fall back to
    ``np.random.poisson`` (the module-level RNG).

    Post-audit D-1 — a non-positive ``price`` now short-circuits to zero
    demand. Upstream handlers (``do_set_price``) reject ``price <= 0``,
    but the defensive guard here makes it impossible for a hand-crafted
    state dict to poison the Poisson lambda through the ``max(price, 1.0)``
    fallback. Post-audit D-3 — ``max_ad_multiplier`` is now a parameter
    so ``actions.max_ad_multiplier`` in the business config can tune the
    ceiling without patching the module constant.
    """
    if base <= 0:
        if record_factors is not None:
            record_factors.update({
                "base": float(base),
                "ad_multiplier": 1.0,
                "price_ratio": 0.0,
                "season": float(seasonality_multiplier),
                "effective_lambda": 0.0,
                "short_circuit": "non_positive_base",
            })
        return 0
    if price <= 0:
        # Defence-in-depth — the action layer rejects non-positive prices
        # but the reward engine / graders should never see an ambiguous
        # demand draw if an external caller passes a bad price dict.
        if record_factors is not None:
            record_factors.update({
                "base": float(base),
                "ad_multiplier": 1.0,
                "price_ratio": 0.0,
                "season": float(seasonality_multiplier),
                "effective_lambda": 0.0,
                "short_circuit": "non_positive_price",
            })
        return 0
    # Ads increase effective lambda with sub-linear (log1p) scaling so the
    # marginal return on spend diminishes past ~$100. The result is hard-capped
    # at ``max_ad_multiplier`` (default: module constant) so pathological
    # budgets cannot explode demand.
    cap = float(max_ad_multiplier) if max_ad_multiplier is not None else MAX_AD_MULTIPLIER
    cap = max(1.0, cap)  # a cap < 1.0 would be nonsense (ads hurt sales)
    raw_ad_mult = 1.0 + math.log1p(max(0.0, ad_spend) / 100.0) * max(ad_elasticity, 0.0)
    ad_multiplier = min(raw_ad_mult, cap)
    # Cheaper than competitor -> higher demand. Clamp to prevent runaway values.
    # Post-audit round-2 (A2-24) — ``price_ratio_bounds`` harmonises the
    # clamp with the configured set_price band so a policy at the band
    # edge doesn't clip against a different window than the one the
    # action validator let it into.
    lo, hi = (0.25, 4.0)
    if price_ratio_bounds is not None:
        try:
            rb_lo = float(price_ratio_bounds[0])
            rb_hi = float(price_ratio_bounds[1])
            if math.isfinite(rb_lo) and math.isfinite(rb_hi) and rb_lo > 0 and rb_hi >= rb_lo:
                lo, hi = rb_lo, rb_hi
        except (TypeError, ValueError, IndexError):
            pass
    price_ratio = competitor_price / max(price, 1.0)
    price_ratio = max(lo, min(hi, price_ratio))
    effective_lambda = max(0.0, base * ad_multiplier * price_ratio * seasonality_multiplier)
    # Post-audit round-2 (A2-25) — numerical safety rail. np.random.poisson
    # becomes numerically unstable well before 1e18; capping at 1e6 is well
    # above any realistic physical demand and keeps the RNG well-behaved.
    effective_lambda = min(effective_lambda, 1e6)
    if record_factors is not None:
        record_factors.update({
            "base": float(base),
            "ad_multiplier": float(ad_multiplier),
            "price_ratio": float(price_ratio),
            "season": float(seasonality_multiplier),
            "effective_lambda": float(effective_lambda),
        })
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
    max_ad_multiplier: Optional[float] = None,
    price_ratio_bounds: Optional[tuple] = None,
    external_multiplier_by_sku: Optional[Dict[str, float]] = None,
    record_factors: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, int]:
    """Compute today's realized sales per SKU (capped by inventory on hand).

    When ``record_factors`` is a dict, it is populated in-place with a
    per-SKU breakdown of the demand factors that went into the Poisson
    lambda: ``{sku: {"base", "ad_multiplier", "price_ratio",
    "season_combined", "season", "external", "effective_lambda",
    "ad_spend", "price", "competitor_price"}}``. Physics is unchanged —
    this is strictly an out-parameter for explainability.
    """
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
        ext_mult = 1.0
        if external_multiplier_by_sku is not None:
            try:
                ext_mult = float(external_multiplier_by_sku.get(sku, 1.0))
            except (TypeError, ValueError):
                ext_mult = 1.0
        if not math.isfinite(ext_mult) or ext_mult <= 0:
            ext_mult = 1.0
        ad_spend_val = float(active_ad_spend.get(sku, 0.0))
        price_val = float(prices.get(sku, 100.0))
        competitor_val = float(competitor_prices.get(sku, 100.0))
        factor_sink: Optional[Dict[str, float]] = None
        if record_factors is not None:
            factor_sink = {}
        raw = generate_demand(
            sku=sku,
            ad_spend=ad_spend_val,
            price=price_val,
            competitor_price=competitor_val,
            base=base,
            ad_elasticity=ad_elasticity,
            seasonality_multiplier=season_mult * ext_mult,
            rng=rng,
            max_ad_multiplier=max_ad_multiplier,
            price_ratio_bounds=price_ratio_bounds,
            record_factors=factor_sink,
        )
        on_hand = int(inventory.get(sku, 0))
        sold = max(0, min(raw, on_hand))
        sales[sku] = sold
        if record_factors is not None and factor_sink is not None:
            # Flesh out the sink with upstream context so downstream
            # explainers don't have to re-thread prices / budgets.
            factor_sink.setdefault("season", float(season_mult))
            factor_sink["external"] = float(ext_mult)
            factor_sink["season_combined"] = float(season_mult * ext_mult)
            factor_sink["ad_spend"] = ad_spend_val
            factor_sink["price"] = price_val
            factor_sink["competitor_price"] = competitor_val
            factor_sink["raw_demand"] = int(raw)
            factor_sink["on_hand"] = int(on_hand)
            factor_sink["sold"] = int(sold)
            record_factors[sku] = factor_sink
    return sales
