"""Phase D.3 - unit tests for the demand model.

Exercises edge cases of ``generate_demand`` / ``generate_all_demand`` that are
expensive to cover through the full WorldEngine integration tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from env.demand_model import (
    MAX_AD_MULTIPLIER,
    generate_all_demand,
    generate_demand,
)


def test_generate_demand_zero_base_is_zero():
    # ``base <= 0`` short-circuits and never touches the RNG.
    assert generate_demand(
        sku="x", ad_spend=1000.0, price=10.0, competitor_price=10.0, base=0.0
    ) == 0
    assert generate_demand(
        sku="x", ad_spend=0.0, price=10.0, competitor_price=10.0, base=-1.0
    ) == 0


def test_price_ratio_clamped_low_end():
    # A crazy-cheap price vs competitor pushes the ratio above 4.0 before
    # clamp; with ad spend disabled we should see bounded lambda.
    np.random.seed(0)
    samples = [
        generate_demand(
            sku="x",
            ad_spend=0.0,
            price=1.0,
            competitor_price=10_000.0,  # ratio raw = 10000, clamped to 4.0
            base=10.0,
        )
        for _ in range(200)
    ]
    # Upper bound via 99.99% Poisson tail for lambda=40: well under 200.
    assert max(samples) < 200


def test_price_ratio_clamped_high_end():
    # Insanely expensive price vs competitor clamps to 0.25, so lambda is
    # floor-bounded at 0.25 * base, not zero.
    np.random.seed(0)
    samples = [
        generate_demand(
            sku="x",
            ad_spend=0.0,
            price=10_000.0,
            competitor_price=1.0,
            base=10.0,
        )
        for _ in range(500)
    ]
    # lambda = 10 * 1.0 * 0.25 = 2.5 -> non-trivial sales occur sometimes.
    assert sum(samples) > 0


def test_ad_multiplier_bounded_by_max():
    # Reconstruct the internal multiplier and assert it can't exceed the cap.
    # We compute the effective_lambda via a controlled draw.
    # For base=1, seasonality=1, price_ratio=1 (price==competitor), the
    # remaining factor IS the ad_multiplier.
    np.random.seed(0)
    draws = []
    for ad in (0.0, 100.0, 10_000.0, 10**9):
        # Average out the Poisson noise with many samples.
        vals = [
            generate_demand(
                sku="x",
                ad_spend=ad,
                price=10.0,
                competitor_price=10.0,
                base=1.0,
                ad_elasticity=100.0,  # huge elasticity forces cap to bite
                seasonality_multiplier=1.0,
            )
            for _ in range(2000)
        ]
        draws.append(sum(vals) / len(vals))
    # No sample mean should exceed MAX_AD_MULTIPLIER (within Poisson noise).
    # Allow a generous noise buffer of ~2x sqrt(lambda/N) for the worst lambda.
    for mean in draws:
        assert mean <= MAX_AD_MULTIPLIER + 0.5, (mean, MAX_AD_MULTIPLIER)


def test_generate_all_demand_respects_inventory_cap():
    np.random.seed(0)
    sales = generate_all_demand(
        inventory={"sku_a": 3},
        active_ad_spend={"sku_a": 0.0},
        prices={"sku_a": 10.0},
        competitor_prices={"sku_a": 50.0},  # high demand lambda
        demand_configs={"sku_a": {"base_units_per_day": 100, "ad_elasticity": 0.0}},
        current_day=1,
    )
    assert sales["sku_a"] <= 3


@pytest.mark.parametrize("weights", [[], None, ["bad", None, float("nan")]])
def test_seasonality_fallback_for_malformed_weights(weights):
    # Malformed / missing weights must NOT raise; demand falls back to [1]*7.
    cfg = {"base_units_per_day": 1.0}
    if weights is not None:
        cfg["seasonality_weights"] = weights
    np.random.seed(1)
    sales = generate_all_demand(
        inventory={"sku_a": 100},
        active_ad_spend={"sku_a": 0.0},
        prices={"sku_a": 10.0},
        competitor_prices={"sku_a": 10.0},
        demand_configs={"sku_a": cfg},
        current_day=3,
    )
    assert sales["sku_a"] >= 0
