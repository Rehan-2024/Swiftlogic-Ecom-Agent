"""Shared fallback constants for the CommerceOps v2 engine.

Kept as a flat module of plain literals so that ``env/*.py`` and the
``ecom_env`` adapter can all import the same canonical values without
creating cross-module cycles. Historically these numbers were scattered as
inline magic values (e.g. ``70.0`` in ``world_engine`` and ``700.0`` in
``supplier_agent``); pinning them here makes bulk-retuning trivial.

v2.3.x Phase D — post-audit remediation added a handful of new keys used
by the reward engine (aging cap), the action layer (min ad budget), and
the day simulator (competitor-price volatility, stockout-driven churn).
Every new knob defaults to a value that preserves pre-audit behaviour so
shipped configs keep loading unchanged; the shipped configs opt into the
tightened semantics explicitly.
"""

from __future__ import annotations


# Used by ``env.actions.do_restock`` when a SKU is present on the inventory
# map but missing from ``unit_costs``. Should never trigger under valid
# configs; existence-of-sku is enforced at config load and a WARNING is
# logged via ``commerceops.actions`` if this fallback is ever consulted.
FALLBACK_UNIT_COST: float = 70.0

# Default base price used by ``SupplierAgent`` when it is asked to quote a
# SKU that was never registered via ``update_base_prices``.
FALLBACK_BASE_PRICE: float = 700.0

# Default profit-task normalizer when a config omits ``graders.profit_task
# .normalizer``. Kept at the legacy v1 value so graders stay in-bounds.
DEFAULT_PROFIT_NORMALIZER: float = 400.0

# Default TTL (in ``step_count`` units) for a supplier quote. May be
# overridden via ``supplier.quote_expiry_steps`` in the business config.
DEFAULT_QUOTE_TTL_STEPS: int = 3

# Hard ceiling on the ad-driven demand multiplier in ``generate_demand``.
# Exposed here as well as in ``demand_model`` so tests can assert both
# paths reference the same value. Configs may opt into a different cap
# via ``actions.max_ad_multiplier``; this constant is the default.
MAX_AD_MULTIPLIER: float = 5.0

# Post-audit round-2 (A2-23) — absolute hard ceiling on ``actions.max_ad_multiplier``.
# Any config that asks for a multiplier above this is rejected at load
# time by ``WorldEngine._validate_actions_section``. Protects the demand
# model from numerical blowups when an operator mis-types an order of
# magnitude (e.g. ``50.0`` instead of ``5.0``).
MAX_AD_MULTIPLIER_HARD_CEILING: float = 10.0

# Post-audit round-2 (A2-63) — default multiplier that derives the
# ``critical_ticket_per_step`` coefficient from ``urgent_ticket_per_step``
# when the config omits the critical key. Kept as a named constant so
# the relationship is discoverable in one place rather than buried as a
# magic ``1.5`` inside the reward engine.
DEFAULT_CRITICAL_MULTIPLIER: float = 1.5

# Post-audit remediation — minimum ad-spend per step. Protects
# ``ad_roi_positive`` from being farmed with pennies. Defaults to ``0.0``
# for backward compatibility with configs authored before the audit;
# shipped configs explicitly raise the floor. See ``_KNOWN_ACTIONS_KEYS``.
DEFAULT_AD_SPEND_MIN_PER_STEP: float = 0.0

# Post-audit remediation — cap on the number of aged tickets that can
# stack penalty. Without a cap the aging term can dominate every step
# on high-spawn configs. ``None`` preserves the pre-audit unbounded
# behaviour; shipped configs set a concrete integer cap.
DEFAULT_TICKET_AGING_PENALTY_CAP: int = 0  # 0 == disabled / unbounded

# Post-audit remediation — per-day relative volatility on
# ``competitor_prices``. Each SKU's competitor price is multiplied by
# ``1 + normal(0, volatility)`` on every ``_simulate_day``, clamped to a
# band anchored at the original config price so the random walk cannot
# escape to absurd values. Defaults to ``0.0`` (static competitor) for
# backward compatibility; shipped configs enable a small positive walk
# so ``set_price`` operates against a live target.
DEFAULT_COMPETITOR_PRICE_VOLATILITY: float = 0.0
COMPETITOR_PRICE_BAND_LO: float = 0.5  # × initial competitor price
COMPETITOR_PRICE_BAND_HI: float = 2.0

# Post-audit remediation — stockout-driven ticket churn. Each SKU that
# transitioned 1→0 on the current tick adds a multiplier on top of the
# configured ``spawn_rate_per_day``. Defaults to ``0.0`` for backward
# compatibility; shipped configs opt in with a small positive factor so
# bad operational decisions feed back into the ticket queue.
DEFAULT_STOCKOUT_CHURN_MULTIPLIER: float = 0.0

# Optional reactive competitor policy defaults. Disabled by default for
# backward compatibility; when enabled the competitor can undercut/follow
# our latest set price movement before demand is sampled.
DEFAULT_REACTIVE_COMPETITOR_ENABLED: bool = False
DEFAULT_REACTIVE_UNDERCUT_MULTIPLIER: float = 0.98
DEFAULT_REACTIVE_FOLLOW_UP_MULTIPLIER: float = 1.01
DEFAULT_REACTIVE_DEADZONE_MULTIPLIER: float = 0.02

# Optional supplier capacity cap per SKU (units per step). ``None`` means
# unlimited (legacy behavior).
DEFAULT_SUPPLIER_CAPACITY_PER_SKU: int = 0

# Optional market-shock defaults (disabled by default).
DEFAULT_MARKET_SHOCK_ENABLED: bool = False
DEFAULT_MARKET_SHOCK_PROBABILITY: float = 0.0
DEFAULT_MARKET_SHOCK_MIN_MULTIPLIER: float = 0.85
DEFAULT_MARKET_SHOCK_MAX_MULTIPLIER: float = 1.25
DEFAULT_MARKET_SHOCK_DURATION_DAYS: int = 2

# Rolling state-history window (ring buffer) used by the AI-CEO
# explainability layer to derive ``info.trend`` and the lightweight
# anomaly detectors. Kept small (N=20) so memory is bounded on long
# episodes without sacrificing trend resolution. A value <= 0 disables
# history collection entirely.
DEFAULT_STATE_HISTORY_WINDOW: int = 20

# Optional simple customer-satisfaction defaults.
DEFAULT_CUSTOMER_SATISFACTION_ENABLED: bool = False
DEFAULT_CUSTOMER_SATISFACTION_INITIAL: float = 1.0
DEFAULT_CUSTOMER_SATISFACTION_MIN: float = 0.3
DEFAULT_CUSTOMER_SATISFACTION_MAX: float = 1.0
DEFAULT_CUSTOMER_SATISFACTION_STOCKOUT_PENALTY: float = 0.03
DEFAULT_CUSTOMER_SATISFACTION_OPEN_TICKET_PENALTY: float = 0.002
DEFAULT_CUSTOMER_SATISFACTION_DAILY_RECOVERY: float = 0.01


__all__ = [
    "FALLBACK_UNIT_COST",
    "FALLBACK_BASE_PRICE",
    "DEFAULT_PROFIT_NORMALIZER",
    "DEFAULT_QUOTE_TTL_STEPS",
    "MAX_AD_MULTIPLIER",
    "MAX_AD_MULTIPLIER_HARD_CEILING",
    "DEFAULT_CRITICAL_MULTIPLIER",
    "DEFAULT_AD_SPEND_MIN_PER_STEP",
    "DEFAULT_TICKET_AGING_PENALTY_CAP",
    "DEFAULT_COMPETITOR_PRICE_VOLATILITY",
    "COMPETITOR_PRICE_BAND_LO",
    "COMPETITOR_PRICE_BAND_HI",
    "DEFAULT_STOCKOUT_CHURN_MULTIPLIER",
    "DEFAULT_REACTIVE_COMPETITOR_ENABLED",
    "DEFAULT_REACTIVE_UNDERCUT_MULTIPLIER",
    "DEFAULT_REACTIVE_FOLLOW_UP_MULTIPLIER",
    "DEFAULT_REACTIVE_DEADZONE_MULTIPLIER",
    "DEFAULT_SUPPLIER_CAPACITY_PER_SKU",
    "DEFAULT_MARKET_SHOCK_ENABLED",
    "DEFAULT_MARKET_SHOCK_PROBABILITY",
    "DEFAULT_MARKET_SHOCK_MIN_MULTIPLIER",
    "DEFAULT_MARKET_SHOCK_MAX_MULTIPLIER",
    "DEFAULT_MARKET_SHOCK_DURATION_DAYS",
    "DEFAULT_CUSTOMER_SATISFACTION_ENABLED",
    "DEFAULT_CUSTOMER_SATISFACTION_INITIAL",
    "DEFAULT_CUSTOMER_SATISFACTION_MIN",
    "DEFAULT_CUSTOMER_SATISFACTION_MAX",
    "DEFAULT_CUSTOMER_SATISFACTION_STOCKOUT_PENALTY",
    "DEFAULT_CUSTOMER_SATISFACTION_OPEN_TICKET_PENALTY",
    "DEFAULT_CUSTOMER_SATISFACTION_DAILY_RECOVERY",
    "DEFAULT_STATE_HISTORY_WINDOW",
]
