"""Shared fallback constants for the CommerceOps v2 engine.

Kept as a flat module of plain literals so that ``env/*.py`` and the
``ecom_env`` adapter can all import the same canonical values without
creating cross-module cycles. Historically these numbers were scattered as
inline magic values (e.g. ``70.0`` in ``world_engine`` and ``700.0`` in
``supplier_agent``); pinning them here makes bulk-retuning trivial.
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
# paths reference the same value.
MAX_AD_MULTIPLIER: float = 5.0


__all__ = [
    "FALLBACK_UNIT_COST",
    "FALLBACK_BASE_PRICE",
    "DEFAULT_PROFIT_NORMALIZER",
    "DEFAULT_QUOTE_TTL_STEPS",
    "MAX_AD_MULTIPLIER",
]
