"""
supplier_agent.py — Rule-based supplier for Swiftlogic CommerceOps v2 (Phase 3 Bonus).

Covers Theme #1 (Multi-Agent) of the hackathon roadmap. Given a SKU, a requested
restock quantity, and a demand signal, the supplier returns a unit price that
the merchant agent can either accept (via a follow-up ``RestockAction``) or
ignore (letting the quote expire).

Pricing formula::

    unit_price = base_price
               * (1 + max(0, qty - volume_free_units) * volume_rate)  # bulk premium
               * (1 + max(0, demand_signal)        * demand_rate)     # hot-item premium

Notes
-----
* Base prices are injected by the ``WorldEngine`` from the active business
  config's ``unit_costs``, so the same supplier logic works across Siyaani,
  MedPlus, and Stackbase without any code changes.
* The supplier has no learnable state — this is a strategic pricing oracle,
  not a second LLM. Keeps the training pipeline simple while still covering
  the multi-agent theme.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from . import constants


logger = logging.getLogger("commerceops.supplier")


class SupplierAgent:
    """Stateless rule-based supplier that quotes unit prices on demand."""

    DEFAULT_VOLUME_FREE_UNITS = 20
    DEFAULT_VOLUME_RATE = 0.01   # +1% per unit above the free threshold
    DEFAULT_DEMAND_RATE = 0.10   # +10% per unit of normalized demand signal
    DEFAULT_PRICE_CAP_MULTIPLIER = 2.5
    # v2.3 Phase 1.2 — small-order volume discount. When a policy negotiates
    # for qty <= ``volume_free_units`` the quote is multiplied by
    # ``(1 - volume_discount)`` so small orders can come in *under* the
    # list price. Combined with ``spot_premium`` on un-negotiated restocks
    # in ``env.actions.do_restock``, this restores economic tension to the
    # ``negotiate -> restock`` loop.
    DEFAULT_VOLUME_DISCOUNT = 0.0
    # Kept as a class-level alias for any external code that reached in for
    # ``SupplierAgent.FALLBACK_BASE_PRICE``; delegates to ``env.constants``.
    FALLBACK_BASE_PRICE = constants.FALLBACK_BASE_PRICE

    def __init__(
        self,
        base_prices: Optional[Dict[str, float]] = None,
        volume_free_units: int = DEFAULT_VOLUME_FREE_UNITS,
        volume_rate: float = DEFAULT_VOLUME_RATE,
        demand_rate: float = DEFAULT_DEMAND_RATE,
        price_cap_multiplier: float = DEFAULT_PRICE_CAP_MULTIPLIER,
        fallback_base_price: float = FALLBACK_BASE_PRICE,
        volume_discount: float = DEFAULT_VOLUME_DISCOUNT,
    ) -> None:
        self.base_prices: Dict[str, float] = dict(base_prices or {})
        # Post-audit round-2 (A2-22) — ``volume_free_units`` must be a
        # strict int. Silent ``int(3.9)`` → 3 was a trap: a config that
        # wrote ``volume_free_units: 3.9`` would behave as ``3`` without
        # the operator being told. Fractional units of inventory don't
        # make sense, so reject explicitly.
        if isinstance(volume_free_units, bool) or not isinstance(volume_free_units, int):
            try:
                as_f = float(volume_free_units)
            except (TypeError, ValueError):
                raise TypeError(
                    f"volume_free_units must be int, got {type(volume_free_units).__name__}"
                )
            if not as_f.is_integer():
                raise TypeError(
                    f"volume_free_units must be whole, got {volume_free_units!r}"
                )
            volume_free_units = int(as_f)
        self.volume_free_units = int(volume_free_units)
        self.volume_rate = float(volume_rate)
        self.demand_rate = float(demand_rate)
        self.price_cap_multiplier = float(price_cap_multiplier)
        self.fallback_base_price = float(fallback_base_price)
        # Clamp to a sane band: a 0% discount is legacy, 50% discount
        # (negative margin) is the practical ceiling. Rejecting here also
        # catches obvious config typos (e.g. 30 instead of 0.3).
        self.volume_discount = max(0.0, min(0.5, float(volume_discount)))

    def quote_price(
        self,
        sku: str,
        quantity_requested: int,
        demand_signal: float,
    ) -> float:
        """Return a supplier unit-price quote for a proposed restock.

        The raw quote is hard-capped at ``base_price * price_cap_multiplier``
        to prevent pricing blowups on pathological (very large) orders or
        sustained demand spikes. For small orders (``qty <= volume_free_units``)
        the quote is additionally multiplied by ``(1 - volume_discount)`` so a
        negotiated quote can beat the list price — restoring economic tension
        that was missing in v2.2 where every quote was >= list price.
        """
        # Post-audit m-5 / round-2 (A2-21) — emit a WARNING when falling
        # back to a generic base price. Instead of the hard-coded module
        # constant, prefer the mean of configured ``base_prices`` (if
        # any) so the fallback is contextually similar to the business
        # the policy is actually running against.
        if sku not in self.base_prices:
            fallback = self._derive_fallback_price()
            logger.warning(
                "supplier_quote_price_fallback sku=%s fallback=%s",
                sku,
                fallback,
            )
            base = float(fallback)
        else:
            base = float(self.base_prices[sku])
        qty = max(0, int(quantity_requested))
        over_free = max(0, qty - self.volume_free_units)
        volume_premium = 1.0 + over_free * self.volume_rate
        demand_premium = 1.0 + max(0.0, float(demand_signal)) * self.demand_rate
        raw = base * volume_premium * demand_premium
        if over_free == 0 and self.volume_discount > 0.0:
            # Small-order discount. Only fires when the order is entirely
            # within the free band AND discount is non-zero.
            raw = raw * (1.0 - self.volume_discount)
        capped = min(raw, base * max(1.0, self.price_cap_multiplier))
        return round(capped, 2)

    # ------------------------------------------------------------------
    # Convenience helpers (used by the WorldEngine and for debugging)
    # ------------------------------------------------------------------
    def update_base_prices(self, base_prices: Dict[str, float]) -> None:
        """Refresh the base-price table, e.g. after a /config hot-swap."""
        self.base_prices = dict(base_prices or {})

    def list_price(self, sku: str) -> float:
        """Return the unmodified list price for a SKU (no premium applied)."""
        # Post-audit m-5 / round-2 (A2-21) — mirror the warning emitted
        # in ``quote_price`` so either entry point surfaces the fallback,
        # and use the same context-aware derivation.
        if sku not in self.base_prices:
            fallback = self._derive_fallback_price()
            logger.warning(
                "supplier_list_price_fallback sku=%s fallback=%s",
                sku,
                fallback,
            )
            return float(fallback)
        return float(self.base_prices[sku])

    def _derive_fallback_price(self) -> float:
        """Return a contextually-sensible fallback base price.

        Post-audit round-2 (A2-21) — previously the supplier fell back
        to a hard-coded ``FALLBACK_BASE_PRICE`` (e.g. $700) whenever an
        unknown SKU was quoted, which makes zero sense on a pharmacy
        config where every SKU costs <$10. We now prefer the mean of
        the live ``base_prices`` map; only if the map is empty do we
        fall back to the module constant.
        """
        if self.base_prices:
            try:
                vals = [float(v) for v in self.base_prices.values() if float(v) > 0]
            except (TypeError, ValueError):
                vals = []
            if vals:
                return float(sum(vals) / len(vals))
        return float(self.fallback_base_price)


__all__ = ["SupplierAgent"]
