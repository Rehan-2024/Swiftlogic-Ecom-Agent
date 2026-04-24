"""env.invariants — post-audit round-2 (A2-3) soft-assertion helper.

A single entry point ``assert_state_invariants(state)`` that checks
the invariants any well-formed ``WorldEngine.state`` must satisfy:

* ``inventory[sku] >= 0`` for every SKU
* ``bank_balance`` is a finite float
* ``active_tickets`` entries are well-formed dicts with a ``status``

Enabled lazily: the check is a no-op unless the ``COMMERCEOPS_ASSERT_INVARIANTS``
environment variable is set to a truthy value (``"1"`` / ``"true"`` /
``"yes"``). Tests flip the flag on in ``conftest.py`` so every
``_process_action`` + ``_simulate_day`` round-trip is sanity-checked.

Failures raise ``AssertionError`` with a descriptive message. The
helper is deliberately cheap (a handful of linear scans) so enabling
it in CI doesn't dominate the per-step time budget.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Mapping


_ENABLED_ENV = "COMMERCEOPS_ASSERT_INVARIANTS"
_TRUTHY = {"1", "true", "yes", "on"}


def invariants_enabled() -> bool:
    """Return True when the ``COMMERCEOPS_ASSERT_INVARIANTS`` flag is set."""
    return str(os.environ.get(_ENABLED_ENV, "")).strip().lower() in _TRUTHY


def assert_state_invariants(state: Mapping[str, object]) -> None:
    """Raise AssertionError if ``state`` violates a basic invariant.

    No-ops when the env flag is off so production runs pay nothing.
    """
    if not invariants_enabled():
        return
    inv = state.get("inventory", {})
    if not isinstance(inv, Mapping):
        raise AssertionError("state.inventory must be a mapping")
    for sku, qty in inv.items():
        try:
            qty_i = int(qty)
        except (TypeError, ValueError):
            raise AssertionError(f"inventory[{sku!r}] is not an int: {qty!r}")
        if qty_i < 0:
            raise AssertionError(
                f"inventory[{sku!r}] is negative: {qty_i}"
            )
    try:
        bank = float(state.get("bank_balance", 0.0))
    except (TypeError, ValueError):
        raise AssertionError("state.bank_balance is not numeric")
    if not math.isfinite(bank):
        raise AssertionError(f"state.bank_balance is not finite: {bank!r}")
    tickets = state.get("active_tickets", [])
    if tickets is None:
        return
    if not isinstance(tickets, list):
        raise AssertionError("state.active_tickets must be a list")
    for t in tickets:
        if not isinstance(t, dict):
            raise AssertionError(f"active_tickets entry is not a dict: {t!r}")
        if "status" not in t or not isinstance(t.get("status"), str):
            raise AssertionError(
                f"active_tickets entry missing/invalid 'status': {t!r}"
            )


__all__ = ["assert_state_invariants", "invariants_enabled"]
