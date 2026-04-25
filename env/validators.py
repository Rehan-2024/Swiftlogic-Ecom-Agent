"""env.validators — post-audit round-2 (A2-60) re-export module.

The config-validation rules currently live inside ``env.world_engine``
as ``WorldEngine._validate_*`` methods. A full extraction into free
functions (per the remediation plan) is a larger refactor that is
intentionally out of scope for the v2.3.x → v2.4.0 bug-fix cycle; the
methods remain testable in place and the invariants they enforce are
covered by ``tests/test_config_validation.py``.

What this module provides instead:

* A stable import path (``from env.validators import ConfigValidationError``)
  that callers can rely on even after the methods get extracted in a
  future release.
* Re-exports of the reward-sign rule table and the centralised
  ``_DEPRECATED_BUT_WHITELISTED`` set so downstream tooling (linting,
  external linters, config-authoring GUIs) can introspect them without
  reaching into the WorldEngine class body.
"""

from __future__ import annotations

from .world_engine import (
    ConfigValidationError,
    _REWARD_SIGN_RULES,
    _DEPRECATED_BUT_WHITELISTED,
)


REWARD_SIGN_RULES = dict(_REWARD_SIGN_RULES)
"""Immutable copy of the canonical reward-sign rules.

Mutating this dict has no effect on the running validator; clone-and-
modify is intentional so external tooling can't accidentally patch the
rules at runtime.
"""

DEPRECATED_BUT_WHITELISTED = frozenset(_DEPRECATED_BUT_WHITELISTED)
"""Frozenset copy of keys that the validator still accepts but will
remove in a future release. Use this to surface the deprecation in
config-authoring tooling without reaching into ``WorldEngine``."""


__all__ = [
    "ConfigValidationError",
    "REWARD_SIGN_RULES",
    "DEPRECATED_BUT_WHITELISTED",
]
