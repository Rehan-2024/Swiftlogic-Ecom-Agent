"""
world_engine.py — Config-driven simulation core for CommerceOps v2.

The WorldEngine owns the mutable world state as a plain dict. It is kept
JSON-friendly so the thin adapter in ``ecom_env.py`` can coerce it into
Pydantic ``EcomObservation`` instances without any further translation.

Design principles:
    * No hardcoded product, price, or reward values — everything comes from the
      business config JSON.
    * Deterministic when seeded (seeds both ``random`` and ``numpy.random``).
    * Supports hot-swapping configs at runtime via ``load_config`` (used by the
      ``POST /config`` endpoint for live business-type swaps on stage).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import constants
from .actions import ACTION_HANDLERS
from .demand_model import generate_all_demand
from .reward_engine import compute_step_reward
from .supplier_agent import SupplierAgent
from .ticket_system import generate_episode_tickets, spawn_daily_tickets


DEFAULT_CONFIG_PATH = "configs/siyaani_fashion.json"


logger = logging.getLogger("commerceops.world_engine")


class ConfigValidationError(ValueError):
    """Raised when a business config file is missing required fields."""


# v2.3 Phase 4.2 — keys that USED to mean something in v2.0/v2.1 but are no
# longer consulted by any module. The validator warns so the operator knows
# to clean them up without breaking existing shipped configs.
_DEPRECATED_CONFIG_KEYS = {
    "financials.solvency_bonus_threshold": (
        "moved to rewards.solvency_threshold — this key is ignored"
    ),
    "demand.demand_model": (
        "only 'poisson' is implemented; this key is ignored"
    ),
    "rewards.bankruptcy_threshold": (
        "moved to financials.bankruptcy_threshold — the rewards mirror is "
        "still honoured for backward compatibility but should be dropped"
    ),
}


# Post-audit m-2 — whitelists of known keys per config section. Anything
# outside these sets is loaded fine but emits a WARNING at load time so
# typos like ``rewards.stockot_penalty`` don't silently take the default.
# These sets are deliberately permissive: purely-cosmetic top-level keys
# (``display_name``, ``currency``, ``_notes``) are included so they don't
# trigger warnings on the shipped configs.
_KNOWN_TOP_LEVEL_KEYS = frozenset({
    "business_id", "display_name", "currency", "_notes",
    "financials", "episode", "products", "tickets", "actions",
    "rewards", "graders", "supplier",
})
_KNOWN_FINANCIALS_KEYS = frozenset({
    "initial_bank_balance", "bankruptcy_threshold",
    "solvency_bonus_threshold",  # deprecated, still whitelisted so the
                                 # dedicated deprecation warning is the
                                 # only signal (no duplicate unknown-key
                                 # warning).
})
_KNOWN_EPISODE_KEYS = frozenset({"max_steps", "steps_per_day"})
_KNOWN_TICKET_KEYS = frozenset({
    "initial_count", "min_initial", "max_initial",
    "spawn_rate_per_day",
    "issue_types", "urgency_levels", "urgency_weights",
    "urgency_age_threshold_days",
    "refund_amount_range",
    "resolved_retention_days",
    "max_active",  # post-audit B.9
})
_KNOWN_ACTIONS_KEYS = frozenset({
    "allowed", "ad_spend_max_per_step",
    "price_min_mult_competitor", "price_max_mult_competitor",
})
_KNOWN_SUPPLIER_KEYS = frozenset({
    "volume_free_units", "volume_rate", "demand_rate",
    "price_cap_multiplier", "volume_discount", "spot_premium",
    "quote_expiry_steps",
})
_KNOWN_REWARD_KEYS = frozenset({
    "restock_success", "refund_success", "ad_roi_positive",
    "negotiate", "wait", "invalid_action",
    "bankruptcy_terminal", "bankruptcy_threshold",
    "solvency_per_step", "solvency_threshold",
    "revenue_multiplier", "revenue_mode", "revenue_cap_per_step",
    "urgent_ticket_per_step", "critical_ticket_per_step",
    "urgent_ticket_age_days",
    "stockout_penalty", "stockout_transition_grace",
    "bank_balance_delta_weight",
    "inventory_target_bonus",
})
_KNOWN_GRADERS_KEYS = frozenset({
    "triage_task", "inventory_task", "profit_task",
})
_KNOWN_GRADER_TRIAGE_KEYS = frozenset({"difficulty", "metric"})
_KNOWN_GRADER_INVENTORY_KEYS = frozenset({
    "difficulty", "metric", "target_sku", "target_units",
})
_KNOWN_GRADER_PROFIT_KEYS = frozenset({"difficulty", "metric", "normalizer"})
_KNOWN_PRODUCT_KEYS = frozenset({
    "sku", "display_name",
    "unit_cost", "sell_price", "competitor_price",
    "initial_stock", "restock_lead_days",
    "demand",
})
_KNOWN_DEMAND_KEYS = frozenset({
    "base_units_per_day", "ad_elasticity", "seasonality_weights",
    "demand_model",  # deprecated; dedicated warning handles it
})


def _warn_unknown_keys(section_name: str, section: Dict, allowed: frozenset) -> None:
    """Emit a WARNING for any key in ``section`` that isn't in ``allowed``.

    Unknown keys never raise — warning-only so forks that extended configs
    with private keys keep loading. The log key ``config_unknown_key``
    makes the messages trivial to grep for in CI.
    """
    if not isinstance(section, dict):
        return
    for key in section:
        if key not in allowed:
            logger.warning(
                "config_unknown_key section=%s key=%s",
                section_name,
                key,
            )


class WorldEngine:
    """Config-driven storefront simulation. One instance per environment."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path: str = config_path
        self.config: Dict = {}
        self.state: Dict = {}
        # v2.3 Phase 5.1 — per-env RNG instances so two engines in the same
        # process don't cross-contaminate via the global ``random`` /
        # ``numpy.random`` state. ``reset(seed)`` reseeds these *only*, so
        # calling one env's ``reset`` never disturbs another's stream.
        self._py_rng: random.Random = random.Random()
        self._np_rng: np.random.Generator = np.random.default_rng()
        self.load_config(config_path)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def load_config(self, config_path: str) -> None:
        """Load and validate a business config JSON. Does NOT auto-reset."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        try:
            with path.open("r", encoding="utf-8") as f:
                self.config = json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            # Post-audit m-3 — raising the base exception type would leak a
            # stack that doesn't mention the config path. Re-wrap as
            # ``ConfigValidationError`` so every config-load failure
            # surfaces with a consistent error class.
            raise ConfigValidationError(
                f"Config at {config_path} is not valid UTF-8 JSON: {exc}"
            ) from exc
        self._validate_config()
        self._build_lookup_tables()
        self.config_path = str(path)

    # Known action types; kept in sync with the discriminated EcomAction union
    # in ecom_env.py. Any allowed[] entry outside this set is rejected at load.
    _KNOWN_ACTIONS = {"restock", "refund", "ad_spend", "negotiate", "wait", "set_price"}

    def _validate_config(self) -> None:
        """Run every validation helper. Orchestrator only — each ``_validate_*``
        method raises :class:`ConfigValidationError` on failure, while the
        ``_warn_*`` helpers log non-fatal issues.

        Post-audit m-8 — split from the historical monolithic helper into
        section-scoped methods so individual rules are testable and
        additive without growing a single >200 LOC function.
        """
        self._validate_required_top_keys()
        self._validate_products()
        self._validate_actions_section()
        self._validate_financials()
        self._validate_rewards()
        self._validate_tickets()
        self._validate_graders()
        self._validate_cross_keys()
        self._warn_deprecated_keys()
        self._warn_unknown_section_keys()

    # ---- Section validators ----------------------------------------------

    def _validate_required_top_keys(self) -> None:
        required_top = [
            "business_id", "financials", "episode", "products",
            "tickets", "actions", "rewards", "graders",
        ]
        for key in required_top:
            if key not in self.config:
                raise ConfigValidationError(f"Config missing required key: {key}")

    def _validate_products(self) -> None:
        if not isinstance(self.config["products"], list) or not self.config["products"]:
            raise ConfigValidationError("Config 'products' must be a non-empty list")
        for p in self.config["products"]:
            for pk in ["sku", "unit_cost", "sell_price", "initial_stock", "demand"]:
                if pk not in p:
                    raise ConfigValidationError(f"Product missing field '{pk}': {p}")
            # Numeric sanity: prices/stocks must be non-negative real numbers.
            for numeric_key in ("unit_cost", "sell_price", "initial_stock"):
                try:
                    val = float(p[numeric_key])
                except (TypeError, ValueError):
                    raise ConfigValidationError(
                        f"Product '{p.get('sku')}' has non-numeric '{numeric_key}': {p[numeric_key]!r}"
                    )
                if val < 0:
                    raise ConfigValidationError(
                        f"Product '{p.get('sku')}' has negative '{numeric_key}': {val}"
                    )
            # Seasonality, if provided, must be a non-empty numeric list.
            demand_cfg = p.get("demand", {}) or {}
            weights = demand_cfg.get("seasonality_weights")
            if weights is not None:
                if not isinstance(weights, list) or not weights:
                    raise ConfigValidationError(
                        f"Product '{p.get('sku')}' has invalid 'seasonality_weights' (must be non-empty list)"
                    )
                for w in weights:
                    try:
                        float(w)
                    except (TypeError, ValueError):
                        raise ConfigValidationError(
                            f"Product '{p.get('sku')}' has non-numeric seasonality weight: {w!r}"
                        )

    def _validate_actions_section(self) -> None:
        actions_cfg = self.config.get("actions", {}) or {}
        allowed = actions_cfg.get("allowed", []) or []
        if not isinstance(allowed, list) or not allowed:
            raise ConfigValidationError("Config 'actions.allowed' must be a non-empty list")
        unknown = [a for a in allowed if a not in self._KNOWN_ACTIONS]
        if unknown:
            raise ConfigValidationError(
                f"Config 'actions.allowed' contains unknown action types: {unknown}"
            )
        ad_cap = actions_cfg.get("ad_spend_max_per_step")
        if ad_cap is not None:
            try:
                if float(ad_cap) < 0:
                    raise ConfigValidationError(
                        f"Config 'actions.ad_spend_max_per_step' must be non-negative: {ad_cap}"
                    )
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"Config 'actions.ad_spend_max_per_step' must be numeric: {ad_cap!r}"
                )

    def _validate_financials(self) -> None:
        fin = self.config.get("financials", {}) or {}
        try:
            if float(fin.get("initial_bank_balance", 0)) < 0:
                raise ConfigValidationError("'financials.initial_bank_balance' must be >= 0")
        except (TypeError, ValueError):
            raise ConfigValidationError("'financials.initial_bank_balance' must be numeric")

    def _validate_rewards(self) -> None:
        # Rewards table must be fully numeric; otherwise the first step call
        # will explode with an opaque TypeError deep inside the reward engine.
        rewards_cfg = self.config.get("rewards", {}) or {}
        if not isinstance(rewards_cfg, dict):
            raise ConfigValidationError("'rewards' must be a JSON object")
        _NON_NUMERIC_REWARD_KEYS = {"revenue_mode"}
        for rk, rv in rewards_cfg.items():
            if rk in _NON_NUMERIC_REWARD_KEYS:
                continue
            try:
                float(rv)
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"rewards.{rk!s} must be numeric, got {rv!r}"
                )
        mode = rewards_cfg.get("revenue_mode")
        if mode is not None and str(mode).lower() not in {"linear", "log", "cap"}:
            raise ConfigValidationError(
                f"rewards.revenue_mode must be one of 'linear'|'log'|'cap', got {mode!r}"
            )

    def _validate_tickets(self) -> None:
        # Refund payout range must be a [lo, hi] pair of non-negative numbers
        # with ``lo <= hi``. v2.3 Phase 2.2 made this *required*.
        tickets_cfg = self.config.get("tickets", {}) or {}
        rr = tickets_cfg.get("refund_amount_range")
        if rr is None:
            raise ConfigValidationError(
                "tickets.refund_amount_range is required and must be a [lo, hi] 2-list of non-negative floats with lo <= hi"
            )
        if not (isinstance(rr, (list, tuple)) and len(rr) == 2):
            raise ConfigValidationError(
                "tickets.refund_amount_range must be a [lo, hi] 2-list"
            )
        try:
            lo_v, hi_v = float(rr[0]), float(rr[1])
        except (TypeError, ValueError):
            raise ConfigValidationError(
                f"tickets.refund_amount_range values must be numeric: {rr!r}"
            )
        if lo_v < 0 or hi_v < 0:
            raise ConfigValidationError(
                f"tickets.refund_amount_range values must be >= 0: {rr!r}"
            )
        if lo_v > hi_v:
            raise ConfigValidationError(
                f"tickets.refund_amount_range must satisfy lo <= hi: {rr!r}"
            )
        # Post-audit B.9 — optional non-negative int cap on open tickets.
        max_active = tickets_cfg.get("max_active")
        if max_active is not None:
            try:
                mv = int(max_active)
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"tickets.max_active must be a non-negative integer: {max_active!r}"
                )
            if mv < 0:
                raise ConfigValidationError(
                    f"tickets.max_active must be >= 0: {max_active!r}"
                )

        # v2.3 Phase 4.4 — reject configs that would produce zero tickets.
        spawn_rate = float(tickets_cfg.get("spawn_rate_per_day", 0.0) or 0.0)
        min_initial = int(tickets_cfg.get("min_initial", 0) or 0)
        initial_count = tickets_cfg.get("initial_count")
        effective_initial = int(initial_count) if initial_count is not None else min_initial
        if effective_initial < 1 and spawn_rate <= 0.0:
            raise ConfigValidationError(
                "tickets: episode would produce zero tickets; require min_initial>=1 OR spawn_rate_per_day>0"
            )

    def _validate_graders(self) -> None:
        # Grader sanity: profit normalizer must be positive, and inventory
        # target_sku (when set) must be one of this config's SKUs.
        graders_cfg = self.config.get("graders", {}) or {}
        profit_cfg = graders_cfg.get("profit_task", {}) or {}
        if "normalizer" in profit_cfg:
            try:
                if float(profit_cfg["normalizer"]) <= 0:
                    raise ConfigValidationError(
                        f"graders.profit_task.normalizer must be > 0: {profit_cfg['normalizer']!r}"
                    )
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"graders.profit_task.normalizer must be numeric: {profit_cfg.get('normalizer')!r}"
                )
        inv_grader = graders_cfg.get("inventory_task", {}) or {}
        tgt = inv_grader.get("target_sku")
        if tgt is not None:
            skus = {p["sku"] for p in self.config["products"]}
            if tgt not in skus:
                raise ConfigValidationError(
                    f"graders.inventory_task.target_sku={tgt!r} not in products {sorted(skus)}"
                )

    def _validate_cross_keys(self) -> None:
        # v2.3 Phase 4.3 — validations that span multiple sections.
        rewards_cfg = self.config.get("rewards", {}) or {}
        mode = rewards_cfg.get("revenue_mode")
        if mode is not None and str(mode).lower() == "cap":
            cap = rewards_cfg.get("revenue_cap_per_step")
            if cap is None:
                raise ConfigValidationError(
                    "rewards.revenue_mode='cap' requires rewards.revenue_cap_per_step to be set"
                )
            try:
                if float(cap) <= 0:
                    raise ConfigValidationError(
                        f"rewards.revenue_cap_per_step must be > 0 when revenue_mode='cap', got {cap!r}"
                    )
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"rewards.revenue_cap_per_step must be numeric, got {cap!r}"
                )
        solvency_th = rewards_cfg.get("solvency_threshold")
        bankruptcy_th = rewards_cfg.get(
            "bankruptcy_threshold",
            (self.config.get("financials", {}) or {}).get("bankruptcy_threshold", 0.0),
        )
        if solvency_th is not None and bankruptcy_th is not None:
            try:
                s_val = float(solvency_th)
                b_val = float(bankruptcy_th)
            except (TypeError, ValueError):
                s_val = b_val = None
            if s_val is not None and b_val is not None and s_val < b_val:
                raise ConfigValidationError(
                    f"rewards.solvency_threshold ({solvency_th}) must be >= bankruptcy_threshold ({bankruptcy_th})"
                )
        # Post-audit C.2 — if ``rewards.bankruptcy_threshold`` is present and
        # the ``financials`` mirror is also present, they must agree. The
        # rewards copy is treated as the deprecated-but-honoured mirror so
        # pre-existing configs keep working.
        fin_cfg = self.config.get("financials", {}) or {}
        if "bankruptcy_threshold" in rewards_cfg and "bankruptcy_threshold" in fin_cfg:
            try:
                rv = float(rewards_cfg["bankruptcy_threshold"])
                fv = float(fin_cfg["bankruptcy_threshold"])
            except (TypeError, ValueError):
                rv = fv = None
            if rv is not None and fv is not None and rv != fv:
                raise ConfigValidationError(
                    "rewards.bankruptcy_threshold and financials.bankruptcy_threshold "
                    f"must be equal when both are set (got {rv} vs {fv}); prefer the "
                    "financials copy — rewards.bankruptcy_threshold is deprecated."
                )

    # ---- Non-fatal warnings ---------------------------------------------

    def _warn_deprecated_keys(self) -> None:
        """v2.3 Phase 4.2 — warn about keys ignored by the engine."""
        fin_cfg = self.config.get("financials", {}) or {}
        if "solvency_bonus_threshold" in fin_cfg:
            logger.warning(
                "config_deprecated key=financials.solvency_bonus_threshold %s",
                _DEPRECATED_CONFIG_KEYS["financials.solvency_bonus_threshold"],
            )
        rewards_cfg = self.config.get("rewards", {}) or {}
        if "bankruptcy_threshold" in rewards_cfg:
            # Post-audit C.2 — emit a soft deprecation notice while keeping
            # the mirror honoured. _validate_cross_keys enforces equality
            # with the financials copy when both are present.
            logger.warning(
                "config_deprecated key=rewards.bankruptcy_threshold %s",
                _DEPRECATED_CONFIG_KEYS["rewards.bankruptcy_threshold"],
            )
        for p in self.config.get("products", []):
            demand_cfg = p.get("demand", {}) or {}
            if "demand_model" in demand_cfg:
                logger.warning(
                    "config_deprecated sku=%s key=demand.demand_model %s",
                    p.get("sku"),
                    _DEPRECATED_CONFIG_KEYS["demand.demand_model"],
                )

    def _warn_unknown_section_keys(self) -> None:
        """Post-audit m-2 — emit WARNINGs for typos / unknown config keys."""
        _warn_unknown_keys("<root>", self.config, _KNOWN_TOP_LEVEL_KEYS)
        _warn_unknown_keys(
            "financials", self.config.get("financials", {}), _KNOWN_FINANCIALS_KEYS
        )
        _warn_unknown_keys(
            "episode", self.config.get("episode", {}), _KNOWN_EPISODE_KEYS
        )
        _warn_unknown_keys(
            "tickets", self.config.get("tickets", {}), _KNOWN_TICKET_KEYS
        )
        _warn_unknown_keys(
            "actions", self.config.get("actions", {}), _KNOWN_ACTIONS_KEYS
        )
        _warn_unknown_keys(
            "supplier", self.config.get("supplier", {}), _KNOWN_SUPPLIER_KEYS
        )
        _warn_unknown_keys(
            "rewards", self.config.get("rewards", {}), _KNOWN_REWARD_KEYS
        )
        graders_cfg = self.config.get("graders", {}) or {}
        _warn_unknown_keys("graders", graders_cfg, _KNOWN_GRADERS_KEYS)
        _warn_unknown_keys(
            "graders.triage_task",
            graders_cfg.get("triage_task", {}),
            _KNOWN_GRADER_TRIAGE_KEYS,
        )
        _warn_unknown_keys(
            "graders.inventory_task",
            graders_cfg.get("inventory_task", {}),
            _KNOWN_GRADER_INVENTORY_KEYS,
        )
        _warn_unknown_keys(
            "graders.profit_task",
            graders_cfg.get("profit_task", {}),
            _KNOWN_GRADER_PROFIT_KEYS,
        )
        for p in self.config.get("products", []):
            sku = p.get("sku", "?")
            _warn_unknown_keys(f"products[{sku}]", p, _KNOWN_PRODUCT_KEYS)
            _warn_unknown_keys(
                f"products[{sku}].demand",
                p.get("demand", {}),
                _KNOWN_DEMAND_KEYS,
            )

    def _build_lookup_tables(self) -> None:
        products = self.config["products"]
        self.unit_costs: Dict[str, float] = {p["sku"]: float(p["unit_cost"]) for p in products}
        self.sell_prices: Dict[str, float] = {p["sku"]: float(p["sell_price"]) for p in products}
        self.competitor_prices: Dict[str, float] = {
            p["sku"]: float(p.get("competitor_price", p["sell_price"] * 1.1)) for p in products
        }
        self.initial_stock: Dict[str, int] = {p["sku"]: int(p["initial_stock"]) for p in products}
        self.demand_configs: Dict[str, dict] = {p["sku"]: p.get("demand", {}) for p in products}

        # Rule-based supplier for Phase 3 Bonus (Theme #1 Multi-Agent).
        # Base prices come from the active config so the same logic works
        # across any business type loaded via /config hot-swap. On a hot-swap
        # we refresh *all* tunables (not only base prices) so the new config's
        # supplier section fully takes effect.
        supplier_cfg = self.config.get("supplier", {}) if isinstance(self.config, dict) else {}
        volume_free = int(supplier_cfg.get("volume_free_units", SupplierAgent.DEFAULT_VOLUME_FREE_UNITS))
        volume_rate = float(supplier_cfg.get("volume_rate", SupplierAgent.DEFAULT_VOLUME_RATE))
        demand_rate = float(supplier_cfg.get("demand_rate", SupplierAgent.DEFAULT_DEMAND_RATE))
        price_cap = float(supplier_cfg.get("price_cap_multiplier", SupplierAgent.DEFAULT_PRICE_CAP_MULTIPLIER))
        # v2.3 Phase 1.2 — supplier combo economics.
        volume_discount = float(
            supplier_cfg.get("volume_discount", SupplierAgent.DEFAULT_VOLUME_DISCOUNT)
        )
        # Cache on the engine so ``env.actions.do_restock`` can read it
        # without rummaging through the config on every restock.
        self._spot_premium = float(supplier_cfg.get("spot_premium", 0.0))
        if hasattr(self, "supplier_agent") and isinstance(self.supplier_agent, SupplierAgent):
            self.supplier_agent.update_base_prices(self.unit_costs)
            self.supplier_agent.volume_free_units = volume_free
            self.supplier_agent.volume_rate = volume_rate
            self.supplier_agent.demand_rate = demand_rate
            self.supplier_agent.price_cap_multiplier = price_cap
            self.supplier_agent.volume_discount = max(0.0, min(0.5, volume_discount))
        else:
            self.supplier_agent = SupplierAgent(
                base_prices=self.unit_costs,
                volume_free_units=volume_free,
                volume_rate=volume_rate,
                demand_rate=demand_rate,
                price_cap_multiplier=price_cap,
                volume_discount=volume_discount,
            )

        # Phase F.3 — cache per-config slices used in the hot path. These
        # dicts are NEVER mutated by the engine (all writes go to self.state),
        # so it's safe to share the same reference for the whole config
        # lifetime and skip a ``.get`` per step.
        self._rewards_cfg: Dict = dict(self.config.get("rewards", {}) or {})
        # Post-audit C.2 — ``bankruptcy_threshold`` logically belongs on the
        # ``financials`` section (it's a financial guardrail), but
        # ``_bankruptcy_term`` still reads it off the reward config for
        # historical reasons. Mirror the financials copy into the cached
        # rewards dict if the rewards section didn't explicitly set one,
        # so operators can drop the deprecated ``rewards.bankruptcy_threshold``
        # without breaking the reward-engine penalty trigger.
        if "bankruptcy_threshold" not in self._rewards_cfg:
            fin_cfg = self.config.get("financials", {}) or {}
            if "bankruptcy_threshold" in fin_cfg:
                self._rewards_cfg["bankruptcy_threshold"] = fin_cfg["bankruptcy_threshold"]
        self._actions_cfg: Dict = dict(self.config.get("actions", {}) or {})

        # v2.3 Phase 6.2 + post-audit M-3 — cache the inventory-target fields
        # from the config's grader block so the reward engine's optional
        # ``inventory_target_bonus`` term can read them without having to
        # import ``ecom_env`` (which would create a circular import).
        #
        # Named ``_reward_shaping_ctx`` (not ``_grader_context``) to
        # deliberately disambiguate from ``EcomEnv.grader_context`` — the
        # latter also carries ``profit_normalizer`` for the profit grader,
        # which this subset intentionally does NOT mirror. Keeping the two
        # namespaces disjoint avoids an accidental contributor assumption
        # that the reward-shaping cache and the grader context are
        # interchangeable.
        graders_cfg = self.config.get("graders", {}) or {}
        inv_cfg = graders_cfg.get("inventory_task", {}) or {}
        try:
            inv_target_units = float(inv_cfg.get("target_units", 0))
        except (TypeError, ValueError):
            inv_target_units = 0.0
        self._reward_shaping_ctx: Dict = {
            "inventory_target_sku": str(inv_cfg.get("target_sku", "")),
            "inventory_target_units": inv_target_units,
        }

    # ------------------------------------------------------------------
    # Fast state snapshot (Phase F.1)
    # ------------------------------------------------------------------
    def _snapshot_state(self) -> Dict:
        """Return a structural copy of ``self.state`` at ~20x deepcopy speed.

        Relies on the fact that our state schema is well-known and shallow:
        every mutable container is either a flat dict of scalars, a list of
        dicts (``active_tickets``), or a dict-of-lists (``daily_sales_history``).
        Because we enumerate every shape explicitly here, ``_snapshot_state``
        is semantically equivalent to ``copy.deepcopy(self.state)`` for the
        reward engine and the observation adapter, while being dramatically
        faster on large step counts.
        """
        snap: Dict = {}
        for k, v in self.state.items():
            if isinstance(v, dict):
                snap[k] = dict(v)
            elif isinstance(v, list):
                snap[k] = [dict(t) if isinstance(t, dict) else t for t in v]
            else:
                snap[k] = v
        hist = self.state.get("daily_sales_history")
        if isinstance(hist, dict):
            snap["daily_sales_history"] = {kk: list(vv) for kk, vv in hist.items()}
        # Post-audit M-1 — ``pending_deliveries`` is a dict of lists-of-tuples.
        # The generic dict branch above only shallow-copies the outer map, so
        # the inner lists would remain shared between ``self.state`` and the
        # snapshot. Dedicated branch mirrors the ``daily_sales_history`` shape
        # treatment. Tuples are immutable so list-level copy is sufficient.
        pending = self.state.get("pending_deliveries")
        if isinstance(pending, dict):
            snap["pending_deliveries"] = {
                kk: list(vv) if isinstance(vv, list) else vv
                for kk, vv in pending.items()
            }
        return snap

    def _drain_pending_deliveries(self) -> None:
        """Move any pending restock deliveries whose day has arrived into stock.

        v2.3 Phase 4.1 — before ``pending_deliveries`` was wired up,
        ``restock_lead_days`` was effectively ignored: all restocks landed
        instantly. Now a restock with ``lead_days > 0`` enqueues the qty
        here; this helper drains arrivals at the start of each simulated
        day and decrements the ``pending_orders`` counter so the
        observation matches reality.
        """
        schedule = self.state.setdefault("pending_deliveries", {})
        if not schedule:
            return
        today = int(self.state.get("current_day", 0))
        pending_orders = self.state.setdefault("pending_orders", {})
        for sku, deliveries in list(schedule.items()):
            if not deliveries:
                continue
            remaining = []
            delivered = 0
            for delivery_day, qty in deliveries:
                if int(delivery_day) <= today:
                    delivered += int(qty)
                else:
                    remaining.append((int(delivery_day), int(qty)))
            if delivered > 0:
                self.state["inventory"][sku] = int(self.state["inventory"].get(sku, 0)) + delivered
                pending_orders[sku] = max(0, int(pending_orders.get(sku, 0)) - delivered)
            schedule[sku] = remaining

    def _get_lead_days(self, sku: str) -> int:
        """Return the ``restock_lead_days`` for a SKU, defaulting to 0."""
        for p in self.config.get("products", []):
            if p.get("sku") == sku:
                try:
                    return max(0, int(p.get("restock_lead_days", 0)))
                except (TypeError, ValueError):
                    return 0
        return 0

    def _recent_sales_signal(self, sku: str) -> float:
        """Return a normalized demand signal from a 3-day rolling sales mean.

        Falls back to the most recent ``daily_sales`` value when the history
        buffer is empty (e.g. right after ``reset``). Clamped to ``[0, 5]`` so
        the supplier's demand premium cannot explode.
        """
        history = self.state.get("daily_sales_history", {}).get(sku, [])
        if isinstance(history, list) and history:
            window = history[-3:]
            avg_sales = float(sum(window)) / float(len(window))
        else:
            avg_sales = float(self.state.get("daily_sales", {}).get(sku, 0))
        base_units = float(self.demand_configs.get(sku, {}).get("base_units_per_day", 1.0) or 1.0)
        if base_units <= 0:
            base_units = 1.0
        return max(0.0, min(5.0, avg_sales / base_units))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> Dict:
        if seed is not None:
            # v2.3 Phase 5.1 + post-audit M-2 — reseed *only* the per-env
            # RNGs. The two global ``random.seed`` / ``np.random.seed`` calls
            # were removed because every consumer (demand_model,
            # ticket_system, refund payout) now accepts an ``rng=`` kwarg
            # that the engine threads through. Reseeding the process-wide
            # globals here coupled every env in the same process, which
            # defeated Phase 5.1's whole purpose for test harnesses / future
            # multi-tenant runners.
            self._py_rng.seed(int(seed))
            self._np_rng = np.random.default_rng(int(seed))

        cfg = self.config
        tickets_cfg = cfg.get("tickets", {})
        initial_count = tickets_cfg.get("initial_count")
        min_c = tickets_cfg.get("min_initial", 3)
        max_c = tickets_cfg.get("max_initial", 5)
        issue_types = tickets_cfg.get("issue_types")
        urgency_levels = tickets_cfg.get("urgency_levels")
        urgency_weights = tickets_cfg.get("urgency_weights")

        active_tickets = generate_episode_tickets(
            num=initial_count,
            current_day=1,
            min_count=min_c,
            max_count=max_c,
            issue_types=issue_types,
            urgency_levels=urgency_levels,
            urgency_weights=urgency_weights,
            rng=self._py_rng,
        )

        skus = [p["sku"] for p in cfg["products"]]
        self.state = {
            "current_day": 1,
            "current_week": 0,
            "step_count": 0,
            "bank_balance": float(cfg["financials"]["initial_bank_balance"]),
            "inventory": {sku: self.initial_stock[sku] for sku in skus},
            "pending_orders": {sku: 0 for sku in skus},
            "active_tickets": active_tickets,
            "daily_sales": {sku: 0 for sku in skus},
            "daily_sales_history": {sku: [] for sku in skus},
            "active_ad_spend": {sku: 0.0 for sku in skus},
            "prices": dict(self.sell_prices),
            "competitor_prices": dict(self.competitor_prices),
            "cumulative_revenue": 0.0,
            # Gross revenue realised on the last simulated day. Surfaced on
            # the observation so training loops don't have to re-derive it
            # from ``daily_sales * prices``. Reset here so the very first
            # /state after /reset doesn't leak a stale value.
            "daily_revenue": 0.0,
            "supplier_quotes": {},
            "supplier_quote_expiry": {},
            # v2.3 Phase 4.1 — pending deliveries schedule. Keys are SKUs,
            # values are lists of ``(delivery_day, quantity)`` tuples. On
            # every ``_simulate_day`` entry we drain any entry whose day has
            # arrived into ``inventory`` and decrement ``pending_orders``.
            "pending_deliveries": {sku: [] for sku in skus},
            "reward": 0.0,
            "done": False,
        }
        return self._snapshot_state()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        # v2.3 Phase 2.3 — decay previous tick's ad budget *before* this
        # tick's action handler runs. If the policy issues another ad_spend
        # this step the handler will overwrite the key; otherwise the budget
        # goes to zero and no longer shapes demand.
        if self.state.get("active_ad_spend"):
            self.state["active_ad_spend"] = {
                sku: 0.0 for sku in self.state["active_ad_spend"]
            }
        # Phase F.1 — structural snapshot is ~20x faster than deepcopy.
        state_before = self._snapshot_state()
        base_reward, info = self._process_action(action)
        # Phase F.2 — thread daily_revenue back out of the day simulation so
        # the reward engine doesn't recompute it from state_after.
        daily_revenue = self._simulate_day()
        self.state["step_count"] += 1

        # Phase G.1 — request a per-term breakdown so callers can log the
        # dynamics of each shaping signal without changing the scalar reward.
        # Breakdown goes into ``info["reward_breakdown"]``; the numeric
        # ``reward`` returned to the RL loop is unchanged.
        #
        # v2.3 Phase 1.1 — surface ``ad_spend_applied`` on ``action_result`` so
        # ``_ad_roi_term`` can detect the budget that was active during
        # ``_simulate_day`` even after it got zeroed in state.
        total_reward, breakdown = compute_step_reward(
            action_result={
                "base_reward": base_reward,
                "daily_revenue": daily_revenue,
                "ad_spend_applied": info.get("ad_spend_applied", {}),
            },
            state_before=state_before,
            state_after=self.state,
            rewards_config=self._rewards_cfg,
            return_breakdown=True,
            grader_context=self._reward_shaping_ctx,
        )
        info["reward_breakdown"] = breakdown

        done = (
            self.state["step_count"] >= int(self.config["episode"].get("max_steps", 50))
            or self.state["bank_balance"] <= float(self.config["financials"].get("bankruptcy_threshold", 0.0))
        )
        self.state["reward"] = total_reward
        self.state["done"] = bool(done)
        return self._snapshot_state(), float(total_reward), bool(done), info

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------
    def _process_action(self, action: Dict) -> Tuple[float, Dict]:
        """Dispatch a validated action dict to its handler in ``env.actions``.

        The handler mutates ``self.state`` directly and returns the
        ``(base_reward, info)`` pair that the step function feeds into
        ``compute_step_reward``. Unknown or disallowed actions short-circuit
        here with the ``invalid_action`` coefficient so ``atype in allowed``
        is a hard gate.
        """
        atype = action.get("action_type")
        rewards_cfg = self.config.get("rewards", {})
        allowed: List[str] = list(self.config.get("actions", {}).get("allowed", []))
        if atype not in allowed:
            return float(rewards_cfg.get("invalid_action", -0.2)), {"error": "action_not_allowed"}

        handler = ACTION_HANDLERS.get(atype)
        if handler is None:
            return float(rewards_cfg.get("invalid_action", -0.2)), {"error": "unknown_action"}
        return handler(self, action)

    # ------------------------------------------------------------------
    # Daily simulation
    # ------------------------------------------------------------------
    def _simulate_day(self) -> float:
        """Advance one business day: realize demand, book revenue, spawn tickets.

        Returns the ``daily_revenue`` realized this tick so ``step`` can pass
        it to ``compute_step_reward`` without having to recompute it from
        ``state_after`` (Phase F.2 optimisation).
        """
        # v2.3 Phase 4.1 — drain any pending supplier deliveries whose
        # delivery day has arrived. ``restock_lead_days`` per product
        # controls how far into the future a restock lands. This closes
        # the long-standing gap where ``pending_orders`` was in state but
        # never consumed.
        self._drain_pending_deliveries()

        sales = generate_all_demand(
            inventory=self.state["inventory"],
            active_ad_spend=self.state["active_ad_spend"],
            prices=self.state["prices"],
            competitor_prices=self.state["competitor_prices"],
            demand_configs=self.demand_configs,
            current_day=int(self.state["current_day"]),
            rng=self._np_rng,
        )
        daily_revenue = 0.0
        for sku, sold in sales.items():
            sold = int(sold)
            self.state["daily_sales"][sku] = sold
            hist = self.state.setdefault("daily_sales_history", {}).setdefault(sku, [])
            hist.append(sold)
            if len(hist) > 3:
                del hist[:-3]
            self.state["inventory"][sku] = max(0, int(self.state["inventory"].get(sku, 0)) - sold)
            daily_revenue += sold * float(self.state["prices"].get(sku, 0.0))
        self.state["bank_balance"] += daily_revenue
        self.state["cumulative_revenue"] = float(self.state.get("cumulative_revenue", 0.0)) + daily_revenue
        # Cache the realised gross revenue so downstream consumers
        # (observation, reward breakdown, debug endpoint) can read it
        # without re-deriving from ``daily_sales * prices``.
        self.state["daily_revenue"] = float(daily_revenue)

        # v2.3 Phase 2.3 — ad budget decay moved to the *top* of ``step`` so
        # the observation returned to the policy correctly reflects the
        # budget that shaped this tick's demand. Previously it was zeroed
        # here, which meant the observation always showed ``{sku: 0.0}``
        # and the budget the policy just spent was invisible. Phase 1.1
        # already decoupled ``_ad_roi_term`` from this state field via
        # ``action_result["ad_spend_applied"]``.

        # v2.3 Phase 5.5 — drop resolved tickets older than the retention
        # window. Keeps ``active_tickets`` bounded on long-horizon runs so
        # observation payloads and grader scans don't degrade linearly.
        retention = int(
            (self.config.get("tickets", {}) or {}).get("resolved_retention_days", 7)
        )
        if retention >= 0:
            today = int(self.state.get("current_day", 0))
            self.state["active_tickets"] = [
                t
                for t in self.state.get("active_tickets", [])
                if not (
                    t.get("status") == "resolved"
                    and today - int(t.get("created_day", today)) > retention
                )
            ]

        # Stochastic ticket spawning.
        tickets_cfg = self.config.get("tickets", {})
        spawn_daily_tickets(
            active_tickets=self.state["active_tickets"],
            current_day=int(self.state["current_day"]),
            spawn_rate_per_day=float(tickets_cfg.get("spawn_rate_per_day", 0.0)),
            issue_types=tickets_cfg.get("issue_types"),
            urgency_levels=tickets_cfg.get("urgency_levels"),
            urgency_weights=tickets_cfg.get("urgency_weights"),
            rng=self._py_rng,
            max_active=tickets_cfg.get("max_active"),  # post-audit B.9
        )

        # Advance the calendar.
        self.state["current_day"] = int(self.state["current_day"]) + 1
        self.state["current_week"] = int(self.state["current_day"]) // 7

        return float(daily_revenue)
