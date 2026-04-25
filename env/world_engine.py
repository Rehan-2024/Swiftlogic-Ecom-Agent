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
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import constants
from .actions import ACTION_HANDLERS
from .demand_model import generate_all_demand
from .invariants import assert_state_invariants as _assert_state_invariants
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
    "rewards", "graders", "supplier", "competitor", "market", "customer",
})
_KNOWN_FINANCIALS_KEYS = frozenset({
    "initial_bank_balance", "bankruptcy_threshold",
    "inventory_holding_cost_per_unit_per_day",
    "solvency_bonus_threshold",  # deprecated, still whitelisted so the
                                 # dedicated deprecation warning is the
                                 # only signal (no duplicate unknown-key
                                 # warning).
})
_KNOWN_EPISODE_KEYS = frozenset({"max_steps", "steps_per_day"})
# Post-audit C.2 (v2.3.x) — ``tickets.max_active`` is an optional
# non-negative integer that caps the number of concurrently open tickets
# a daily spawn can push onto the queue. It is **not** set by default:
# the shipped configs rely on natural churn (agent resolution + retention
# decay) to keep the queue bounded. Explicitly set ``max_active`` when
# tuning exotic configs with high ``spawn_rate_per_day`` or long episode
# horizons, where an unbounded backlog would otherwise dominate the
# triage grader and crowd out the other reward terms. See the README
# "Config status" section for the recommended workflow.
_KNOWN_TICKET_KEYS = frozenset({
    "initial_count", "min_initial", "max_initial",
    "spawn_rate_per_day",
    "issue_types", "urgency_levels", "urgency_weights",
    "urgency_age_threshold_days",
    "refund_amount_range",
    "resolved_retention_days",
    "max_active",  # post-audit B.9 — see C.2 note above.
    # Post-audit R-7 / realism: spawn-rate multiplier driven by the number
    # of SKUs that hit zero stock on the current step (customer churn feedback).
    "stockout_churn_multiplier",
    # Post-audit round-2 (A2-33) — opt-in SKU-biased ticket generation.
    "ticket_issue_bias_by_sku",
    # Audit MEDIUM #8 — opt-in partial refund path. Defaults off so
    # every shipped config behaves identically to pre-audit builds.
    "allow_partial_refund",
    "partial_refund_min_fraction",
})
_KNOWN_ACTIONS_KEYS = frozenset({
    "allowed", "ad_spend_max_per_step",
    "price_min_mult_competitor", "price_max_mult_competitor",
    # Post-audit H-2: per-step minimum ad spend so ``ad_roi_positive``
    # can't be farmed with pennies.
    "ad_spend_min_per_step",
    # Post-audit D-3: config-driven cap for the ad-driven demand multiplier.
    "max_ad_multiplier",
    # Post-audit round-2 (A2-13) — per-step cap on the quantity of a
    # single ``restock`` action. Protects the simulator against a
    # policy that asks for, e.g., 10^9 units in one shot.
    "restock_max_qty_per_step",
})
_KNOWN_SUPPLIER_KEYS = frozenset({
    "volume_free_units", "volume_rate", "demand_rate",
    "price_cap_multiplier", "volume_discount", "spot_premium",
    "quote_expiry_steps", "capacity_per_sku",
})
_KNOWN_COMPETITOR_KEYS = frozenset({
    "reactive_enabled",
    "reactive_undercut_multiplier",
    "reactive_follow_up_multiplier",
    "reactive_deadzone_multiplier",
})
_KNOWN_MARKET_KEYS = frozenset({
    "shock_enabled",
    "shock_probability",
    "shock_min_multiplier",
    "shock_max_multiplier",
    "shock_duration_days",
    "shock_sku_multipliers",
})
_KNOWN_CUSTOMER_KEYS = frozenset({
    "satisfaction_enabled",
    "satisfaction_initial",
    "satisfaction_min",
    "satisfaction_max",
    "stockout_penalty",
    "open_ticket_penalty",
    "daily_recovery",
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
    # Post-audit L-2 / L-4: ``set_price`` is an opt-in per-step reward
    # coefficient for the SetPriceAction. Post-audit R-6: optional cap
    # on the aggregate ticket-aging penalty so long-horizon episodes
    # can't be dominated by a single queue metric.
    "set_price",
    "ticket_aging_penalty_cap",
    # Post-audit round-2 (A2-29) — optional per-urgency penalty map.
    # When set, overrides the ``urgent_ticket_per_step`` /
    # ``critical_ticket_per_step`` scalars on a per-urgency-label basis
    # so exotic configs can fine-tune (or add entirely new) levels.
    "urgency_penalty_map",
    # Post-audit round-2 (A2-31) — opt-in scaled ad-ROI bonus.
    "ad_roi_scaled",
    # Post-audit round-2 (A2-32) — optional cap on the refund-payout
    # amount subtracted from the bank-delta shaping term so a single
    # large refund doesn't dominate the reward.
    "refund_payout_delta_cap",
})
_KNOWN_GRADERS_KEYS = frozenset({
    "triage_task", "inventory_task", "profit_task",
})
_KNOWN_GRADER_TRIAGE_KEYS = frozenset({
    "difficulty", "metric",
    # Post-audit round-2 (A2-34) — optional additive bonus applied when
    # the agent resolved tickets on the SKUs flagged as problematic.
    "sku_match_bonus",
})
_KNOWN_GRADER_INVENTORY_KEYS = frozenset({
    "difficulty", "metric", "target_sku", "target_units",
})
_KNOWN_GRADER_PROFIT_KEYS = frozenset({"difficulty", "metric", "normalizer"})
_KNOWN_PRODUCT_KEYS = frozenset({
    "sku", "display_name",
    "unit_cost", "sell_price", "competitor_price",
    "initial_stock", "restock_lead_days",
    "demand",
    # Post-audit realism: per-step competitor-price relative volatility.
    # ``0.0`` (default) preserves the static-competitor behaviour.
    "competitor_price_volatility",
})
_KNOWN_DEMAND_KEYS = frozenset({
    "base_units_per_day", "ad_elasticity", "seasonality_weights",
    "demand_model",  # deprecated; dedicated warning handles it
})


# Post-audit round-2 (A2-26) — canonical reward-sign rules. ``<= 0`` keys
# are coefficients on penalty events (agent did something bad); ``>= 0``
# keys are bonuses. Zero is always allowed so configs can disable a term
# by setting it to 0 without tripping the sign rule.
_REWARD_SIGN_RULES: Dict[str, str] = {
    "bankruptcy_terminal":      "<= 0",
    "stockout_penalty":         "<= 0",
    "urgent_ticket_per_step":   "<= 0",
    "critical_ticket_per_step": "<= 0",
    "invalid_action":           "<= 0",
    "revenue_multiplier":       ">= 0",
    "solvency_per_step":        ">= 0",
    "ad_roi_positive":          ">= 0",
    "restock_success":          ">= 0",
    "refund_success":           ">= 0",
    "inventory_target_bonus":   ">= 0",
    "set_price":                ">= 0",
    "ticket_aging_penalty_cap": ">= 0",
    "refund_payout_delta_cap":  ">= 0",
    "revenue_cap_per_step":     ">= 0",
}

# Post-audit round-2 (A2-62) — single source of truth for keys that are
# deprecated but still accepted (so we don't emit duplicate "unknown key"
# warnings alongside the dedicated deprecation warning).
_DEPRECATED_BUT_WHITELISTED: frozenset = frozenset({
    "rewards.bankruptcy_threshold",
    "financials.solvency_bonus_threshold",
    "demand.demand_model",
})


def _warn_unknown_keys(
    section_name: str,
    section: Dict,
    allowed: frozenset,
    nested: bool = False,
) -> None:
    """Emit a WARNING for any key in ``section`` that isn't in ``allowed``.

    Unknown keys never raise — warning-only so forks that extended configs
    with private keys keep loading. The log key ``config_unknown_key``
    makes the messages trivial to grep for in CI.

    Audit MEDIUM #4 — ``nested=True`` switches the log key to
    ``config_unknown_nested_key`` so operators can distinguish root-level
    typos (which are usually fatal operator errors) from sub-section
    typos (which are often silently ignored by downstream code). This
    matches the docstring promise on ``_warn_unknown_section_keys``.
    """
    if not isinstance(section, dict):
        return
    log_key = "config_unknown_nested_key" if nested else "config_unknown_key"
    for key in section:
        if key not in allowed:
            logger.warning(
                "%s section=%s key=%s",
                log_key,
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
        self._validate_supplier()
        self._validate_competitor()
        self._validate_market()
        self._validate_customer()
        self._validate_episode()
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
        seen_skus = set()
        for p in self.config["products"]:
            for pk in ["sku", "unit_cost", "sell_price", "initial_stock", "demand"]:
                if pk not in p:
                    raise ConfigValidationError(f"Product missing field '{pk}': {p}")
            sku = p.get("sku")
            if not isinstance(sku, str) or not sku:
                raise ConfigValidationError(f"Product has empty/non-string 'sku': {p!r}")
            if sku in seen_skus:
                raise ConfigValidationError(f"Duplicate product sku={sku!r}")
            seen_skus.add(sku)
            # Numeric sanity: prices/stocks must be non-negative real numbers.
            for numeric_key in ("unit_cost", "sell_price", "initial_stock"):
                try:
                    val = float(p[numeric_key])
                except (TypeError, ValueError):
                    raise ConfigValidationError(
                        f"Product '{sku}' has non-numeric '{numeric_key}': {p[numeric_key]!r}"
                    )
                if val < 0 or not math.isfinite(val):
                    raise ConfigValidationError(
                        f"Product '{sku}' has invalid '{numeric_key}': {val}"
                    )
            # Post-audit round-2 (A2-17) — ``sell_price`` and ``unit_cost``
            # must be strictly positive. A zero unit_cost makes every
            # supplier quote land at 0 and lets the agent restock an
            # arbitrary quantity for free; a zero sell_price collapses
            # revenue, breaks the competitor-ratio math in the demand
            # model, and makes ``price_ratio`` divide-by-zero.
            for positive_key in ("unit_cost", "sell_price"):
                if float(p[positive_key]) <= 0.0:
                    raise ConfigValidationError(
                        f"Product '{sku}' has non-positive '{positive_key}': "
                        f"{p[positive_key]!r} (must be > 0)"
                    )
            if float(p["sell_price"]) <= float(p["unit_cost"]):
                # Negative margin is rarely what the author intended;
                # surface it as a soft warning so audits catch it but
                # diagnostic / stress-test configs still load.
                logger.warning(
                    "config_soft_warn sku=%s sell_price=%s <= unit_cost=%s "
                    "(negative / zero margin)",
                    sku,
                    p["sell_price"],
                    p["unit_cost"],
                )
            # Post-audit round-2 (A2-54) — ``initial_stock`` must be a
            # strict integer (fractional units of inventory don't make
            # sense and would surface as fractional restock quantities).
            raw_init = p["initial_stock"]
            if isinstance(raw_init, bool) or not isinstance(raw_init, int):
                # Allow floats only if they are whole numbers (e.g. 20.0
                # written by a YAML-to-JSON converter) — reject anything
                # with a fractional part.
                try:
                    as_float = float(raw_init)
                except (TypeError, ValueError):
                    raise ConfigValidationError(
                        f"Product '{sku}' has non-integer 'initial_stock': {raw_init!r}"
                    )
                if not math.isfinite(as_float) or not float(as_float).is_integer():
                    raise ConfigValidationError(
                        f"Product '{sku}' has non-integer 'initial_stock': {raw_init!r}"
                    )
            # Post-audit P-4 — ``restock_lead_days`` is optional but when
            # set must be a non-negative integer. Post-audit round-2
            # (A2-54) strict-int check: fractional floats are rejected.
            lead = p.get("restock_lead_days")
            if lead is not None:
                if isinstance(lead, bool):
                    raise ConfigValidationError(
                        f"Product '{sku}' has non-integer 'restock_lead_days': {lead!r}"
                    )
                if not isinstance(lead, int):
                    try:
                        as_float = float(lead)
                    except (TypeError, ValueError):
                        raise ConfigValidationError(
                            f"Product '{sku}' has non-integer 'restock_lead_days': {lead!r}"
                        )
                    if not math.isfinite(as_float) or not float(as_float).is_integer():
                        raise ConfigValidationError(
                            f"Product '{sku}' has non-integer 'restock_lead_days': {lead!r}"
                        )
                    lead_i = int(as_float)
                else:
                    lead_i = int(lead)
                if lead_i < 0:
                    raise ConfigValidationError(
                        f"Product '{sku}' has negative 'restock_lead_days': {lead_i}"
                    )
            # Post-audit realism — optional per-product competitor-price
            # volatility. Must be a finite, non-negative float when set.
            vol = p.get("competitor_price_volatility")
            if vol is not None:
                try:
                    vol_f = float(vol)
                except (TypeError, ValueError):
                    raise ConfigValidationError(
                        f"Product '{sku}' has non-numeric 'competitor_price_volatility': {vol!r}"
                    )
                if vol_f < 0 or not math.isfinite(vol_f):
                    raise ConfigValidationError(
                        f"Product '{sku}' has invalid 'competitor_price_volatility': {vol_f}"
                    )
            # Seasonality, if provided, must be a non-empty numeric list.
            demand_cfg = p.get("demand", {}) or {}
            # Post-audit P-4 — demand numerics: ``base_units_per_day`` and
            # ``ad_elasticity`` (optional) must be non-negative finite floats.
            for dkey in ("base_units_per_day", "ad_elasticity"):
                if dkey in demand_cfg:
                    try:
                        dv = float(demand_cfg[dkey])
                    except (TypeError, ValueError):
                        raise ConfigValidationError(
                            f"Product '{sku}' has non-numeric demand.{dkey}: {demand_cfg[dkey]!r}"
                        )
                    if dv < 0 or not math.isfinite(dv):
                        raise ConfigValidationError(
                            f"Product '{sku}' has invalid demand.{dkey}: {dv}"
                        )
            weights = demand_cfg.get("seasonality_weights")
            if weights is not None:
                if not isinstance(weights, list) or not weights:
                    raise ConfigValidationError(
                        f"Product '{sku}' has invalid 'seasonality_weights' (must be non-empty list)"
                    )
                for w in weights:
                    try:
                        wf = float(w)
                    except (TypeError, ValueError):
                        raise ConfigValidationError(
                            f"Product '{sku}' has non-numeric seasonality weight: {w!r}"
                        )
                    if not math.isfinite(wf) or wf < 0:
                        raise ConfigValidationError(
                            f"Product '{sku}' has invalid seasonality weight: {wf!r}"
                        )
                # Post-audit round-2 (A2-18) — the demand model walks
                # ``day_of_week`` mod ``len(weights)``. Non-7 lengths
                # *work*, but the resulting rhythm is rarely what the
                # config author intended. Soft-warn so audits catch it.
                if len(weights) not in (0, 7):
                    logger.warning(
                        "config_soft_warn sku=%s seasonality_weights has "
                        "length=%d (expected 7 for weekly seasonality)",
                        sku,
                        len(weights),
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
        # Post-audit H-2 / R-1 — optional min ad spend.
        ad_min = actions_cfg.get("ad_spend_min_per_step")
        if ad_min is not None:
            try:
                ad_min_f = float(ad_min)
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"Config 'actions.ad_spend_min_per_step' must be numeric: {ad_min!r}"
                )
            if ad_min_f < 0 or not math.isfinite(ad_min_f):
                raise ConfigValidationError(
                    f"Config 'actions.ad_spend_min_per_step' must be >= 0: {ad_min!r}"
                )
            if ad_cap is not None:
                try:
                    if ad_min_f > float(ad_cap):
                        raise ConfigValidationError(
                            "actions.ad_spend_min_per_step must be <= "
                            f"actions.ad_spend_max_per_step ({ad_min_f} > {ad_cap})"
                        )
                except (TypeError, ValueError):
                    # ad_cap numeric-ness already checked above.
                    pass
        # Post-audit D-3 — max_ad_multiplier (ad-driven demand ceiling).
        # Post-audit round-2 (A2-23) — also enforce a hard absolute
        # ceiling coming from ``constants.MAX_AD_MULTIPLIER_HARD_CEILING``
        # so a typo (``50.0`` instead of ``5.0``) can never push the
        # Poisson lambda into the numerical danger zone.
        mam = actions_cfg.get("max_ad_multiplier")
        if mam is not None:
            try:
                mam_f = float(mam)
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"Config 'actions.max_ad_multiplier' must be numeric: {mam!r}"
                )
            if not math.isfinite(mam_f) or mam_f < 1.0:
                raise ConfigValidationError(
                    "actions.max_ad_multiplier must be a finite float >= 1.0 "
                    f"(got {mam!r})"
                )
            hard_ceil = float(constants.MAX_AD_MULTIPLIER_HARD_CEILING)
            if mam_f > hard_ceil:
                raise ConfigValidationError(
                    f"actions.max_ad_multiplier must be <= {hard_ceil} "
                    f"(got {mam_f}; hard ceiling protects the Poisson "
                    "lambda from numerical blowup)"
                )

        # Post-audit round-2 (A2-13) — optional per-step restock cap.
        # Zero is disallowed (would refuse every restock); negative and
        # non-integer values are rejected.
        rmq = actions_cfg.get("restock_max_qty_per_step")
        if rmq is not None:
            if isinstance(rmq, bool):
                raise ConfigValidationError(
                    "actions.restock_max_qty_per_step must be a positive "
                    f"integer: {rmq!r}"
                )
            try:
                rmq_i = int(rmq)
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    "actions.restock_max_qty_per_step must be an integer: "
                    f"{rmq!r}"
                )
            if rmq_i <= 0:
                raise ConfigValidationError(
                    "actions.restock_max_qty_per_step must be > 0: "
                    f"{rmq!r}"
                )

        # Post-audit round-2 (A2-57) — ``ad_spend_min_per_step`` without a
        # paired ``ad_spend_max_per_step`` is allowed (fallback to +inf),
        # but when the two are specified together we've already checked
        # ``min <= max`` above. No additional logic needed here; the
        # comment preserves the contract.

    def _validate_financials(self) -> None:
        fin = self.config.get("financials", {}) or {}
        try:
            initial_balance = float(fin.get("initial_bank_balance", 0))
        except (TypeError, ValueError):
            raise ConfigValidationError("'financials.initial_bank_balance' must be numeric")
        if initial_balance < 0:
            raise ConfigValidationError("'financials.initial_bank_balance' must be >= 0")

        # Post-audit B.2 (v2.3.x) — bankruptcy_threshold must be numeric and
        # finite when present. A stray string / NaN slipping through here
        # would poison every ``bank_balance <= threshold`` branch downstream.
        bankruptcy_threshold: Optional[float] = None
        if "bankruptcy_threshold" in fin:
            raw = fin["bankruptcy_threshold"]
            try:
                bankruptcy_threshold = float(raw)
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"'financials.bankruptcy_threshold' must be numeric: {raw!r}"
                )
            if not math.isfinite(bankruptcy_threshold):
                raise ConfigValidationError(
                    f"'financials.bankruptcy_threshold' must be finite: {raw!r}"
                )

        # Soft check: capital should start above the bankruptcy floor; otherwise
        # the episode starts already-bankrupt. We warn instead of raising so
        # exotic diagnostic configs (e.g. bankruptcy_threshold=initial for
        # stress testing) are not rejected outright.
        if bankruptcy_threshold is not None:
            floor = max(0.0, bankruptcy_threshold)
            if initial_balance < floor:
                logger.warning(
                    "config_soft_warn financials.initial_bank_balance (%s) is below "
                    "max(0, financials.bankruptcy_threshold) (%s); episodes will start "
                    "bankrupt.",
                    initial_balance,
                    floor,
                )
        if "inventory_holding_cost_per_unit_per_day" in fin:
            raw = fin["inventory_holding_cost_per_unit_per_day"]
            try:
                hv = float(raw)
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"'financials.inventory_holding_cost_per_unit_per_day' must be numeric: {raw!r}"
                )
            if not math.isfinite(hv) or hv < 0.0:
                raise ConfigValidationError(
                    f"'financials.inventory_holding_cost_per_unit_per_day' must be >= 0: {raw!r}"
                )

    def _validate_rewards(self) -> None:
        # Rewards table must be fully numeric; otherwise the first step call
        # will explode with an opaque TypeError deep inside the reward engine.
        rewards_cfg = self.config.get("rewards", {}) or {}
        if not isinstance(rewards_cfg, dict):
            raise ConfigValidationError("'rewards' must be a JSON object")
        _NON_NUMERIC_REWARD_KEYS = {
            "revenue_mode",
            # Post-audit round-2 (A2-29) — urgency map is a nested dict,
            # validated separately below. Exclude from the flat numeric
            # sweep so its dict value doesn't blow up ``float()``.
            "urgency_penalty_map",
            # Post-audit round-2 (A2-31) — scaled-ad-ROI flag is boolean.
            "ad_roi_scaled",
        }
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
        # Post-audit R-6 — ticket aging penalty cap must be a non-negative int.
        if "ticket_aging_penalty_cap" in rewards_cfg:
            raw = rewards_cfg["ticket_aging_penalty_cap"]
            try:
                cap_i = int(raw)
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"rewards.ticket_aging_penalty_cap must be an integer: {raw!r}"
                )
            if cap_i < 0:
                raise ConfigValidationError(
                    f"rewards.ticket_aging_penalty_cap must be >= 0: {raw!r}"
                )

        # Post-audit round-2 (A2-26) — reward-sign rules. Each coefficient
        # has a natural sign (e.g. bankruptcy_terminal must be a penalty,
        # solvency_per_step must be a bonus). A sign-inverted config
        # silently trains the policy in the wrong direction; we now reject
        # it at load time.
        for key, rule in _REWARD_SIGN_RULES.items():
            if key not in rewards_cfg:
                continue
            try:
                val = float(rewards_cfg[key])
            except (TypeError, ValueError):
                # Already rejected by the numeric sweep above; defensive.
                continue
            if not math.isfinite(val):
                continue
            if rule == "<= 0" and val > 0:
                raise ConfigValidationError(
                    f"rewards.{key} must be <= 0 (it is a penalty coefficient): {val}"
                )
            if rule == ">= 0" and val < 0:
                raise ConfigValidationError(
                    f"rewards.{key} must be >= 0 (it is a bonus coefficient): {val}"
                )

        # Post-audit round-2 (A2-29) — urgency_penalty_map must be a dict
        # of {str: non-positive float}. Empty dict is allowed (disables
        # the per-urgency override).
        upm = rewards_cfg.get("urgency_penalty_map")
        if upm is not None:
            if not isinstance(upm, dict):
                raise ConfigValidationError(
                    f"rewards.urgency_penalty_map must be an object: {upm!r}"
                )
            for lbl, coef in upm.items():
                if not isinstance(lbl, str) or not lbl:
                    raise ConfigValidationError(
                        f"rewards.urgency_penalty_map has invalid key: {lbl!r}"
                    )
                try:
                    cf = float(coef)
                except (TypeError, ValueError):
                    raise ConfigValidationError(
                        f"rewards.urgency_penalty_map[{lbl!r}] must be numeric: {coef!r}"
                    )
                if not math.isfinite(cf) or cf > 0:
                    raise ConfigValidationError(
                        f"rewards.urgency_penalty_map[{lbl!r}] must be <= 0: {cf}"
                    )
        # Post-audit round-2 (A2-31) — ad_roi_scaled is a bool flag.
        ars = rewards_cfg.get("ad_roi_scaled")
        if ars is not None and not isinstance(ars, bool):
            raise ConfigValidationError(
                f"rewards.ad_roi_scaled must be a boolean: {ars!r}"
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
        # Post-audit round-2 (A2-35) — ``hi == 0`` collapses every refund
        # to a free resolve. If ``rewards.refund_success`` is a positive
        # bonus (as shipped), that becomes a pure farming loop. Reject
        # any config that asks for positive refund bonus AND zero payout
        # simultaneously; also warn on the degenerate ``[0, 0]`` case.
        rewards_cfg = self.config.get("rewards", {}) or {}
        try:
            refund_bonus = float(rewards_cfg.get("refund_success", 0.0))
        except (TypeError, ValueError):
            refund_bonus = 0.0
        if hi_v <= 0.0:
            if refund_bonus > 0.0:
                raise ConfigValidationError(
                    "tickets.refund_amount_range upper bound must be > 0 "
                    "when rewards.refund_success > 0 (otherwise refunds "
                    "become a free reward-pump): "
                    f"range={rr!r}, refund_success={refund_bonus!r}"
                )
            logger.warning(
                "config_soft_warn tickets.refund_amount_range=[0, 0] "
                "disables refund payouts entirely"
            )

        # Post-audit round-2 (A2-36) — urgency_levels and urgency_weights
        # must be parallel lists with at least one positive weight.
        levels = tickets_cfg.get("urgency_levels")
        weights = tickets_cfg.get("urgency_weights")
        if levels is not None and not isinstance(levels, list):
            raise ConfigValidationError(
                f"tickets.urgency_levels must be a list: {levels!r}"
            )
        if weights is not None and not isinstance(weights, list):
            raise ConfigValidationError(
                f"tickets.urgency_weights must be a list: {weights!r}"
            )
        if levels is not None and weights is not None:
            if len(levels) != len(weights):
                raise ConfigValidationError(
                    "tickets.urgency_levels and urgency_weights must have "
                    f"the same length (got {len(levels)} vs {len(weights)})"
                )
        if weights is not None:
            try:
                numeric_weights = [float(w) for w in weights]
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"tickets.urgency_weights must be numeric: {weights!r}"
                )
            if any((not math.isfinite(w)) or w < 0 for w in numeric_weights):
                raise ConfigValidationError(
                    f"tickets.urgency_weights values must be >= 0 and finite: {weights!r}"
                )
            if numeric_weights and sum(numeric_weights) <= 0:
                raise ConfigValidationError(
                    f"tickets.urgency_weights must sum to > 0: {weights!r}"
                )

        # Post-audit round-2 (A2-37) — ``initial_count`` must be null or a
        # non-negative integer (not a string, not a float with fractional
        # part, etc.).
        ic = tickets_cfg.get("initial_count")
        if ic is not None:
            if isinstance(ic, bool):
                raise ConfigValidationError(
                    f"tickets.initial_count must be an integer or null: {ic!r}"
                )
            if isinstance(ic, float):
                if not math.isfinite(ic) or not ic.is_integer():
                    raise ConfigValidationError(
                        f"tickets.initial_count must be an integer or null: {ic!r}"
                    )
            elif not isinstance(ic, int):
                raise ConfigValidationError(
                    f"tickets.initial_count must be an integer or null: {ic!r}"
                )
            if int(ic) < 0:
                raise ConfigValidationError(
                    f"tickets.initial_count must be >= 0: {ic!r}"
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

        # Post-audit — optional stockout-driven churn multiplier.
        churn = tickets_cfg.get("stockout_churn_multiplier")
        if churn is not None:
            try:
                cf = float(churn)
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"tickets.stockout_churn_multiplier must be numeric: {churn!r}"
                )
            if not math.isfinite(cf) or cf < 0:
                raise ConfigValidationError(
                    f"tickets.stockout_churn_multiplier must be >= 0: {churn!r}"
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
        """Grader sanity — profit normalizer positive, inventory target
        sku present in products, inventory target_units positive.

        Post-audit L-1 — previously ``target_units`` was silently coerced
        via ``float(...)`` with no bound check. A ``target_units`` of
        ``0`` or a negative value would make the inventory grader always
        score max (``stock >= 0`` is trivially true for every SKU),
        quietly breaking the RL signal. We now raise at config load.
        """
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
        if "target_units" in inv_grader:
            raw = inv_grader["target_units"]
            try:
                tu = float(raw)
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"graders.inventory_task.target_units must be numeric: {raw!r}"
                )
            if not math.isfinite(tu) or tu <= 0:
                raise ConfigValidationError(
                    f"graders.inventory_task.target_units must be > 0: {raw!r}"
                )

    def _validate_supplier(self) -> None:
        """Post-audit S-4 / L-3 — enforce supplier numerics at load time.

        All keys are optional (defaults come from ``SupplierAgent``), but
        when a config sets one we require a numeric, finite value in the
        appropriate range. Unbounded or non-numeric entries previously
        flowed through to the ``SupplierAgent`` constructor and surfaced
        as opaque ``float('nan')`` quote prices at runtime.
        """
        supplier_cfg = self.config.get("supplier", {}) or {}
        if not isinstance(supplier_cfg, dict):
            raise ConfigValidationError("'supplier' must be a JSON object")
        # (key, type, min_value, max_value_or_None)
        numeric_specs = [
            ("volume_rate", "float", 0.0, None),
            ("demand_rate", "float", 0.0, None),
            ("price_cap_multiplier", "float", 1.0, None),
            ("volume_discount", "float", 0.0, 0.5),
            ("spot_premium", "float", 0.0, 1.0),
            ("volume_free_units", "int", 0, None),
            ("quote_expiry_steps", "int", 0, None),
        ]
        for key, kind, lo, hi in numeric_specs:
            if key not in supplier_cfg:
                continue
            raw = supplier_cfg[key]
            try:
                val = int(raw) if kind == "int" else float(raw)
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"supplier.{key} must be a {kind}: {raw!r}"
                )
            if kind == "float" and not math.isfinite(val):
                raise ConfigValidationError(
                    f"supplier.{key} must be finite: {raw!r}"
                )
            if val < lo:
                raise ConfigValidationError(
                    f"supplier.{key} must be >= {lo}: {val}"
                )
            if hi is not None and val > hi:
                raise ConfigValidationError(
                    f"supplier.{key} must be <= {hi}: {val}"
                )
        cap_map = supplier_cfg.get("capacity_per_sku")
        if cap_map is not None:
            if not isinstance(cap_map, dict):
                raise ConfigValidationError(
                    f"supplier.capacity_per_sku must be an object: {cap_map!r}"
                )
            for sku, cap in cap_map.items():
                if not isinstance(sku, str) or not sku:
                    raise ConfigValidationError(
                        f"supplier.capacity_per_sku has invalid sku key: {sku!r}"
                    )
                try:
                    cap_i = int(cap)
                except (TypeError, ValueError):
                    raise ConfigValidationError(
                        f"supplier.capacity_per_sku[{sku!r}] must be an integer: {cap!r}"
                    )
                if cap_i < 0:
                    raise ConfigValidationError(
                        f"supplier.capacity_per_sku[{sku!r}] must be >= 0: {cap_i!r}"
                    )

    def _validate_competitor(self) -> None:
        cfg = self.config.get("competitor", {}) or {}
        if not isinstance(cfg, dict):
            raise ConfigValidationError("'competitor' must be a JSON object")
        for bool_key in ("reactive_enabled",):
            val = cfg.get(bool_key)
            if val is not None and not isinstance(val, bool):
                raise ConfigValidationError(f"competitor.{bool_key} must be boolean: {val!r}")
        for num_key in (
            "reactive_undercut_multiplier",
            "reactive_follow_up_multiplier",
            "reactive_deadzone_multiplier",
        ):
            if num_key not in cfg:
                continue
            try:
                nv = float(cfg[num_key])
            except (TypeError, ValueError):
                raise ConfigValidationError(f"competitor.{num_key} must be numeric: {cfg[num_key]!r}")
            if not math.isfinite(nv) or nv < 0:
                raise ConfigValidationError(f"competitor.{num_key} must be finite and >= 0: {nv!r}")

    def _validate_market(self) -> None:
        cfg = self.config.get("market", {}) or {}
        if not isinstance(cfg, dict):
            raise ConfigValidationError("'market' must be a JSON object")
        val = cfg.get("shock_enabled")
        if val is not None and not isinstance(val, bool):
            raise ConfigValidationError(f"market.shock_enabled must be boolean: {val!r}")
        for num_key in ("shock_probability", "shock_min_multiplier", "shock_max_multiplier"):
            if num_key not in cfg:
                continue
            try:
                nv = float(cfg[num_key])
            except (TypeError, ValueError):
                raise ConfigValidationError(f"market.{num_key} must be numeric: {cfg[num_key]!r}")
            if not math.isfinite(nv):
                raise ConfigValidationError(f"market.{num_key} must be finite: {cfg[num_key]!r}")
        prob = float(cfg.get("shock_probability", constants.DEFAULT_MARKET_SHOCK_PROBABILITY) or 0.0)
        if prob < 0.0 or prob > 1.0:
            raise ConfigValidationError(f"market.shock_probability must be in [0, 1]: {prob!r}")
        lo = float(cfg.get("shock_min_multiplier", constants.DEFAULT_MARKET_SHOCK_MIN_MULTIPLIER) or 0.0)
        hi = float(cfg.get("shock_max_multiplier", constants.DEFAULT_MARKET_SHOCK_MAX_MULTIPLIER) or 0.0)
        if lo <= 0.0 or hi <= 0.0 or hi < lo:
            raise ConfigValidationError(
                f"market shock multipliers must satisfy 0 < min <= max (got min={lo}, max={hi})"
            )
        if "shock_duration_days" in cfg:
            try:
                dd = int(cfg["shock_duration_days"])
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    f"market.shock_duration_days must be integer: {cfg['shock_duration_days']!r}"
                )
            if dd < 1:
                raise ConfigValidationError("market.shock_duration_days must be >= 1")
        sku_mult = cfg.get("shock_sku_multipliers")
        if sku_mult is not None:
            if not isinstance(sku_mult, dict):
                raise ConfigValidationError("market.shock_sku_multipliers must be an object")
            for sku, mv in sku_mult.items():
                if not isinstance(sku, str) or not sku:
                    raise ConfigValidationError("market.shock_sku_multipliers has invalid sku key")
                try:
                    mf = float(mv)
                except (TypeError, ValueError):
                    raise ConfigValidationError(
                        f"market.shock_sku_multipliers[{sku!r}] must be numeric: {mv!r}"
                    )
                if not math.isfinite(mf) or mf <= 0:
                    raise ConfigValidationError(
                        f"market.shock_sku_multipliers[{sku!r}] must be > 0: {mf!r}"
                    )

    def _validate_customer(self) -> None:
        cfg = self.config.get("customer", {}) or {}
        if not isinstance(cfg, dict):
            raise ConfigValidationError("'customer' must be a JSON object")
        val = cfg.get("satisfaction_enabled")
        if val is not None and not isinstance(val, bool):
            raise ConfigValidationError(f"customer.satisfaction_enabled must be boolean: {val!r}")
        for num_key in (
            "satisfaction_initial",
            "satisfaction_min",
            "satisfaction_max",
            "stockout_penalty",
            "open_ticket_penalty",
            "daily_recovery",
        ):
            if num_key not in cfg:
                continue
            try:
                nv = float(cfg[num_key])
            except (TypeError, ValueError):
                raise ConfigValidationError(f"customer.{num_key} must be numeric: {cfg[num_key]!r}")
            if not math.isfinite(nv):
                raise ConfigValidationError(f"customer.{num_key} must be finite: {cfg[num_key]!r}")
        sat_min = float(cfg.get("satisfaction_min", constants.DEFAULT_CUSTOMER_SATISFACTION_MIN))
        sat_max = float(cfg.get("satisfaction_max", constants.DEFAULT_CUSTOMER_SATISFACTION_MAX))
        if sat_min < 0.0 or sat_max > 1.0 or sat_max < sat_min:
            raise ConfigValidationError(
                f"customer satisfaction bounds must satisfy 0 <= min <= max <= 1 (got {sat_min}, {sat_max})"
            )

    def _validate_episode(self) -> None:
        """Post-audit round-2 (A2-44) — soft-warn on tiny episode horizons.

        ``max_steps`` < 10 OR shorter than 4x the longest ``restock_lead_days``
        means the policy can't realistically see a restock land before
        ``done`` fires, so the inventory-management signal collapses.
        This is a config-authoring hint, not a hard failure.
        """
        episode_cfg = self.config.get("episode", {}) or {}
        try:
            max_steps = int(episode_cfg.get("max_steps", 50))
        except (TypeError, ValueError):
            return
        if max_steps <= 0:
            raise ConfigValidationError(
                f"episode.max_steps must be > 0: {max_steps!r}"
            )
        if max_steps < 10:
            logger.warning(
                "config_soft_warn episode.max_steps=%d is < 10; the agent "
                "may not see enough variety to train.",
                max_steps,
            )
        products = self.config.get("products", []) or []
        max_lead = 0
        for p in products:
            try:
                lead = int(p.get("restock_lead_days", 0) or 0)
            except (TypeError, ValueError):
                lead = 0
            if lead > max_lead:
                max_lead = lead
        if max_lead > 0 and max_steps < max_lead * 4:
            logger.warning(
                "config_soft_warn episode.max_steps=%d is < 4 * max "
                "restock_lead_days=%d (=%d); policies will have a hard "
                "time observing deliveries land.",
                max_steps,
                max_lead,
                max_lead * 4,
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
        # Post-audit B.1 (v2.3.x) — if price bounds are configured, they must
        # be strictly positive and form a valid lo<=hi interval; otherwise the
        # SetPriceAction path degenerates into an empty feasible set and the
        # policy silently collapses to the lower bound.
        actions_cfg = self.config.get("actions", {}) or {}
        pmin = actions_cfg.get("price_min_mult_competitor")
        pmax = actions_cfg.get("price_max_mult_competitor")
        if pmin is not None or pmax is not None:
            try:
                pmin_f = float(pmin) if pmin is not None else None
                pmax_f = float(pmax) if pmax is not None else None
            except (TypeError, ValueError):
                raise ConfigValidationError(
                    "actions.price_min_mult_competitor / "
                    "actions.price_max_mult_competitor must be numeric when set "
                    f"(got min={pmin!r} max={pmax!r})"
                )
            if pmin_f is not None and pmin_f <= 0:
                raise ConfigValidationError(
                    f"actions.price_min_mult_competitor must be > 0: {pmin_f}"
                )
            if pmax_f is not None and pmax_f <= 0:
                raise ConfigValidationError(
                    f"actions.price_max_mult_competitor must be > 0: {pmax_f}"
                )
            if pmin_f is not None and pmax_f is not None and pmin_f > pmax_f:
                raise ConfigValidationError(
                    "actions.price_min_mult_competitor must be <= "
                    f"actions.price_max_mult_competitor (got {pmin_f} > {pmax_f})"
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
        # Post-audit round-2 (A2-12) — ``bankruptcy_threshold`` MUST live
        # under ``financials``. A config that only sets
        # ``rewards.bankruptcy_threshold`` (without the canonical
        # financials copy) used to "work" thanks to the mirror, but that
        # splits the source of truth between two sections. Force the
        # canonical placement now; the dedicated deprecation warning
        # still fires for the rewards mirror when both are set.
        if (
            "bankruptcy_threshold" in rewards_cfg
            and "bankruptcy_threshold" not in fin_cfg
        ):
            raise ConfigValidationError(
                "rewards.bankruptcy_threshold is set but the canonical "
                "'financials.bankruptcy_threshold' is missing — "
                "bankruptcy_threshold must live on the financials section "
                "(the rewards mirror is deprecated)."
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
        """Post-audit m-2 — emit WARNINGs for typos / unknown config keys.

        Post-audit round-2 (A2-59) — we now also walk one level *into*
        known sections whose values are dicts with unrecognised sub-keys.
        Recursion depth is capped at 2 so the traversal is O(config) even
        on pathologically nested forks. Unknown nested keys emit a
        ``config_unknown_nested_key`` warning (distinct log key so it can
        be filtered independently in CI).
        """
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
            "competitor", self.config.get("competitor", {}), _KNOWN_COMPETITOR_KEYS
        )
        _warn_unknown_keys(
            "market", self.config.get("market", {}), _KNOWN_MARKET_KEYS
        )
        _warn_unknown_keys(
            "customer", self.config.get("customer", {}), _KNOWN_CUSTOMER_KEYS
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
            nested=True,
        )
        _warn_unknown_keys(
            "graders.inventory_task",
            graders_cfg.get("inventory_task", {}),
            _KNOWN_GRADER_INVENTORY_KEYS,
            nested=True,
        )
        _warn_unknown_keys(
            "graders.profit_task",
            graders_cfg.get("profit_task", {}),
            _KNOWN_GRADER_PROFIT_KEYS,
            nested=True,
        )
        for p in self.config.get("products", []):
            sku = p.get("sku", "?")
            _warn_unknown_keys(f"products[{sku}]", p, _KNOWN_PRODUCT_KEYS)
            _warn_unknown_keys(
                f"products[{sku}].demand",
                p.get("demand", {}),
                _KNOWN_DEMAND_KEYS,
                nested=True,
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
        self._competitor_cfg: Dict = dict(self.config.get("competitor", {}) or {})
        self._market_cfg: Dict = dict(self.config.get("market", {}) or {})
        self._customer_cfg: Dict = dict(self.config.get("customer", {}) or {})
        self._supplier_capacity_per_sku: Dict[str, int] = {
            str(k): int(v)
            for k, v in ((self.config.get("supplier", {}) or {}).get("capacity_per_sku", {}) or {}).items()
            if isinstance(k, str)
        }

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

        # Post-audit round-2 (A2-27) — normalise ``revenue_mode`` once so
        # every downstream reader sees a canonical lower-case string.
        mode_raw = self._rewards_cfg.get("revenue_mode", "linear")
        self._rewards_cfg["revenue_mode"] = str(mode_raw or "linear").lower()

        # Post-audit round-2 (A2-45) — cache the per-SKU restock lead-day
        # map so ``_get_lead_days`` is O(1) instead of a linear scan of
        # ``config["products"]``. Built from the same products list; on
        # hot-swap via ``_build_lookup_tables`` the cache is refreshed.
        self._lead_days: Dict[str, int] = {}
        for p in products:
            sku = p.get("sku")
            if not isinstance(sku, str):
                continue
            try:
                ld = int(p.get("restock_lead_days", 0) or 0)
            except (TypeError, ValueError):
                ld = 0
            self._lead_days[sku] = max(0, ld)

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

    def _walk_competitor_prices(self) -> Dict[str, float]:
        """Drift each SKU's competitor price by a small Gaussian nudge.

        Per-product volatility comes from ``products[*].competitor_price_volatility``.
        A volatility of ``0.0`` (default) keeps the competitor static, preserving
        backward compatibility with pre-audit configs. When volatility is
        positive, today's competitor price = ``yesterday * (1 + normal(0, vol))``,
        clamped to a multiplicative band around the original config value
        so the walk cannot drift to zero or ∞.

        The band bounds come from ``env.constants.COMPETITOR_PRICE_BAND_LO``
        and ``COMPETITOR_PRICE_BAND_HI`` and are deliberately generous
        (0.5x .. 2.0x) so a 50-step episode with ``volatility=0.02`` still
        spends most of its time inside the band.
        """
        state_prices: Dict[str, float] = self.state.setdefault(
            "competitor_prices", {}
        )
        # Per-SKU relative price delta observed this step (post / pre − 1).
        # Returned so ``step`` can surface the magnitude of the random walk
        # in the explainability payload without re-deriving from a snapshot.
        deltas: Dict[str, float] = {}
        for p in self.config.get("products", []) or []:
            sku = p.get("sku")
            if sku is None:
                continue
            try:
                vol = float(p.get("competitor_price_volatility", 0.0) or 0.0)
            except (TypeError, ValueError):
                vol = 0.0
            if vol <= 0.0 or not math.isfinite(vol):
                continue
            base = float(self.competitor_prices.get(sku, state_prices.get(sku, 0.0)))
            if base <= 0.0 or not math.isfinite(base):
                continue
            current = float(state_prices.get(sku, base))
            # Bounded Gaussian nudge. Clamp the standard deviation at 0.5
            # regardless of config to prevent a mis-tuned value from
            # producing 90% daily swings.
            sigma = min(0.5, vol)
            shock = float(self._np_rng.normal(0.0, sigma))
            # Cap the per-step relative change at +/-3σ so an unlucky
            # draw doesn't push the price outside the band in one hop.
            shock = max(-3.0 * sigma, min(3.0 * sigma, shock))
            nxt = current * (1.0 + shock)
            lo = base * float(constants.COMPETITOR_PRICE_BAND_LO)
            hi = base * float(constants.COMPETITOR_PRICE_BAND_HI)
            # Post-audit round-2 (A2-47) — keep full precision in state
            # so a long episode doesn't accumulate ``round(x, 2)``
            # discretisation drift. The observation layer rounds on
            # serialisation (see ``EcomObservation.competitor_prices``).
            final = max(lo, min(hi, nxt))
            if current > 0.0 and math.isfinite(current):
                deltas[sku] = (final - current) / current
            state_prices[sku] = final
        return deltas

    def _reactive_competitor_step(self, action: Dict) -> None:
        """Optional reactive competitor policy applied before demand generation.

        When enabled, a ``set_price`` action nudges the competitor price for
        that SKU to undercut/follow our move. This is additive to the random
        walk and preserves determinism through env-local RNG.

        Post-audit remediation (explainability-truth wave) — always writes a
        ``_competitor_reaction_flag`` marker into ``self.state`` describing
        whether a *causal* reaction fired and why. ``step`` lifts the flag
        into ``info["competitor_reaction"]`` so inference can emit the
        ``competitor_action`` narrative only when a true reaction occurred;
        random-walk drift is narrated separately as ``competitor_walk``.

        Reason codes:
            * ``"disabled"``            — reactive policy off in the config.
            * ``"no_set_price_action"`` — our action wasn't set_price.
            * ``"invalid_state"``       — SKU unknown / non-positive price.
            * ``"undercut"``            — we dropped below the deadzone; the
                                          competitor undercut our price.
            * ``"follow"``              — we raised above the deadzone; the
                                          competitor followed us up.
            * ``"deadzone"``            — we stayed inside the deadzone; the
                                          small mean-reverting jitter fires
                                          but is NOT a true reaction, so
                                          ``triggered`` stays False.
        """
        # Default marker: not triggered. Keys are always present so the
        # info payload is stable even when the config opts out.
        flag: Dict[str, object] = {
            "triggered": False,
            "reason": "none",
            "magnitude": 0.0,
            "sku": None,
            "our_price": None,
            "competitor_before": None,
            "competitor_after": None,
        }
        try:
            cfg = self._competitor_cfg if isinstance(getattr(self, "_competitor_cfg", None), dict) else {}
            enabled = bool(cfg.get("reactive_enabled", constants.DEFAULT_REACTIVE_COMPETITOR_ENABLED))
            if not enabled:
                flag["reason"] = "disabled"
                return
            if (action or {}).get("action_type") != "set_price":
                flag["reason"] = "no_set_price_action"
                return
            sku = action.get("sku")
            if not isinstance(sku, str) or not sku:
                flag["reason"] = "invalid_state"
                return
            prices = self.state.get("prices", {}) or {}
            comp = self.state.get("competitor_prices", {}) or {}
            if sku not in prices or sku not in comp:
                flag["reason"] = "invalid_state"
                flag["sku"] = sku
                return
            our_price = float(prices.get(sku, 0.0) or 0.0)
            current_comp = float(comp.get(sku, 0.0) or 0.0)
            base = float(self.competitor_prices.get(sku, current_comp))
            if our_price <= 0.0 or current_comp <= 0.0 or base <= 0.0:
                flag["reason"] = "invalid_state"
                flag["sku"] = sku
                return
            undercut = float(cfg.get("reactive_undercut_multiplier", constants.DEFAULT_REACTIVE_UNDERCUT_MULTIPLIER))
            follow_up = float(cfg.get("reactive_follow_up_multiplier", constants.DEFAULT_REACTIVE_FOLLOW_UP_MULTIPLIER))
            deadzone = float(cfg.get("reactive_deadzone_multiplier", constants.DEFAULT_REACTIVE_DEADZONE_MULTIPLIER))
            deadzone = max(0.0, deadzone)
            upper_dead = current_comp * (1.0 + deadzone)
            lower_dead = current_comp * (1.0 - deadzone)
            triggered = False
            if our_price < lower_dead:
                target = our_price * max(0.5, undercut)
                reason = "undercut"
                triggered = True
            elif our_price > upper_dead:
                target = our_price * max(0.5, follow_up)
                reason = "follow"
                triggered = True
            else:
                # small mean-reverting nudge to avoid flat locks.
                # This is NOT a true causal reaction — flag stays False.
                jitter = float(self._np_rng.normal(0.0, 0.005))
                target = current_comp * (1.0 + max(-0.02, min(0.02, jitter)))
                reason = "deadzone"
            lo = base * float(constants.COMPETITOR_PRICE_BAND_LO)
            hi = base * float(constants.COMPETITOR_PRICE_BAND_HI)
            final_comp = max(lo, min(hi, target))
            magnitude = 0.0
            if current_comp > 0.0:
                magnitude = (final_comp - current_comp) / current_comp
            comp[sku] = final_comp
            flag.update({
                "triggered": bool(triggered),
                "reason": reason,
                "magnitude": float(magnitude),
                "sku": sku,
                "our_price": round(our_price, 4),
                "competitor_before": round(current_comp, 4),
                "competitor_after": round(final_comp, 4),
            })
        finally:
            # Always write the marker (even on early-return) so ``step``
            # always has something deterministic to lift into ``info``.
            self.state["_competitor_reaction_flag"] = flag

    def _market_event_multiplier_by_sku(self) -> Dict[str, float]:
        """Return per-SKU demand multipliers from optional market shocks."""
        skus = list((self.state.get("inventory", {}) or {}).keys())
        out = {sku: 1.0 for sku in skus}
        cfg = self._market_cfg if isinstance(getattr(self, "_market_cfg", None), dict) else {}
        if not bool(cfg.get("shock_enabled", constants.DEFAULT_MARKET_SHOCK_ENABLED)):
            return out

        active = self.state.get("_active_market_shock")
        if isinstance(active, dict):
            remaining = int(active.get("remaining_days", 0) or 0)
            mults = active.get("sku_multipliers", {}) or {}
            if remaining > 0 and isinstance(mults, dict):
                for sku in skus:
                    out[sku] = float(mults.get(sku, 1.0) or 1.0)
                active["remaining_days"] = remaining - 1
                return out

        prob = float(cfg.get("shock_probability", constants.DEFAULT_MARKET_SHOCK_PROBABILITY) or 0.0)
        prob = max(0.0, min(1.0, prob))
        if self._py_rng.random() > prob:
            self.state["_active_market_shock"] = {"remaining_days": 0, "sku_multipliers": {}}
            return out
        lo = float(cfg.get("shock_min_multiplier", constants.DEFAULT_MARKET_SHOCK_MIN_MULTIPLIER) or 1.0)
        hi = float(cfg.get("shock_max_multiplier", constants.DEFAULT_MARKET_SHOCK_MAX_MULTIPLIER) or 1.0)
        duration = int(cfg.get("shock_duration_days", constants.DEFAULT_MARKET_SHOCK_DURATION_DAYS) or 1)
        duration = max(1, duration)
        sku_base = cfg.get("shock_sku_multipliers", {}) or {}
        mults: Dict[str, float] = {}
        for sku in skus:
            raw = float(self._py_rng.uniform(lo, hi))
            if isinstance(sku_base, dict) and sku in sku_base:
                raw *= float(sku_base.get(sku, 1.0) or 1.0)
            mults[sku] = max(0.1, min(3.0, raw))
            out[sku] = mults[sku]
        self.state["_active_market_shock"] = {
            "remaining_days": duration - 1,
            "sku_multipliers": dict(mults),
        }
        return out

    def _peek_customer_satisfaction(self) -> float:
        """Return the current satisfaction scalar without mutating state.

        Used by ``_simulate_day`` to thread a demand multiplier through
        ``generate_all_demand`` before the end-of-day update. Calling
        ``_update_customer_satisfaction`` twice per tick was an audit
        MEDIUM bug: the open-ticket penalty and recovery were applied
        both pre- and post-sales, double-counting the decay/recovery.
        The peek helper fixes that by reading state without mutating.
        """
        cfg = self._customer_cfg if isinstance(getattr(self, "_customer_cfg", None), dict) else {}
        enabled = bool(cfg.get("satisfaction_enabled", constants.DEFAULT_CUSTOMER_SATISFACTION_ENABLED))
        if not enabled:
            return 1.0
        return float(
            self.state.get(
                "customer_satisfaction",
                cfg.get("satisfaction_initial", 1.0),
            ) or 1.0
        )

    def _update_customer_satisfaction(self, stockout_events: int) -> float:
        """Update and return simple bounded customer satisfaction scalar.

        Mutates ``state["customer_satisfaction"]`` with today's decay and
        recovery. Call **once per simulated day** (audit MEDIUM #3 fix).
        """
        cfg = self._customer_cfg if isinstance(getattr(self, "_customer_cfg", None), dict) else {}
        enabled = bool(cfg.get("satisfaction_enabled", constants.DEFAULT_CUSTOMER_SATISFACTION_ENABLED))
        if not enabled:
            self.state["customer_satisfaction"] = 1.0
            return 1.0
        sat_min = float(cfg.get("satisfaction_min", constants.DEFAULT_CUSTOMER_SATISFACTION_MIN))
        sat_max = float(cfg.get("satisfaction_max", constants.DEFAULT_CUSTOMER_SATISFACTION_MAX))
        sat = float(self.state.get("customer_satisfaction", cfg.get("satisfaction_initial", 1.0)) or 1.0)
        stockout_pen = float(cfg.get("stockout_penalty", constants.DEFAULT_CUSTOMER_SATISFACTION_STOCKOUT_PENALTY) or 0.0)
        ticket_pen = float(cfg.get("open_ticket_penalty", constants.DEFAULT_CUSTOMER_SATISFACTION_OPEN_TICKET_PENALTY) or 0.0)
        recovery = float(cfg.get("daily_recovery", constants.DEFAULT_CUSTOMER_SATISFACTION_DAILY_RECOVERY) or 0.0)
        open_tickets = sum(
            1 for t in (self.state.get("active_tickets") or [])
            if isinstance(t, dict) and t.get("status") == "open"
        )
        sat -= max(0.0, stockout_pen) * float(max(0, stockout_events))
        sat -= max(0.0, ticket_pen) * float(max(0, open_tickets))
        if stockout_events == 0:
            sat += max(0.0, recovery)
        sat = max(sat_min, min(sat_max, sat))
        self.state["customer_satisfaction"] = sat
        return sat

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
        """Return the ``restock_lead_days`` for a SKU, defaulting to 0.

        Post-audit round-2 (A2-45) — O(1) lookup backed by the
        pre-built ``_lead_days`` dict. Falls back to a linear scan on
        the (legacy) chance the cache is missing.
        """
        cache = getattr(self, "_lead_days", None)
        if isinstance(cache, dict) and sku in cache:
            return int(cache[sku])
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
    def reseed(self, seed: int) -> None:
        """Reseed the per-env RNGs without touching simulation state.

        Post-audit round-2 (A2-2) — exposed as a public method so
        ``EcomEnv.seed`` can reseed without calling ``reset``.
        """
        self._py_rng.seed(int(seed))
        self._np_rng = np.random.default_rng(int(seed))

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
            # Post-audit M-3 — quantity each live quote is bound to; drives
            # the "quote covers the first N units, overflow pays spot" split
            # inside ``env.actions.do_restock``. Populated by ``do_negotiate``.
            "supplier_quoted_qty": {},
            "customer_satisfaction": float(
                (self._customer_cfg or {}).get(
                    "satisfaction_initial",
                    constants.DEFAULT_CUSTOMER_SATISFACTION_INITIAL,
                )
            ),
            "_active_market_shock": {"remaining_days": 0, "sku_multipliers": {}},
            # v2.3 Phase 4.1 — pending deliveries schedule. Keys are SKUs,
            # values are lists of ``(delivery_day, quantity)`` tuples. On
            # every ``_simulate_day`` entry we drain any entry whose day has
            # arrived into ``inventory`` and decrement ``pending_orders``.
            "pending_deliveries": {sku: [] for sku in skus},
            # Post-audit round-2 (A2-39) — monotonic ticket-id counter
            # seeded to the number of initial tickets so spawn_daily_tickets
            # never reuses an id even after retention-pruning drops older
            # resolved tickets from ``active_tickets``.
            "_ticket_id_hwm": [len(active_tickets)],
            "reward": 0.0,
            "done": False,
        }
        return self._snapshot_state()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        # Post-audit B-1 — reject any ``step`` call issued *after* the
        # episode already terminated. Previously the engine quietly
        # advanced the simulated day, spawned tickets, and accumulated
        # revenue past ``done=True``, which (a) broke the OpenEnv-style
        # expectation that clients should ``/reset`` after ``done`` and
        # (b) made off-by-one comparisons between training harnesses
        # dependent on whether they called ``step`` one extra time.
        # We now short-circuit with a no-op that preserves the final
        # snapshot and flags the error in ``info`` — the state is NOT
        # mutated and no rewards are emitted.
        if bool(self.state.get("done")):
            snapshot = self._snapshot_state()
            return (
                snapshot,
                0.0,
                True,
                {
                    "error": "episode_terminated",
                    "hint": "call /reset before the next /step",
                },
            )
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
        # Post-audit round-2 (A2-11) — capture the target-SKU stock
        # *before* both the action handler runs and the day simulation
        # advances (which applies pending-order deliveries). The reward
        # engine compares this to ``stock_after`` when deciding whether
        # the inventory-target bonus can be attributed to deliberate
        # action on this step vs. a passive drift from an earlier
        # negotiate→wait→wait sequence.
        inv_target_sku = str(self._reward_shaping_ctx.get("inventory_target_sku", ""))
        target_stock_pre_step = 0
        if inv_target_sku:
            try:
                target_stock_pre_step = int(
                    self.state.get("inventory", {}).get(inv_target_sku, 0) or 0
                )
            except (TypeError, ValueError):
                target_stock_pre_step = 0
        base_reward, info = self._process_action(action)
        # Phase F.2 — thread daily_revenue back out of the day simulation so
        # the reward engine doesn't recompute it from state_after.
        daily_revenue = self._simulate_day(action)
        info["inventory_holding_cost"] = float(
            self.state.get("daily_inventory_holding_cost", 0.0) or 0.0
        )
        info["customer_satisfaction"] = float(
            self.state.get("customer_satisfaction", 1.0) or 1.0
        )
        info["market_shock"] = dict(
            self.state.get("_active_market_shock", {}) or {}
        )
        # Explainability-truth (Wave 1) — lift the per-step competitor
        # reaction marker set by ``_reactive_competitor_step`` into info.
        # Fall back to a stable "not triggered" payload so downstream
        # consumers never see an undefined key.
        info["competitor_reaction"] = dict(
            self.state.pop("_competitor_reaction_flag", {})
            or {
                "triggered": False,
                "reason": "none",
                "magnitude": 0.0,
                "sku": None,
            }
        )
        info["competitor_walk"] = {
            sku: round(float(delta), 6)
            for sku, delta in (self.state.pop("_competitor_walk_deltas", {}) or {}).items()
        }
        # Wave 2 — demand decomposition per SKU. Physics untouched.
        info["demand_factors"] = {
            sku: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in fac.items()}
            for sku, fac in (self.state.pop("_demand_factors_last", {}) or {}).items()
        }
        info["market_event_multipliers"] = {
            sku: round(float(m), 6)
            for sku, m in (self.state.pop("_market_mult_last", {}) or {}).items()
        }
        # Gap 1 — expose the satisfaction scalar that modulated today's
        # demand (distinct from ``info.customer_satisfaction`` which is
        # the end-of-day *post-mutation* value). Naming is explicit so
        # downstream consumers can't confuse the two.
        info["satisfaction_for_demand"] = round(
            float(self.state.pop("_satisfaction_for_demand", 1.0) or 1.0), 6
        )
        self.state["step_count"] += 1

        # Post-audit round-2 (A2-11) — compute the NET landed units on
        # the target SKU attributable to *this* step. The reward engine
        # accepts a positive landing as attribution for the inventory
        # bonus (covers the "negotiate→restock→wait until delivery"
        # pattern where the delivery day, not the restock day, is the
        # one that crosses the threshold).
        target_sku_net_landed_units = 0
        if inv_target_sku:
            try:
                target_stock_post_step = int(
                    self.state.get("inventory", {}).get(inv_target_sku, 0) or 0
                )
            except (TypeError, ValueError):
                target_stock_post_step = target_stock_pre_step
            # Net = (after day simulation) − (before action). A positive
            # delta means inventory landed (either via this step's
            # restock or a previously-scheduled delivery). Consumption
            # during the day is *netted out*, matching the intent of
            # the reward-shaping rule.
            target_sku_net_landed_units = max(
                0, target_stock_post_step - target_stock_pre_step
            )

        # Phase G.1 — request a per-term breakdown so callers can log the
        # dynamics of each shaping signal without changing the scalar reward.
        # v2.3 Phase 1.1 / post-audit M-2 + round-2 (A2-11, A2-14, A2-32) —
        # thread action context through to the reward engine.
        total_reward, breakdown = compute_step_reward(
            action_result={
                "base_reward": base_reward,
                "daily_revenue": daily_revenue,
                "ad_spend_applied": info.get("ad_spend_applied", {}),
                "restock_cost": info.get("restock_cost", 0.0),
                "restock_cost_amortised": info.get(
                    "restock_cost_amortised", info.get("restock_cost", 0.0)
                ),
                "restock_cost_punitive": info.get("restock_cost_punitive", 0.0),
                "refund_payout": info.get("refund_payout", 0.0),
                "restock_sku": (info.get("restock") or {}).get("sku", ""),
                "target_sku_net_landed_units": int(target_sku_net_landed_units),
                # Audit MEDIUM #6 — thread the action metadata so
                # ``_solvency_term`` can credit ad_spend / negotiate /
                # set_price (which have zero base reward in shipped
                # configs but are genuine productive actions) without
                # re-opening the ``wait``-loop farm. ``action_error``
                # is the reason string emitted by failing handlers.
                "action_type": str(action.get("action_type", "wait")),
                "action_error": info.get("error"),
            },
            state_before=state_before,
            state_after=self.state,
            rewards_config=self._rewards_cfg,
            return_breakdown=True,
            grader_context=self._reward_shaping_ctx,
        )
        info["reward_breakdown"] = breakdown

        # Wave 2 — action → effect trace. Pure diff of the pre-action
        # snapshot vs post-simulation state. No physics changes.
        inventory_before_total = 0
        inventory_after_total = 0
        try:
            inv_before_map = dict(state_before.get("inventory", {}) or {})
            inv_after_map = dict(self.state.get("inventory", {}) or {})
            tickets_before = state_before.get("active_tickets", []) or []
            tickets_after = self.state.get("active_tickets", []) or []
            open_before = sum(
                1 for t in tickets_before
                if isinstance(t, dict) and t.get("status") == "open"
            )
            open_after = sum(
                1 for t in tickets_after
                if isinstance(t, dict) and t.get("status") == "open"
            )
            # Gap 2 — split ``tickets_change`` into the two independent
            # causes so explainers can distinguish "agent triaged 2
            # refunds" from "2 fresh churn tickets spawned". We
            # identify tickets by ``ticket_id`` so we catch resolves
            # even when spawns offset the open-count delta exactly.
            before_ids = {
                str(t.get("ticket_id"))
                for t in tickets_before
                if isinstance(t, dict) and t.get("ticket_id") is not None
            }
            before_open_ids = {
                str(t.get("ticket_id"))
                for t in tickets_before
                if isinstance(t, dict)
                and t.get("ticket_id") is not None
                and t.get("status") == "open"
            }
            after_by_id = {
                str(t.get("ticket_id")): t
                for t in tickets_after
                if isinstance(t, dict) and t.get("ticket_id") is not None
            }
            tickets_spawned = sum(
                1 for tid in after_by_id.keys() if tid not in before_ids
            )
            tickets_resolved = sum(
                1
                for tid in before_open_ids
                if tid in after_by_id
                and after_by_id[tid].get("status") == "resolved"
            )
            inventory_before_total = sum(int(v or 0) for v in inv_before_map.values())
            inventory_after_total = sum(int(v or 0) for v in inv_after_map.values())
            sales_total = sum(
                int(v or 0) for v in (self.state.get("daily_sales", {}) or {}).values()
            )
            bank_before_val = float(state_before.get("bank_balance", 0.0) or 0.0)
            bank_after_val = float(self.state.get("bank_balance", 0.0) or 0.0)
            info["action_effect"] = {
                "inventory_before": inventory_before_total,
                "inventory_after": inventory_after_total,
                "inventory_change": inventory_after_total - inventory_before_total,
                "bank_before": round(bank_before_val, 2),
                "bank_after": round(bank_after_val, 2),
                "bank_change": round(bank_after_val - bank_before_val, 2),
                "demand_change": int(sales_total),
                "daily_revenue": round(float(daily_revenue), 2),
                "tickets_open_before": int(open_before),
                "tickets_open_after": int(open_after),
                "tickets_change": int(open_after - open_before),
                "tickets_spawned": int(tickets_spawned),
                "tickets_resolved": int(tickets_resolved),
                "tickets_total_after": int(len(tickets_after)),
            }
            # Back-compat convenience mirror — some downstream consumers
            # read ``bank_balance_delta`` directly off info.
            info.setdefault(
                "bank_balance_delta",
                info["action_effect"]["bank_change"],
            )
        except Exception:
            # Pure diagnostic; never let it break the step.
            info.setdefault("action_effect", {})

        # Wave 2 — KPI layer. Divide-by-zero guarded.
        try:
            daily_rev_f = float(daily_revenue or 0.0)
            cost_of_goods = 0.0
            daily_sales_map = self.state.get("daily_sales", {}) or {}
            for sku, qty in daily_sales_map.items():
                cost_of_goods += float(self.unit_costs.get(sku, 0.0) or 0.0) * int(qty or 0)
            profit_margin = 0.0
            if daily_rev_f > 0.0:
                profit_margin = (daily_rev_f - cost_of_goods) / daily_rev_f
            inventory_after = self.state.get("inventory", {}) or {}
            inv_total = sum(int(v or 0) for v in inventory_after.values())
            stockout_skus = sum(1 for v in inventory_after.values() if int(v or 0) == 0)
            total_skus = max(1, len(inventory_after))
            stockout_rate = stockout_skus / total_skus
            # Inventory turnover (rough): sold / avg_on_hand this step.
            sold_total = sum(int(v or 0) for v in daily_sales_map.values())
            avg_inv = max(1.0, (inventory_before_total + inv_total) / 2.0)
            inventory_turnover = sold_total / avg_inv
            # Gap 3 — revenue_trend. Compare today's revenue to the
            # mean of the prior history window (which at this point
            # still holds the PREVIOUS step's values — the Wave 3
            # ring buffer is pushed below this block). Small relative
            # deltas register as "flat" to avoid flapping on Poisson
            # noise. Emitted as a string so CEO narratives can inline
            # it ("revenue trending up").
            prior_rev = (self.state.get("history", {}) or {}).get("revenue", []) or []
            revenue_trend = "flat"
            revenue_rel_change = 0.0
            if prior_rev:
                baseline_rev = sum(float(v) for v in prior_rev) / max(1, len(prior_rev))
                if baseline_rev > 0.0:
                    rel = (daily_rev_f - baseline_rev) / baseline_rev
                    revenue_rel_change = rel
                    if rel >= 0.05:
                        revenue_trend = "up"
                    elif rel <= -0.05:
                        revenue_trend = "down"
                elif daily_rev_f > 0.0:
                    revenue_trend = "up"
            kpi_dict = {
                "profit_margin": round(float(profit_margin), 4),
                "stockout_rate": round(float(stockout_rate), 4),
                "inventory_turnover": round(float(inventory_turnover), 4),
                "cost_of_goods_sold": round(float(cost_of_goods), 2),
                "gross_profit": round(float(daily_rev_f - cost_of_goods), 2),
                "units_sold": int(sold_total),
                "inventory_on_hand": int(inv_total),
                "sku_stockouts": int(stockout_skus),
                "revenue_trend": revenue_trend,
                "revenue_change_pct": round(float(revenue_rel_change) * 100.0, 2),
                "daily_revenue": round(float(daily_rev_f), 2),
            }
            info["kpis"] = kpi_dict
        except Exception:
            info.setdefault("kpis", {})

        # Wave 3 — state history ring buffer + trend/intent/why_failed.
        try:
            window = int(
                (self._rewards_cfg.get("state_history_window")
                 if isinstance(self._rewards_cfg.get("state_history_window"), int)
                 else constants.DEFAULT_STATE_HISTORY_WINDOW)
            )
        except (TypeError, ValueError):
            window = int(constants.DEFAULT_STATE_HISTORY_WINDOW)
        window = max(0, window)
        history = self.state.setdefault("history", {})
        if window > 0:
            def _push(key: str, value: float) -> None:
                buf = history.setdefault(key, [])
                buf.append(float(value))
                if len(buf) > window:
                    del buf[: len(buf) - window]

            _push("revenue", float(daily_revenue or 0.0))
            _push("bank_balance", float(self.state.get("bank_balance", 0.0) or 0.0))
            _push(
                "inventory_total",
                float(sum(int(v or 0) for v in (self.state.get("inventory", {}) or {}).values())),
            )
            _push(
                "sales_total",
                float(sum(int(v or 0) for v in (self.state.get("daily_sales", {}) or {}).values())),
            )
            _push(
                "open_tickets",
                float(sum(
                    1 for t in (self.state.get("active_tickets", []) or [])
                    if isinstance(t, dict) and t.get("status") == "open"
                )),
            )
            _push("customer_satisfaction", float(self.state.get("customer_satisfaction", 1.0) or 1.0))
            _push("reward", float(total_reward))

        # Wave 5 — policy stability. Ring buffer of the last N action
        # types; score = (plurality count / buffer size). A policy that
        # flips between every known action scores low; a stable policy
        # scores high. Always emitted so inference can consume it.
        recent_actions = self.state.setdefault("_recent_actions", [])
        atype_for_history = str(action.get("action_type", "wait"))
        recent_actions.append(atype_for_history)
        if window > 0 and len(recent_actions) > window:
            del recent_actions[: len(recent_actions) - window]
        if recent_actions:
            counts: Dict[str, int] = {}
            for at in recent_actions:
                counts[at] = counts.get(at, 0) + 1
            plurality = max(counts.values())
            stability = plurality / float(len(recent_actions))
        else:
            counts = {}
            stability = 1.0
        info["policy_stability"] = {
            "score": round(float(stability), 4),
            "window": int(len(recent_actions)),
            "distribution": dict(counts),
            "last_action": atype_for_history,
        }

        # Wave 5 — anomaly detection. Baseline = mean of the prior window
        # values (excluding the latest). Flags are strictly threshold-
        # driven so replays produce identical output.
        anomalies: List[Dict[str, object]] = []
        try:
            rev_buf = history.get("revenue", []) or []
            if len(rev_buf) >= 3:
                latest_rev = float(rev_buf[-1])
                base_rev = sum(rev_buf[:-1]) / max(1, len(rev_buf) - 1)
                if base_rev > 0.0:
                    rel = (latest_rev - base_rev) / base_rev
                    if rel >= 0.75:
                        anomalies.append({
                            "type": "demand_spike",
                            "metric": "revenue",
                            "relative_change": round(rel, 4),
                            "baseline": round(base_rev, 2),
                            "observed": round(latest_rev, 2),
                        })
                    elif rel <= -0.5:
                        anomalies.append({
                            "type": "demand_collapse",
                            "metric": "revenue",
                            "relative_change": round(rel, 4),
                            "baseline": round(base_rev, 2),
                            "observed": round(latest_rev, 2),
                        })
            rew_buf = history.get("reward", []) or []
            if rew_buf:
                latest_rew = float(rew_buf[-1])
                if latest_rew < 0.0 and float(daily_revenue or 0.0) > 0.0:
                    sold_total = sum(
                        int(v or 0)
                        for v in (self.state.get("daily_sales", {}) or {}).values()
                    )
                    if sold_total > 0:
                        anomalies.append({
                            "type": "loss_despite_sales",
                            "metric": "reward",
                            "reward": round(latest_rew, 4),
                            "daily_revenue": round(float(daily_revenue or 0.0), 2),
                            "units_sold": int(sold_total),
                        })
            inv_buf = history.get("inventory_total", []) or []
            if len(inv_buf) >= 3 and float(inv_buf[-1]) <= 0.0 and float(inv_buf[-2]) > 0.0:
                anomalies.append({
                    "type": "stockout_cliff",
                    "metric": "inventory_total",
                    "previous": round(float(inv_buf[-2]), 2),
                    "observed": 0.0,
                })
            bank_buf = history.get("bank_balance", []) or []
            if len(bank_buf) >= 3:
                recent_dec = all(
                    float(bank_buf[i]) < float(bank_buf[i - 1])
                    for i in range(-1, -3, -1)
                )
                try:
                    _bk_th = float(
                        (self.config.get("financials", {}) or {}).get(
                            "bankruptcy_threshold", 0.0
                        )
                    )
                except (TypeError, ValueError):
                    _bk_th = 0.0
                if recent_dec and float(bank_buf[-1]) <= _bk_th * 1.2:
                    anomalies.append({
                        "type": "cash_slide",
                        "metric": "bank_balance",
                        "observed": round(float(bank_buf[-1]), 2),
                        "threshold": round(float(_bk_th), 2),
                    })
        except Exception:
            pass
        info["anomalies"] = anomalies

        # Derive trend direction ("up" / "down" / "flat") from the ring
        # buffer. Compares the latest value to the mean of the prior
        # window. Small relative deltas register as "flat" to avoid
        # flapping between directions on noisy signals.
        def _trend(key: str, tol: float = 0.02) -> str:
            buf = history.get(key, []) or []
            if len(buf) < 2:
                return "flat"
            latest = float(buf[-1])
            prior = buf[:-1]
            avg_prior = sum(prior) / max(1, len(prior))
            if avg_prior == 0.0:
                if latest > 0.0:
                    return "up"
                if latest < 0.0:
                    return "down"
                return "flat"
            rel = (latest - avg_prior) / max(1e-9, abs(avg_prior))
            if rel > tol:
                return "up"
            if rel < -tol:
                return "down"
            return "flat"

        info["trend"] = {
            "revenue": _trend("revenue"),
            "inventory": _trend("inventory_total"),
            "demand": _trend("sales_total"),
            "bank_balance": _trend("bank_balance"),
            "open_tickets": _trend("open_tickets"),
            "customer_satisfaction": _trend("customer_satisfaction"),
            "reward": _trend("reward"),
        }

        # Wave 3 — CEO intent derived purely from observable signals.
        try:
            kpis = info.get("kpis", {}) or {}
            stockout_rate = float(kpis.get("stockout_rate", 0.0) or 0.0)
            open_tickets_now = int(info.get("action_effect", {}).get("tickets_open_after", 0) or 0)
            urgent_age_threshold = int(self._rewards_cfg.get("urgent_ticket_age_days", 3) or 3)
            today_for_intent = int(self.state.get("current_day", 1))
            aged_tickets = sum(
                1 for t in (self.state.get("active_tickets", []) or [])
                if isinstance(t, dict) and t.get("status") == "open"
                and today_for_intent - int(t.get("created_day", 1) or 1) >= urgent_age_threshold
            )
            # Gap 4 — deterministic intent ladder with an explicit
            # ``maintain_balance`` fallback for the "everything is
            # healthy" regime. Priority (highest → lowest):
            #   1. avoid_stockout — ≥50% of SKUs at zero stock.
            #   2. clear_tickets  — 3+ aged tickets OR 8+ open tickets.
            #   3. increase_profit — profit margin ≤10%, or revenue
            #      actively trending down.
            #   4. maintain_balance — no pressing signal; stay the
            #      course (previously misclassified as
            #      ``increase_profit`` which made the CEO narrative
            #      always sound profit-obsessed even in equilibrium).
            profit_margin = float(kpis.get("profit_margin", 0.0) or 0.0)
            revenue_trend = str(kpis.get("revenue_trend", "flat"))
            if stockout_rate >= 0.5:
                intent = "avoid_stockout"
            elif aged_tickets >= 3 or open_tickets_now >= 8:
                intent = "clear_tickets"
            elif profit_margin <= 0.1 or revenue_trend == "down":
                intent = "increase_profit"
            else:
                intent = "maintain_balance"
            info["intent"] = intent
            info["intent_signals"] = {
                "stockout_rate": round(stockout_rate, 4),
                "aged_tickets": int(aged_tickets),
                "open_tickets": int(open_tickets_now),
                "profit_margin": round(profit_margin, 4),
                "revenue_trend": revenue_trend,
            }
        except Exception:
            info.setdefault("intent", "maintain_balance")
            info.setdefault("intent_signals", {})

        # Wave 3 — why_failed: aggregate negative reward terms + action
        # errors + operational triggers into a human-readable list.
        try:
            reasons: List[str] = []
            action_error = info.get("error")
            if isinstance(action_error, str) and action_error:
                reasons.append(f"action_rejected:{action_error}")
            bd = info.get("reward_breakdown", {}) or {}
            for term in ("bankruptcy", "stockout", "ticket_aging", "delta"):
                try:
                    val = float(bd.get(term, 0.0) or 0.0)
                except (TypeError, ValueError):
                    val = 0.0
                if val < -0.05:
                    reasons.append(f"reward_penalty:{term}={round(val, 3)}")
            if int(info.get("kpis", {}).get("sku_stockouts", 0) or 0) > 0:
                reasons.append("operational:stockouts_present")
            if float(info.get("customer_satisfaction", 1.0) or 1.0) < 0.6:
                reasons.append("operational:low_customer_satisfaction")
            unfilled = int(info.get("restock_unfilled_qty", 0) or 0)
            if unfilled > 0:
                reasons.append(f"supplier:unfilled_units={unfilled}")
            # De-duplicate while preserving order.
            seen = set()
            deduped = []
            for r in reasons:
                if r not in seen:
                    seen.add(r)
                    deduped.append(r)
            info["why_failed"] = deduped
        except Exception:
            info.setdefault("why_failed", [])

        # Post-audit round-2 (A2-38) — stall guard. If the bank balance
        # sat below the bankruptcy threshold for ``stall_terminate_steps``
        # consecutive steps AND no revenue was booked over that window,
        # mark the episode done. Prevents zero-revenue "stall" policies
        # from dragging an episode out to ``max_steps`` with no hope of
        # recovery.
        bank_after = float(self.state.get("bank_balance", 0.0))
        fin_cfg = self.config.get("financials", {}) or {}
        bk_threshold = float(fin_cfg.get("bankruptcy_threshold", 0.0))
        stall_window = int(self._rewards_cfg.get("stall_terminate_steps", 0) or 0)
        stall_done = False
        if stall_window > 0:
            stall = self.state.setdefault("_stall_tracker", {"count": 0, "revenue": 0.0})
            if bank_after <= bk_threshold:
                stall["count"] = int(stall.get("count", 0)) + 1
                stall["revenue"] = float(stall.get("revenue", 0.0)) + float(daily_revenue)
                if stall["count"] >= stall_window and stall["revenue"] <= 0.0:
                    stall_done = True
            else:
                stall["count"] = 0
                stall["revenue"] = 0.0

        done = (
            self.state["step_count"] >= int(self.config["episode"].get("max_steps", 50))
            or bank_after <= bk_threshold
            or stall_done
        )
        self.state["reward"] = total_reward
        self.state["done"] = bool(done)
        if stall_done:
            info.setdefault("termination_reason", "stall")

        # Wave 7 — explainability confidence. Deterministic scalar derived
        # from (a) how concentrated the reward is in a single term vs
        # spread across many (more diffuse = less confident) and (b)
        # whether any causal attribution exists (true competitor reaction,
        # shock, action error). ``1.0`` means "fully explained by visible
        # signals", ``0.0`` means "we observed a reward with no visible
        # cause". Pure derivation — never mutates state.
        #
        # ---------------------------------------------------------------
        # Gap 5 — documented formula
        # ---------------------------------------------------------------
        # Let ``T = {revenue, solvency, stockout, ticket_aging, ad_roi,
        # bankruptcy, delta, inventory_target_bonus}`` be the reward
        # breakdown terms. Define:
        #
        #     term_sum       = sum(|bd[t]|  for t in T)
        #     concentration  = max(|bd[t]|) / term_sum        (∈ [0, 1])
        #                    = 1.0   when term_sum ≤ 1e-9     (no reward to explain)
        #
        # A single dominant term (e.g. stockout penalty swamping
        # everything else) gives concentration → 1.0; a diffuse
        # spread across five terms gives concentration → 0.2.
        #
        # Causal bonuses (additive, bounded so total ≤ 0.25):
        #     + 0.10  if info.competitor_reaction.triggered is True
        #     + 0.05  if a market shock is currently active
        #     + 0.10  if the action was rejected with a structured error
        #
        # The final score is a weighted combination, clamped to [0, 1]:
        #
        #     confidence = min(1.0,
        #                      0.60 * concentration        # how focused
        #                    + 0.40                        # baseline: we always have
        #                                                  #   reward_breakdown + action
        #                                                  #   diagnostics to show
        #                    + causal_bonus)               # bump for each causal handle
        #
        # Interpretation:
        #   * 1.00 — a single reward term dominates AND at least one
        #     causal handle fires (reactive competitor / shock / error).
        #   * 0.80 — single term dominates with no extra causal handle.
        #   * 0.50 — reward spread across ~5 roughly-equal terms.
        #   * 0.40 — lower bound when reward is fully diffuse with
        #     zero causal handles. The 0.40 floor reflects the fact
        #     that ``reward_breakdown`` alone is already a strong
        #     explanation.
        try:
            bd = info.get("reward_breakdown", {}) or {}
            term_values = [
                abs(float(bd.get(k, 0.0) or 0.0))
                for k in (
                    "revenue", "solvency", "stockout", "ticket_aging",
                    "ad_roi", "bankruptcy", "delta", "inventory_target_bonus",
                )
            ]
            term_sum = sum(term_values)
            if term_sum <= 1e-9:
                concentration = 1.0  # trivial reward -> fully explained by zero
            else:
                concentration = max(term_values) / term_sum
            causal_bonus = 0.0
            if info.get("competitor_reaction", {}).get("triggered"):
                causal_bonus += 0.1
            if (info.get("market_shock", {}) or {}).get("remaining_days", 0):
                causal_bonus += 0.05
            if isinstance(info.get("error"), str) and info["error"]:
                causal_bonus += 0.1
            confidence = min(1.0, 0.6 * concentration + 0.4 + causal_bonus)
            info["confidence"] = round(float(confidence), 4)
            # Gap 5 — surface the raw components alongside the scalar so
            # auditors/judges can verify the formula without reading
            # this docstring. All values are deterministic and bounded.
            info["confidence_breakdown"] = {
                "score": round(float(confidence), 4),
                "concentration": round(float(concentration), 4),
                "baseline": 0.4,
                "causal_bonus": round(float(causal_bonus), 4),
                "term_sum": round(float(term_sum), 6),
                "formula": "min(1.0, 0.6*concentration + 0.4 + causal_bonus)",
            }
        except Exception:
            info.setdefault("confidence", 0.5)
            info.setdefault("confidence_breakdown", {})

        # Wave 7 — episode summary, emitted only on the terminal step so
        # clients get a one-shot closeout without having to poll per-step.
        if bool(done):
            try:
                bank_start = float(self.config.get("financials", {}).get("initial_bank_balance", 0.0) or 0.0)
                bank_end = float(self.state.get("bank_balance", 0.0) or 0.0)
                total_profit = bank_end - bank_start
                cum_rev = float(self.state.get("cumulative_revenue", 0.0) or 0.0)
                resolved = sum(
                    1 for t in (self.state.get("active_tickets", []) or [])
                    if isinstance(t, dict) and t.get("status") == "resolved"
                )
                total_tickets = len(self.state.get("active_tickets", []) or [])
                summary_mistakes: List[str] = []
                hist = self.state.get("history", {}) or {}
                stockout_trail = 0
                inv_buf = hist.get("inventory_total", []) or []
                if inv_buf:
                    stockout_trail = sum(1 for v in inv_buf if float(v) <= 0.0)
                if stockout_trail > 0:
                    summary_mistakes.append(f"stockout_days={stockout_trail}")
                if total_tickets > 0 and resolved / max(1, total_tickets) < 0.5:
                    summary_mistakes.append("low_ticket_resolution")
                if total_profit < 0:
                    summary_mistakes.append("bank_declined")
                if stall_done:
                    summary_mistakes.append("episode_stalled")
                if bank_after <= bk_threshold:
                    summary_mistakes.append("bankruptcy")
                termination = (
                    "stall" if stall_done
                    else "bankruptcy" if bank_after <= bk_threshold
                    else "max_steps"
                )
                # Strategy label = dominant intent over the episode.
                intent_counter = self.state.setdefault("_intent_counter", {})
                intent_counter[info.get("intent", "maintain_balance")] = (
                    intent_counter.get(info.get("intent", "maintain_balance"), 0) + 1
                )
                strategy = max(intent_counter.items(), key=lambda kv: kv[1])[0] if intent_counter else "maintain_balance"
                info["episode_summary"] = {
                    "total_profit": round(float(total_profit), 2),
                    "bank_start": round(float(bank_start), 2),
                    "bank_end": round(float(bank_end), 2),
                    "cumulative_revenue": round(float(cum_rev), 2),
                    "resolved_tickets": int(resolved),
                    "total_tickets_touched": int(total_tickets),
                    "steps": int(self.state.get("step_count", 0)),
                    "strategy": strategy,
                    "termination_reason": termination,
                    "mistakes": summary_mistakes,
                }
            except Exception:
                info.setdefault("episode_summary", {})
        else:
            # Still count the intent so the terminal summary has data.
            intent_counter = self.state.setdefault("_intent_counter", {})
            current_intent = info.get("intent", "maintain_balance")
            intent_counter[current_intent] = intent_counter.get(current_intent, 0) + 1

        # --- Part A5 (additive, post-hoc) -----------------------------
        # ``action_quality`` and ``strategy_phase`` are *derived* labels
        # that piggyback on information already present in ``info`` and
        # ``self.state``. They DO NOT mutate state, DO NOT feed reward,
        # and DO NOT change action / observation schemas — the env
        # remains frozen per the global contract. Their sole purpose
        # is to give judges / the scripted demo a one-line read on
        # "was this step good and what phase is the agent in".
        try:
            action_type = str(action.get("action_type", "wait") or "wait")
            action_err = info.get("error")
            breakdown = info.get("reward_breakdown", {}) or {}
            stockout_pen = abs(float(breakdown.get("stockout_penalty", 0.0) or 0.0))
            wait_loop_pen = abs(float(breakdown.get("wait_loop_penalty", 0.0) or 0.0))

            if action_err:
                quality, quality_reason = "bad", f"action_error={action_err}"
            elif total_reward <= -1.0 or stockout_pen > 0.5 or wait_loop_pen > 0.5:
                reason_bits = []
                if stockout_pen > 0.5:
                    reason_bits.append("stockout")
                if wait_loop_pen > 0.5:
                    reason_bits.append("wait_loop")
                if total_reward <= -1.0 and not reason_bits:
                    reason_bits.append("negative_reward")
                quality, quality_reason = "bad", ",".join(reason_bits)
            elif total_reward >= 0.2 and action_type != "wait":
                quality, quality_reason = "good", f"reward={total_reward:.2f}+productive"
            elif total_reward >= 0.0:
                quality, quality_reason = "neutral", "low_impact"
            else:
                quality, quality_reason = "neutral", "mild_loss"
            info["action_quality"] = quality
            info["action_quality_reason"] = quality_reason
        except Exception:
            info.setdefault("action_quality", "neutral")
            info.setdefault("action_quality_reason", "unknown")

        try:
            max_steps = int(self.config.get("episode", {}).get("max_steps", 50) or 50)
            step_count = int(self.state.get("step_count", 0) or 0)
            frac = step_count / max(1, max_steps)
            hist = self.state.get("history", {}) or {}
            reward_buf = [float(x) for x in (hist.get("reward") or [])][-5:]
            recent_neg = sum(1 for r in reward_buf if r < -0.5)
            bank_buf = [float(x) for x in (hist.get("bank_balance") or [])]
            bank_growing = (
                len(bank_buf) >= 2 and bank_buf[-1] > bank_buf[0]
            )

            if frac < 0.2:
                phase, conf, note = "explore", 0.6, f"step {step_count}/{max_steps}"
            elif recent_neg >= 2:
                phase, conf, note = "recover", 0.7, f"recent_neg={recent_neg}"
            elif bank_growing and recent_neg == 0:
                phase, conf, note = "exploit", 0.8, "bank_trending_up"
            else:
                phase, conf, note = "stabilize", 0.55, "steady_state"
            info["strategy_phase"] = phase
            info["strategy_phase_confidence"] = round(float(conf), 3)
            info["strategy_phase_note"] = note
        except Exception:
            info.setdefault("strategy_phase", "stabilize")
            info.setdefault("strategy_phase_confidence", 0.5)
            info.setdefault("strategy_phase_note", "unknown")

        # Invariants soft-check (audit MEDIUM #5) — no-op unless the
        # COMMERCEOPS_ASSERT_INVARIANTS env flag is set.
        try:
            _assert_state_invariants(self.state)
        except AssertionError:
            # Re-raise so tests / CI catch regressions. Production runs
            # with the flag disabled pay nothing.
            raise

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
    def _simulate_day(self, action: Dict | None = None) -> float:
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

        # Post-audit realism — evolve competitor prices as a small,
        # clamped random walk. Per-SKU volatility is read once from the
        # config; SKUs without a configured volatility are static (legacy
        # behaviour). The band ``[LO, HI] * initial_competitor_price``
        # keeps the walk from ever escaping to absurd values and makes
        # the policy's competitor-ratio signal stable.
        walk_deltas = self._walk_competitor_prices()
        # Stash on state so ``step`` can lift it into the explainability
        # payload without having to re-diff snapshots. Cleared below.
        self.state["_competitor_walk_deltas"] = dict(walk_deltas or {})
        # Optional reactive competitor step: if we just set a price, the
        # competitor may undercut/follow before today's demand draw.
        self._reactive_competitor_step(action or {})

        inv_before_day = dict(self.state["inventory"])
        actions_cfg = self.config.get("actions", {}) or {}
        max_ad_multiplier = actions_cfg.get("max_ad_multiplier")
        # Post-audit round-2 (A2-24) — harmonise the demand-model price
        # ratio clamp with the config's set_price band. The agent can
        # only move its price into ``[pmin_mult_competitor,
        # pmax_mult_competitor] * competitor_price``; the demand model
        # should clamp the *ratio* (competitor / price) to the inverse of
        # that window so the two stay consistent. Fall back to the
        # legacy ``(0.25, 4.0)`` clamp when either bound is missing.
        price_ratio_bounds = None
        pmin = actions_cfg.get("price_min_mult_competitor")
        pmax = actions_cfg.get("price_max_mult_competitor")
        try:
            if pmin is not None and pmax is not None:
                pmin_f = float(pmin)
                pmax_f = float(pmax)
                if pmin_f > 0 and pmax_f > 0 and pmax_f >= pmin_f:
                    price_ratio_bounds = (1.0 / pmax_f, 1.0 / pmin_f)
        except (TypeError, ValueError):
            price_ratio_bounds = None
        market_mult = self._market_event_multiplier_by_sku()
        # Audit MEDIUM #3 fix — peek the current satisfaction without
        # mutating it. The real update runs once, AFTER sales are
        # booked, so the open-ticket decay / recovery applies once per
        # day instead of twice.
        cust_sat = self._peek_customer_satisfaction()
        demand_external_mult = {
            sku: float(market_mult.get(sku, 1.0)) * float(cust_sat)
            for sku in self.state["inventory"]
        }
        # Capture per-SKU demand factor decomposition for explainability.
        demand_factors: Dict[str, Dict[str, float]] = {}
        sales = generate_all_demand(
            inventory=self.state["inventory"],
            active_ad_spend=self.state["active_ad_spend"],
            prices=self.state["prices"],
            competitor_prices=self.state["competitor_prices"],
            demand_configs=self.demand_configs,
            current_day=int(self.state["current_day"]),
            rng=self._np_rng,
            max_ad_multiplier=max_ad_multiplier,
            price_ratio_bounds=price_ratio_bounds,
            external_multiplier_by_sku=demand_external_mult,
            record_factors=demand_factors,
        )
        # Post-audit Gap 1 — ``generate_all_demand`` collapses the
        # market-shock and satisfaction multipliers into a single
        # ``external`` scalar (so demand_model.py stays agnostic about
        # *why* the market is hotter/cooler). Here we decompose them
        # back into the explicit ``shock`` and ``satisfaction`` keys
        # that the explainability contract advertises, plus rename
        # the ambiguous ``season`` key to ``seasonality`` for clarity
        # (``season_combined`` is preserved for diff-ability with the
        # lambda math). Physics is untouched: we only enrich the sink.
        for sku, sink in demand_factors.items():
            if not isinstance(sink, dict):
                continue
            shock_val = float(market_mult.get(sku, 1.0) or 1.0)
            sink["shock"] = shock_val
            sink["satisfaction"] = float(cust_sat)
            # Canonical name for the day-of-week multiplier. The
            # legacy ``season`` key is kept as an alias for back-compat
            # with any downstream consumer that grep'd for it.
            if "season" in sink and "seasonality" not in sink:
                sink["seasonality"] = float(sink["season"])
        # Stash demand_factors + the market_mult snapshot so ``step`` can
        # promote them into ``info`` without another trip through the
        # demand model.
        self.state["_demand_factors_last"] = demand_factors
        self.state["_market_mult_last"] = {
            sku: float(market_mult.get(sku, 1.0) or 1.0)
            for sku in self.state["inventory"]
        }
        # Stash the satisfaction scalar used for this tick's demand
        # draw so ``step`` can surface it in ``info`` alongside the
        # end-of-day mutated value (the pair lets the explainer show
        # "satisfaction at demand time" vs "satisfaction after today").
        self.state["_satisfaction_for_demand"] = float(cust_sat)
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
        # Optional simple inventory holding cost (off by default).
        fin_cfg = self.config.get("financials", {}) or {}
        hold_rate = float(fin_cfg.get("inventory_holding_cost_per_unit_per_day", 0.0) or 0.0)
        holding_cost = 0.0
        if hold_rate > 0.0:
            holding_cost = hold_rate * float(sum(int(v) for v in self.state.get("inventory", {}).values()))
            if holding_cost > 0.0:
                self.state["bank_balance"] = float(self.state.get("bank_balance", 0.0)) - holding_cost
        self.state["daily_inventory_holding_cost"] = float(holding_cost)

        # Post-audit realism / R-7 — count SKUs that transitioned from
        # in-stock to out-of-stock on this step. Each such stockout pokes
        # the churn multiplier that scales today's ticket spawn rate.
        stockout_events = 0
        inv_after = self.state["inventory"]
        for sku, before_qty in inv_before_day.items():
            if int(before_qty) > 0 and int(inv_after.get(sku, 0)) == 0:
                stockout_events += 1
        # Update satisfaction after observing today's outcomes.
        self._update_customer_satisfaction(stockout_events=stockout_events)

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

        # Stochastic ticket spawning. Post-audit R-7 / realism — the
        # effective spawn rate is scaled up by any stockouts observed
        # this tick times ``tickets.stockout_churn_multiplier`` so bad
        # operational decisions (letting inventory go to zero) feed
        # back into the ticket queue that the policy has to triage.
        tickets_cfg = self.config.get("tickets", {})
        base_spawn = float(tickets_cfg.get("spawn_rate_per_day", 0.0))
        try:
            churn_mult = float(
                tickets_cfg.get(
                    "stockout_churn_multiplier",
                    constants.DEFAULT_STOCKOUT_CHURN_MULTIPLIER,
                )
            )
        except (TypeError, ValueError):
            churn_mult = float(constants.DEFAULT_STOCKOUT_CHURN_MULTIPLIER)
        effective_spawn = base_spawn * (1.0 + max(0.0, churn_mult) * stockout_events)
        # Post-audit round-2 (A2-39) — monotonic ticket-id counter. A
        # one-element list is used as a shared mutable box so the helper
        # can update the high-water mark in place.
        hwm_list = self.state.setdefault("_ticket_id_hwm", [0])
        spawn_daily_tickets(
            active_tickets=self.state["active_tickets"],
            current_day=int(self.state["current_day"]),
            spawn_rate_per_day=effective_spawn,
            issue_types=tickets_cfg.get("issue_types"),
            urgency_levels=tickets_cfg.get("urgency_levels"),
            urgency_weights=tickets_cfg.get("urgency_weights"),
            rng=self._py_rng,
            max_active=tickets_cfg.get("max_active"),  # post-audit B.9
            ticket_id_high_water_mark=hwm_list,
        )

        # Advance the calendar.
        self.state["current_day"] = int(self.state["current_day"]) + 1
        self.state["current_week"] = int(self.state["current_day"]) // 7

        return float(daily_revenue)
