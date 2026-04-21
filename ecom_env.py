"""
ecom_env.py — Swiftlogic CommerceOps v2 Pydantic + Adapter layer.

This module is the stable, import-compatible surface for the OpenEnv validator
and the FastAPI server. It provides:

    * Pydantic action/observation/reward models (unchanged public shape, with
      new optional fields for CommerceOps v2).
    * An ``EcomEnv`` class that delegates all simulation logic to a
      ``env.world_engine.WorldEngine`` instance loaded from a business config
      JSON under ``configs/``.
    * Three deterministic grading functions whose outputs are strictly clamped
      to (0.01, 0.99) as required by OpenEnv Phase 2 Deep Validation.

The server imports the following symbols and they must remain available:

    EcomEnv, EcomAction, EcomObservation, EcomReward,
    RestockAction, RefundAction, AdSpendAction, NegotiateAction, WaitAction,
    grade_triage_task, grade_inventory_task, grade_profit_task
"""

from __future__ import annotations

import os
import warnings
from typing import Annotated, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, RootModel

from env import constants
from env.world_engine import WorldEngine
from env.ticket_system import generate_episode_tickets  # re-export for compat


# ---------------------------------------------------------------------------
# Default config path (can be overridden via env var for tests/demo)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = os.environ.get(
    "COMMERCEOPS_CONFIG", "configs/siyaani_fashion.json"
)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class Ticket(BaseModel):
    ticket_id: str
    issue_type: str
    status: str
    urgency: str = "normal"
    created_day: int = 1


class EcomObservation(BaseModel):
    current_day: int
    step_count: int
    bank_balance: float
    inventory: Dict[str, int]
    pending_orders: Dict[str, int]
    active_tickets: List[Ticket]
    daily_sales: Dict[str, int]
    active_ad_spend: Dict[str, float]

    # CommerceOps v2 additions (all defaulted for backwards compatibility).
    current_week: int = 0
    prices: Dict[str, float] = Field(default_factory=dict)
    competitor_prices: Dict[str, float] = Field(default_factory=dict)
    cumulative_revenue: float = 0.0
    # Gross revenue realised on the most recent simulated day. Surfaced on
    # the observation so policies/graders can read a scalar instead of
    # re-deriving from ``daily_sales`` * ``prices``. Defaulted to ``0.0``
    # for wire compatibility with older serialised observations.
    daily_revenue: float = 0.0
    supplier_quotes: Dict[str, float] = Field(default_factory=dict)
    # Per-SKU ``step_count`` at which the paired supplier quote goes stale.
    # Exposing this alongside ``supplier_quotes`` lets a policy reason about
    # its negotiate -> restock window without having to remember when it
    # issued the negotiation. Defaults to an empty dict for clients that
    # serialised observations from older server versions.
    supplier_quote_expiry: Dict[str, int] = Field(default_factory=dict)
    # Post-audit m-10 — detailed per-SKU delivery schedule. Each entry is
    # a list of ``[delivery_day, quantity]`` pairs (both ints), projected
    # from ``WorldEngine.state['pending_deliveries']``. Exposing the
    # schedule lets a policy distinguish "one big order arriving tomorrow"
    # from "three small orders arriving over the next five days" without
    # having to re-derive it from ``pending_orders``'s aggregate counter.
    # List-of-lists (not list-of-tuples) is used so JSON roundtrips cleanly.
    pending_orders_schedule: Dict[str, List[List[int]]] = Field(default_factory=dict)

    reward: float = 0.0
    done: bool = False


class RestockAction(BaseModel):
    action_type: Literal["restock"] = "restock"
    sku: str
    quantity: int


class RefundAction(BaseModel):
    action_type: Literal["refund"] = "refund"
    ticket_id: str


class AdSpendAction(BaseModel):
    action_type: Literal["ad_spend"] = "ad_spend"
    sku: str
    budget: float


class NegotiateAction(BaseModel):
    action_type: Literal["negotiate"] = "negotiate"
    sku: str
    quantity: int


class WaitAction(BaseModel):
    action_type: Literal["wait"] = "wait"


class SetPriceAction(BaseModel):
    """v2.3 Phase 2.4 — agent-controlled repricing.

    Previously ``prices`` was observable but immutable: the policy could see
    competitor prices and its own list price but had no lever to shift them.
    ``SetPriceAction`` closes the loop and plugs into the existing demand
    model via the price/competitor-price ratio. Bounds are drawn from
    ``actions.price_min_mult_competitor`` / ``actions.price_max_mult_competitor``
    (defaults to ``0.25`` / ``4.0`` to mirror the engine's demand clamps).
    """

    action_type: Literal["set_price"] = "set_price"
    sku: str
    price: float


class EcomAction(RootModel):
    root: Annotated[
        Union[
            RestockAction,
            RefundAction,
            AdSpendAction,
            NegotiateAction,
            WaitAction,
            SetPriceAction,
        ],
        Field(discriminator="action_type"),
    ]


class EcomReward(BaseModel):
    value: float


# ---------------------------------------------------------------------------
# Grader context
# ---------------------------------------------------------------------------
# Every ``EcomEnv`` instance carries its own ``grader_context`` dict, and on
# construction / load_config it also mirrors that context to the module-level
# ``_GRADER_CONTEXT`` below. The mirror preserves backward compatibility with
# the server, which imports bare ``grade_*`` functions and calls them with
# only ``(initial_state, final_state)``. New callers can pass
# ``context=env.grader_context`` explicitly and bypass the shared mirror,
# which is how test harnesses with multiple ``EcomEnv`` instances avoid
# racing on the module global.

_DEFAULT_GRADER_CONTEXT: Dict[str, object] = {
    "profit_normalizer": constants.DEFAULT_PROFIT_NORMALIZER,
    "inventory_target_sku": "cotton_set",
    "inventory_target_units": 10,
}

_GRADER_CONTEXT: Dict[str, object] = dict(_DEFAULT_GRADER_CONTEXT)


def _build_grader_context(config: Dict) -> Dict[str, object]:
    """Pure helper: distil a grader-context dict from a business config."""
    graders = config.get("graders", {}) if isinstance(config, dict) else {}
    profit_cfg = graders.get("profit_task", {}) if isinstance(graders, dict) else {}
    inv_cfg = graders.get("inventory_task", {}) if isinstance(graders, dict) else {}
    return {
        "profit_normalizer": float(
            profit_cfg.get("normalizer", constants.DEFAULT_PROFIT_NORMALIZER)
        ),
        "inventory_target_sku": str(inv_cfg.get("target_sku", "cotton_set")),
        "inventory_target_units": float(inv_cfg.get("target_units", 10)),
    }


def _refresh_grader_context(config: Dict) -> Dict[str, object]:
    """Update the module-level mirror and return the fresh context.

    Kept for backward compatibility with any external caller that imported
    this symbol from v2.0/v2.1.
    """
    ctx = _build_grader_context(config)
    _GRADER_CONTEXT.update(ctx)
    return ctx


# ---------------------------------------------------------------------------
# EcomEnv adapter
# ---------------------------------------------------------------------------

class EcomEnv:
    """OpenEnv-compatible e-commerce storefront environment (CommerceOps v2).

    Backed by a :class:`env.world_engine.WorldEngine` driven by a business
    config JSON. The server instantiates it with no args (preserving v1
    compatibility); pass ``config_path`` to load a different business.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.world_engine: WorldEngine = WorldEngine(config_path)
        # Each EcomEnv owns its grader context; we also mirror it to the
        # module-level singleton so bare ``grade_*`` imports keep working.
        self.grader_context: Dict[str, object] = _refresh_grader_context(
            self.world_engine.config
        )
        self._current_state: EcomObservation = self._wrap_state(self.world_engine.reset())

    # -- Seeding --------------------------------------------------------
    def seed(self, seed: int) -> None:
        self.world_engine.reset(seed=int(seed))

    # -- Config hot-swap ------------------------------------------------
    def load_config(self, config_path: str, seed: Optional[int] = None) -> EcomObservation:
        """Hot-swap the world's business config and reset."""
        self.world_engine.load_config(config_path)
        self.grader_context = _refresh_grader_context(self.world_engine.config)
        return self.reset(seed=seed)

    # -- Reset ----------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> EcomObservation:
        raw = self.world_engine.reset(seed=seed)
        self._current_state = self._wrap_state(raw)
        return self._current_state

    # -- Step -----------------------------------------------------------
    def step(self, action) -> Tuple[EcomObservation, EcomReward, bool, Dict]:
        if hasattr(action, "root"):
            action = action.root
        if hasattr(action, "model_dump"):
            action_dict = action.model_dump()
        elif isinstance(action, dict):
            action_dict = action
        else:
            action_dict = {"action_type": "wait"}

        raw, reward_val, done, info = self.world_engine.step(action_dict)
        self._current_state = self._wrap_state(raw)
        return self._current_state, EcomReward(value=float(reward_val)), bool(done), info

    # -- State ----------------------------------------------------------
    def state(self) -> EcomObservation:
        return self._current_state

    # -- Framework compatibility (OpenEnv v0.2.3) -----------------------
    async def reset_async(self, seed=None, **kwargs):
        return self.reset(seed=seed)

    async def step_async(self, action, **kwargs):
        obs, _reward, _done, _info = self.step(action)
        return obs

    def close(self):
        pass

    # -- Internal helpers -----------------------------------------------
    @staticmethod
    def _wrap_state(raw: Dict) -> EcomObservation:
        """Coerce a WorldEngine state dict into an EcomObservation.

        Post-audit m-10 — project ``pending_deliveries`` (which lives in the
        internal state as ``{sku: [(delivery_day, qty), ...]}``) into the
        observation-friendly ``pending_orders_schedule`` shape, a dict of
        JSON-clean ``{sku: [[delivery_day, qty], ...]}`` lists. We project
        here rather than mutating the engine's state so the engine layer
        stays the single source of truth.
        """
        projected: Dict[str, List[List[int]]] = {}
        raw_pending = raw.get("pending_deliveries") or {}
        if isinstance(raw_pending, dict):
            for sku, entries in raw_pending.items():
                if not isinstance(entries, list):
                    continue
                clean: List[List[int]] = []
                for entry in entries:
                    try:
                        day, qty = entry
                        clean.append([int(day), int(qty)])
                    except (TypeError, ValueError):
                        continue
                if clean:
                    projected[str(sku)] = clean
        obs_input = dict(raw)
        obs_input["pending_orders_schedule"] = projected
        return EcomObservation(**obs_input)


# ---------------------------------------------------------------------------
# Deterministic Grading Functions (config-aware, clamped to (0.01, 0.99))
# ---------------------------------------------------------------------------

def grade_triage_task(
    initial_state: EcomObservation,
    final_state: EcomObservation,
    *,
    context: Optional[Dict[str, object]] = None,  # unused, present for API symmetry
) -> float:
    """Ratio of resolved tickets to total tickets (initial + spawned).

    v2.3 Phase 4.4 — when an episode genuinely has no tickets (which the
    validator now flags as a likely config bug, but still permits), we
    return the neutral ``0.5`` rather than the near-perfect ``0.99``.
    Crediting the agent almost-full score for doing nothing on an empty
    ticket queue was a freebie that distorted training signals on exotic
    configs with ``min_initial=0, spawn_rate_per_day=0``.
    """
    tickets = final_state.active_tickets or []
    if not tickets:
        return 0.5
    resolved = sum(1 for t in tickets if t.status == "resolved")
    ratio = resolved / len(tickets)
    return max(0.01, min(0.99, ratio))


def grade_inventory_task(
    initial_state: EcomObservation,
    final_state: EcomObservation,
    *,
    context: Optional[Dict[str, object]] = None,
) -> float:
    """Ratio of target-SKU stock vs the config-defined target units.

    ``context`` (optional) lets the caller bind a specific env's grader
    context, bypassing the module-level mirror. This is how multi-env test
    harnesses avoid racing on shared global state.
    """
    if context is None:
        # Post-audit m-6 — relying on the shared module mirror is racy once
        # more than one EcomEnv lives in the same process. Emit a
        # ``DeprecationWarning`` so callers migrate to ``context=``.
        warnings.warn(
            "Calling grade_inventory_task without an explicit context= kwarg "
            "is deprecated; pass context=env.grader_context. The module-level "
            "_GRADER_CONTEXT mirror will be removed in v2.4.",
            DeprecationWarning,
            stacklevel=2,
        )
    ctx = context if context is not None else _GRADER_CONTEXT
    target_sku = str(ctx.get("inventory_target_sku", "cotton_set"))
    target_units = float(ctx.get("inventory_target_units", 10)) or 1.0
    stock = float(final_state.inventory.get(target_sku, 0))
    ratio = stock / target_units
    return max(0.01, min(0.99, ratio))


def grade_profit_task(
    initial_state: EcomObservation,
    final_state: EcomObservation,
    *,
    context: Optional[Dict[str, object]] = None,
) -> float:
    """Normalize bank-balance growth around break-even (0.5).

    Uses the active config's profit normalizer when available, falling back
    to a capital-scaled heuristic so the grader stays meaningful across v1 /
    v2 / alternate business configs.
    """
    if context is None:
        # Post-audit m-6 — see ``grade_inventory_task`` for the rationale.
        warnings.warn(
            "Calling grade_profit_task without an explicit context= kwarg "
            "is deprecated; pass context=env.grader_context. The module-level "
            "_GRADER_CONTEXT mirror will be removed in v2.4.",
            DeprecationWarning,
            stacklevel=2,
        )
    ctx = context if context is not None else _GRADER_CONTEXT
    profit = final_state.bank_balance - initial_state.bank_balance
    configured = float(ctx.get("profit_normalizer", constants.DEFAULT_PROFIT_NORMALIZER))
    normalizer = max(
        configured,
        0.4 * max(initial_state.bank_balance, 1.0),
        constants.DEFAULT_PROFIT_NORMALIZER,
    )
    score = 0.5 + (profit / normalizer)
    return max(0.01, min(0.99, score))


__all__ = [
    "Ticket",
    "EcomObservation",
    "RestockAction",
    "RefundAction",
    "AdSpendAction",
    "NegotiateAction",
    "WaitAction",
    "SetPriceAction",
    "EcomAction",
    "EcomReward",
    "EcomEnv",
    "grade_triage_task",
    "grade_inventory_task",
    "grade_profit_task",
    "generate_episode_tickets",
]
