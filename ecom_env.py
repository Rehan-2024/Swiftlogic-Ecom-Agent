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

from pydantic import BaseModel, ConfigDict, Field, RootModel

from env import constants
from env.world_engine import WorldEngine
from env.ticket_system import generate_episode_tickets  # re-export for compat


# ---------------------------------------------------------------------------
# Default config path (can be overridden via env var for tests/demo)
# ---------------------------------------------------------------------------
# Canonical (non-demo) default. ``DEFAULT_CONFIG_PATH`` is preserved as a
# module-level attribute for backwards compatibility with external
# callers that import it; the active default is re-resolved dynamically
# in ``_resolve_default_config_path`` so toggling ``COMMERCEOPS_DEMO_MODE``
# or ``COMMERCEOPS_CONFIG`` at runtime takes effect on the next
# ``EcomEnv()`` construction (the audit flagged the old import-time
# freeze as a bug).

_DEFAULT_BASELINE_CONFIG = "configs/siyaani_fashion.json"
_DEFAULT_DEMO_CONFIG = "configs/siyaani_fashion_demo.json"
_DEMO_MODE_ENV = "COMMERCEOPS_DEMO_MODE"
_CONFIG_ENV = "COMMERCEOPS_CONFIG"
_DEMO_TRUTHY = {"1", "true", "yes", "on"}


def _demo_mode_enabled() -> bool:
    """Return True when ``COMMERCEOPS_DEMO_MODE`` is explicitly truthy."""
    return str(os.environ.get(_DEMO_MODE_ENV, "")).strip().lower() in _DEMO_TRUTHY


def _resolve_default_config_path() -> str:
    """Pick the active default config at construction time.

    Resolution precedence (highest → lowest):
    1. ``COMMERCEOPS_CONFIG`` — explicit override (unchanged from v2.3.x).
    2. ``COMMERCEOPS_DEMO_MODE`` truthy and the demo config file exists on
       disk → ``configs/siyaani_fashion_demo.json``.
    3. ``configs/siyaani_fashion.json`` — canonical shipping default.

    Kept as a function (not a bare module constant) so tests and the
    server's ``/config`` hot-swap flow see fresh values when the env
    vars are flipped mid-process.
    """
    override = os.environ.get(_CONFIG_ENV)
    if override:
        return override
    if _demo_mode_enabled():
        try:
            from pathlib import Path as _Path
            if _Path(_DEFAULT_DEMO_CONFIG).exists():
                return _DEFAULT_DEMO_CONFIG
        except Exception:
            # Filesystem errors are non-fatal; fall through to the baseline.
            pass
    return _DEFAULT_BASELINE_CONFIG


DEFAULT_CONFIG_PATH = _resolve_default_config_path()


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class Ticket(BaseModel):
    # Post-audit L-14 — silently drop unknown ticket fields (e.g. forks
    # that add ``notes`` or ``priority``) so the adapter layer doesn't
    # explode on observation coercion. The internal engine dict shape is
    # the single source of truth; the Pydantic model is a projection.
    model_config = ConfigDict(extra="ignore")

    ticket_id: str
    issue_type: str
    status: str
    urgency: str = "normal"
    created_day: int = 1
    # Post-audit round-2 (A2-33) — optional SKU tag. Tickets generated
    # when ``tickets.ticket_issue_bias_by_sku=True`` carry the SKU that
    # most likely caused the issue (e.g. a stockout-linked churn
    # ticket). Defaulted to ``None`` so every existing client keeps
    # deserialising cleanly.
    sku: Optional[str] = None


class EcomObservation(BaseModel):
    # Post-audit L-14 — ignore unknown fields on the observation payload.
    # The WorldEngine state dict accumulates private bookkeeping keys
    # (e.g. ``supplier_quoted_qty``, ``pending_deliveries``, ``reward``,
    # ``done``) that the observation model doesn't need to surface; the
    # default "extra=allow" behaviour would pass them through, while
    # "extra=forbid" would break on any new key added to state. Pinning
    # to "ignore" makes the model stable against state-shape evolution.
    model_config = ConfigDict(extra="ignore")

    # Post-audit round-2 (A2-16) — ``current_day`` is the calendar day
    # that will be *played next*, not the day whose sales were just
    # booked. The sync ``_simulate_day`` in WorldEngine books revenue
    # against ``current_day`` and then advances it by 1, so the
    # observation returned from ``/step`` sees the incremented value.
    # The derived field ``current_day_played`` (= ``current_day - 1``)
    # is provided below for client clarity; existing clients that
    # only read ``current_day`` keep their exact behaviour.
    current_day: int
    step_count: int
    bank_balance: float
    inventory: Dict[str, int]
    pending_orders: Dict[str, int]
    active_tickets: List[Ticket]
    daily_sales: Dict[str, int]
    active_ad_spend: Dict[str, float]

    # Derived, defaulted field (A2-16). ``current_day_played`` surfaces
    # the day whose sales are represented by ``daily_sales`` /
    # ``daily_revenue``. Set from ``_wrap_state``; defaulted here so
    # older serialised observations still load.
    current_day_played: int = 0

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
    # Optional simple bounded scalar representing customer sentiment.
    customer_satisfaction: float = 1.0
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
    triage_cfg = graders.get("triage_task", {}) if isinstance(graders, dict) else {}
    # Post-audit round-2 (A2-34) — triage sku_match_bonus is optional
    # and clamped to ``[0, 0.1]``. Reading it into the context cache
    # keeps the grader function pure (no config lookup) and lets
    # external callers introspect the active value.
    try:
        triage_bonus = float(triage_cfg.get("sku_match_bonus", 0.0) or 0.0)
    except (TypeError, ValueError):
        triage_bonus = 0.0
    triage_bonus = max(0.0, min(0.1, triage_bonus))
    return {
        "profit_normalizer": float(
            profit_cfg.get("normalizer", constants.DEFAULT_PROFIT_NORMALIZER)
        ),
        "inventory_target_sku": str(inv_cfg.get("target_sku", "cotton_set")),
        "inventory_target_units": float(inv_cfg.get("target_units", 10)),
        "triage_sku_match_bonus": triage_bonus,
    }


def _refresh_grader_context(config: Dict) -> Dict[str, object]:
    """Update the module-level mirror and return the fresh context.

    Kept for backward compatibility with any external caller that imported
    this symbol from v2.0/v2.1.
    """
    ctx = _build_grader_context(config)
    _GRADER_CONTEXT.update(ctx)
    return ctx


def _strict_grader_context_enabled() -> bool:
    """Return True if ``COMMERCEOPS_STRICT_GRADER_CONTEXT`` is truthy.

    Read at call-time (not import-time) so tests can toggle the flag via
    ``monkeypatch.setenv`` without reloading the module.
    """
    return os.environ.get("COMMERCEOPS_STRICT_GRADER_CONTEXT", "") in {"1", "true", "yes"}


def _warn_or_raise_missing_grader_context(fn_name: str) -> None:
    """Emit a ``DeprecationWarning`` (or ``RuntimeError`` in strict mode)
    when a grader is called without an explicit ``context=`` kwarg.

    v2.3.x Phase A.2 — the module-level ``_GRADER_CONTEXT`` mirror is still
    consulted for backward compatibility with the pre-v2.3 server, but the
    warning now nudges callers toward ``context=env.grader_context`` and
    ships an opt-in strict flag so CI can fail loudly. The warning uses
    ``stacklevel=3`` so the traceback points at the real caller (through
    this helper + the grader function) rather than at this module.
    """
    message = (
        f"Calling {fn_name} without an explicit context= kwarg is "
        "deprecated; pass context=env.grader_context to bind the per-env "
        "grader state. The module-level _GRADER_CONTEXT mirror will be "
        "removed in v2.4. Set COMMERCEOPS_STRICT_GRADER_CONTEXT=1 to fail "
        "fast on this path today."
    )
    if _strict_grader_context_enabled():
        raise RuntimeError(message)
    warnings.warn(message, DeprecationWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# EcomEnv adapter
# ---------------------------------------------------------------------------

class EcomEnv:
    """OpenEnv-compatible e-commerce storefront environment (CommerceOps v2).

    Backed by a :class:`env.world_engine.WorldEngine` driven by a business
    config JSON. The server instantiates it with no args (preserving v1
    compatibility); pass ``config_path`` to load a different business.
    """

    def __init__(self, config_path: Optional[str] = None):
        # Post-audit: resolve the default at construction time (not at
        # module import) so flipping ``COMMERCEOPS_DEMO_MODE`` or
        # ``COMMERCEOPS_CONFIG`` after ``ecom_env`` has been imported
        # still takes effect on the next ``EcomEnv()`` instance.
        if config_path is None:
            config_path = _resolve_default_config_path()
        self.world_engine: WorldEngine = WorldEngine(config_path)
        # Each EcomEnv owns its grader context; we also mirror it to the
        # module-level singleton so bare ``grade_*`` imports keep working.
        self.grader_context: Dict[str, object] = _refresh_grader_context(
            self.world_engine.config
        )
        self._current_state: EcomObservation = self._wrap_state(self.world_engine.reset())

    # -- Seeding --------------------------------------------------------
    def seed(self, seed: int) -> None:
        """Reseed the per-env RNGs WITHOUT wiping state.

        Post-audit round-2 (A2-2) — previously ``seed`` proxied to
        ``world_engine.reset``, which also cleared inventory, bank
        balance, tickets, etc. Gymnasium-style harnesses that issue
        ``env.seed(42)`` mid-episode expected a no-mutation reseed;
        the old shape silently wiped their state.

        Calls ``WorldEngine.reseed`` (a new method that touches only
        ``_py_rng`` and ``_np_rng``). Callers who want the old
        seed-and-reset behaviour should call ``env.reset(seed=...)``
        explicitly.
        """
        if hasattr(self.world_engine, "reseed"):
            self.world_engine.reseed(int(seed))
        else:
            # Defensive fallback for engines predating ``reseed``.
            self.world_engine.reset(seed=int(seed))

    # -- Graders helper -------------------------------------------------
    def graders(self):
        """Return a dict of grader callables pre-bound to ``self.grader_context``.

        Post-audit round-2 (A2-1) — callers (``inference.py``, the
        server's ``/grader`` route, external test harnesses) previously
        had to pass ``env.grader_context`` on every call; forgetting it
        silently fell back to the deprecated module-level mirror
        (``_GRADER_CONTEXT``). The helper centralises the wiring so the
        mirror can be removed in a future release without chasing every
        caller.
        """
        ctx = self.grader_context
        return {
            "triage_task": lambda i, f: grade_triage_task(i, f, context=ctx),
            "inventory_task": lambda i, f: grade_inventory_task(i, f, context=ctx),
            "profit_task": lambda i, f: grade_profit_task(i, f, context=ctx),
            "stability_task": lambda i, f: grade_stability_task(i, f, context=ctx),
            "competitor_response_task": lambda i, f: grade_competitor_response_task(
                i, f, context=ctx
            ),
            "crisis_recovery_task": lambda i, f: grade_crisis_recovery_task(
                i, f, context=ctx
            ),
        }

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
        # Post-audit round-2 (A2-6) — direct (non-HTTP) callers of
        # ``EcomEnv.step`` must hit the same discriminated-union
        # validation path as the ``/step`` route. Previously a raw dict
        # with an unknown ``action_type`` slipped straight through to
        # ``WorldEngine.step``, which surfaced as a generic
        # ``unknown_action`` inside the info dict instead of a clean
        # Pydantic ``ValidationError`` the caller can introspect.
        if hasattr(action, "root"):
            action = action.root
        if hasattr(action, "model_dump"):
            action_dict = action.model_dump()
        elif isinstance(action, dict):
            action_dict = EcomAction.model_validate(action).root.model_dump()
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
        """Async wrapper over :meth:`reset` provided for OpenEnv compatibility.

        .. note:: Audit MINOR #17 — this wrapper is ``async`` for
           protocol conformance only; the underlying ``reset`` is a
           synchronous, CPU-bound call that does *not* yield to the
           event loop. In a single-worker Uvicorn deployment that's
           fine (the framework already serialises requests). If you
           host multiple envs inside the same event loop and require
           non-blocking behaviour, wrap this call in
           ``asyncio.to_thread`` at the caller.
        """
        return self.reset(seed=seed)

    async def step_async(self, action, **kwargs):
        """Async wrapper over ``step``.

        Post-audit M-1 — previously this method returned the observation
        only, silently discarding the reward, ``done`` flag, and ``info``.
        OpenEnv v0.2.3 async harnesses that awaited ``step_async`` never
        saw terminal signals and could run past ``done=True`` unnoticed.
        The wrapper now returns the full ``(obs, reward, done, info)``
        tuple, matching the sync ``step`` contract. Legacy callers that
        only destructured the first element keep working because Python
        tuples unpack positionally.

        .. note:: Audit MINOR #17 — ``step_async`` is not actually
           non-blocking. ``WorldEngine.step`` executes the full
           day-simulation synchronously while holding the event loop.
           This is intentional: the single-worker server topology
           serialises ``/step`` calls under ``state["lock"]`` anyway,
           so a real ``await`` would add scheduling overhead without
           any throughput win. Callers running multiple ``EcomEnv``s
           in a shared event loop should wrap this in
           ``asyncio.to_thread`` to restore parallelism.
        """
        return self.step(action)

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
        # Post-audit round-2 (A2-16) — surface the day whose sales are
        # reflected in ``daily_sales`` / ``daily_revenue``. ``current_day``
        # is already ``play-next``; ``current_day_played`` is ``current_day - 1``,
        # clamped at 0 for the initial reset observation where no day
        # has been played yet.
        try:
            cd_next = int(obs_input.get("current_day", 0))
        except (TypeError, ValueError):
            cd_next = 0
        obs_input["current_day_played"] = max(0, cd_next - 1)
        # Audit MINOR #12 — the engine keeps competitor prices in full
        # precision in state (A2-47) so a long episode doesn't compound
        # ``round(x, 2)`` drift. Agent prices, in contrast, are rounded
        # to 2 dp on ``do_set_price``. That asymmetry leaks into the
        # observation: one SKU reports ``1799.99874...`` while another
        # reports ``1800.00``. We harmonise on serialisation only — the
        # internal float is preserved for the next day's drift.
        for _key in ("prices", "competitor_prices"):
            _raw_map = obs_input.get(_key)
            if isinstance(_raw_map, dict):
                obs_input[_key] = {
                    str(sku): round(float(val), 2)
                    for sku, val in _raw_map.items()
                    if isinstance(val, (int, float))
                }
        return EcomObservation(**obs_input)


# ---------------------------------------------------------------------------
# Deterministic Grading Functions (config-aware, clamped to (0.01, 0.99))
# ---------------------------------------------------------------------------

def grade_triage_task(
    initial_state: EcomObservation,
    final_state: EcomObservation,
    *,
    context: Optional[Dict[str, object]] = None,
) -> float:
    """Ratio of resolved tickets to total tickets (initial + spawned).

    v2.3 Phase 4.4 — when an episode genuinely has no tickets (which the
    validator now flags as a likely config bug, but still permits), we
    return the neutral ``0.5`` rather than the near-perfect ``0.99``.

    Post-audit round-2 (A2-34) — if the context supplies a non-zero
    ``triage_sku_match_bonus`` (clamped to ``[0, 0.1]``) the base ratio
    is bumped when any resolved ticket carries an ``sku`` tag, since
    that implies the policy's refund targeted a specific problem SKU.
    Bonus is capped so a single match can never push the grade past
    the clamped 0.99 ceiling.
    """
    tickets = final_state.active_tickets or []
    if not tickets:
        return 0.5
    resolved_tickets = [t for t in tickets if t.status == "resolved"]
    ratio = len(resolved_tickets) / len(tickets)
    bonus = 0.0
    if context is not None:
        try:
            bonus_cfg = float(context.get("triage_sku_match_bonus", 0.0) or 0.0)
        except (TypeError, ValueError):
            bonus_cfg = 0.0
        bonus_cfg = max(0.0, min(0.1, bonus_cfg))
        if bonus_cfg > 0 and any(getattr(t, "sku", None) for t in resolved_tickets):
            bonus = bonus_cfg
    return max(0.01, min(0.99, ratio + bonus))


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
        _warn_or_raise_missing_grader_context("grade_inventory_task")
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
        _warn_or_raise_missing_grader_context("grade_profit_task")
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


# ---------------------------------------------------------------------------
# Evaluation-only graders (Part A3)
# ---------------------------------------------------------------------------
# The three graders below are *additive* and *evaluation-only*. They are
# registered on ``/tasks`` and ``/grader`` so judges can see the full
# picture, but they are explicitly tagged ``evaluation_only: true`` and
# are NEVER summed into the training reward (see roadmap A.4 boundary).
# Rationale: keeping training on the 3 stable signals (triage, inventory,
# profit) preserves GRPO variance properties; adding three more signals
# to the reward at the same time as the policy is learning is a known
# source of instability (guide §7).
#
# All three are pure functions of ``initial_state`` and ``final_state`` —
# no hidden state, no history buffer access, no physics touch — so they
# respect the env-freeze contract.

def grade_stability_task(
    initial_state: EcomObservation,
    final_state: EcomObservation,
    *,
    context: Optional[Dict[str, object]] = None,
) -> float:
    """Customer-satisfaction retention (evaluation-only, clamped (0.01, 0.99)).

    Measures how well the agent preserved the end-of-episode customer
    satisfaction signal. ``1.0`` implies perfect retention; ``0.0`` implies
    full churn. The grader linearly maps this scalar to the clamp range
    using a configurable ``stability_target`` (default ``0.75`` — above
    which the score saturates near 0.99).
    """
    sat = float(getattr(final_state, "customer_satisfaction", 1.0) or 0.0)
    target = 0.75
    if context is not None:
        try:
            target = float(context.get("stability_target", target) or target)
        except (TypeError, ValueError):
            target = 0.75
    target = max(1e-6, float(target))
    score = sat / target
    return max(0.01, min(0.99, score))


def grade_competitor_response_task(
    initial_state: EcomObservation,
    final_state: EcomObservation,
    *,
    context: Optional[Dict[str, object]] = None,
) -> float:
    """Agent priced at-or-below competitor across observed SKUs (clamped).

    Computes the mean per-SKU ratio ``our_price / competitor_price`` over
    every SKU with both values observed, then maps via a small linear
    window:

      * ratio ≤ 0.80 → score ≈ 0.99 (decisively undercutting)
      * ratio == 1.0 → score == 0.5 (matched)
      * ratio ≥ 1.20 → score ≈ 0.01 (decisively over-priced)

    Returns a neutral ``0.5`` if no SKU has both prices observable. Pure
    function of the observation — no history needed.
    """
    prices = final_state.prices or {}
    comp = final_state.competitor_prices or {}
    ratios = []
    for sku, p in prices.items():
        c = comp.get(sku, 0.0)
        try:
            c_f = float(c)
            p_f = float(p)
        except (TypeError, ValueError):
            continue
        if c_f <= 0 or p_f <= 0:
            continue
        ratios.append(p_f / c_f)
    if not ratios:
        return 0.5
    mean_ratio = sum(ratios) / len(ratios)
    score = 1.0 - (mean_ratio - 0.8) / 0.4
    return max(0.01, min(0.99, score))


def grade_crisis_recovery_task(
    initial_state: EcomObservation,
    final_state: EcomObservation,
    *,
    context: Optional[Dict[str, object]] = None,
) -> float:
    """Bank-balance resilience across the episode (clamped (0.01, 0.99)).

    Scores ``final_bank / initial_bank`` with a linear window:

      * ratio ≥ 1.5 → score ≈ 0.99 (thrived through crises)
      * ratio == 1.0 → score == 0.5 (survived flat)
      * ratio ≤ 0.5 → score ≈ 0.01 (bank cratered)

    Distinct from the profit grader: profit is absolute growth, crisis
    recovery is *resilience* measured as a ratio so small and large
    businesses score comparably.
    """
    init_bal = float(initial_state.bank_balance or 0.0)
    final_bal = float(final_state.bank_balance or 0.0)
    if init_bal <= 0:
        return 0.5
    ratio = final_bal / init_bal
    score = 0.5 + (ratio - 1.0)
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
    "grade_stability_task",
    "grade_competitor_response_task",
    "grade_crisis_recovery_task",
    "generate_episode_tickets",
]
