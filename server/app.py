"""
server/app.py — FastAPI front-end for the CommerceOps v2.3 environment.

v2.3 Phase 3.1 refactored the single module-level ``app`` + ``env`` globals
into a :func:`create_app` factory so tests (and future multi-tenant runners)
can build isolated :class:`EcomEnv` instances without tripping on shared
state. The module-level ``app = create_app()`` is preserved so the OpenEnv
container entrypoint keeps working unchanged.

When ``EcomEnv()`` construction fails (missing or malformed config) the
factory now returns a "degraded" app that responds 503 on every mutating
endpoint and exposes a diagnostic ``/health`` payload, instead of crashing
the process at import time.
"""

from __future__ import annotations

import copy
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Type, get_args

import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError


logger = logging.getLogger("commerceops")

# Hard cap on inbound JSON body size. The simulation only ever accepts small
# control payloads ({seed}, {business_id}, action dicts), so 64 KiB is ample.
MAX_BODY_BYTES = 64 * 1024

# Ensure ecom_env is importable from both local and Docker contexts.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from ecom_env import (
        EcomEnv,
        EcomAction,
        EcomObservation,
        EcomReward,
        RestockAction,
        RefundAction,
        AdSpendAction,
        NegotiateAction,
        WaitAction,
        SetPriceAction,
        grade_triage_task,
        grade_inventory_task,
        grade_profit_task,
    )
except ImportError as e:
    raise RuntimeError(f"Cannot import ecom_env — check PYTHONPATH. Error: {e}")


# ---------------------------------------------------------------------------
# Centralized action dispatch: one mapping keeps the server layer and the
# discriminated EcomAction union aligned as actions evolve.
#
# Post-audit m-4 — previously hand-maintained. Now derived from the
# ``EcomAction`` discriminated union so adding a new variant in
# ``ecom_env.py`` automatically propagates here; forgetting to update this
# table is no longer possible. The runtime assertion below (against the
# engine's ``_KNOWN_ACTIONS`` whitelist) catches any drift at import time.
# ---------------------------------------------------------------------------

def _derive_action_models() -> Dict[str, Type[BaseModel]]:
    """Walk the ``EcomAction`` discriminated union and build a
    ``{action_type_literal: model_cls}`` map.

    Pydantic v2 strips the ``Annotated[...]`` wrapper from the root field
    before we see it and exposes the inner ``Union`` members directly, so
    ``get_args(root_annotation)`` returns the member tuple. Be defensive
    and handle the ``Annotated``-wrapped shape too in case a future
    Pydantic release changes this.
    """
    root_annotation = EcomAction.model_fields["root"].annotation
    args = get_args(root_annotation)
    if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
        members = args
    else:
        # Legacy/alternative shape: Annotated[Union[...], discriminator-field]
        union_type = args[0]
        members = get_args(union_type)
    out: Dict[str, Type[BaseModel]] = {}
    for member in members:
        default = member.model_fields["action_type"].default
        out[default] = member
    return out


_ACTION_MODELS: Dict[str, Type[BaseModel]] = _derive_action_models()

# Drift guard: every action exposed on the wire must also be known to the
# world engine's allowlist validator, otherwise ``actions.allowed`` could
# accept values the server refuses to dispatch (or vice versa).
try:
    from env.world_engine import WorldEngine as _WorldEngineForCheck

    _SERVER_ACTIONS = set(_ACTION_MODELS.keys())
    _ENGINE_ACTIONS = set(_WorldEngineForCheck._KNOWN_ACTIONS)
    assert _SERVER_ACTIONS == _ENGINE_ACTIONS, (
        f"Action registry drift: server={_SERVER_ACTIONS} "
        f"engine={_ENGINE_ACTIONS}"
    )
except ImportError:  # pragma: no cover — engine import is a hard dep at runtime
    pass

# Only filesystem-safe slugs are allowed for /config business ids.
_BUSINESS_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")


def _available_business_ids() -> set:
    """Return the set of config slugs currently discoverable on disk."""
    cfg_dir = Path(ROOT) / "configs"
    if not cfg_dir.exists():
        return set()
    return {p.stem for p in cfg_dir.glob("*.json") if p.is_file()}


class _BodyTooLarge(Exception):
    """Raised internally when inbound body exceeds MAX_BODY_BYTES."""


async def _safe_json(request: Request) -> Any:
    """Best-effort JSON body read that never raises to the caller.

    Returns the parsed JSON, ``None`` on malformed/empty bodies, or raises
    ``_BodyTooLarge`` when the *actual* body size exceeds ``MAX_BODY_BYTES``.
    Callers translate that into a stable 413.

    v2.3 Phase 2.1 — previously the cap was header-only, so chunked or
    lying clients could skip it. We now also measure ``len(raw)`` after
    reading the full body so oversized payloads are rejected regardless
    of transfer encoding. The fast-fail on the header is preserved as a
    cheap short-circuit when honest clients announce their size.
    """
    import json

    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > MAX_BODY_BYTES:
                raise _BodyTooLarge()
        except ValueError:
            return None
    try:
        raw = await request.body()
    except _BodyTooLarge:
        raise
    except Exception:
        return None
    if raw is None:
        return None
    if len(raw) > MAX_BODY_BYTES:
        raise _BodyTooLarge()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _bad_request(detail: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": detail, **extra})


def _payload_too_large() -> JSONResponse:
    # v2.3 Phase 3.3 — enrich the 413 body so clients can introspect the
    # limit without parsing the English ``detail`` string.
    return JSONResponse(
        status_code=413,
        content={
            "detail": f"Request body exceeds {MAX_BODY_BYTES} bytes",
            "max_bytes": MAX_BODY_BYTES,
        },
    )


def _service_unavailable(reason: str) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"status": "degraded", "reason": reason},
    )


TASKS = [
    {
        "id": "triage_task",
        "name": "Customer Ticket Triage",
        "description": "Resolve all open customer support tickets by issuing refund actions.",
        "difficulty": "easy",
        "grader": {
            "module": "ecom_env",
            "function": "grade_triage_task",
        },
    },
    {
        "id": "inventory_task",
        "name": "Inventory Management",
        "description": "Maintain the target SKU stock above its configured safety threshold.",
        "difficulty": "medium",
        "grader": {
            "module": "ecom_env",
            "function": "grade_inventory_task",
        },
    },
    {
        "id": "profit_task",
        "name": "Profit Maximization",
        "description": "Grow the bank balance beyond the initial seed capital over a 50-day cycle.",
        "difficulty": "hard",
        "grader": {
            "module": "ecom_env",
            "function": "grade_profit_task",
        },
    },
]


# ---------------------------------------------------------------------------
# App factory (v2.3 Phase 3.1)
# ---------------------------------------------------------------------------

def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Build a fresh FastAPI app + EcomEnv pair.

    v2.3 Phase 3.1 — previously the module imported ``app`` / ``env`` as
    globals, which crashed the process when the default config was
    missing. The factory now wraps construction in a try/except and falls
    back to a "degraded" app where every mutating endpoint returns 503
    with a diagnostic ``/health`` payload. Healthy endpoints come back the
    moment a successful ``/config`` call succeeds.
    """
    app = FastAPI(title="Swiftlogic CommerceOps v2")

    debug_enabled = os.environ.get("COMMERCEOPS_DEBUG", "") == "1"

    # All per-app mutable state lives here so route handlers capture a
    # single namespace via closure. This makes it trivial to construct
    # multiple independent apps in tests.
    state = {
        "env": None,
        "initial_state": None,
        "degraded_reason": None,
        "debug_enabled": debug_enabled,
        "last_step_info": {},
        "lock": threading.Lock(),
    }

    try:
        if config_path:
            state["env"] = EcomEnv(config_path=config_path)
        else:
            state["env"] = EcomEnv()
    except Exception as exc:
        # Log full traceback but don't crash — the process stays up and
        # serves 503s until a successful /config swap installs a working env.
        logger.exception("EcomEnv construction failed; starting in degraded mode")
        state["degraded_reason"] = f"{exc.__class__.__name__}: {exc}"

    # Expose for tests / debugging without leaking into the HTTP surface.
    app.state.commerceops = state

    def _require_env() -> Optional[JSONResponse]:
        """Return a 503 JSONResponse when the env is missing, else None."""
        if state["env"] is None:
            return _service_unavailable(
                state["degraded_reason"] or "Environment not initialized"
            )
        return None

    # ---- Routes --------------------------------------------------------

    @app.get("/health")
    async def health():
        if state["env"] is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "reason": state["degraded_reason"] or "Environment not initialized",
                },
            )
        return {"status": "ok"}

    @app.get("/")
    async def root():
        env = state["env"]
        if env is None:
            return {
                "status": "degraded",
                "reason": state["degraded_reason"],
                "endpoints": ["/health", "/config"],
            }
        return {
            "status": "online",
            "environment": "Swiftlogic CommerceOps v2",
            "version": "2.3.0",
            "framework": "OpenEnv v0.2.3",
            "business_id": env.world_engine.config.get("business_id"),
            "display_name": env.world_engine.config.get("display_name"),
            "endpoints": [
                "/reset",
                "/step",
                "/state",
                "/tasks",
                "/grader",
                "/config",
                "/health",
            ],
        }

    @app.post("/reset")
    async def reset_env(request: Request):
        unavailable = _require_env()
        if unavailable is not None:
            return unavailable
        try:
            body = await _safe_json(request)
        except _BodyTooLarge:
            return _payload_too_large()
        seed = 42
        if isinstance(body, dict):
            try:
                seed = int(body.get("seed", 42))
            except (TypeError, ValueError):
                return _bad_request("Field 'seed' must be an integer")

        with state["lock"]:
            obs = state["env"].reset(seed=seed)
            state["initial_state"] = obs.model_copy(deep=True)
        return {"observation": obs.model_dump(), "reward": 0.0, "done": False}

    @app.post("/step")
    async def step_env(request: Request):
        unavailable = _require_env()
        if unavailable is not None:
            return unavailable
        try:
            body = await _safe_json(request)
        except _BodyTooLarge:
            return _payload_too_large()
        if body is None:
            return _bad_request("Request body must be valid JSON")

        action_data = body.get("action", body) if isinstance(body, dict) else body
        if not isinstance(action_data, dict):
            return _bad_request("Action payload must be a JSON object")

        a_type = action_data.get("action_type")
        model_cls = _ACTION_MODELS.get(a_type)
        if model_cls is None:
            return _bad_request(
                f"Unknown action_type: {a_type!r}",
                allowed=sorted(_ACTION_MODELS.keys()),
            )

        try:
            action = model_cls(**action_data)
        except ValidationError as e:
            return _bad_request("Action validation failed", errors=e.errors())
        except TypeError as e:
            return _bad_request(f"Action validation failed: {e}")

        try:
            with state["lock"]:
                obs, reward, done, info = state["env"].step(action)
        except Exception:
            logger.exception("Environment step failed")
            return JSONResponse(
                status_code=500,
                content={"detail": "Environment step failed"},
            )

        if state["debug_enabled"]:
            state["last_step_info"].clear()
            state["last_step_info"].update({"action_type": a_type, **info})

        return {
            "observation": obs.model_dump(),
            "reward": reward.value,
            "done": done,
            "info": info,
        }

    @app.get("/state")
    async def get_state():
        unavailable = _require_env()
        if unavailable is not None:
            return unavailable
        with state["lock"]:
            obs = state["env"].state()
        return {"observation": obs.model_dump()}

    @app.get("/debug/last_step_info")
    async def debug_last_step_info():
        """Return the cached ``info`` dict from the most recent /step call.

        v2.3 Phase 3.2 — the response is a ``copy.deepcopy`` of the cache
        so mutating callers (e.g. tests that pop reward_breakdown before
        asserting) can't poison the next debug read.
        """
        if not state["debug_enabled"]:
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        return {"info": copy.deepcopy(state["last_step_info"])}

    @app.get("/tasks")
    async def get_tasks():
        return TASKS

    @app.post("/grader")
    async def run_grader(request: Request):
        """Run all graders against current state and return scores.

        Returns 409 when no baseline has been snapshotted. A grader call is
        a strictly read-only operation, so this endpoint never mutates env
        state. v2.3 Phase 3.4 — each grader is called with
        ``context=env.grader_context`` explicitly so the per-env context
        bypass the module-level mirror and two processes running different
        configs can't race.
        """
        unavailable = _require_env()
        if unavailable is not None:
            return unavailable

        with state["lock"]:
            if state["initial_state"] is None:
                return JSONResponse(
                    status_code=409,
                    content={
                        "detail": "Grader has no baseline. Call /reset before /grader.",
                    },
                )
            final_state = state["env"].state()
            initial_state = state["initial_state"]
            grader_ctx = state["env"].grader_context

        scores = {
            "triage_task": grade_triage_task(
                initial_state, final_state, context=grader_ctx
            ),
            "inventory_task": grade_inventory_task(
                initial_state, final_state, context=grader_ctx
            ),
            "profit_task": grade_profit_task(
                initial_state, final_state, context=grader_ctx
            ),
        }

        results = []
        for task in TASKS:
            tid = task["id"]
            score = scores.get(tid, 0.0)
            results.append(
                {
                    "task_id": tid,
                    "score": round(score, 4),
                    "grader": task["grader"],
                }
            )

        return {"scores": results}

    @app.post("/config")
    async def load_config(request: Request):
        """Hot-reload the environment with a different business config.

        Example::

            curl -X POST http://localhost:7860/config \
                -H "Content-Type: application/json" \
                -d '{"business_id": "medplus_pharmacy"}'

        Only business_ids matching ``^[a-z0-9][a-z0-9_\\-]{0,63}$`` and
        backed by a real ``configs/<id>.json`` file on disk are accepted;
        anything else returns a stable 4xx instead of touching the
        filesystem. In degraded mode (env construction failed at startup)
        a successful /config lifts the degraded flag.
        """
        try:
            body = await _safe_json(request)
        except _BodyTooLarge:
            return _payload_too_large()

        business_id = "siyaani_fashion"
        seed = 42
        if isinstance(body, dict):
            business_id = body.get("business_id", business_id) or business_id
            try:
                seed = int(body.get("seed", 42))
            except (TypeError, ValueError):
                return _bad_request("Field 'seed' must be an integer")

        if not isinstance(business_id, str) or not _BUSINESS_ID_RE.match(business_id):
            return _bad_request(
                "Invalid business_id format",
                business_id=str(business_id),
                pattern=_BUSINESS_ID_RE.pattern,
            )

        available = _available_business_ids()
        if business_id not in available:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"Config '{business_id}' not found",
                    "available": sorted(available),
                },
            )

        config_path = Path(ROOT) / "configs" / f"{business_id}.json"
        try:
            with state["lock"]:
                if state["env"] is None:
                    # Recover from degraded mode: build a fresh env now.
                    state["env"] = EcomEnv(config_path=str(config_path))
                    state["degraded_reason"] = None
                    obs = state["env"].state()
                    if seed != 42:
                        obs = state["env"].reset(seed=seed)
                else:
                    obs = state["env"].load_config(str(config_path), seed=seed)
                state["initial_state"] = obs.model_copy(deep=True)
        except Exception:
            logger.exception("Failed to load config %s", business_id)
            return _bad_request(f"Failed to load config '{business_id}'")

        env = state["env"]
        return {
            "status": "config_loaded",
            "business_id": business_id,
            "display_name": env.world_engine.config.get("display_name"),
            "currency": env.world_engine.config.get("currency"),
            "products": [p["sku"] for p in env.world_engine.config.get("products", [])],
            "observation": obs.model_dump(),
        }

    return app


# Module-level ``app`` for OpenEnv runtime compatibility. The container
# entrypoint does ``uvicorn server.app:app`` so this name MUST exist.
app = create_app()


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
