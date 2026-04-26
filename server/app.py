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

.. note::
    **Concurrency model** (post-audit M-4 / B-2) — the factory creates
    ONE :class:`EcomEnv` per process. The env's state is serialized
    behind ``state["lock"]``, so every endpoint runs strictly sequentially
    even on a multi-threaded event loop. This intentionally keeps the
    simulation deterministic.

    **Do NOT run this app with ``uvicorn --workers N`` (N>1) or behind a
    pre-forking WSGI gateway.** Each worker gets its own process with its
    own env, its own per-env RNG, and its own initial-state snapshot, so
    grader scores become non-deterministic with respect to the caller's
    seed. The shipped ``Dockerfile`` sticks to a single uvicorn worker
    for exactly this reason. For horizontal scaling, run N independent
    single-worker containers behind a sticky-routing load balancer so
    each client hits the same env for its entire episode.
"""

from __future__ import annotations

import copy
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Type, get_args

import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, ValidationError


logger = logging.getLogger("commerceops")


class _WarningCollector(logging.Handler):
    """Buffer WARNING-level records emitted by the ``commerceops.*`` loggers.

    Post-audit C-5 — ``/config`` previously returned the loaded business
    metadata but left the operator to scrape stderr for any warnings the
    config validator emitted (unknown keys, deprecated keys, soft
    bankruptcy warnings). We now attach a short-lived collector during
    :func:`EcomEnv` construction / ``load_config`` and return the
    captured warnings in the response so clients can surface them in
    their UI without re-parsing the log stream.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(self.format(record))
        except Exception:  # pragma: no cover — never let logging crash.
            pass


def _capture_config_warnings():
    """Context-manager helper: attach a collector to the commerceops loggers.

    Used by ``/config`` (and ``create_app``) to surface validator
    warnings without changing any existing log routing; the collector is
    removed again on exit so it never leaks across requests.
    """
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        collector = _WarningCollector()
        collector.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        # Attach to the root "commerceops" logger so all submodule
        # warnings (commerceops.world_engine, commerceops.actions, ...)
        # are captured.
        logger.addHandler(collector)
        try:
            yield collector
        finally:
            logger.removeHandler(collector)

    return _cm()

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
        grade_stability_task,
        grade_competitor_response_task,
        grade_crisis_recovery_task,
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

# Post-audit round-2 (A2-48) — Windows reserves a handful of legacy
# device names that cannot be used as file stems even on case-insensitive
# filesystems. Even though the regex above already gates the slug to a
# small alphabet, a literal ``con`` or ``nul`` would still collide with
# the reserved names and crash ``open()`` on Windows during config load.
# Reject these up front so the failure surfaces as a clean 400 instead
# of a 500-level OS error.
_WINDOWS_RESERVED_SLUGS: frozenset = frozenset({
    "con", "prn", "aux", "nul",
    "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
})


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

    v2.3.x Phase A.1 — the body is now streamed chunk-by-chunk and the
    accumulator is aborted as soon as it crosses ``MAX_BODY_BYTES + 1``.
    This prevents oversized / chunked / lying clients from forcing the
    ASGI layer to buffer arbitrarily large payloads in memory before we
    get a chance to reject them. The ``content-length`` fast-path is
    preserved so honest clients short-circuit without touching the
    stream at all.
    """
    import json

    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > MAX_BODY_BYTES:
                raise _BodyTooLarge()
        except ValueError:
            return None

    limit = MAX_BODY_BYTES
    buf = bytearray()
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) > limit:
                raise _BodyTooLarge()
    except _BodyTooLarge:
        raise
    except (OSError, RuntimeError, ConnectionError):
        # Narrow on stream-read failures (A2-49 post-audit round-2).
        # Anything broader propagates so FastAPI's 500 handler can log
        # with a traceback rather than swallowing genuine bugs.
        return None

    if not buf:
        return None
    try:
        return json.loads(bytes(buf))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        # Post-audit round-2 (A2-49) — narrow the JSON-parse handler so
        # unexpected exceptions (e.g. OOM, keyboard interrupt) bubble up
        # instead of being silently coerced to ``None``.
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
        "evaluation_only": False,
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
        "evaluation_only": False,
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
        "evaluation_only": False,
        "grader": {
            "module": "ecom_env",
            "function": "grade_profit_task",
        },
    },
    {
        "id": "stability_task",
        "name": "Satisfaction Stability",
        "description": "Keep end-of-episode customer satisfaction at or above the stability target.",
        "difficulty": "easy",
        "evaluation_only": True,
        "grader": {
            "module": "ecom_env",
            "function": "grade_stability_task",
        },
    },
    {
        "id": "competitor_response_task",
        "name": "Competitor Response",
        "description": "Keep the agent's prices at or below the competitor's across observed SKUs.",
        "difficulty": "medium",
        "evaluation_only": True,
        "grader": {
            "module": "ecom_env",
            "function": "grade_competitor_response_task",
        },
    },
    {
        "id": "crisis_recovery_task",
        "name": "Crisis Recovery",
        "description": "Preserve or grow bank balance across market shocks and stockout cascades.",
        "difficulty": "hard",
        "evaluation_only": True,
        "grader": {
            "module": "ecom_env",
            "function": "grade_crisis_recovery_task",
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

    # Post-audit round-2 (A2-9) — OpenEnv validators sometimes hit
    # ``/grader`` before issuing an explicit ``/reset`` (they treat the
    # freshly-constructed env as the implicit baseline). Capture the
    # current observation here so that workflow succeeds with a sane
    # baseline even without an explicit reset. The /reset and /config
    # routes still overwrite this with the post-reset observation.
    if state["env"] is not None:
        try:
            state["initial_state"] = state["env"].state().model_copy(deep=True)
        except Exception:
            # Defensive — constructing the initial obs should never
            # fail, but if it does we fall back to the 409 behaviour.
            state["initial_state"] = None

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
    async def root(request: Request):
        env = state["env"]
        # C2 (Phase C) — content-negotiated landing page. JSON stays the
        # default to preserve the existing OpenEnv contract; HTML is
        # served only when the browser explicitly prefers it (or when
        # ``?format=html`` is passed). Tests using TestClient send
        # ``Accept: */*`` and continue to receive JSON.
        wants_html = request.query_params.get("format") == "html"
        if not wants_html:
            accept = request.headers.get("accept", "")
            if "text/html" in accept and "application/json" not in accept:
                wants_html = True
        if wants_html:
            # Browsers land on the long-form Gradio storytelling app
            # (mounted under /demo/ by demo/entry.py). The OpenEnv JSON
            # contract below stays the default for non-browser clients.
            return RedirectResponse(url="/demo/", status_code=307)

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
                "/demo",
            ],
        }

    @app.get("/demo")
    async def run_landing_demo(request: Request):
        """Server-Sent Events stream that runs a deterministic scripted demo.

        Used by the landing page "Run Demo" button (Phase C2). This
        endpoint is read-write w.r.t. env state and is therefore
        serialized behind the same lock as ``/step``. It does NOT alter
        the OpenEnv contract — clients that don't care can ignore it.
        """
        unavailable = _require_env()
        if unavailable is not None:
            return unavailable
        try:
            steps = int(request.query_params.get("steps", "30"))
        except (TypeError, ValueError):
            steps = 30
        try:
            seed = int(request.query_params.get("seed", "20260425"))
        except (TypeError, ValueError):
            seed = 20260425
        steps = max(1, min(steps, 50))

        from server.landing import stream_scripted_demo  # local import

        def _generator():
            with state["lock"]:
                obs = state["env"].reset(seed=seed)
                state["initial_state"] = obs.model_copy(deep=True)
                yield from stream_scripted_demo(state["env"], seed=seed, steps=steps)

        return StreamingResponse(_generator(), media_type="text/event-stream")

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
            if state["debug_enabled"]:
                logger.debug(
                    "commerceops.step action_type=%s invalid_action=1 reason=validation",
                    a_type,
                )
            return _bad_request("Action validation failed", errors=e.errors())
        except TypeError as e:
            if state["debug_enabled"]:
                logger.debug(
                    "commerceops.step action_type=%s invalid_action=1 reason=type_error",
                    a_type,
                )
            return _bad_request(f"Action validation failed: {e}")

        # Post-audit C.4 (v2.3.x) — lightweight structured telemetry gated on
        # ``COMMERCEOPS_DEBUG=1``. We emit one debug line per accepted step
        # with the action type, measured latency, and the ``info.invalid``
        # flag if the engine downgraded the action at dispatch. No metrics
        # backend is wired up here; this is purely a logging hook so
        # operators can tail the container log and spot anomalies.
        t_start = time.perf_counter() if state["debug_enabled"] else 0.0
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
            latency_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
            engine_invalid = 1 if bool(info.get("invalid", False)) else 0
            logger.debug(
                "commerceops.step action_type=%s latency_ms=%s invalid_action=%s",
                a_type,
                latency_ms,
                engine_invalid,
            )
            state["last_step_info"].clear()
            state["last_step_info"].update(
                {"action_type": a_type, "latency_ms": latency_ms, **info}
            )

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
        state and — as of v2.3.x Phase B.3 — never reads the request body
        either. The ``request`` parameter is kept purely for FastAPI route
        typing so the OpenAPI schema and OpenEnv validator keep seeing the
        same signature. v2.3 Phase 3.4 — each grader is called with
        ``context=env.grader_context`` explicitly so the per-env context
        bypasses the module-level mirror and two processes running
        different configs can't race.
        """
        del request  # explicitly unused; see docstring.
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
            # A2-1 — compute scores under the same lock as state/context
            # capture so a concurrent /step or /config cannot mutate
            # ``grader_context`` (or the observation backing ``final_state``)
            # between snapshot and grader execution.
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
                "stability_task": grade_stability_task(
                    initial_state, final_state, context=grader_ctx
                ),
                "competitor_response_task": grade_competitor_response_task(
                    initial_state, final_state, context=grader_ctx
                ),
                "crisis_recovery_task": grade_crisis_recovery_task(
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
        # Post-audit round-2 (A2-48) — reject Windows-reserved slugs.
        if business_id.lower() in _WINDOWS_RESERVED_SLUGS:
            return _bad_request(
                "Reserved business_id (Windows device name)",
                business_id=business_id,
                reserved=sorted(_WINDOWS_RESERVED_SLUGS),
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
        # Post-audit C-5 — capture any non-fatal warnings the validator
        # emits while loading this config so the response body can
        # surface them to the client. Fatal errors still come back as
        # 400s via the existing exception handler.
        warnings_captured: list[str] = []
        try:
            with state["lock"], _capture_config_warnings() as collector:
                if state["env"] is None:
                    state["env"] = EcomEnv(config_path=str(config_path))
                    state["degraded_reason"] = None
                    obs = state["env"].state()
                    if seed != 42:
                        obs = state["env"].reset(seed=seed)
                else:
                    obs = state["env"].load_config(str(config_path), seed=seed)
                state["initial_state"] = obs.model_copy(deep=True)
                warnings_captured = list(collector.records)
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
            "warnings": warnings_captured,
            "observation": obs.model_dump(),
        }

    return app


# Module-level ``app`` for OpenEnv runtime compatibility. The container
# entrypoint does ``uvicorn server.app:app`` so this name MUST exist.
app = create_app()


def main():
    # Audit MINOR #18 — soft runtime guard against the documented
    # anti-pattern ``uvicorn --workers N`` (N>1). Multiple workers
    # each construct an independent ``EcomEnv`` with a fresh RNG, so
    # graders go non-deterministic and the ``state["lock"]`` only
    # serialises *per-worker* requests. We check a couple of common
    # signals (``UVICORN_WORKERS`` env var, ``WEB_CONCURRENCY`` used
    # by Heroku-style buildpacks, and the ``--workers`` CLI flag) and
    # log a loud warning. We do NOT refuse to start — operators have
    # legitimate reasons to run multi-worker tests — but the warning
    # is hard to miss.
    import sys
    try:
        env_workers = int(os.environ.get("UVICORN_WORKERS", "1") or 1)
    except (TypeError, ValueError):
        env_workers = 1
    try:
        web_concurrency = int(os.environ.get("WEB_CONCURRENCY", "1") or 1)
    except (TypeError, ValueError):
        web_concurrency = 1
    cli_workers = 0
    for i, arg in enumerate(sys.argv):
        if arg == "--workers" and i + 1 < len(sys.argv):
            try:
                cli_workers = int(sys.argv[i + 1])
            except ValueError:
                cli_workers = 0
        elif arg.startswith("--workers="):
            try:
                cli_workers = int(arg.split("=", 1)[1])
            except ValueError:
                cli_workers = 0
    worker_count = max(env_workers, web_concurrency, cli_workers)
    if worker_count > 1:
        logger.warning(
            "multiworker_anti_pattern workers=%s "
            "CommerceOps v2 runs one EcomEnv per worker (independent RNG). "
            "Graders will be non-deterministic across workers. "
            "Deploy behind a single-worker container with sticky routing.",
            worker_count,
        )
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
