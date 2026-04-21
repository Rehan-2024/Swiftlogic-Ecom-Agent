"""Shared pytest fixtures for CommerceOps v2 regression tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def fresh_app():
    """Return a FastAPI TestClient over an isolated ``create_app()`` instance.

    v2.3 Phase 3.1 — the server moved from module-level ``env`` globals to
    a ``create_app(config_path=...)`` factory, so every test gets its own
    :class:`EcomEnv` instance without racing on shared state.
    """
    from fastapi.testclient import TestClient
    from server.app import create_app

    os.environ.setdefault("COMMERCEOPS_CONFIG", "configs/siyaani_fashion.json")

    app = create_app("configs/siyaani_fashion.json")
    client = TestClient(app)
    client.post("/reset", json={"seed": 42})
    # Expose the live app on the client so tests that need to poke the
    # internal state (e.g. toggle debug) can reach it via ``client.app``.
    return client


@pytest.fixture()
def world():
    """Return a freshly-built WorldEngine pre-reset with a deterministic seed."""
    from env.world_engine import WorldEngine
    w = WorldEngine("configs/siyaani_fashion.json")
    w.reset(seed=123)
    return w


@pytest.fixture()
def configs_root():
    return ROOT / "configs"
