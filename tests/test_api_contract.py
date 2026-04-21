"""Phase 6.1 — endpoint contract tests.

Guardrails these tests enforce:
  * Required OpenEnv routes all respond with stable JSON shapes.
  * Both flat and wrapped action payloads are accepted on /step.
  * Unknown / malformed payloads yield 4xx, never 500.
  * /config rejects unsafe business_ids and only loads known configs.
"""

from __future__ import annotations


def test_health_and_root(fresh_app):
    r = fresh_app.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

    r = fresh_app.get("/")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "online"
    assert "endpoints" in payload


def test_reset_returns_observation_shape(fresh_app):
    r = fresh_app.post("/reset", json={"seed": 7})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"observation", "reward", "done"}
    obs = body["observation"]
    for key in (
        "current_day",
        "step_count",
        "bank_balance",
        "inventory",
        "daily_sales",
        "active_ad_spend",
        "supplier_quotes",
    ):
        assert key in obs, f"missing obs field: {key}"


def test_step_accepts_flat_and_wrapped_action(fresh_app):
    flat = fresh_app.post("/step", json={"action_type": "wait"})
    assert flat.status_code == 200
    wrapped = fresh_app.post("/step", json={"action": {"action_type": "wait"}})
    assert wrapped.status_code == 200


def test_step_unknown_action_is_400_not_500(fresh_app):
    r = fresh_app.post("/step", json={"action_type": "teleport"})
    assert r.status_code == 400
    body = r.json()
    assert "allowed" in body
    assert "wait" in body["allowed"]


def test_step_missing_fields_returns_400(fresh_app):
    # restock without sku/quantity should fail validation, not crash.
    r = fresh_app.post("/step", json={"action_type": "restock"})
    assert r.status_code == 400
    assert "errors" in r.json()


def test_step_invalid_json_returns_400(fresh_app):
    r = fresh_app.post(
        "/step",
        content="not-json-at-all",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400


def test_step_non_dict_body_returns_400(fresh_app):
    r = fresh_app.post("/step", json=[1, 2, 3])
    assert r.status_code == 400


def test_grader_scores_always_in_open_interval(fresh_app):
    r = fresh_app.post("/grader")
    assert r.status_code == 200
    for entry in r.json()["scores"]:
        assert 0.01 <= entry["score"] <= 0.99


def test_tasks_endpoint_returns_three_tasks(fresh_app):
    r = fresh_app.get("/tasks")
    assert r.status_code == 200
    ids = [t["id"] for t in r.json()]
    assert set(ids) == {"triage_task", "inventory_task", "profit_task"}


def test_config_rejects_path_traversal(fresh_app):
    r = fresh_app.post("/config", json={"business_id": "../etc/passwd"})
    assert r.status_code == 400


def test_config_rejects_unknown_id_with_available_list(fresh_app):
    r = fresh_app.post("/config", json={"business_id": "ghost_corp"})
    assert r.status_code == 404
    assert isinstance(r.json().get("available"), list)


def test_config_hot_swap_works(fresh_app):
    r = fresh_app.post("/config", json={"business_id": "medplus_pharmacy"})
    assert r.status_code == 200
    assert r.json()["business_id"] == "medplus_pharmacy"


def test_wait_step_observation_is_backward_compatible(fresh_app):
    r = fresh_app.post("/step", json={"action_type": "wait"})
    assert r.status_code == 200
    obs = r.json()["observation"]
    # v2 additions must stay defaulted (present but may be zero/empty).
    assert "prices" in obs and "competitor_prices" in obs
    assert "supplier_quotes" in obs
    # Phase A.3 — quote expiry map is always surfaced on the observation.
    assert "supplier_quote_expiry" in obs


def test_grader_without_reset_returns_409():
    """Phase B.1 — /grader must not silently reset the env.

    We build a fresh server singleton without calling /reset, then hit
    /grader and assert a 409. The server must also NOT have mutated state.
    """
    from fastapi.testclient import TestClient
    from server import app as app_module
    from ecom_env import EcomEnv

    app_module.env = EcomEnv("configs/siyaani_fashion.json")
    app_module._initial_state = None
    client = TestClient(app_module.app)
    before = client.get("/state").json()

    r = client.post("/grader")
    assert r.status_code == 409
    assert "Call /reset" in r.json()["detail"]

    after = client.get("/state").json()
    assert before == after, "/grader call must be read-only when no baseline"


def test_oversized_body_returns_413(fresh_app):
    """Phase B.3 — body size cap rejects malicious payloads with 413."""
    huge = {"action_type": "wait", "padding": "x" * (128 * 1024)}
    r = fresh_app.post("/step", json=huge)
    assert r.status_code == 413


def test_step_500_path_does_not_leak_internal_error(monkeypatch, fresh_app):
    """Phase B.2 — the last-resort 500 path must not leak exception strings.

    v2.3 Phase 3.1 — the server is now built via ``create_app`` and each
    test client owns its own :class:`EcomEnv`, stored on
    ``app.state.commerceops["env"]``. Patching the module-level ``env`` no
    longer affects the fixture's env, so we monkey-patch the per-client
    instance directly.
    """

    def _boom(_action):
        raise RuntimeError("SECRET_INTERNAL_STATE_x7y9")

    env = fresh_app.app.state.commerceops["env"]
    monkeypatch.setattr(env, "step", _boom)
    r = fresh_app.post("/step", json={"action_type": "wait"})
    assert r.status_code == 500
    body = r.json()
    assert body == {"detail": "Environment step failed"}, body
    assert "SECRET_INTERNAL_STATE_x7y9" not in r.text


def test_action_models_cover_discriminated_union():
    """Post-audit m-4 — the server's ``_ACTION_MODELS`` dispatch table must
    exactly match the discriminated union in :mod:`ecom_env`. Deriving one
    from the other closes the v2.3 manual-sync footgun.
    """
    from server.app import _ACTION_MODELS
    from env.world_engine import WorldEngine

    assert set(_ACTION_MODELS.keys()) == {
        "restock", "refund", "ad_spend", "negotiate", "wait", "set_price"
    }
    assert set(_ACTION_MODELS.keys()) == set(WorldEngine._KNOWN_ACTIONS)
