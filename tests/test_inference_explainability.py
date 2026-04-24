from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference import (
    build_step_trace,
    build_reward_summary,
    validate_trace_schema,
)


def _obs(day: int, bank: float, inv: dict, comp: dict, prices: dict, sales: dict, tickets: list):
    return SimpleNamespace(
        current_day=day,
        bank_balance=bank,
        inventory=inv,
        competitor_prices=comp,
        prices=prices,
        daily_sales=sales,
        active_tickets=tickets,
    )


def test_reward_summary_top_drivers_uses_actual_breakdown():
    info = {
        "reward_breakdown": {
            "revenue": 0.4,
            "solvency": 0.01,
            "delta": -0.2,
            "daily_revenue": 10.0,
            "scale_hint": "small",
        }
    }
    rs = build_reward_summary(0.21, info)
    assert rs["total"] == pytest.approx(0.21)
    # absolute values: revenue(0.4), delta(0.2)
    assert rs["top_drivers"] == ["revenue", "delta"]


def test_step_trace_schema_short_and_consistent():
    before = _obs(
        day=1,
        bank=1000.0,
        inv={"sku_a": 3},
        comp={"sku_a": 100.0},
        prices={"sku_a": 105.0},
        sales={"sku_a": 1},
        tickets=[{"status": "open", "urgency": "urgent"}],
    )
    after = _obs(
        day=2,
        bank=1020.0,
        inv={"sku_a": 1},
        comp={"sku_a": 98.0},
        prices={"sku_a": 99.0},
        sales={"sku_a": 4},
        tickets=[
            {"status": "open", "urgency": "urgent"},
            {"status": "open", "urgency": "critical"},
        ],
    )
    action = SimpleNamespace(action_type="set_price", model_dump=lambda: {"action_type": "set_price", "sku": "sku_a", "price": 99.0})
    info = {"reward_breakdown": {"revenue": 0.2, "delta": -0.1, "solvency": 0.05}}
    trace = build_step_trace(before, after, action, info, reward_val=0.15)
    validate_trace_schema(trace)
    for field in ("reasoning", "causal_chain", "why_it_worked"):
        vals = trace[field]
        assert 2 <= len(vals) <= 3
        assert all(len(str(v).split()) <= 12 for v in vals)


def test_integration_real_env_drives_valid_trace():
    """Audit MEDIUM #7 — exercise the builders against a real ``EcomEnv``.

    The ``SimpleNamespace`` fixtures above only cover fields the
    builders directly read, which misses Pydantic-coercion bugs and
    schema drift. This test wires the full pipeline: ``EcomEnv.reset``
    → ``EcomEnv.step`` → ``build_step_trace`` → ``validate_trace_schema``
    across one of every action type so the new Wave-3/4 ``info`` keys
    (``intent``, ``kpis``, ``trend``, ``competitor_reaction``, etc.) are
    actually populated and consumed end-to-end.
    """
    from ecom_env import (
        EcomEnv,
        WaitAction,
        AdSpendAction,
        NegotiateAction,
        RestockAction,
        SetPriceAction,
    )

    env = EcomEnv("configs/siyaani_fashion.json")
    obs_before = env.reset(seed=42)
    inv0 = getattr(obs_before, "inventory", {}) or {}
    sku = sorted(inv0.keys())[0]

    actions = [
        WaitAction(),
        SetPriceAction(sku=sku, price=1700.0),
        NegotiateAction(sku=sku, quantity=5),
        RestockAction(sku=sku, quantity=5),
        AdSpendAction(sku=sku, budget=150.0),
    ]

    for act in actions:
        obs_after, reward, done, info = env.step(act)
        trace = build_step_trace(obs_before, obs_after, act, info, reward_val=float(reward.value))
        validate_trace_schema(trace)
        # Engine-side Wave 1/2/3 fields must be present in info.
        for k in (
            "competitor_reaction",
            "demand_factors",
            "action_effect",
            "kpis",
            "trend",
            "intent",
            "why_failed",
            "confidence",
            "policy_stability",
            "anomalies",
        ):
            assert k in info, f"missing engine info key: {k}"
        # Explainability-truth: when the reaction is not triggered the
        # trace must not claim undercut/follow.
        triggered = bool(info["competitor_reaction"].get("triggered", False))
        narrated = trace["market_reaction"]["competitor_action"]
        if not triggered:
            assert narrated not in {"undercut", "follow"}, (
                f"explainer narrated causal competitor reaction {narrated!r} "
                f"but engine did not trigger one"
            )
        # Wave 4: department suggestions + decision_context shape.
        dept = trace.get("departments", {})
        assert set(dept.keys()) == {"inventory", "marketing", "support"}
        ctx = trace.get("decision_context", {})
        assert ctx.get("chosen_action") == act.action_type
        assert isinstance(ctx.get("based_on"), list) and ctx["based_on"], \
            "decision_context.based_on must be a non-empty list"
        obs_before = obs_after
        if done:
            break


def test_trace_deterministic_same_inputs_same_output():
    before = _obs(
        day=1,
        bank=500.0,
        inv={"sku_a": 10},
        comp={"sku_a": 100.0},
        prices={"sku_a": 100.0},
        sales={"sku_a": 2},
        tickets=[],
    )
    after = _obs(
        day=2,
        bank=510.0,
        inv={"sku_a": 8},
        comp={"sku_a": 100.0},
        prices={"sku_a": 100.0},
        sales={"sku_a": 2},
        tickets=[],
    )
    action = SimpleNamespace(action_type="wait", model_dump=lambda: {"action_type": "wait"})
    info = {"reward_breakdown": {"revenue": 0.1, "delta": 0.0}}
    t1 = build_step_trace(before, after, action, info, reward_val=0.1)
    t2 = build_step_trace(before, after, action, info, reward_val=0.1)
    assert t1 == t2
