"""Phase 6.4 — grader bounds + multi-config hot-swap regression."""

from __future__ import annotations

import itertools

import pytest

from ecom_env import (
    EcomEnv,
    grade_triage_task,
    grade_inventory_task,
    grade_profit_task,
)


CONFIGS = [
    "configs/siyaani_fashion.json",
    "configs/medplus_pharmacy.json",
    "configs/stackbase_saas.json",
]

SEEDS = [0, 7, 42, 101]


def _score(fn, initial, final, env):
    """Helper — invoke a grader with the env's explicit context to dodge
    the m-6 DeprecationWarning that fires for callers relying on the
    module mirror.
    """
    ctx = getattr(env, "grader_context", None)
    if fn is grade_triage_task:
        return fn(initial, final)
    return fn(initial, final, context=ctx)


@pytest.mark.parametrize("cfg,seed", list(itertools.product(CONFIGS, SEEDS)))
def test_grader_scores_stay_in_open_interval(cfg, seed):
    env = EcomEnv(cfg)
    initial = env.reset(seed=seed).model_copy(deep=True)
    for _ in range(20):
        env.step({"action_type": "wait"})
    final = env.state()

    for fn in (grade_triage_task, grade_inventory_task, grade_profit_task):
        score = _score(fn, initial, final, env)
        assert 0.01 <= score <= 0.99, f"{fn.__name__} out of bounds on {cfg}/{seed}: {score}"


def test_negotiate_is_whitelisted_across_all_configs():
    for cfg in CONFIGS:
        env = EcomEnv(cfg)
        allowed = env.world_engine.config["actions"]["allowed"]
        assert "negotiate" in allowed, f"{cfg} missing 'negotiate' in allowed actions"


def test_hot_swap_preserves_grader_contract():
    env = EcomEnv(CONFIGS[0])
    env.reset(seed=5)
    for cfg in CONFIGS:
        initial = env.load_config(cfg, seed=5).model_copy(deep=True)
        for _ in range(10):
            env.step({"action_type": "wait"})
        final = env.state()
        for fn in (grade_triage_task, grade_inventory_task, grade_profit_task):
            score = _score(fn, initial, final, env)
            assert 0.01 <= score <= 0.99, (cfg, fn.__name__, score)


# ---------------------------------------------------------------------------
# Post-audit m-6 — DeprecationWarning when relying on the module mirror
# ---------------------------------------------------------------------------

def test_deprecated_grader_call_emits_warning():
    """Calling ``grade_inventory_task`` / ``grade_profit_task`` without
    ``context=`` must emit a :class:`DeprecationWarning`. Callers are
    expected to migrate to ``context=env.grader_context``.
    """
    env = EcomEnv(CONFIGS[0])
    initial = env.reset(seed=1).model_copy(deep=True)
    env.step({"action_type": "wait"})
    final = env.state()

    with pytest.warns(DeprecationWarning, match="grade_inventory_task"):
        grade_inventory_task(initial, final)
    with pytest.warns(DeprecationWarning, match="grade_profit_task"):
        grade_profit_task(initial, final)


def test_grader_explicit_context_does_not_warn(recwarn):
    """The fix path — passing ``context=env.grader_context`` — must not
    emit the DeprecationWarning.
    """
    env = EcomEnv(CONFIGS[0])
    initial = env.reset(seed=1).model_copy(deep=True)
    env.step({"action_type": "wait"})
    final = env.state()

    grade_inventory_task(initial, final, context=env.grader_context)
    grade_profit_task(initial, final, context=env.grader_context)
    depr = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
    assert not depr, [str(w.message) for w in depr]
