"""Tests for the additive ``info['strategy_phase']`` derivation (Part A5).

The phase label is a pure function of already-emitted info / state and
must never mutate state or alter reward.
"""

from __future__ import annotations

import pytest

from ecom_env import EcomEnv


PHASES = {"explore", "stabilize", "recover", "exploit"}


@pytest.fixture()
def env():
    e = EcomEnv("configs/siyaani_fashion.json")
    e.reset(seed=42)
    return e


def test_phase_key_present_and_enumerated(env):
    _obs, _reward, _done, info = env.step({"action_type": "wait"})
    assert "strategy_phase" in info
    assert info["strategy_phase"] in PHASES
    assert 0.0 <= float(info["strategy_phase_confidence"]) <= 1.0
    assert isinstance(info.get("strategy_phase_note"), str)


def test_early_steps_are_explore(env):
    # max_steps in siyaani_fashion is 50, so steps 1..10 are in the
    # "explore" window (frac < 0.2).
    for i in range(5):
        _, _, _, info = env.step({"action_type": "wait"})
    assert info["strategy_phase"] == "explore"


def test_phase_does_not_change_reward(env):
    """Determinism regression — label never affects the reward tape."""
    env.reset(seed=999)
    rewards_a = []
    infos_a = []
    for _ in range(8):
        _, r, _, info = env.step({"action_type": "wait"})
        rewards_a.append(float(r.value) if hasattr(r, "value") else float(r))
        infos_a.append(info.get("strategy_phase"))

    env2 = EcomEnv("configs/siyaani_fashion.json")
    env2.reset(seed=999)
    rewards_b = []
    infos_b = []
    for _ in range(8):
        _, r, _, info = env2.step({"action_type": "wait"})
        rewards_b.append(float(r.value) if hasattr(r, "value") else float(r))
        infos_b.append(info.get("strategy_phase"))

    assert rewards_a == rewards_b
    assert infos_a == infos_b


def test_confidence_is_bounded(env):
    for _ in range(15):
        _, _, done, info = env.step({"action_type": "wait"})
        c = float(info["strategy_phase_confidence"])
        assert 0.0 <= c <= 1.0
        if done:
            break
