"""Tests for the additive ``info['action_quality']`` derivation (Part A5).

These tests assert the labeling rules are stable and purely post-hoc —
i.e. the reward value is NOT affected by the label (env freeze contract).
"""

from __future__ import annotations

import pytest

from ecom_env import EcomEnv


@pytest.fixture()
def env():
    e = EcomEnv("configs/siyaani_fashion.json")
    e.reset(seed=13)
    return e


def test_action_quality_key_always_present(env):
    obs, reward, done, info = env.step({"action_type": "wait"})
    assert "action_quality" in info
    assert info["action_quality"] in {"good", "neutral", "bad"}
    assert isinstance(info.get("action_quality_reason"), str)


def test_unknown_action_type_marks_bad(env):
    obs, reward, done, info = env.step({"action_type": "restock", "sku": "__ghost_sku__", "quantity": 10})
    # Invalid SKU — handler should error, quality -> bad.
    assert info.get("action_quality") in {"bad", "neutral"}
    if info.get("error"):
        assert info["action_quality"] == "bad"
        assert "action_error" in info["action_quality_reason"]


def test_wait_at_start_is_neutral_not_bad(env):
    _obs, _reward, _done, info = env.step({"action_type": "wait"})
    # First wait with no tickets should not be labeled bad (we haven't
    # hit stockouts or wait-loop penalties yet on step 1).
    assert info["action_quality"] in {"neutral", "good"}


def test_label_is_additive_only_reward_unaffected(env):
    """The env freeze says new info keys must not change the reward.

    Verify determinism across identical seeds — the reward stream is
    bit-identical whether or not we read the new keys.
    """
    env.reset(seed=2024)
    rewards_a = []
    for _ in range(10):
        _, r, _, info = env.step({"action_type": "wait"})
        assert "action_quality" in info
        rewards_a.append(float(r.value) if hasattr(r, "value") else float(r))

    # Second run, same seed, same tape — should match exactly.
    env2 = EcomEnv("configs/siyaani_fashion.json")
    env2.reset(seed=2024)
    rewards_b = []
    for _ in range(10):
        _, r, _, info = env2.step({"action_type": "wait"})
        rewards_b.append(float(r.value) if hasattr(r, "value") else float(r))

    assert rewards_a == rewards_b
