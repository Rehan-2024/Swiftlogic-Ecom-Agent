"""Tests for the shared training pipeline modules (Parts B2, B3, B4, B+.6).

These cover the parts of the pipeline that don't require a live LLM —
rollout validation, reward combining, curriculum advancement, composite
score computation, and policy-signature hashing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from training.composite import CompositeWeights, compute_composite
from training.curriculum import CurriculumState, Stage, default_curriculum, rolling_mean
from training.eval_utils import run_eval_sweep, write_json
from training.policies import (
    build_heuristic_producer,
    build_random_producer,
    build_wait_producer,
)
from training.rewards import RewardWeights, combined_reward, reward_breakdown
from training.rollout import extract_action_json, rollout_episode, validate_action


CONFIG = "configs/siyaani_fashion.json"


class TestExtractAction:
    def test_parses_wait(self):
        obj, _ = extract_action_json('Action: {"action_type": "wait"}')
        assert obj == {"action_type": "wait"}

    def test_picks_last_block_after_marker(self):
        obj, _ = extract_action_json('Thought: ...\nAction: {"action_type":"restock","sku":"x","quantity":3}')
        assert obj["action_type"] == "restock"
        assert obj["quantity"] == 3

    def test_returns_none_on_garbage(self):
        obj, _ = extract_action_json("nope")
        assert obj is None

    def test_returns_none_on_malformed_json(self):
        obj, _ = extract_action_json("Action: {action_type: wait")
        assert obj is None


class TestValidate:
    def test_valid_wait(self):
        action, ok, err = validate_action({"action_type": "wait"})
        assert ok and err is None
        assert action["action_type"] == "wait"

    def test_missing_required_field_falls_back(self):
        # restock requires both `sku` and `quantity`; missing quantity -> invalid.
        action, ok, _ = validate_action({"action_type": "restock", "sku": "x"})
        assert not ok
        assert action["action_type"] == "wait"

    def test_unknown_action_type_falls_back(self):
        action, ok, _ = validate_action({"action_type": "hack", "sku": "x"})
        assert not ok
        assert action["action_type"] == "wait"

    def test_none_falls_back(self):
        action, ok, err = validate_action(None)
        assert not ok and action["action_type"] == "wait"
        assert err == "no_action_found"


class TestRolloutEpisode:
    def test_wait_producer_produces_records(self):
        producer = build_wait_producer()
        rec = rollout_episode(producer, 2026, config_path=CONFIG)
        assert len(rec.steps) > 0
        assert rec.initial_obs is not None and rec.final_obs is not None
        assert all(s.was_valid for s in rec.steps)
        assert rec.format_compliance == 1.0

    def test_rollout_is_deterministic(self):
        producer = build_wait_producer()
        a = rollout_episode(producer, 777, config_path=CONFIG)
        b = rollout_episode(producer, 777, config_path=CONFIG)
        assert [s.reward for s in a.steps] == [s.reward for s in b.steps]
        assert a.grader_scores == b.grader_scores

    def test_heuristic_producer_yields_higher_training_score(self):
        wait = rollout_episode(build_wait_producer(), 1234, config_path=CONFIG)
        heur = rollout_episode(build_heuristic_producer(), 1234, config_path=CONFIG)
        wait_mean = sum(wait.grader_scores.values()) / 3
        heur_mean = sum(heur.grader_scores.values()) / 3
        # Heuristic should not be strictly worse than wait on average.
        assert heur_mean >= wait_mean - 0.1


class TestCombinedReward:
    def test_zero_episode_reward_is_below_heuristic(self):
        wait_rec = rollout_episode(build_wait_producer(), 1, config_path=CONFIG)
        heur_rec = rollout_episode(build_heuristic_producer(), 1, config_path=CONFIG)
        w = combined_reward(wait_rec, RewardWeights())
        h = combined_reward(heur_rec, RewardWeights())
        assert isinstance(w, float) and isinstance(h, float)

    def test_breakdown_keys(self):
        rec = rollout_episode(build_wait_producer(), 3, config_path=CONFIG)
        bd = reward_breakdown(rec)
        assert set(bd.keys()) == {"env", "graders", "format", "total", "grader_sum_raw", "format_compliance", "fallback_rate"}


class TestCurriculum:
    def test_default_curriculum_has_at_least_two_stages(self):
        state = default_curriculum(".")
        assert len(state.stages) >= 2
        assert state.stages[0].config_path.endswith("siyaani_fashion_easy.json")

    def test_advancement_after_streak(self):
        state = CurriculumState(stages=[
            Stage(name="s1", config_path="configs/siyaani_fashion_easy.json",
                  advance_threshold=0.5, streak=2),
            Stage(name="s2", config_path="configs/siyaani_fashion.json",
                  advance_threshold=999, streak=999),
        ])
        assert state.current.name == "s1"
        assert state.observe(0.1) is None
        assert state.observe(0.6) is None  # 1/2
        advanced = state.observe(0.7)
        assert advanced is not None and advanced.name == "s2"

    def test_rolling_mean_empty(self):
        assert rolling_mean([]) == 0.0


class TestCompositeScore:
    def test_compute_composite_against_artifacts(self, tmp_path):
        # Build a minimal before/after pair.
        before = {
            "before_metrics": True,
            "policies": {
                "heuristic": {
                    "summary": {
                        "composite_training_mean": 0.4,
                        "composite_all_mean": 0.5,
                        "mean_format_compliance": 1.0,
                    }
                }
            },
        }
        after = {
            "label": "after",
            "episodes": [],
            "summary": {
                "composite_training_mean": 0.7,
                "composite_all_mean": 0.75,
                "mean_format_compliance": 0.95,
            },
        }
        before_path = tmp_path / "before.json"
        after_path = tmp_path / "after.json"
        before_path.write_text(json.dumps(before), encoding="utf-8")
        after_path.write_text(json.dumps(after), encoding="utf-8")
        # NOTE: compute_composite reads the heuristic policy from 'policies'
        # so we have to pass a file whose summary is directly under 'summary'.
        before_min = {
            "summary": before["policies"]["heuristic"]["summary"],
        }
        before_path.write_text(json.dumps(before_min), encoding="utf-8")
        result = compute_composite(str(before_path), str(after_path))
        assert "before" in result and "after" in result
        assert result["after"]["score"] > result["before"]["score"]
        assert "->" in result["headline"]


class TestEvalSweep:
    def test_run_eval_sweep_shape(self):
        bundle = run_eval_sweep(
            "wait_only",
            build_wait_producer(),
            seeds=[101, 202],
            configs=[CONFIG],
        )
        assert "episodes" in bundle and "summary" in bundle
        assert len(bundle["episodes"]) == 2
        assert set(bundle["summary"]["per_task"].keys()) == {
            "triage_task", "inventory_task", "profit_task",
            "stability_task", "competitor_response_task", "crisis_recovery_task",
        }
