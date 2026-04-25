"""Tests for the three evaluation-only graders added in Part A3.

These graders are additive (they do not feed the training reward) and must
be pure, deterministic, clamped to (0.01, 0.99), and reachable through
both the Python ``env.graders()`` map and the HTTP ``/grader`` / ``/tasks``
routes.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ecom_env import (
    EcomObservation,
    grade_competitor_response_task,
    grade_crisis_recovery_task,
    grade_stability_task,
)


def _obs(**overrides) -> EcomObservation:
    defaults = dict(
        current_day=1,
        step_count=0,
        bank_balance=100_000.0,
        inventory={},
        pending_orders={},
        active_tickets=[],
        daily_sales={},
        active_ad_spend={},
        customer_satisfaction=1.0,
        prices={},
        competitor_prices={},
    )
    defaults.update(overrides)
    return EcomObservation(**defaults)


# ---------------------------------------------------------------------------
# grade_stability_task
# ---------------------------------------------------------------------------

class TestStabilityGrader:
    def test_perfect_satisfaction_is_clamped_near_one(self):
        score = grade_stability_task(_obs(customer_satisfaction=1.0), _obs(customer_satisfaction=1.0))
        assert 0.9 <= score <= 0.99

    def test_zero_satisfaction_is_clamped_near_zero(self):
        score = grade_stability_task(_obs(customer_satisfaction=1.0), _obs(customer_satisfaction=0.0))
        assert 0.01 <= score <= 0.05

    def test_score_is_monotonic_in_final_satisfaction(self):
        initial = _obs(customer_satisfaction=1.0)
        scores = [
            grade_stability_task(initial, _obs(customer_satisfaction=c)) for c in (0.1, 0.3, 0.6, 0.9)
        ]
        assert scores == sorted(scores)

    def test_custom_target_lowers_required_bar(self):
        initial = _obs(customer_satisfaction=1.0)
        final = _obs(customer_satisfaction=0.5)
        strict = grade_stability_task(initial, final, context={"stability_target": 1.0})
        lenient = grade_stability_task(initial, final, context={"stability_target": 0.5})
        assert lenient > strict

    def test_score_is_clamped_tightly(self):
        for sat in (-5.0, 0.0, 0.5, 0.75, 1.0, 5.0):
            s = grade_stability_task(_obs(), _obs(customer_satisfaction=sat))
            assert 0.01 <= s <= 0.99


# ---------------------------------------------------------------------------
# grade_competitor_response_task
# ---------------------------------------------------------------------------

class TestCompetitorResponseGrader:
    def test_decisive_undercut_scores_high(self):
        final = _obs(
            prices={"sku_a": 80.0, "sku_b": 80.0},
            competitor_prices={"sku_a": 100.0, "sku_b": 100.0},
        )
        assert grade_competitor_response_task(_obs(), final) >= 0.9

    def test_matched_prices_score_neutral(self):
        final = _obs(
            prices={"sku_a": 100.0}, competitor_prices={"sku_a": 100.0}
        )
        score = grade_competitor_response_task(_obs(), final)
        assert 0.45 <= score <= 0.55

    def test_over_priced_scores_low(self):
        final = _obs(
            prices={"sku_a": 120.0}, competitor_prices={"sku_a": 100.0}
        )
        assert grade_competitor_response_task(_obs(), final) <= 0.1

    def test_missing_competitor_prices_returns_neutral(self):
        final = _obs(prices={"sku_a": 100.0}, competitor_prices={})
        assert grade_competitor_response_task(_obs(), final) == 0.5

    def test_zero_or_negative_entries_are_skipped(self):
        final = _obs(
            prices={"sku_a": 80.0, "sku_b": -5.0},
            competitor_prices={"sku_a": 100.0, "sku_b": 100.0},
        )
        score = grade_competitor_response_task(_obs(), final)
        assert score >= 0.9


# ---------------------------------------------------------------------------
# grade_crisis_recovery_task
# ---------------------------------------------------------------------------

class TestCrisisRecoveryGrader:
    def test_flat_balance_scores_neutral(self):
        s = grade_crisis_recovery_task(_obs(bank_balance=100.0), _obs(bank_balance=100.0))
        assert s == 0.5

    def test_growth_scores_high(self):
        s = grade_crisis_recovery_task(_obs(bank_balance=100.0), _obs(bank_balance=150.0))
        assert s >= 0.9

    def test_crash_scores_low(self):
        s = grade_crisis_recovery_task(_obs(bank_balance=100.0), _obs(bank_balance=40.0))
        assert s <= 0.1

    def test_zero_initial_balance_returns_neutral(self):
        s = grade_crisis_recovery_task(_obs(bank_balance=0.0), _obs(bank_balance=1000.0))
        assert s == 0.5


# ---------------------------------------------------------------------------
# wire-up — /tasks and /grader expose the new graders
# ---------------------------------------------------------------------------

class TestEvaluationOnlyWiring:
    def test_tasks_endpoint_lists_six_tasks_with_eval_flag(self, fresh_app: TestClient):
        resp = fresh_app.get("/tasks")
        assert resp.status_code == 200
        tasks = resp.json()
        ids = {t["id"] for t in tasks}
        assert ids == {
            "triage_task",
            "inventory_task",
            "profit_task",
            "stability_task",
            "competitor_response_task",
            "crisis_recovery_task",
        }
        eval_only = {t["id"] for t in tasks if t.get("evaluation_only")}
        assert eval_only == {"stability_task", "competitor_response_task", "crisis_recovery_task"}

    def test_grader_endpoint_returns_all_six_scores(self, fresh_app: TestClient):
        # Drive one real step so final_state differs from initial_state.
        fresh_app.post("/step", json={"action_type": "wait"})
        resp = fresh_app.post("/grader")
        assert resp.status_code == 200
        scores = {s["task_id"]: float(s["score"]) for s in resp.json()["scores"]}
        assert len(scores) == 6
        for name, v in scores.items():
            assert 0.01 <= v <= 0.99, f"{name} score {v} outside clamp"

    def test_env_graders_map_is_six(self):
        from ecom_env import EcomEnv

        env = EcomEnv(config_path="configs/siyaani_fashion.json")
        env.reset(seed=7)
        graders = env.graders()
        assert set(graders) == {
            "triage_task",
            "inventory_task",
            "profit_task",
            "stability_task",
            "competitor_response_task",
            "crisis_recovery_task",
        }
