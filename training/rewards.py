"""Combined training reward (Part B3).

``training_reward = alpha * env_reward + beta * sum(3 training graders) +
                    gamma * format_reward``

Only the three *training* graders (triage, inventory, profit) feed this
sum. The three evaluation-only graders (stability, competitor_response,
crisis_recovery) are deliberately excluded — they are scored separately
at evaluation time so we can measure generalization across reward
surfaces without baking them into the policy gradient (roadmap B.3 +
guide §7 "add gradually").

Format reward = ``episode.format_compliance`` — the fraction of steps
whose raw model output parsed into a valid ``EcomAction``. A model that
hallucinates JSON structure is penalised even if its semantic actions
would have worked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .rollout import EpisodeRecord


@dataclass
class RewardWeights:
    alpha_env: float = 1.0 / 50.0  # normalise per-episode env reward (max_steps=50)
    beta_grader: float = 2.0        # scale graders up — their [0.01,0.99] range dominates
    gamma_format: float = 0.5       # format compliance is a soft constraint

    def as_dict(self) -> Dict[str, float]:
        return {
            "alpha_env": self.alpha_env,
            "beta_grader": self.beta_grader,
            "gamma_format": self.gamma_format,
        }


TRAINING_GRADERS = ("triage_task", "inventory_task", "profit_task")


def combined_reward(rec: EpisodeRecord, weights: RewardWeights | None = None) -> float:
    """Compute the scalar combined reward for a full episode rollout."""
    weights = weights or RewardWeights()
    env_r = weights.alpha_env * float(rec.total_reward)
    grader_sum = sum(float(rec.grader_scores.get(g, 0.0)) for g in TRAINING_GRADERS)
    grader_r = weights.beta_grader * grader_sum
    format_r = weights.gamma_format * float(rec.format_compliance)
    return env_r + grader_r + format_r


def reward_breakdown(rec: EpisodeRecord, weights: RewardWeights | None = None) -> Dict[str, float]:
    """Expose per-term scalar contributions for logging / plotting."""
    weights = weights or RewardWeights()
    env_r = weights.alpha_env * float(rec.total_reward)
    grader_sum = sum(float(rec.grader_scores.get(g, 0.0)) for g in TRAINING_GRADERS)
    grader_r = weights.beta_grader * grader_sum
    format_r = weights.gamma_format * float(rec.format_compliance)
    return {
        "env": round(env_r, 4),
        "graders": round(grader_r, 4),
        "format": round(format_r, 4),
        "total": round(env_r + grader_r + format_r, 4),
        "grader_sum_raw": round(grader_sum, 4),
        "format_compliance": round(rec.format_compliance, 4),
        "fallback_rate": round(rec.fallback_rate, 4),
    }
