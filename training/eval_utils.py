"""Evaluation sweeps — before/after metrics and generalization (B5/B7/B+.1).

Runs a given ``ActionProducer`` across a configurable list of seeds and
configs, collects full grader scores (6 tasks) per episode, and emits a
normalized JSON blob. The same code path is used to compute
``before_metrics.json`` (baseline) and ``after_metrics.json`` (trained
adapter), plus generalization sweeps across alternate configs.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .rollout import (
    ActionProducer,
    EpisodeRecord,
    full_grader_scores,
    rollout_episode,
)


GRADERS_TRAINING = ("triage_task", "inventory_task", "profit_task")
GRADERS_ALL = GRADERS_TRAINING + (
    "stability_task",
    "competitor_response_task",
    "crisis_recovery_task",
)


def _episode_summary(rec: EpisodeRecord, full_scores: Dict[str, float]) -> Dict[str, Any]:
    return {
        "seed": rec.seed,
        "config": rec.config_path,
        "steps": len(rec.steps),
        "total_reward": round(float(rec.total_reward), 4),
        "format_compliance": round(float(rec.format_compliance), 4),
        "fallback_rate": round(float(rec.fallback_rate), 4),
        "final_bank": round(float(rec.final_obs.bank_balance) if rec.final_obs else 0.0, 2),
        "final_customer_satisfaction": round(
            float(rec.final_obs.customer_satisfaction) if rec.final_obs else 0.0, 4
        ),
        "grader_scores": {k: round(float(v), 4) for k, v in full_scores.items()},
    }


def run_eval_sweep(
    label: str,
    producer: ActionProducer,
    seeds: List[int],
    configs: List[str],
    *,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    episodes: List[Dict[str, Any]] = []
    for config in configs:
        for seed in seeds:
            rec = rollout_episode(producer, seed, config_path=config, max_steps=max_steps)
            scores = full_grader_scores(rec)
            episodes.append(_episode_summary(rec, scores))
    return {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": seeds,
        "configs": configs,
        "episodes": episodes,
        "summary": summarize_episodes(episodes),
    }


def summarize_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _mean(xs: List[float]) -> float:
        return round(statistics.mean(xs), 4) if xs else 0.0

    def _stdev(xs: List[float]) -> float:
        return round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0

    per_task = {
        g: {
            "mean": _mean([ep["grader_scores"][g] for ep in episodes]),
            "stdev": _stdev([ep["grader_scores"][g] for ep in episodes]),
        }
        for g in GRADERS_ALL
    }
    composite_training = _mean([
        statistics.mean([ep["grader_scores"][g] for g in GRADERS_TRAINING])
        for ep in episodes
    ])
    composite_all = _mean([
        statistics.mean([ep["grader_scores"][g] for g in GRADERS_ALL])
        for ep in episodes
    ])
    return {
        "n_episodes": len(episodes),
        "mean_total_reward": _mean([e["total_reward"] for e in episodes]),
        "mean_format_compliance": _mean([e["format_compliance"] for e in episodes]),
        "mean_fallback_rate": _mean([e["fallback_rate"] for e in episodes]),
        "mean_final_bank": _mean([e["final_bank"] for e in episodes]),
        "per_task": per_task,
        "composite_training_mean": composite_training,
        "composite_all_mean": composite_all,
    }


def write_json(obj: Dict[str, Any], path: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return str(out)
