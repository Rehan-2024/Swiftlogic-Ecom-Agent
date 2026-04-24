"""Baseline policy sweep for Part A4 / Part B5.

Runs ``N_SEEDS`` (default 10) episodes of each baseline policy against the
frozen ``EcomEnv``, computes the 6 grader scores (3 training + 3
evaluation-only) per episode, and emits a JSON report for downstream
plots (``artifacts/task_baselines.json``, ``artifacts/before_metrics.json``).

Policies:

* ``wait_only`` — issues ``wait`` every step. Weak but cheap baseline.
* ``random`` — uniform random across 6 action types + random args.
* ``heuristic`` — simple rule set: refund tickets > restock low-stock >
  ad_spend on target SKU > wait.
* ``zero_shot_llm`` — **prepared locally, filled in Colab**. Requires a
  transformers model to run. If unavailable, writes a ``pending_colab``
  marker that the notebook later overwrites.

All randomness is driven by ``random.Random(seed)`` — the script is bit-
deterministic for a given seed. The LLM policy is the only non-
deterministic entry and is explicitly quarantined behind a CLI flag.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecom_env import (
    EcomEnv,
    EcomObservation,
    grade_competitor_response_task,
    grade_crisis_recovery_task,
    grade_inventory_task,
    grade_profit_task,
    grade_stability_task,
    grade_triage_task,
)


DEFAULT_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
DEFAULT_CONFIG = "configs/siyaani_fashion.json"

GRADERS_TRAINING = ("triage_task", "inventory_task", "profit_task")
GRADERS_EVALUATION = ("stability_task", "competitor_response_task", "crisis_recovery_task")
GRADER_NAMES = GRADERS_TRAINING + GRADERS_EVALUATION


Policy = Callable[[EcomObservation, random.Random], Dict[str, Any]]


def policy_wait(_obs: EcomObservation, _rng: random.Random) -> Dict[str, Any]:
    return {"action_type": "wait"}


def policy_random(obs: EcomObservation, rng: random.Random) -> Dict[str, Any]:
    """Uniform across the 6 action types, with random valid-looking args."""
    skus: List[str] = list(obs.inventory.keys()) or ["cotton_set"]
    ticket_ids: List[str] = [t.ticket_id for t in obs.active_tickets]
    choice = rng.choice(["wait", "restock", "ad_spend", "negotiate", "set_price", "refund"])
    if choice == "wait":
        return {"action_type": "wait"}
    if choice == "restock":
        return {"action_type": "restock", "sku": rng.choice(skus), "quantity": rng.randint(1, 30)}
    if choice == "ad_spend":
        return {
            "action_type": "ad_spend",
            "sku": rng.choice(skus),
            "budget": round(rng.uniform(50, 1500), 2),
        }
    if choice == "negotiate":
        return {
            "action_type": "negotiate",
            "sku": rng.choice(skus),
            "quantity": rng.randint(5, 40),
        }
    if choice == "set_price":
        current = float(obs.prices.get(skus[0], 1000.0))
        mult = rng.uniform(0.7, 1.3)
        return {
            "action_type": "set_price",
            "sku": rng.choice(skus),
            "price": round(max(1.0, current * mult), 2),
        }
    # refund
    if not ticket_ids:
        return {"action_type": "wait"}
    return {"action_type": "refund", "ticket_id": rng.choice(ticket_ids)}


def policy_heuristic(obs: EcomObservation, rng: random.Random) -> Dict[str, Any]:
    """Rule priority: refund > restock critical stock > price match > wait.

    * Any open ticket → refund the oldest one.
    * Any SKU stock < 5 → restock 20 units.
    * Any SKU priced > competitor by 5 % → undercut by 5 %.
    * Otherwise → wait.
    """
    if obs.active_tickets:
        return {"action_type": "refund", "ticket_id": obs.active_tickets[0].ticket_id}

    for sku, stock in obs.inventory.items():
        if stock < 5:
            return {"action_type": "restock", "sku": sku, "quantity": 20}

    for sku, our_price in (obs.prices or {}).items():
        comp = float((obs.competitor_prices or {}).get(sku, 0.0))
        if comp > 0 and our_price >= comp * 1.05:
            return {"action_type": "set_price", "sku": sku, "price": round(comp * 0.95, 2)}

    return {"action_type": "wait"}


def build_policies() -> Dict[str, Policy]:
    return {
        "wait_only": policy_wait,
        "random": policy_random,
        "heuristic": policy_heuristic,
    }


def run_episode(
    env: EcomEnv,
    seed: int,
    policy: Policy,
    max_steps: int,
    rng: random.Random,
) -> Dict[str, Any]:
    obs = env.reset(seed=seed)
    initial = obs
    total_reward = 0.0
    step_rewards: List[float] = []
    action_counts: Dict[str, int] = {}
    done = False
    steps = 0
    last_info: Dict[str, Any] = {}
    while not done and steps < max_steps:
        action = policy(obs, rng)
        action_counts[action["action_type"]] = action_counts.get(action["action_type"], 0) + 1
        obs, reward, done, info = env.step(action)
        step_rewards.append(float(reward.value))
        total_reward += float(reward.value)
        last_info = info
        steps += 1
    final_obs = obs

    grader_ctx = env.grader_context
    grader_scores = {
        "triage_task": grade_triage_task(initial, final_obs, context=grader_ctx),
        "inventory_task": grade_inventory_task(initial, final_obs, context=grader_ctx),
        "profit_task": grade_profit_task(initial, final_obs, context=grader_ctx),
        "stability_task": grade_stability_task(initial, final_obs, context=grader_ctx),
        "competitor_response_task": grade_competitor_response_task(initial, final_obs, context=grader_ctx),
        "crisis_recovery_task": grade_crisis_recovery_task(initial, final_obs, context=grader_ctx),
    }
    return {
        "seed": seed,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "mean_step_reward": round(statistics.mean(step_rewards) if step_rewards else 0.0, 6),
        "final_bank_balance": round(float(final_obs.bank_balance), 2),
        "final_customer_satisfaction": round(float(final_obs.customer_satisfaction), 4),
        "action_counts": action_counts,
        "grader_scores": {k: round(float(v), 4) for k, v in grader_scores.items()},
    }


def summarize_policy(
    policy_name: str,
    episodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    def _mean(xs: List[float]) -> float:
        return round(statistics.mean(xs), 4) if xs else 0.0

    def _stdev(xs: List[float]) -> float:
        return round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0

    rewards = [ep["total_reward"] for ep in episodes]
    banks = [ep["final_bank_balance"] for ep in episodes]
    per_task = {
        name: {
            "mean": _mean([ep["grader_scores"][name] for ep in episodes]),
            "stdev": _stdev([ep["grader_scores"][name] for ep in episodes]),
        }
        for name in GRADER_NAMES
    }
    composite_training = _mean([
        statistics.mean([ep["grader_scores"][n] for n in GRADERS_TRAINING])
        for ep in episodes
    ])
    composite_all = _mean([
        statistics.mean([ep["grader_scores"][n] for n in GRADER_NAMES])
        for ep in episodes
    ])
    return {
        "policy": policy_name,
        "n_episodes": len(episodes),
        "mean_total_reward": _mean(rewards),
        "stdev_total_reward": _stdev(rewards),
        "mean_final_bank": _mean(banks),
        "per_task": per_task,
        "composite_training_mean": composite_training,
        "composite_all_mean": composite_all,
    }


def run_sweep(
    seeds: List[int],
    config_path: str,
    max_steps: Optional[int],
    policies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    all_policies = build_policies()
    if policies is not None:
        chosen = {k: v for k, v in all_policies.items() if k in policies}
    else:
        chosen = all_policies

    env = EcomEnv(config_path=config_path)
    effective_steps = max_steps or env.world_engine.config["episode"]["max_steps"]
    report: Dict[str, Any] = {
        "config": str(config_path),
        "seeds": seeds,
        "max_steps": effective_steps,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "policies": {},
    }
    for name, fn in chosen.items():
        print(f"[baselines] policy={name}", flush=True)
        episodes: List[Dict[str, Any]] = []
        for seed in seeds:
            rng = random.Random(seed)
            ep = run_episode(env, seed, fn, effective_steps, rng)
            ep["policy"] = name
            episodes.append(ep)
            print(
                f"  seed={seed:>5} reward={ep['total_reward']:>9.2f} "
                f"bank={ep['final_bank_balance']:>10.2f} "
                f"triage={ep['grader_scores']['triage_task']:.2f} "
                f"inv={ep['grader_scores']['inventory_task']:.2f} "
                f"profit={ep['grader_scores']['profit_task']:.2f}"
            )
        report["policies"][name] = {
            "episodes": episodes,
            "summary": summarize_policy(name, episodes),
        }

    report["policies"]["zero_shot_llm"] = {
        "status": "pending_colab",
        "note": (
            "Zero-shot LLM baseline is populated by the training notebook "
            "(swiftlogic_grpo_training.ipynb, Part B5) which has access to "
            "the Qwen2.5-0.5B-Instruct weights. This stub is overwritten in "
            "place on merge."
        ),
    }
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Baseline policy sweep")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--policies",
        nargs="*",
        default=None,
        help="Subset of [wait_only, random, heuristic]. zero_shot_llm always stubbed.",
    )
    parser.add_argument(
        "--out",
        default="artifacts/task_baselines.json",
        help="Output path for the merged report.",
    )
    args = parser.parse_args(argv)

    report = run_sweep(args.seeds, args.config, args.max_steps, args.policies)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[baselines] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
