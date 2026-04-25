"""End-to-end pipeline orchestrator (Part C3 + B6/B7/B+.* fallback).

Modes:
  --smoke-test   Fast sanity run (seeds 42 only, tiny). Exits 0 on green.
  --fast-mode    5 seeds / 20-step episodes / shorter plots. ~2 min.
  (default)      Full 10-seed, full-length run. ~10 min on CPU.

Each artifact produced carries a ``provenance`` tag:
  'trained_adapter' — produced with a LoRA adapter loaded from disk
  'heuristic_fallback' — produced with the hand-coded heuristic policy
                          as a stand-in before the Colab notebook has
                          trained and dropped the adapter.

The notebook on Colab runs the same underlying ``training.*`` functions
and overwrites these JSONs / PNGs with real trained-model numbers.

Usage:
  python scripts/run_full_pipeline.py                          # heuristic fallback
  python scripts/run_full_pipeline.py --trained-adapter ARTIFACTS/adapter
  python scripts/run_full_pipeline.py --fast-mode --smoke-test
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.behavior import (
    plot_behavior_evolution,
    plot_policy_evolution_line,
    policy_signature,
)
from training.composite import compute_composite
from training.eval_utils import run_eval_sweep, summarize_episodes, write_json
from training.plots import (
    plot_before_after_bars,
    plot_exploration_curve,
    plot_failure_vs_recovery,
    plot_generalization,
    plot_reward_curve,
)
from training.policies import (
    build_heuristic_producer,
    build_random_producer,
    build_wait_producer,
)
from training.rewards import RewardWeights, combined_reward, reward_breakdown
from training.rollout import rollout_episode


ARTIFACTS = ROOT / "artifacts"
CONFIG_PROD = str(ROOT / "configs" / "siyaani_fashion.json")
CONFIG_EASY = str(ROOT / "configs" / "siyaani_fashion_easy.json")


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

def _mode_params(mode: str) -> Dict[str, Any]:
    if mode == "smoke":
        return {
            "seeds_before": [101, 202],
            "seeds_after": [111, 222],
            "gen_seeds": [1212, 1313],
            "pseudo_curve_episodes": 20,
            "baseline_configs": [CONFIG_PROD],
        }
    if mode == "fast":
        return {
            "seeds_before": [101, 202, 303, 404, 505],
            "seeds_after": [111, 222, 333, 444, 555],
            "gen_seeds": [1212, 1313, 1414, 1515, 1616],
            "pseudo_curve_episodes": 60,
            "baseline_configs": [CONFIG_PROD],
        }
    return {
        "seeds_before": [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010],
        "seeds_after": [111, 222, 333, 444, 555, 666, 777, 888, 999, 1111],
        "gen_seeds": [1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020, 2121],
        "pseudo_curve_episodes": 200,
        "baseline_configs": [CONFIG_PROD],
    }


# ---------------------------------------------------------------------------
# Adapter loading (B6/B7 "trained" arm)
# ---------------------------------------------------------------------------

def _build_trained_producer(adapter_dir: Optional[str]):
    if not adapter_dir:
        return None, "missing"
    if not Path(adapter_dir, "adapter_config.json").exists():
        return None, "no_adapter_config"
    try:
        from training.policies import build_trained_producer
        return build_trained_producer(adapter_dir), "loaded"
    except Exception as exc:
        return None, f"error:{exc!r}"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_baselines(params: Dict[str, Any], provenance: str) -> Dict[str, Any]:
    print("[pipeline] stage: baselines")
    per_policy: Dict[str, Any] = {}
    for name, prod in [
        ("wait_only", build_wait_producer()),
        ("random", build_random_producer()),
        ("heuristic", build_heuristic_producer()),
    ]:
        bundle = run_eval_sweep(
            name, prod, params["seeds_before"], params["baseline_configs"]
        )
        bundle["provenance"] = "deterministic"
        per_policy[name] = bundle
        write_json(bundle, str(ARTIFACTS / f"baseline_{name}.json"))

    zero_shot_path = ARTIFACTS / "baseline_zero_shot_llm.json"
    if not zero_shot_path.exists():
        write_json(
            {
                "status": "pending_colab",
                "note": (
                    "Run swiftlogic_grpo_training.ipynb on Colab to populate "
                    "baseline_zero_shot_llm.json (requires the Qwen2.5-0.5B weights)."
                ),
            },
            str(zero_shot_path),
        )
    per_policy["zero_shot_llm"] = json.loads(zero_shot_path.read_text(encoding="utf-8"))

    combined = {
        "before_metrics": True,
        "provenance": provenance,
        "policies": per_policy,
        "baseline_for_composite": "heuristic",
    }
    write_json(combined, str(ARTIFACTS / "before_metrics.json"))
    return combined


def stage_reward_curve(params: Dict[str, Any], provenance: str) -> List[float]:
    """Produce a learning-style curve.

    With ``provenance == 'heuristic_fallback'``: we do not hallucinate a
    fake learning curve. Instead we emit a flat line reflecting the
    heuristic's average combined reward, clearly labeled. The Colab
    notebook overwrites this with the real GRPO curve when training
    completes.
    """
    print("[pipeline] stage: reward_curve")
    producer = build_heuristic_producer()
    rewards: List[float] = []
    weights = RewardWeights()
    for seed in range(params["pseudo_curve_episodes"]):
        rec = rollout_episode(producer, seed, config_path=CONFIG_PROD)
        rewards.append(combined_reward(rec, weights))
    title = (
        "GRPO training reward — POPULATED BY COLAB NOTEBOOK"
        if provenance == "heuristic_fallback"
        else "GRPO training reward (per episode)"
    )
    plot_reward_curve(rewards, str(ARTIFACTS / "reward_curve.png"), title=title)
    log_path = ARTIFACTS / "training_log.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# provenance={provenance}\n")
        f.write(f"# n_episodes={len(rewards)}\n")
        for i, r in enumerate(rewards):
            f.write(f"episode={i} reward={r:.4f}\n")
    return rewards


def stage_eval_after(
    params: Dict[str, Any],
    trained_producer,
    provenance: str,
) -> Dict[str, Any]:
    print("[pipeline] stage: eval_after")
    producer = trained_producer or build_heuristic_producer()
    bundle = run_eval_sweep(
        "after_training", producer, params["seeds_after"], [CONFIG_PROD]
    )
    bundle["provenance"] = provenance
    write_json(bundle, str(ARTIFACTS / "after_metrics.json"))

    before = json.loads((ARTIFACTS / "before_metrics.json").read_text(encoding="utf-8"))
    ref = before["policies"]["heuristic"]["summary"]["per_task"]
    plot_before_after_bars(
        ref,
        bundle["summary"]["per_task"],
        str(ARTIFACTS / "before_after_comparison.png"),
        title=f"Grader scores — baseline vs {'trained' if trained_producer else 'heuristic (fallback)'}",
    )
    return bundle


def stage_generalization(
    params: Dict[str, Any],
    trained_producer,
    provenance: str,
) -> Dict[str, Any]:
    print("[pipeline] stage: generalization")
    configs = [CONFIG_PROD, CONFIG_EASY]
    demo_cfg = ROOT / "configs" / "siyaani_fashion_demo.json"
    if demo_cfg.exists():
        configs.append(str(demo_cfg))
    # Round-2 plan section 3.3 + 10.6 - the unseen-config gate.
    # The dashboard's anti-fake audit (C6_generalization_unseen) blocks merge
    # unless these two business configs appear in generalization.json.
    for unseen_name in ("medplus_pharmacy.json", "stackbase_saas.json"):
        unseen_cfg = ROOT / "configs" / unseen_name
        if unseen_cfg.exists():
            configs.append(str(unseen_cfg))

    producer = trained_producer or build_heuristic_producer()
    bundle = run_eval_sweep("generalization", producer, params["gen_seeds"], configs)
    bundle["provenance"] = provenance
    write_json(bundle, str(ARTIFACTS / "generalization.json"))

    per_cfg: Dict[str, Dict[str, Any]] = {}
    for cfg in configs:
        eps = [e for e in bundle["episodes"] if e["config"] == cfg]
        per_cfg[Path(cfg).name] = summarize_episodes(eps)["per_task"]
    plot_generalization(
        per_cfg,
        str(ARTIFACTS / "generalization.png"),
        title=f"Generalization across configs ({'trained' if trained_producer else 'heuristic'})",
    )
    return bundle


def stage_hard_seed(
    params: Dict[str, Any],
    trained_producer,
    provenance: str,
    after_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    print("[pipeline] stage: hard_seed_retraining")
    producer = trained_producer or build_heuristic_producer()
    # Pick the 3 worst seeds from the after bundle.
    sorted_eps = sorted(
        after_bundle["episodes"],
        key=lambda e: statistics.mean(e["grader_scores"].values()),
    )
    hard_seeds = [e["seed"] for e in sorted_eps[:3]]
    if not hard_seeds:
        hard_seeds = params["seeds_after"][:3]

    pre = run_eval_sweep(
        "hard_seeds_before", producer, hard_seeds, [CONFIG_PROD]
    )
    # "Retraining" proxy in fallback mode: use the adapter again on same
    # seeds — numbers should be identical (deterministic), signalling a
    # real improvement will only show after Colab training. Mark it.
    post = run_eval_sweep("hard_seeds_after", producer, hard_seeds, [CONFIG_PROD])

    result = {
        "provenance": provenance,
        "hard_seeds": hard_seeds,
        "before": {"summary": pre["summary"]},
        "after": {"summary": post["summary"]},
        "note": (
            "In fallback mode the before / after are identical (pure "
            "heuristic). The Colab notebook runs a real 30-step GRPO "
            "burst targeting these exact seeds and replaces this file."
        ) if provenance == "heuristic_fallback" else "real hard-seed GRPO burst",
    }
    write_json(result, str(ARTIFACTS / "hard_seed_retraining.json"))
    return result


def stage_behavior_evolution(params: Dict[str, Any], trained_producer, provenance: str) -> None:
    print("[pipeline] stage: behavior_evolution")
    checkpoints = {
        "wait_only": _collect_action_stream(build_wait_producer(), [101, 202], CONFIG_PROD),
        "random": _collect_action_stream(build_random_producer(), [101, 202], CONFIG_PROD),
        "heuristic": _collect_action_stream(build_heuristic_producer(), [101, 202], CONFIG_PROD),
    }
    if trained_producer:
        checkpoints["trained"] = _collect_action_stream(trained_producer, [101, 202], CONFIG_PROD)
    else:
        checkpoints["trained_fallback"] = checkpoints["heuristic"]

    plot_behavior_evolution(checkpoints, str(ARTIFACTS / "behavior_evolution.png"))
    plot_policy_evolution_line(checkpoints, str(ARTIFACTS / "policy_evolution.png"))
    sigs = {k: policy_signature(v) for k, v in checkpoints.items()}
    write_json({"provenance": provenance, "signatures": sigs}, str(ARTIFACTS / "policy_signature.json"))
    plot_exploration_curve(
        [sigs[k]["entropy"] for k in checkpoints],
        str(ARTIFACTS / "exploration_curve.png"),
        checkpoint_labels=list(checkpoints.keys()),
    )


def _collect_action_stream(producer, seeds, config_path):
    out = []
    for s in seeds:
        rec = rollout_episode(producer, s, config_path=config_path)
        out.extend({"action_type": step.action.get("action_type", "wait")} for step in rec.steps)
    return out


def stage_composite(provenance: str) -> Dict[str, Any]:
    print("[pipeline] stage: composite_score")
    # compute_composite expects before/after to each have a 'summary'
    # field directly. Adapt by extracting the heuristic summary.
    before_path = ARTIFACTS / "before_metrics.json"
    before = json.loads(before_path.read_text(encoding="utf-8"))
    before_summary = {"summary": before["policies"]["heuristic"]["summary"]}
    tmp_path = ARTIFACTS / "_before_summary_view.json"
    tmp_path.write_text(json.dumps(before_summary), encoding="utf-8")

    result = compute_composite(
        before_metrics_path=str(tmp_path),
        after_metrics_path=str(ARTIFACTS / "after_metrics.json"),
        generalization_path=str(ARTIFACTS / "generalization.json"),
        out_path=str(ARTIFACTS / "composite_score.json"),
    )
    result["provenance"] = provenance
    write_json(result, str(ARTIFACTS / "composite_score.json"))
    tmp_path.unlink(missing_ok=True)
    print(f"[pipeline] headline: {result['headline']}")
    return result


def stage_failure_vs_recovery(trained_producer, provenance: str) -> Dict[str, Any]:
    print("[pipeline] stage: failure_vs_recovery (Part B+.7)")
    seed = 20260425
    # Baseline = wait_only (deliberately failing)
    wait_rec = rollout_episode(
        build_wait_producer(), seed, config_path=CONFIG_PROD
    )
    producer = trained_producer or build_heuristic_producer()
    trained_rec = rollout_episode(producer, seed, config_path=CONFIG_PROD)

    baseline_banks = _bank_trajectory(wait_rec)
    trained_banks = _bank_trajectory(trained_rec)
    plot_failure_vs_recovery(
        baseline_banks,
        trained_banks,
        str(ARTIFACTS / "failure_vs_recovery.png"),
        title=(
            f"Failure vs recovery — seed={seed}  baseline={wait_rec.final_obs.bank_balance:.0f}  "
            f"{'trained' if trained_producer else 'heuristic'}={trained_rec.final_obs.bank_balance:.0f}"
        ),
    )
    result = {
        "provenance": provenance,
        "seed": seed,
        "baseline_final_bank": round(float(wait_rec.final_obs.bank_balance), 2),
        "trained_final_bank": round(float(trained_rec.final_obs.bank_balance), 2),
        "baseline_grader_scores": wait_rec.grader_scores,
        "trained_grader_scores": trained_rec.grader_scores,
    }
    write_json(result, str(ARTIFACTS / "failure_vs_recovery.json"))
    return result


def _bank_trajectory(rec) -> List[float]:
    banks = [float(rec.initial_obs.bank_balance)]
    # We don't have a per-step bank trace on the record right now; recompute
    # by re-playing. Cheap — fewer than 50 steps.
    from ecom_env import EcomEnv

    env = EcomEnv(config_path=rec.config_path)
    env.reset(seed=rec.seed)
    for s in rec.steps:
        obs, _, _, _ = env.step(s.action)
        banks.append(float(obs.bank_balance))
    return banks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _git_info() -> Dict[str, Any]:
    import shutil
    import subprocess

    if shutil.which("git") is None:
        return {"available": False}
    info: Dict[str, Any] = {"available": True}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        info["sha"] = sha.stdout.strip() if sha.returncode == 0 else None
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        info["branch"] = branch.stdout.strip() if branch.returncode == 0 else None
        tag = subprocess.run(
            ["git", "tag", "--list", "release/env-frozen-v2.3"],
            capture_output=True, text=True, check=False,
        )
        info["env_frozen_tag_present"] = bool(tag.stdout.strip())
    except Exception as exc:  # noqa: BLE001
        info["error"] = repr(exc)
    return info


def _write_run_config(mode: str, params: Dict[str, Any], provenance: str,
                      adapter_status: str, weights: RewardWeights) -> Path:
    import platform

    payload = {
        "mode": mode,
        "provenance": provenance,
        "adapter_status": adapter_status,
        "params": params,
        "reward_weights": weights.as_dict(),
        "interpreter": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "git": _git_info(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = ARTIFACTS / "run_config.json"
    write_json(payload, str(out))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--trained-adapter", default=None,
                        help="Path to a LoRA adapter dir produced by Colab training.")
    parser.add_argument("--with-pytest", action="store_true",
                        help="Run `pytest -q` before stage execution (slow).")
    args = parser.parse_args(argv)

    mode = "smoke" if args.smoke_test else ("fast" if args.fast_mode else "full")
    params = _mode_params(mode)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    trained_producer, adapter_status = _build_trained_producer(args.trained_adapter)
    if trained_producer:
        provenance = "trained_adapter"
    else:
        provenance = "heuristic_fallback"
    print(f"[pipeline] mode={mode} provenance={provenance} adapter_status={adapter_status}")

    weights = RewardWeights()
    _write_run_config(mode, params, provenance, adapter_status, weights)

    if args.with_pytest:
        import shutil
        import subprocess

        print("[pipeline] running pytest -q (this can take a while)...")
        py = shutil.which("pytest") or "pytest"
        rc = subprocess.run([py, "-q"], cwd=str(ROOT)).returncode
        if rc != 0:
            print(f"[pipeline] pytest failed (rc={rc}); aborting.")
            return rc

    t0 = time.time()

    # Part B5 — baselines
    stage_baselines(params, provenance)

    # Part B6 — training curve (fallback-labeled)
    stage_reward_curve(params, provenance)

    # Part B7 — after evaluation + before/after comparison
    after_bundle = stage_eval_after(params, trained_producer, provenance)

    # Part B+.1 — generalization
    stage_generalization(params, trained_producer, provenance)

    # Part B+.2 — hard-seed retraining proxy
    stage_hard_seed(params, trained_producer, provenance, after_bundle)

    # Part B+.3 — behavior evolution
    stage_behavior_evolution(params, trained_producer, provenance)

    # Part B+.7 — failure vs recovery (also triggered by scripted_demo.py)
    stage_failure_vs_recovery(trained_producer, provenance)

    # Part B+.6 — composite score
    stage_composite(provenance)

    # Provenance manifest
    provenance_manifest = {
        "pipeline_run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline_mode": mode,
        "provenance": provenance,
        "adapter_status": adapter_status,
        "wall_seconds": round(time.time() - t0, 1),
        "artifact_hash_policy": "colab overwrites on next run with trained numbers",
    }
    write_json(provenance_manifest, str(ARTIFACTS / "pipeline_manifest.json"))
    print(f"[pipeline] done in {provenance_manifest['wall_seconds']}s — {provenance}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
