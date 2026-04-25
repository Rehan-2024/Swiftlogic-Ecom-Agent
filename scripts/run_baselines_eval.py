"""Run the 4 mandatory baselines through the training eval sweep (Part B5).

Populates:
  - artifacts/before_metrics.json       (combined bundle)
  - artifacts/baseline_wait_only.json
  - artifacts/baseline_random.json
  - artifacts/baseline_heuristic.json
  - artifacts/baseline_zero_shot_llm.json  (only if --with-llm and a model
    is available; otherwise a 'pending' stub preserved for Colab merge)

Usage:
  python scripts/run_baselines_eval.py                 # 3 baselines only
  python scripts/run_baselines_eval.py --with-llm      # run zero-shot via transformers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.eval_utils import run_eval_sweep, write_json
from training.policies import (
    build_heuristic_producer,
    build_random_producer,
    build_wait_producer,
    build_zero_shot_producer,
)


DEFAULT_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
DEFAULT_CONFIG = str(ROOT / "configs" / "siyaani_fashion.json")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--with-llm", action="store_true",
                        help="Load Qwen2.5-0.5B-Instruct and run zero-shot baseline.")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = {
        "wait_only": build_wait_producer(),
        "random": build_random_producer(),
        "heuristic": build_heuristic_producer(),
    }

    per_policy: Dict[str, Dict[str, Any]] = {}
    for name, producer in policies.items():
        print(f"[baselines-eval] running {name}", flush=True)
        bundle = run_eval_sweep(
            label=name,
            producer=producer,
            seeds=args.seeds,
            configs=[args.config],
            max_steps=args.max_steps,
        )
        per_policy[name] = bundle
        write_json(bundle, str(out_dir / f"baseline_{name}.json"))
        print(
            f"[baselines-eval]   mean_reward={bundle['summary']['mean_total_reward']} "
            f"composite_training={bundle['summary']['composite_training_mean']} "
            f"composite_all={bundle['summary']['composite_all_mean']}"
        )

    if args.with_llm:
        print(f"[baselines-eval] loading zero-shot LLM {args.llm_model}", flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
        model = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype="auto")
        model.eval()
        producer = build_zero_shot_producer(model, tokenizer)
        bundle = run_eval_sweep(
            label="zero_shot_llm",
            producer=producer,
            seeds=args.seeds,
            configs=[args.config],
            max_steps=args.max_steps,
        )
        per_policy["zero_shot_llm"] = bundle
        write_json(bundle, str(out_dir / "baseline_zero_shot_llm.json"))
    else:
        per_policy["zero_shot_llm"] = {
            "status": "pending_colab",
            "note": "populate via --with-llm or the training notebook Part B5.",
        }
        write_json(
            per_policy["zero_shot_llm"], str(out_dir / "baseline_zero_shot_llm.json")
        )

    combined = {
        "before_metrics": True,
        "policies": per_policy,
        "baseline_for_composite": "heuristic",
        "note": (
            "For Part B+.6 composite scoring we treat 'heuristic' as the "
            "primary before-training reference — it is the strongest "
            "non-LLM policy and matches what a domain-expert CEO would do."
        ),
    }
    write_json(combined, str(out_dir / "before_metrics.json"))
    print(f"[baselines-eval] wrote {out_dir / 'before_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
