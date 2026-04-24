"""Scripted demo trajectory for CommerceOps v2 (Part A6, updated in B+.7).

Runs a fixed-seed, deterministic episode against the frozen ``EcomEnv``
with a selectable policy and prints a human-readable running commentary
that surfaces the additive explainability keys (``action_quality``,
``strategy_phase``, ``reward_breakdown``). The output is identical
across runs for the same ``--seed``.

Policies:

* ``wait_only`` / ``random`` / ``heuristic`` — same definitions as
  ``scripts/baselines.py``, imported to avoid drift.
* ``scripted`` (default) — a hand-tuned 50-step trajectory that showcases
  a refund sprint → restock → ad-spend → price-match → stabilize arc.
* ``trained`` — loads the LoRA adapter from ``artifacts/adapter`` if
  available and plays its actions. Falls back to the scripted trajectory
  with a clear banner if the adapter or its dependencies are missing.

Usage:

    python scripted_demo.py                      # scripted, quiet
    python scripted_demo.py --policy random      # baseline
    python scripted_demo.py --verbose            # full breakdown per step
    python scripted_demo.py --policy trained -v  # trained-vs-baseline for B+.7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecom_env import EcomEnv, EcomObservation
from scripts.baselines import (  # noqa: E402 — local script
    policy_heuristic,
    policy_random,
    policy_wait,
)


DEFAULT_SEED = 20260425  # hackathon submission date — deterministic "wow" seed
DEFAULT_CONFIG = "configs/siyaani_fashion.json"
ADAPTER_DIR = ROOT / "artifacts" / "adapter"


# ---------------------------------------------------------------------------
# Scripted trajectory — a hand-tuned 50-step tape designed to recover a
# struggling boutique without LLM inference. Deterministic for judges.
# ---------------------------------------------------------------------------
_SCRIPTED_TAPE: List[Dict[str, Any]] = [
    # Days 1–4 — triage inbound tickets
    {"action_type": "wait"},
    {"action_type": "refund", "ticket_id": "__first_ticket__"},
    {"action_type": "refund", "ticket_id": "__first_ticket__"},
    {"action_type": "refund", "ticket_id": "__first_ticket__"},
    # Days 5–10 — price-match competitor on best seller
    {"action_type": "set_price", "sku": "cotton_set", "price": 1050.0},
    {"action_type": "ad_spend", "sku": "cotton_set", "budget": 400.0},
    {"action_type": "wait"},
    {"action_type": "negotiate", "sku": "cotton_set", "quantity": 30},
    {"action_type": "restock", "sku": "cotton_set", "quantity": 30},
    {"action_type": "wait"},
    # Days 11–20 — grow silk_kurta category
    {"action_type": "set_price", "sku": "silk_kurta", "price": 1700.0},
    {"action_type": "ad_spend", "sku": "silk_kurta", "budget": 600.0},
    {"action_type": "wait"},
    {"action_type": "negotiate", "sku": "silk_kurta", "quantity": 20},
    {"action_type": "restock", "sku": "silk_kurta", "quantity": 20},
    {"action_type": "refund", "ticket_id": "__first_ticket__"},
    {"action_type": "refund", "ticket_id": "__first_ticket__"},
    {"action_type": "wait"},
    {"action_type": "ad_spend", "sku": "silk_saree", "budget": 350.0},
    {"action_type": "wait"},
    # Days 21–35 — stabilize, keep stock above safety
    {"action_type": "restock", "sku": "linen_dupatta", "quantity": 15},
    {"action_type": "set_price", "sku": "linen_dupatta", "price": 1250.0},
    {"action_type": "wait"},
    {"action_type": "refund", "ticket_id": "__first_ticket__"},
    {"action_type": "wait"},
    {"action_type": "negotiate", "sku": "silk_saree", "quantity": 8},
    {"action_type": "restock", "sku": "silk_saree", "quantity": 8},
    {"action_type": "wait"},
    {"action_type": "refund", "ticket_id": "__first_ticket__"},
    {"action_type": "wait"},
    {"action_type": "ad_spend", "sku": "cotton_set", "budget": 300.0},
    {"action_type": "wait"},
    {"action_type": "refund", "ticket_id": "__first_ticket__"},
    {"action_type": "wait"},
    {"action_type": "wait"},
    # Days 36–50 — harvest
    {"action_type": "set_price", "sku": "cotton_set", "price": 1080.0},
    {"action_type": "refund", "ticket_id": "__first_ticket__"},
    {"action_type": "negotiate", "sku": "cotton_set", "quantity": 20},
    {"action_type": "restock", "sku": "cotton_set", "quantity": 20},
    {"action_type": "wait"},
    {"action_type": "ad_spend", "sku": "silk_kurta", "budget": 450.0},
    {"action_type": "wait"},
    {"action_type": "refund", "ticket_id": "__first_ticket__"},
    {"action_type": "wait"},
    {"action_type": "wait"},
    {"action_type": "wait"},
    {"action_type": "refund", "ticket_id": "__first_ticket__"},
    {"action_type": "wait"},
    {"action_type": "wait"},
    {"action_type": "wait"},
]


def _resolve_action(action: Dict[str, Any], obs: EcomObservation) -> Dict[str, Any]:
    """Resolve placeholder tokens like ``__first_ticket__`` against live state."""
    out = dict(action)
    if out.get("ticket_id") == "__first_ticket__":
        tickets = [t.ticket_id for t in obs.active_tickets if t.status != "resolved"]
        if tickets:
            out["ticket_id"] = tickets[0]
        else:
            # No open tickets — fall back to a safe wait so we never
            # emit a refund against a non-existent ticket.
            return {"action_type": "wait"}
    return out


def policy_scripted(obs: EcomObservation, rng: random.Random) -> Dict[str, Any]:
    step = getattr(policy_scripted, "_step", 0)
    action = _SCRIPTED_TAPE[step % len(_SCRIPTED_TAPE)]
    policy_scripted._step = step + 1
    return _resolve_action(action, obs)


def policy_trained(obs: EcomObservation, rng: random.Random) -> Dict[str, Any]:
    """Trained-adapter policy. Falls through to scripted if unavailable.

    The adapter is loaded once on first call; subsequent calls reuse it.
    """
    state = getattr(policy_trained, "_state", None)
    if state is None:
        state = _load_trained_policy()
        policy_trained._state = state
    if state.get("fallback"):
        return policy_scripted(obs, rng)
    return state["callable"](obs, rng)


def _load_trained_policy() -> Dict[str, Any]:
    """Load Qwen2.5 + LoRA adapter from ``artifacts/adapter`` if available."""
    marker = ADAPTER_DIR / "adapter_config.json"
    if not marker.exists():
        print(
            f"[scripted_demo] trained adapter not found at {ADAPTER_DIR} — "
            "falling back to scripted trajectory.",
            file=sys.stderr,
        )
        return {"fallback": True}
    try:
        # Lazy import so missing torch doesn't break every other policy.
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto")
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
        model.eval()

        def _policy(obs: EcomObservation, rng: random.Random) -> Dict[str, Any]:
            return _infer_action(model, tokenizer, obs)

        return {"fallback": False, "callable": _policy}
    except Exception as exc:
        print(
            f"[scripted_demo] trained adapter load failed ({exc}); falling back to scripted.",
            file=sys.stderr,
        )
        return {"fallback": True}


def _infer_action(model, tokenizer, obs: EcomObservation) -> Dict[str, Any]:
    """Prompt the LoRA-trained model for a JSON action; safe-fallback to wait."""
    prompt = (
        "You are the CEO of a small commerce business. Given the current "
        "business state, respond with exactly one JSON action of the form "
        '{"action_type": "wait"|"restock"|"refund"|"ad_spend"|"negotiate"|"set_price", ...}.\n\n'
        f"State: bank={obs.bank_balance:.0f}, "
        f"inventory={dict(obs.inventory)}, "
        f"open_tickets={len(obs.active_tickets)}, "
        f"customer_satisfaction={obs.customer_satisfaction:.2f}.\n\n"
        "Action (JSON only):"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Try to parse a JSON object out of the text.
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = json.loads(text[start : end + 1])
            if isinstance(candidate, dict) and candidate.get("action_type"):
                return _resolve_action(candidate, obs)
    except (json.JSONDecodeError, ValueError):
        pass
    return {"action_type": "wait"}


POLICIES = {
    "wait_only": policy_wait,
    "random": policy_random,
    "heuristic": policy_heuristic,
    "scripted": policy_scripted,
    "trained": policy_trained,
}


def _fmt_action(action: Dict[str, Any]) -> str:
    atype = action.get("action_type", "?")
    parts = [atype]
    for k, v in action.items():
        if k == "action_type":
            continue
        parts.append(f"{k}={v}")
    return " ".join(parts)


def _print_step(step_idx: int, action: Dict[str, Any], reward_value: float, info: Dict[str, Any], obs: EcomObservation, verbose: bool) -> None:
    quality = info.get("action_quality", "?")
    phase = info.get("strategy_phase", "?")
    quality_glyph = {"good": "+", "neutral": "=", "bad": "-"}.get(quality, "?")
    headline = (
        f"step {step_idx:02d} [{phase:>9s}] [{quality_glyph}{quality}] "
        f"{_fmt_action(action):<42s}"
        f" reward={reward_value:>+7.3f}  bank={obs.bank_balance:>9.0f}"
    )
    print(headline)
    if verbose:
        bd = info.get("reward_breakdown", {}) or {}
        top = sorted(
            ((k, float(v)) for k, v in bd.items() if isinstance(v, (int, float))),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:4]
        if top:
            print("         breakdown:", " ".join(f"{k}={v:+.2f}" for k, v in top))
        if info.get("action_quality_reason"):
            print(f"         quality_reason: {info['action_quality_reason']}")
        if info.get("strategy_phase_note"):
            print(f"         phase_note: {info['strategy_phase_note']}")


def run_demo(
    policy_name: str,
    seed: int,
    config_path: str,
    max_steps: Optional[int],
    verbose: bool,
    out_json: Optional[str] = None,
) -> Dict[str, Any]:
    if policy_name not in POLICIES:
        raise SystemExit(f"unknown policy: {policy_name}. choose from: {sorted(POLICIES)}")
    policy = POLICIES[policy_name]

    # Reset scripted policy's internal counter so reruns are deterministic.
    if hasattr(policy_scripted, "_step"):
        policy_scripted._step = 0

    env = EcomEnv(config_path=config_path)
    obs = env.reset(seed=seed)
    initial = obs
    rng = random.Random(seed)
    limit = max_steps or env.world_engine.config["episode"]["max_steps"]

    print(f"# CommerceOps scripted demo — policy={policy_name} seed={seed} config={config_path}")
    print(f"# initial: bank={obs.bank_balance:.0f} tickets={len(obs.active_tickets)} SKUs={list(obs.inventory)}")

    done = False
    step_idx = 0
    tape: List[Dict[str, Any]] = []
    while not done and step_idx < limit:
        action = policy(obs, rng)
        obs, reward, done, info = env.step(action)
        r_val = float(reward.value) if hasattr(reward, "value") else float(reward)
        _print_step(step_idx + 1, action, r_val, info, obs, verbose)
        tape.append({
            "step": step_idx + 1,
            "action": action,
            "reward": round(r_val, 4),
            "bank": round(float(obs.bank_balance), 2),
            "quality": info.get("action_quality"),
            "phase": info.get("strategy_phase"),
        })
        step_idx += 1

    grader_ctx = env.grader_context
    from ecom_env import (
        grade_competitor_response_task,
        grade_crisis_recovery_task,
        grade_inventory_task,
        grade_profit_task,
        grade_stability_task,
        grade_triage_task,
    )
    grader_scores = {
        "triage_task": grade_triage_task(initial, obs, context=grader_ctx),
        "inventory_task": grade_inventory_task(initial, obs, context=grader_ctx),
        "profit_task": grade_profit_task(initial, obs, context=grader_ctx),
        "stability_task": grade_stability_task(initial, obs, context=grader_ctx),
        "competitor_response_task": grade_competitor_response_task(initial, obs, context=grader_ctx),
        "crisis_recovery_task": grade_crisis_recovery_task(initial, obs, context=grader_ctx),
    }
    print()
    print("# final grader scores:")
    for name, v in grader_scores.items():
        print(f"  {name:<26s} {v:.3f}")
    print(f"# final bank: {obs.bank_balance:.0f}  cust_satisfaction: {obs.customer_satisfaction:.2f}")

    result = {
        "policy": policy_name,
        "seed": seed,
        "steps": step_idx,
        "final_bank": round(float(obs.bank_balance), 2),
        "final_customer_satisfaction": round(float(obs.customer_satisfaction), 4),
        "grader_scores": {k: round(float(v), 4) for k, v in grader_scores.items()},
        "tape": tape,
    }
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"# wrote {out_json}")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CommerceOps scripted demo")
    parser.add_argument("--policy", default="scripted", choices=sorted(POLICIES))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--out", default=None, help="Optional JSON tape output path")
    args = parser.parse_args(argv)

    run_demo(args.policy, args.seed, args.config, args.max_steps, args.verbose, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
