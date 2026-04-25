"""Policy adapters — wraps the deterministic baselines from
``scripts/baselines.py`` into the ``ActionProducer`` signature expected
by ``training.rollout.rollout_episode``.

The zero-shot LLM policy is declared here too but imports transformers
lazily. The trained-adapter policy (`build_trained_producer`) composes
a Qwen2.5 + LoRA model that the notebook produces in Part B6.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from ecom_env import EcomObservation

from .rollout import ActionProducer


# ---------------------------------------------------------------------------
# Deterministic baselines (wrappers around scripts/baselines.py)
# ---------------------------------------------------------------------------

def _wrap_simple(fn: Callable[[EcomObservation, random.Random], Dict[str, Any]]) -> ActionProducer:
    def _producer(obs: EcomObservation, state: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        rng = state.setdefault("rng", random.Random(state.get("seed", 0)))
        action = fn(obs, rng)
        return action, None
    return _producer


def build_wait_producer() -> ActionProducer:
    from scripts.baselines import policy_wait
    return _wrap_simple(policy_wait)


def build_random_producer() -> ActionProducer:
    from scripts.baselines import policy_random
    return _wrap_simple(policy_random)


def build_heuristic_producer() -> ActionProducer:
    from scripts.baselines import policy_heuristic
    return _wrap_simple(policy_heuristic)


# ---------------------------------------------------------------------------
# Zero-shot LLM policy (the 4th baseline, Part B5)
# ---------------------------------------------------------------------------

def _format_obs_for_prompt(obs: EcomObservation) -> str:
    return (
        f"day={obs.current_day} step={obs.step_count} "
        f"bank={float(obs.bank_balance):.0f} "
        f"inventory={dict(obs.inventory)} "
        f"open_tickets={[t.ticket_id for t in obs.active_tickets if t.status == 'open']} "
        f"customer_satisfaction={float(obs.customer_satisfaction):.2f} "
        f"prices={dict(obs.prices)} "
        f"competitor_prices={dict(obs.competitor_prices)}"
    )


SYSTEM_PROMPT = (
    "You are a commerce agent. Each turn, reply with exactly one JSON "
    "object of the form "
    '{"action_type": "wait"|"restock"|"refund"|"ad_spend"|"negotiate"|"set_price", ...}. '
    "Return ONLY the JSON, no prose. Fields: for 'restock'/'negotiate' use "
    "{sku, quantity}; 'ad_spend' {sku, budget}; 'set_price' {sku, price}; "
    "'refund' {ticket_id}; 'wait' {}. Choose a productive action."
)


def build_zero_shot_producer(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 96,
    thought_log: Optional[list] = None,
) -> ActionProducer:
    """Prompt ``model`` zero-shot; fallback handled downstream by validator."""

    def _producer(obs: EcomObservation, state: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\nState: {_format_obs_for_prompt(obs)}\nAction (JSON only):\n<|assistant|>\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            import torch  # noqa: F401
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        except ImportError:
            pass
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if thought_log is not None:
            thought_log.append({"step": state.get("step", 0) + 1, "prompt_tail": prompt[-200:], "output": gen[:400]})
        match = re.search(r"\{[^{}]*\}", gen, re.DOTALL)
        if not match:
            return None, gen
        try:
            cand = json.loads(match.group(0))
            if isinstance(cand, dict) and cand.get("action_type"):
                return cand, gen
        except (json.JSONDecodeError, ValueError):
            pass
        return None, gen

    return _producer


# ---------------------------------------------------------------------------
# Trained adapter policy (Part B7 / B+ evaluation)
# ---------------------------------------------------------------------------

def build_trained_producer(
    adapter_dir: str,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    *,
    thought_log: Optional[list] = None,
) -> ActionProducer:
    """Load a LoRA adapter saved by the Part B6 training loop."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")
    if Path(adapter_dir).exists():
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return build_zero_shot_producer(model, tokenizer, thought_log=thought_log)
