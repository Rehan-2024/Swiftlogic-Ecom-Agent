"""Policy layer for the dashboard.

Three policies, one interface:
  - baseline_wait_only:  always emits WaitAction.
  - baseline_zero_shot:  base Qwen2.5-1.5B-Instruct, no adapter, T=0.0.
  - trained:             base + LoRA adapter from ADAPTER_DIR, T=0.7, top_p=0.9.

Reuses inference._build_action and inference.build_step_trace so the
on-screen CEO trace is byte-identical to what the OpenEnv eval harness
produces. (Round-2 plan section 5.2.)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.artifact_loader import ADAPTER_DIR  # noqa: E402

logger = logging.getLogger("commerceops.demo.policy")


HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

POLICY_BASELINE_WAIT = "baseline_wait_only"
POLICY_BASELINE_ZERO_SHOT = "baseline_zero_shot"
POLICY_TRAINED = "trained"
ALL_POLICIES = (POLICY_BASELINE_WAIT, POLICY_BASELINE_ZERO_SHOT, POLICY_TRAINED)


SYSTEM_PROMPT = (
    "You are an autonomous digital storefront operator for an Indian ethnic-wear "
    "brand (Siyaani). Maximise profit over a 50-day cycle without going bankrupt.\n\n"
    "Return ONLY raw JSON matching exactly one of these schemas:\n"
    '1. {"action_type":"restock","sku":"<string>","quantity":<int>}\n'
    '2. {"action_type":"refund","ticket_id":"<string>"}\n'
    '3. {"action_type":"ad_spend","sku":"<string>","budget":<float>}\n'
    '4. {"action_type":"negotiate","sku":"<string>","quantity":<int>}\n'
    '5. {"action_type":"set_price","sku":"<string>","price":<float>}\n'
    '6. {"action_type":"wait"}\n\n'
    "No markdown, no commentary, JSON only."
)


@dataclass
class PolicyHandle:
    name: str                      # one of ALL_POLICIES
    available: bool
    reason: str = ""               # why unavailable, if applicable
    model: Any = None              # transformers/peft model when loaded
    tokenizer: Any = None          # transformers tokenizer when loaded
    device: str = "cpu"


_handles: Dict[str, PolicyHandle] = {}


def _adapter_present() -> bool:
    return (Path(ADAPTER_DIR) / "adapter_config.json").exists()


def get_policy(name: str) -> PolicyHandle:
    """Lazy-load + cache the requested policy. Idempotent and safe to call repeatedly."""
    if name not in ALL_POLICIES:
        raise ValueError(f"unknown policy: {name}")
    if name in _handles:
        return _handles[name]

    if name == POLICY_BASELINE_WAIT:
        h = PolicyHandle(name=name, available=True, reason="")
        _handles[name] = h
        return h

    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except ImportError as exc:
        h = PolicyHandle(name=name, available=False, reason=f"transformers/torch not installed: {exc}")
        _handles[name] = h
        return h

    if name == POLICY_TRAINED and not _adapter_present():
        h = PolicyHandle(
            name=name,
            available=False,
            reason=f"adapter not found at {ADAPTER_DIR}; run Phase 0 training first",
        )
        _handles[name] = h
        return h

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        base_kwargs: Dict[str, Any] = {}
        if device == "cuda":
            base_kwargs["torch_dtype"] = torch.float16
            base_kwargs["device_map"] = "auto"
        else:
            base_kwargs["torch_dtype"] = torch.float32
        base = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, **base_kwargs)
        if device != "cuda":
            base = base.to(device)
        if name == POLICY_TRAINED:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        else:
            model = base
        model.eval()
        h = PolicyHandle(name=name, available=True, model=model, tokenizer=tokenizer, device=device)
    except Exception as exc:
        logger.exception("policy load failed for %s", name)
        h = PolicyHandle(name=name, available=False, reason=f"load failed: {exc.__class__.__name__}: {exc}")
    _handles[name] = h
    return h


def _build_messages(obs_json: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Current Observation:\n{obs_json}"},
    ]


def infer_action(handle: PolicyHandle, obs: Dict[str, Any]) -> Dict[str, Any]:
    """Return a raw action dict (action_type + parameters). Falls back to wait on parse error.

    The fallback is recorded by the episode_runner; this function does not log silently.
    """
    if handle.name == POLICY_BASELINE_WAIT:
        return {"action_type": "wait"}

    if not handle.available or handle.model is None or handle.tokenizer is None:
        return {"action_type": "wait", "_fallback_reason": handle.reason or "policy unavailable"}

    import torch
    obs_json = json.dumps(obs, default=str)
    msgs = _build_messages(obs_json)
    try:
        prompt = handle.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = SYSTEM_PROMPT + "\n\nObservation:\n" + obs_json + "\n\nAction JSON:"
    inputs = handle.tokenizer(prompt, return_tensors="pt").to(handle.device)
    gen_kwargs: Dict[str, Any] = {"max_new_tokens": 96, "pad_token_id": handle.tokenizer.eos_token_id}
    if handle.name == POLICY_TRAINED:
        gen_kwargs.update({"do_sample": True, "temperature": 0.7, "top_p": 0.9})
    else:
        gen_kwargs.update({"do_sample": False, "temperature": 0.0})
    with torch.no_grad():
        out = handle.model.generate(**inputs, **gen_kwargs)
    text = handle.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    text = text.strip().strip("`").strip()
    if text.startswith("json"):
        text = text[4:].strip()
    try:
        action = json.loads(text)
        if not isinstance(action, dict) or "action_type" not in action:
            raise ValueError("missing action_type")
        return action
    except (ValueError, json.JSONDecodeError):
        return {"action_type": "wait", "_fallback_reason": "parse_failed", "_raw": text[:200]}
