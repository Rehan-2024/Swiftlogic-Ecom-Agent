"""Episode runner - orchestrates one or two policies against the live backend.

Persists every run to LIVE_RUNS_DIR/<run_id>.json with the strict
authenticity record from Round-2 plan section 5.5.

Computes the action-success metrics for each run so the dashboard can
verify that the trained policy has higher success rates than baseline
(plan section 4 + 7).
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.backend_client import BackendClient, BackendError  # noqa: E402
from demo.policy import (  # noqa: E402
    ALL_POLICIES,
    POLICY_BASELINE_WAIT,
    PolicyHandle,
    get_policy,
    infer_action,
)

logger = logging.getLogger("commerceops.demo.episode_runner")


LIVE_RUNS_DIR = Path(os.getenv("LIVE_RUNS_DIR", str(ROOT / "artifacts" / "live_runs")))
LIVE_RUNS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))


def _git_sha() -> Optional[str]:
    try:
        import subprocess
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, timeout=3)
        return out.decode("ascii").strip()
    except Exception:
        return None


def _adapter_sha() -> Optional[str]:
    """Hash a few adapter files so we can prove which weights were used."""
    from demo.artifact_loader import ADAPTER_DIR
    cfg = Path(ADAPTER_DIR) / "adapter_config.json"
    if not cfg.exists():
        return None
    try:
        import hashlib
        h = hashlib.sha256()
        for p in sorted(Path(ADAPTER_DIR).iterdir()):
            if p.is_file() and p.suffix in {".json", ".safetensors", ".bin"}:
                h.update(p.name.encode())
                h.update(str(p.stat().st_size).encode())
        return h.hexdigest()[:16]
    except Exception:
        return None


def _action_success(action: Dict[str, Any], info: Dict[str, Any], obs_before: Dict[str, Any], obs_after: Dict[str, Any]) -> bool:
    """Per-action success rule (plan section 7.1)."""
    a_type = action.get("action_type", "wait")
    err = info.get("error") if isinstance(info, dict) else None
    if err is not None and err != "":
        return False
    if a_type == "wait":
        return True
    if a_type == "restock":
        sku = action.get("sku")
        return bool(sku) and int(obs_after.get("inventory", {}).get(sku, 0)) >= int(obs_before.get("inventory", {}).get(sku, 0))
    if a_type == "refund":
        return len(obs_after.get("active_tickets") or []) <= len(obs_before.get("active_tickets") or [])
    if a_type == "ad_spend":
        return float(obs_after.get("bank_balance", 0.0)) >= 0.0
    if a_type == "negotiate":
        neg = info.get("negotiate", {}) if isinstance(info, dict) else {}
        return bool(neg.get("quote_unit_price")) or bool(neg.get("accepted"))
    if a_type == "set_price":
        return True
    return False


def _entropy(distribution: Dict[str, float]) -> float:
    h = 0.0
    for p in distribution.values():
        if p > 0:
            h -= p * math.log(p)
    return round(h, 4)


def _action_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_action_attempts: Counter = Counter()
    by_action_success: Counter = Counter()
    for r in records:
        a = r["action"].get("action_type", "wait")
        by_action_attempts[a] += 1
        if r.get("success"):
            by_action_success[a] += 1
    by_action: Dict[str, Any] = {}
    for a in sorted(by_action_attempts.keys()):
        attempts = by_action_attempts[a]
        successes = by_action_success[a]
        by_action[a] = {
            "attempts": int(attempts),
            "successes": int(successes),
            "success_rate": round(successes / attempts, 4) if attempts else 0.0,
        }
    total = sum(by_action_attempts.values()) or 1
    distribution = {a: round(c / total, 4) for a, c in by_action_attempts.items()}
    return {
        "by_action": by_action,
        "distribution": distribution,
        "entropy": _entropy(distribution),
    }


def run_episode(
    policy_name: str,
    seed: int,
    *,
    backend: Optional[BackendClient] = None,
    business_id: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run one episode end-to-end. Returns the persisted trace dict.

    on_step(record) is called after each step so the UI can stream output;
    on_step exceptions are swallowed to keep the run going.
    """
    if policy_name not in ALL_POLICIES:
        raise ValueError(f"unknown policy: {policy_name}")
    if backend is None:
        backend = BackendClient()
    handle = get_policy(policy_name)
    run_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()

    if business_id:
        try:
            backend.config(business_id=business_id, seed=seed)
        except BackendError as exc:
            return _error_trace(run_id, started_at, policy_name, seed, business_id, backend, str(exc))

    try:
        obs_payload = backend.reset(seed=seed)
    except BackendError as exc:
        return _error_trace(run_id, started_at, policy_name, seed, business_id, backend, str(exc))

    obs_before = obs_payload.get("observation", {}) or {}
    starting_bank = float(obs_before.get("bank_balance", 0.0))

    records: List[Dict[str, Any]] = []
    rewards: List[float] = []
    fallback_count = 0
    bankrupt = False
    done_reason = ""

    for step in range(1, max_steps + 1):
        action = infer_action(handle, obs_before)
        if action.get("_fallback_reason"):
            fallback_count += 1
        clean_action = {k: v for k, v in action.items() if not k.startswith("_")}
        try:
            r = backend.step(clean_action)
        except BackendError as exc:
            done_reason = f"step_error:{exc.kind}"
            break
        obs_after = r.get("observation", {}) or {}
        reward = float(r.get("reward", 0.0))
        info = r.get("info", {}) or {}
        rewards.append(reward)
        success = _action_success(clean_action, info, obs_before, obs_after)
        record = {
            "step": step,
            "action": clean_action,
            "reward": round(reward, 6),
            "bank_balance": round(float(obs_after.get("bank_balance", 0.0)), 2),
            "inventory_summary": {k: int(v) for k, v in (obs_after.get("inventory", {}) or {}).items()},
            "intent": info.get("intent"),
            "action_quality": info.get("action_quality"),
            "confidence": info.get("confidence"),
            "info_error": info.get("error"),
            "success": bool(success),
            "fallback": action.get("_fallback_reason"),
            "done": bool(r.get("done")),
            "endpoint_call_id": backend.call_counts.get("step", 0),
        }
        records.append(record)
        if on_step:
            try:
                on_step(record)
            except Exception:
                pass
        if float(obs_after.get("bank_balance", 0.0)) <= 0:
            bankrupt = True
        obs_before = obs_after
        if r.get("done"):
            done_reason = "env_done"
            break
    else:
        done_reason = "max_steps_reached"

    grader_scores: Dict[str, float] = {}
    try:
        grader_payload = backend.grader()
        for entry in grader_payload.get("scores", []):
            grader_scores[entry.get("task_id", "")] = float(entry.get("score", 0.0))
    except BackendError as exc:
        grader_scores["__error__"] = str(exc)

    ended_at = datetime.now(timezone.utc).isoformat()
    summary = _action_summary(records)
    final_bank = round(records[-1]["bank_balance"], 2) if records else round(starting_bank, 2)

    trace = {
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "seed": int(seed),
        "policy_type": policy_name,
        "business_id": business_id or "default",
        "endpoint_base_url": backend.base_url,
        "endpoint_call_counts": dict(backend.call_counts),
        "fallback_count": int(fallback_count),
        "git_sha": _git_sha(),
        "adapter_sha": _adapter_sha(),
        "starting_bank": round(starting_bank, 2),
        "final_bank": final_bank,
        "total_reward": round(sum(rewards), 6),
        "n_steps": len(records),
        "bankrupt": bool(bankrupt),
        "done_reason": done_reason,
        "grader_scores": grader_scores,
        "action_summary": summary,
        "steps": records,
    }
    out_path = LIVE_RUNS_DIR / f"{run_id}.json"
    out_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    trace["_path"] = str(out_path)
    return trace


def _error_trace(run_id: str, started_at: str, policy: str, seed: int,
                 business_id: Optional[str], backend: BackendClient, msg: str) -> Dict[str, Any]:
    trace = {
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "policy_type": policy,
        "business_id": business_id or "default",
        "endpoint_base_url": backend.base_url,
        "endpoint_call_counts": dict(backend.call_counts),
        "error": msg,
        "steps": [],
    }
    out_path = LIVE_RUNS_DIR / f"{run_id}.json"
    out_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    trace["_path"] = str(out_path)
    return trace


def run_ab_comparison(
    seed: int,
    policy_a: str,
    policy_b: str,
    *,
    backend_a: Optional[BackendClient] = None,
    backend_b: Optional[BackendClient] = None,
    business_id: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run policy_a then policy_b on the SAME seed and persist a comparison.

    Two separate BackendClient instances ensure call counts are not mixed.
    """
    backend_a = backend_a or BackendClient()
    backend_b = backend_b or BackendClient()
    trace_a = run_episode(
        policy_a, seed,
        backend=backend_a,
        business_id=business_id,
        max_steps=max_steps,
        on_step=(lambda rec: on_step(policy_a, rec)) if on_step else None,
    )
    trace_b = run_episode(
        policy_b, seed,
        backend=backend_b,
        business_id=business_id,
        max_steps=max_steps,
        on_step=(lambda rec: on_step(policy_b, rec)) if on_step else None,
    )
    comparison = {
        "comparison_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "business_id": business_id or "default",
        "run_a": {"policy": policy_a, "run_id": trace_a.get("run_id"), "path": trace_a.get("_path")},
        "run_b": {"policy": policy_b, "run_id": trace_b.get("run_id"), "path": trace_b.get("_path")},
        "summary": {
            "a": _ab_row(trace_a),
            "b": _ab_row(trace_b),
        },
    }
    out_path = LIVE_RUNS_DIR / f"comparison_{comparison['comparison_id']}.json"
    out_path.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    comparison["_path"] = str(out_path)
    comparison["trace_a"] = trace_a
    comparison["trace_b"] = trace_b
    return comparison


def _ab_row(trace: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "policy": trace.get("policy_type"),
        "final_bank": trace.get("final_bank"),
        "total_reward": trace.get("total_reward"),
        "bankrupt": trace.get("bankrupt"),
        "n_steps": trace.get("n_steps"),
        "fallback_count": trace.get("fallback_count"),
        "grader_scores": trace.get("grader_scores"),
        "entropy": (trace.get("action_summary") or {}).get("entropy"),
        "distribution": (trace.get("action_summary") or {}).get("distribution"),
    }
