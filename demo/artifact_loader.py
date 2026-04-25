"""Artifact loader + provenance/freshness gate.

The dashboard reads ALL training proof from disk (via this module) so the
UI never needs to know anything about how the artifacts were produced.

Provenance contract (Round-2 plan section 1, rule 5):
    pipeline_manifest.json["provenance"] in {"grpo_trained", "heuristic_fallback"}
    pipeline_manifest.json["adapter_status"] in {"available", "missing", "unknown"}

Ready-for-judge gate is true only when:
    provenance == "grpo_trained" AND adapter_status == "available"

If READY_FOR_JUDGE=force_on is set, this module raises at import time
(strict authenticity rule - no silent overrides).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", str(ARTIFACTS_DIR / "swiftlogic_grpo_adapter")))


class Provenance(str, Enum):
    GRPO_TRAINED = "grpo_trained"
    HEURISTIC_FALLBACK = "heuristic_fallback"
    UNKNOWN = "unknown"


_FORCE = os.getenv("READY_FOR_JUDGE", "auto").strip().lower()
if _FORCE == "force_on":
    raise RuntimeError(
        "READY_FOR_JUDGE=force_on is rejected by the authenticity rules. "
        "The judge-ready badge can only flip on when artifacts are real."
    )
_FORCE_OFF = (_FORCE == "force_off")


@dataclass
class JudgeReadiness:
    ready: bool
    provenance: Provenance
    adapter_status: str
    reasons: List[str] = field(default_factory=list)
    pipeline_run_at: Optional[str] = None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def load_pipeline_manifest() -> Dict[str, Any]:
    return _read_json(ARTIFACTS_DIR / "pipeline_manifest.json") or {}


def load_composite_score() -> Dict[str, Any]:
    return _read_json(ARTIFACTS_DIR / "composite_score.json") or {}


def load_policy_signature() -> Dict[str, Any]:
    return _read_json(ARTIFACTS_DIR / "policy_signature.json") or {}


def load_generalization() -> Dict[str, Any]:
    return _read_json(ARTIFACTS_DIR / "generalization.json") or {}


def load_failure_vs_recovery() -> Dict[str, Any]:
    return _read_json(ARTIFACTS_DIR / "failure_vs_recovery.json") or {}


def load_before_metrics() -> Dict[str, Any]:
    return _read_json(ARTIFACTS_DIR / "before_metrics.json") or {}


def load_after_metrics() -> Dict[str, Any]:
    return _read_json(ARTIFACTS_DIR / "after_metrics.json") or {}


def load_action_success(policy: str) -> Dict[str, Any]:
    """policy in {'baseline_wait_only','baseline_zero_shot','trained'}."""
    return _read_json(ARTIFACTS_DIR / f"action_success_{policy}.json") or {}


def artifact_image_path(filename: str) -> Optional[str]:
    p = ARTIFACTS_DIR / filename
    return str(p) if p.exists() else None


def judge_readiness() -> JudgeReadiness:
    manifest = load_pipeline_manifest()
    raw_prov = (manifest.get("provenance") or "unknown").strip().lower()
    try:
        prov = Provenance(raw_prov)
    except ValueError:
        prov = Provenance.UNKNOWN
    adapter_status = (manifest.get("adapter_status") or "unknown").strip().lower()
    pipeline_run_at = manifest.get("pipeline_run_at")
    reasons: List[str] = []
    if prov != Provenance.GRPO_TRAINED:
        reasons.append(f"provenance is {prov.value}, expected grpo_trained")
    if adapter_status != "available":
        reasons.append(f"adapter_status is {adapter_status}, expected available")
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        reasons.append(f"adapter_config.json not found at {ADAPTER_DIR}")
    if _FORCE_OFF:
        reasons.append("READY_FOR_JUDGE=force_off in environment")
    ready = not reasons
    return JudgeReadiness(
        ready=ready,
        provenance=prov,
        adapter_status=adapter_status,
        reasons=reasons,
        pipeline_run_at=pipeline_run_at,
    )


def freshness_summary() -> Dict[str, Any]:
    """Return mtime / age info for the key artifacts so the UI can flag stale files."""
    keys = [
        "pipeline_manifest.json",
        "composite_score.json",
        "policy_signature.json",
        "generalization.json",
        "before_metrics.json",
        "after_metrics.json",
        "reward_curve.png",
        "exploration_curve.png",
        "policy_evolution.png",
        "before_after_comparison.png",
        "failure_vs_recovery.png",
        "generalization.png",
    ]
    now = time.time()
    out: Dict[str, Any] = {}
    for key in keys:
        p = ARTIFACTS_DIR / key
        if p.exists():
            mt = p.stat().st_mtime
            out[key] = {
                "exists": True,
                "size": p.stat().st_size,
                "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mt)),
                "age_minutes": round((now - mt) / 60.0, 1),
            }
        else:
            out[key] = {"exists": False}
    return out


def generalization_covers_unseen_configs() -> Dict[str, Any]:
    """Section 6.3 + 10.6 - hard guard: medplus_pharmacy and stackbase_saas must appear."""
    data = load_generalization()
    episodes = data.get("episodes", []) if isinstance(data, dict) else []
    seen_configs = set()
    for ep in episodes:
        cfg = (ep.get("config") or "").replace("\\", "/").lower()
        if "medplus_pharmacy" in cfg:
            seen_configs.add("medplus_pharmacy")
        if "stackbase_saas" in cfg:
            seen_configs.add("stackbase_saas")
    required = {"medplus_pharmacy", "stackbase_saas"}
    missing = sorted(required - seen_configs)
    return {
        "covered": sorted(seen_configs),
        "missing": missing,
        "ok": not missing,
    }


def policy_signatures_distinct() -> Dict[str, Any]:
    """Section 10.4 + 10.7.4 - trained signature hash must differ from heuristic."""
    sig = load_policy_signature().get("signatures", {}) if isinstance(load_policy_signature(), dict) else {}
    hashes: Dict[str, str] = {}
    for name, body in sig.items() if isinstance(sig, dict) else []:
        if isinstance(body, dict) and "hash" in body:
            hashes[name] = str(body.get("hash"))
    distinct = len(set(hashes.values())) == len(hashes) if hashes else False
    return {"hashes": hashes, "distinct": distinct, "ok": distinct}
