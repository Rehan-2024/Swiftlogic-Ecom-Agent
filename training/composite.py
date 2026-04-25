"""Composite score calculation (Part B+.6).

Produces a single headline scalar in ``[0, 1]`` that combines:

* ``composite_training_mean``  — average of 3 training graders
* ``composite_all_mean``       — average of 6 graders (training + eval-only)
* ``format_compliance_mean``   — share of steps with valid JSON
* ``generalization_dropoff``   — 1 - max(|train - test| across configs)

The weights are fixed so judges can see a deterministic recipe (roadmap
B+.6: "no magic numbers, show the formula").
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CompositeWeights:
    w_training: float = 0.35
    w_all: float = 0.35
    w_format: float = 0.15
    w_generalization: float = 0.15


def _safe_get(d: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def _generalization_score(after_bundle: Optional[Dict[str, Any]]) -> float:
    """1 - max(|train_mean - alt_mean|) across configs in the generalization bundle."""
    if not after_bundle or "episodes" not in after_bundle:
        return 0.5
    episodes = after_bundle["episodes"]
    by_config: Dict[str, List[float]] = {}
    for ep in episodes:
        config = ep.get("config", "unknown")
        scores = ep.get("grader_scores", {})
        vals = [float(v) for v in scores.values() if isinstance(v, (int, float))]
        if vals:
            by_config.setdefault(config, []).append(statistics.mean(vals))
    if len(by_config) < 2:
        return 0.5
    means = [statistics.mean(v) for v in by_config.values()]
    spread = max(means) - min(means)
    return max(0.0, min(1.0, 1.0 - spread))


def compute_composite(
    before_metrics_path: str,
    after_metrics_path: str,
    *,
    generalization_path: Optional[str] = None,
    weights: Optional[CompositeWeights] = None,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    weights = weights or CompositeWeights()

    def _load(path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    before = _load(before_metrics_path) or {}
    after = _load(after_metrics_path) or {}
    gen = _load(generalization_path)

    def _score(bundle: Dict[str, Any]) -> float:
        if not bundle:
            return 0.0
        s = bundle.get("summary", {})
        training = _safe_get(s, "composite_training_mean")
        all_scores = _safe_get(s, "composite_all_mean")
        fmt = _safe_get(s, "mean_format_compliance")
        g = _generalization_score(gen) if bundle is after else 0.5
        return (
            weights.w_training * training
            + weights.w_all * all_scores
            + weights.w_format * fmt
            + weights.w_generalization * g
        )

    before_score = _score(before)
    after_score = _score(after)
    delta_abs = after_score - before_score
    delta_pct = (delta_abs / before_score * 100.0) if before_score > 0 else 0.0
    arrow = " -> "
    headline = f"{before_score:.2f}{arrow}{after_score:.2f} (+{delta_pct:.0f}%)"

    result = {
        "weights": weights.__dict__,
        "before": {
            "score": round(before_score, 4),
            "composite_training": round(_safe_get(before, "summary", "composite_training_mean"), 4),
            "composite_all": round(_safe_get(before, "summary", "composite_all_mean"), 4),
            "format_compliance": round(_safe_get(before, "summary", "mean_format_compliance"), 4),
        },
        "after": {
            "score": round(after_score, 4),
            "composite_training": round(_safe_get(after, "summary", "composite_training_mean"), 4),
            "composite_all": round(_safe_get(after, "summary", "composite_all_mean"), 4),
            "format_compliance": round(_safe_get(after, "summary", "mean_format_compliance"), 4),
            "generalization": round(_generalization_score(gen), 4),
        },
        "delta": {
            "abs": round(delta_abs, 4),
            "pct": round(delta_pct, 2),
        },
        "headline": headline,
    }
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
