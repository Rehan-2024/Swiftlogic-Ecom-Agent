"""Behavior-evolution + policy-signature utilities (Part B+.3).

Given a list of episode traces (one per checkpoint / baseline / trained),
this module:

* Counts action-type frequencies (Shannon entropy → exploration curve).
* Builds a ``policy_signature.json`` that hashes the action distribution
  so two policies with identical preferences collide cleanly.
* Plots ``behavior_evolution.png`` — stacked-bar per checkpoint with the
  share of each action type.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ACTION_TYPES = ("wait", "restock", "refund", "ad_spend", "negotiate", "set_price")


def _counter_to_probs(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values()) or 1
    return {a: counter.get(a, 0) / total for a in ACTION_TYPES}


def action_entropy(actions: Iterable[Dict[str, Any]]) -> float:
    counter = Counter(a.get("action_type", "wait") for a in actions)
    probs = _counter_to_probs(counter)
    entropy = 0.0
    for p in probs.values():
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def policy_signature(actions: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    counter = Counter(a.get("action_type", "wait") for a in actions)
    probs = _counter_to_probs(counter)
    payload = json.dumps(probs, sort_keys=True)
    return {
        "distribution": {k: round(v, 4) for k, v in probs.items()},
        "entropy": round(action_entropy(actions), 4),
        "hash": hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16],
    }


def plot_behavior_evolution(
    checkpoint_label_to_actions: Dict[str, List[Dict[str, Any]]],
    out_path: str,
    *,
    title: str = "Behavior evolution (action distribution per checkpoint)",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(checkpoint_label_to_actions.keys())
    if not labels:
        # Nothing to plot — emit a blank so downstream artifacts still exist.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no checkpoints", ha="center", va="center")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path

    data = {a: [] for a in ACTION_TYPES}
    for label in labels:
        probs = _counter_to_probs(Counter(a.get("action_type", "wait") for a in checkpoint_label_to_actions[label]))
        for a in ACTION_TYPES:
            data[a].append(probs[a])

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.3), 5))
    bottom = [0.0] * len(labels)
    for a in ACTION_TYPES:
        ax.bar(labels, data[a], bottom=bottom, label=a)
        bottom = [b + v for b, v in zip(bottom, data[a])]
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of steps")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(ACTION_TYPES))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_policy_evolution_line(
    checkpoint_label_to_actions: Dict[str, List[Dict[str, Any]]],
    out_path: str,
    *,
    title: str = "Policy evolution — key action shares across checkpoints",
) -> str:
    """Line chart of restock / refund / ad_spend shares over checkpoints."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(checkpoint_label_to_actions.keys())
    if not labels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no checkpoints", ha="center", va="center")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path

    data: Dict[str, List[float]] = {a: [] for a in ACTION_TYPES}
    for label in labels:
        probs = _counter_to_probs(Counter(a.get("action_type", "wait") for a in checkpoint_label_to_actions[label]))
        for a in ACTION_TYPES:
            data[a].append(probs[a])

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.3), 5))
    x = list(range(len(labels)))
    for a in ("restock", "refund", "ad_spend", "set_price"):
        ax.plot(x, data[a], marker="o", label=a)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of steps")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
