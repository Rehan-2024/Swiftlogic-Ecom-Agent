"""Plotting utilities for training artifacts.

Matplotlib-only (no seaborn) so the stack is thin. Produces the PNGs
listed in the roadmap: ``reward_curve.png``, ``before_after_comparison.png``,
``behavior_evolution.png``, ``generalization.png``, etc.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _lazy_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_reward_curve(
    rewards: Sequence[float],
    out_path: str,
    *,
    title: str = "Training reward (per episode)",
    window: int = 10,
    stage_boundaries: Optional[Sequence[Tuple[int, str]]] = None,
) -> str:
    plt = _lazy_plt()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rewards, alpha=0.4, label="episode reward")
    if len(rewards) >= window:
        smooth = [
            statistics.mean(rewards[max(0, i - window + 1) : i + 1]) for i in range(len(rewards))
        ]
        ax.plot(smooth, color="tab:red", label=f"rolling mean ({window})")
    if stage_boundaries:
        for idx, name in stage_boundaries:
            ax.axvline(idx, linestyle="--", color="gray", alpha=0.7)
            ax.text(idx, max(rewards) * 0.9 if rewards else 1.0, name, rotation=90, fontsize=8)
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_before_after_bars(
    before_summary: Dict[str, Dict[str, float]],
    after_summary: Dict[str, Dict[str, float]],
    out_path: str,
    *,
    tasks: Optional[Sequence[str]] = None,
    title: str = "Grader scores — before vs after",
) -> str:
    plt = _lazy_plt()
    tasks = tasks or list(before_summary.keys())
    before_vals = [float(before_summary[t]["mean"]) for t in tasks]
    after_vals = [float(after_summary[t]["mean"]) for t in tasks]
    x = range(len(tasks))
    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 1.3), 5))
    ax.bar([i - 0.18 for i in x], before_vals, width=0.36, label="before (baseline)", color="#b0b0b0")
    ax.bar([i + 0.18 for i in x], after_vals, width=0.36, label="after (trained)", color="#e55934")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tasks, rotation=25, ha="right")
    ax.set_ylabel("grader score (0..1)")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_generalization(
    per_config_summaries: Dict[str, Dict[str, Dict[str, float]]],
    out_path: str,
    *,
    tasks: Optional[Sequence[str]] = None,
    title: str = "Generalization across configs",
) -> str:
    plt = _lazy_plt()
    configs = list(per_config_summaries.keys())
    tasks = tasks or list(next(iter(per_config_summaries.values())).keys())
    fig, ax = plt.subplots(figsize=(max(9, len(tasks) * 1.3), 5))
    width = 0.8 / max(1, len(configs))
    for i, config in enumerate(configs):
        offset = -0.4 + width * (i + 0.5)
        vals = [float(per_config_summaries[config][t]["mean"]) for t in tasks]
        ax.bar([x + offset for x in range(len(tasks))], vals, width=width, label=config)
    ax.set_xticks(list(range(len(tasks))))
    ax.set_xticklabels(tasks, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("grader score (0..1)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_exploration_curve(
    action_entropy_per_checkpoint: List[float],
    out_path: str,
    *,
    checkpoint_labels: Optional[Sequence[str]] = None,
    title: str = "Exploration curve (action-type entropy)",
) -> str:
    plt = _lazy_plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(len(action_entropy_per_checkpoint)))
    ax.plot(x, action_entropy_per_checkpoint, marker="o", color="#2a9d8f")
    if checkpoint_labels:
        ax.set_xticks(x)
        ax.set_xticklabels(list(checkpoint_labels), rotation=20, ha="right")
    ax.set_ylabel("Shannon entropy over action types")
    ax.set_xlabel("checkpoint")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_failure_vs_recovery(
    baseline_banks: Iterable[float],
    trained_banks: Iterable[float],
    out_path: str,
    *,
    title: str = "Failure vs recovery — scripted baseline vs trained",
) -> str:
    plt = _lazy_plt()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(baseline_banks), label="baseline (scripted)", color="#888", linewidth=2)
    ax.plot(list(trained_banks), label="trained (LoRA)", color="#e63946", linewidth=2)
    ax.set_xlabel("day")
    ax.set_ylabel("bank balance")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
