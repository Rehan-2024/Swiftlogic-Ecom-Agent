"""Live Demo Theater - the showcase interactive component.

A Gradio generator that runs one episode against the live OpenEnv backend
and yields, on every step:

  1. The current step card (Day N) showing the four phases:
        Observe -> Reason -> Act -> React
  2. A live-updating bank-balance line chart (matplotlib).
  3. A live-updating action-distribution bar chart (matplotlib).
  4. The accumulated step log.
  5. A progress bar / episode meta line.
  6. A final scorecard at episode end.

The phase visualisation is honest: every phase shows a piece of the
real per-step record returned by the backend - no scripted animation
is passed off as agent behaviour.

A small ``STEP_PACE_SECONDS`` delay is applied between yields so the
four phase cards have time to fade in (CSS animation is ~330ms total)
and so the judge can read each Day's decision before it's replaced.
The delay is purely cosmetic; the underlying backend calls happen at
their natural pace.
"""

from __future__ import annotations

import html
import json
import logging
import os
import queue
import threading
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from demo.backend_client import BackendClient, BackendError  # noqa: E402
from demo.components import banner, patience_box  # noqa: E402
from demo.episode_runner import _action_success, run_ab_comparison  # noqa: E402
from demo.policy import (  # noqa: E402
    ALL_POLICIES,
    POLICY_BASELINE_WAIT,
    POLICY_BASELINE_ZERO_SHOT,
    POLICY_TRAINED,
    get_policy,
    infer_action,
)

logger = logging.getLogger("commerceops.demo.live_theater")

# Palette - mirrors theme.css so static and matplotlib visuals feel native.
PAL_INK = "#2c3e50"
PAL_INK_SOFT = "#475569"
PAL_INK_MUTED = "#6b7280"
PAL_BG = "#faf7f2"
PAL_SURFACE = "#ffffff"
PAL_BORDER = "#e7e1d6"
PAL_ACCENT = "#b08968"
PAL_ACCENT2 = "#6b8e6b"
PAL_DANGER = "#b3543c"

DEMO_MAX_STEPS = int(os.getenv("DEMO_MAX_STEPS", "30"))
# Pacing is purely cosmetic; default OFF so streaming feels live. Set
# STEP_PACE_SECONDS to a small value (e.g. 0.1) to slow down for demos.
STEP_PACE_SECONDS = float(os.getenv("STEP_PACE_SECONDS", "0.0"))
# Compare-mode coalesces step events so the heavy matplotlib redraw runs
# at most this often. Per-event yields stay correct (the queue keeps
# all data); we just collapse rapid bursts.
COMPARE_MIN_REDRAW_SEC = float(os.getenv("COMPARE_MIN_REDRAW_SEC", "0.22"))
# Single-policy stream: redrawing 3 matplotlib figures every step is the main
# UI cost. Refresh charts on a stride + time budget; HTML still every step.
LIVE_CHART_EVERY_N = max(1, int(os.getenv("LIVE_CHART_EVERY_N", "2")))
LIVE_MIN_REDRAW_SEC = float(os.getenv("LIVE_MIN_REDRAW_SEC", "0.12"))
# Smaller, snappier matplotlib defaults for the live theater.
CHART_FIGSIZE = (5.0, 2.4)
CHART_DPI = 84


def _legend_if_any(ax: "plt.Axes") -> None:
    _h, lab = ax.get_legend_handles_labels()
    if lab:
        ax.legend(loc="best", fontsize=7)


# ---------------------------------------------------------------------------
# Phase / step rendering
# ---------------------------------------------------------------------------

def _truncate_text(text: str, n: int = 240) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    return text[: n - 1] + "..."


def _observe_phase_html(obs: Dict[str, Any]) -> str:
    bank = float(obs.get("bank_balance", 0.0))
    inv = obs.get("inventory", {}) or {}
    tickets = obs.get("active_tickets") or []
    competitor = (obs.get("competitor_prices") or {})
    rows: List[str] = []
    rows.append(f"bank   INR {bank:>12,.2f}")
    rows.append(f"sku-ct {len(inv):>12d}")
    if inv:
        top = sorted(inv.items(), key=lambda kv: -int(kv[1]))[:3]
        for sku, qty in top:
            rows.append(f"  {sku[:10]:<10} {int(qty):>5d} u")
    if competitor:
        rows.append(f"comp   {len(competitor):>12d} skus")
    rows.append(f"tickets {len(tickets):>11d}")
    body = html.escape("\n".join(rows))
    return (
        '<div class="r2-phase">'
        '<div class="phase-eyebrow">phase 1 / observe</div>'
        '<div class="phase-title">Read environment state</div>'
        f'<pre class="phase-body">{body}</pre>'
        '</div>'
    )


def _reason_phase_html(record: Dict[str, Any], policy_name: str) -> str:
    intent = record.get("intent")
    quality = record.get("action_quality")
    confidence = record.get("confidence")
    fallback = record.get("fallback")
    lines: List[str] = []
    if intent:
        lines.append(f"intent: {intent}")
    if quality is not None:
        lines.append(f"quality: {quality}")
    if confidence is not None:
        try:
            lines.append(f"confidence: {float(confidence):.2f}")
        except (TypeError, ValueError):
            pass
    if not lines:
        if policy_name == POLICY_BASELINE_WAIT:
            lines.append("policy: WaitOnly - no model invocation; emits 'wait' deterministically.")
        else:
            lines.append("policy emitted action without a parseable intent label.")
    if fallback:
        lines.append(f"fallback: {fallback}")
    body = html.escape(_truncate_text("\n".join(lines), 320))
    cls = " is-warn" if fallback else ""
    return (
        f'<div class="r2-phase{cls}">'
        '<div class="phase-eyebrow">phase 2 / reason</div>'
        '<div class="phase-title">CEO trace</div>'
        f'<div class="phase-body is-text">{body}</div>'
        '</div>'
    )


def _act_phase_html(action: Dict[str, Any]) -> str:
    a_type = action.get("action_type", "wait")
    pretty = json.dumps(action, indent=2, sort_keys=True)
    body = html.escape(_truncate_text(pretty, 360))
    return (
        '<div class="r2-phase">'
        '<div class="phase-eyebrow">phase 3 / act</div>'
        f'<div class="phase-title">action_type: {html.escape(str(a_type))}</div>'
        f'<pre class="phase-body">{body}</pre>'
        '</div>'
    )


def _react_phase_html(record: Dict[str, Any]) -> str:
    reward = float(record.get("reward", 0.0))
    bank = float(record.get("bank_balance", 0.0))
    success = bool(record.get("success"))
    err = record.get("info_error")
    lines = [
        f"reward    {reward:>+9.3f}",
        f"new bank  INR {bank:>11,.2f}",
        f"success   {'yes' if success else 'no'}",
    ]
    if err:
        lines.append(f"env error {str(err)[:60]}")
    body = html.escape("\n".join(lines))
    cls = " is-success" if success and not err else (" is-failure" if err else " is-warn")
    if reward > 0 and success and not err:
        cls = " is-success"
    elif reward < 0 and not success:
        cls = " is-failure"
    return (
        f'<div class="r2-phase{cls}">'
        '<div class="phase-eyebrow">phase 4 / react</div>'
        '<div class="phase-title">Environment response</div>'
        f'<pre class="phase-body">{body}</pre>'
        '</div>'
    )


def _step_card_html(step_idx: int, total: int, obs: Dict[str, Any], action: Dict[str, Any], record: Dict[str, Any], policy_name: str) -> str:
    reward = float(record.get("reward", 0.0))
    if reward > 0.05:
        badge_cls, badge_label = "is-pos", f"+{reward:.3f}"
    elif reward < -0.05:
        badge_cls, badge_label = "is-neg", f"{reward:.3f}"
    else:
        badge_cls, badge_label = "is-neu", f"{reward:+.3f}"
    return (
        '<div class="r2-step-card">'
        '<div class="day-bar">'
        f'<span class="day-label">Day {step_idx:02d} / {total}</span>'
        f'<span class="reward-badge {badge_cls}">reward {badge_label}</span>'
        '</div>'
        '<div class="r2-phase-grid">'
        f'{_observe_phase_html(obs)}'
        f'{_reason_phase_html(record, policy_name)}'
        f'{_act_phase_html(action)}'
        f'{_react_phase_html(record)}'
        '</div>'
        '</div>'
    )


def build_pipeline_html(pct: float) -> str:
    return f"""
<div class="pipeline-flow">
  <div class="p-node {'active' if pct >= 0 else ''}">
    <div class="icon">🚀</div>
    <div class="label">Start</div>
  </div>
  <div class="p-edge"><div class="p-fill" style="width: {min(100, max(0, pct / 33.3 * 100))}%"></div></div>
  <div class="p-node {'active' if pct >= 33.3 else ''}">
    <div class="icon">⚙️</div>
    <div class="label">Mid-shift</div>
  </div>
  <div class="p-edge"><div class="p-fill" style="width: {min(100, max(0, (pct - 33.3) / 33.3 * 100))}%"></div></div>
  <div class="p-node {'active' if pct >= 66.6 else ''}">
    <div class="icon">📈</div>
    <div class="label">Late-shift</div>
  </div>
  <div class="p-edge"><div class="p-fill" style="width: {min(100, max(0, (pct - 66.6) / 33.3 * 100))}%"></div></div>
  <div class="p-node {'active' if pct >= 99.9 else ''}">
    <div class="icon">🏁</div>
    <div class="label">Complete</div>
  </div>
</div>
"""


def _theater_head_html(policy: str, seed: int, business_id: str, base_url: str, current: int, total: int, status: str) -> str:
    pct = max(0.0, min(100.0, (current / max(total, 1)) * 100.0))
    return (
        '<div class="r2-theater-head">'
        '<div>'
        f'<h3 class="episode-title">Episode - policy <code>{html.escape(policy)}</code> @ seed {int(seed)}</h3>'
        '</div>'
        '</div>'
        f'{build_pipeline_html(pct)}'
    )


def _scorecard_html(record: Dict[str, Any]) -> str:
    starting = record.get("starting_bank", 0.0)
    final = record.get("final_bank", starting)
    delta = float(final) - float(starting)
    bankrupt = bool(record.get("bankrupt"))
    grader = record.get("grader_scores", {}) or {}
    cells: List[str] = []

    def _cell(k: str, v: str) -> str:
        return f'<div class="cell"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div></div>'

    cells.append(_cell("Starting bank", f"INR {float(starting):,.0f}"))
    cells.append(_cell("Final bank", f"INR {float(final):,.0f}"))
    cells.append(_cell("Delta", f"{'+' if delta >= 0 else ''}{delta:,.0f}"))
    cells.append(_cell("Total reward", f"{record.get('total_reward', 0.0):+.2f}"))
    cells.append(_cell("Steps", str(record.get("n_steps", 0))))
    cells.append(_cell("Fallbacks", str(record.get("fallback_count", 0))))
    cells.append(_cell("Bankrupt", "yes" if bankrupt else "no"))
    cells.append(_cell("Action entropy", f"{record.get('entropy', 0.0):.3f}"))
    if grader and "__error__" not in grader:
        avg = sum(grader.values()) / max(1, len(grader))
        cells.append(_cell("Grader avg", f"{avg:.3f}"))
    grid = "".join(cells)
    return (
        '<div class="r2-scorecard">'
        '<h3>Episode scorecard</h3>'
        f'<div class="grid">{grid}</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Live charts
# ---------------------------------------------------------------------------

def _style_axis(ax: "plt.Axes") -> None:
    ax.set_facecolor(PAL_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PAL_BORDER)
    ax.tick_params(colors=PAL_INK_MUTED, labelsize=9)
    ax.grid(True, color=PAL_BORDER, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.title.set_color(PAL_INK)


def _bank_chart(banks: List[float], starting_bank: float):
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE, dpi=CHART_DPI)
    fig.patch.set_facecolor(PAL_BG)
    _style_axis(ax)
    if banks:
        x = list(range(len(banks)))
        ax.plot(x, banks, color=PAL_ACCENT, linewidth=2.0)
        ax.fill_between(x, banks, color=PAL_ACCENT, alpha=0.10)
        ax.scatter(x[-1:], banks[-1:], color=PAL_ACCENT, s=22, zorder=5)
    ax.axhline(y=float(starting_bank), color=PAL_INK_MUTED, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axhline(y=0.0, color=PAL_DANGER, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_title("Bank balance (live, INR)", fontsize=11, pad=8)
    ax.set_xlabel("step", color=PAL_INK_MUTED, fontsize=9)
    fig.tight_layout()
    return fig


def _action_dist_chart(action_counts: Counter, *, mode: str = "bar", title: str = "Action distribution (live)"):
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE, dpi=CHART_DPI)
    fig.patch.set_facecolor(PAL_BG)
    _style_axis(ax)
    if not action_counts:
        ax.text(0.5, 0.5, "no actions yet", color=PAL_INK_MUTED,
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
    else:
        items = sorted(action_counts.items(), key=lambda kv: -kv[1])
        labels = [k for k, _ in items]
        counts = [v for _, v in items]
        if mode == "pie":
            ax.grid(False)
            ax.pie(
                counts,
                labels=labels,
                autopct="%1.0f%%",
                textprops={"color": PAL_INK_SOFT, "fontsize": 9},
                wedgeprops={"linewidth": 0.8, "edgecolor": PAL_BORDER},
            )
            ax.axis("equal")
        else:
            bars = ax.bar(labels, counts, color=PAL_ACCENT, edgecolor=PAL_BORDER, linewidth=0.8)
            for b, c in zip(bars, counts):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, str(c),
                        ha="center", va="bottom", fontsize=9, color=PAL_INK_SOFT)
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
            ax.set_ylabel("count", color=PAL_INK_MUTED, fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)
    fig.tight_layout()
    return fig


def _action_success_chart(attempts: Counter, successes: Counter, *, title: str = "Action success rates"):
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE, dpi=CHART_DPI)
    fig.patch.set_facecolor(PAL_BG)
    _style_axis(ax)
    if not attempts:
        ax.text(0.5, 0.5, "no actions yet", color=PAL_INK_MUTED, ha="center", va="center", transform=ax.transAxes, fontsize=10)
    else:
        actions = sorted(attempts.keys())
        rates = [((successes.get(a, 0) / attempts[a]) if attempts[a] else 0.0) for a in actions]
        bars = ax.bar(actions, rates, color=PAL_ACCENT2, edgecolor=PAL_BORDER, linewidth=0.8)
        for b, r in zip(bars, rates):
            ax.text(
                b.get_x() + b.get_width() / 2,
                min(1.0, b.get_height() + 0.03),
                f"{r:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=PAL_INK_SOFT,
            )
        ax.set_ylim(0, 1.0)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_ylabel("success rate", color=PAL_INK_MUTED, fontsize=9)
    fig.tight_layout()
    return fig


def _flow_graph(obs: Dict[str, Any], action: Dict[str, Any], reward: float):
    nodes = {
        "State": (0, 0, "🔍"),
        "Reasoner": (1, 0, "🧠"),
        "Action": (2, 0, "⚡"),
        "Market": (3, 0, "📈")
    }
    edge_x, edge_y = [], []
    for start, end in [("State", "Reasoner"), ("Reasoner", "Action"), ("Action", "Market")]:
        x0, y0, _ = nodes[start]
        x1, y1, _ = nodes[end]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=3, color=PAL_BORDER), hoverinfo='none', mode='lines'))
    node_x = [v[0] for v in nodes.values()]
    node_y = [v[1] for v in nodes.values()]
    node_text = [f"{v[2]} {k}" for k, v in nodes.items()]
    colors = [PAL_ACCENT2, PAL_ACCENT, PAL_ACCENT, PAL_DANGER if reward < 0 else PAL_ACCENT2]
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="bottom center",
                            marker=dict(size=40, color=colors, line=dict(width=2, color=PAL_INK)), textfont=dict(size=10, color=PAL_INK_SOFT)))
    fig.update_layout(showlegend=False, margin=dict(b=40,l=40,r=40,t=20),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 3.5]),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
                      paper_bgcolor=PAL_BG, plot_bgcolor=PAL_BG, height=180)
    return fig


def _entropy(distribution: Dict[str, float]) -> float:
    import math
    h = 0.0
    for p in distribution.values():
        if p > 0:
            h -= p * math.log(p)
    return round(h, 4)


# ---------------------------------------------------------------------------
# Step log helpers
# ---------------------------------------------------------------------------

def _format_log_line(record: Dict[str, Any]) -> str:
    a = record["action"]
    a_type = a.get("action_type", "wait")
    params = {k: v for k, v in a.items() if k != "action_type"}
    params_str = json.dumps(params, separators=(",", ":")) if params else ""
    success = "ok" if record.get("success") else "fail"
    fb = f" [fallback:{record.get('fallback')}]" if record.get("fallback") else ""
    return (
        f"day {record['step']:02d}  {a_type:<10} {params_str:<46}  "
        f"reward={record['reward']:>+7.3f}  bank={record['bank_balance']:>10.2f}  "
        f"[{success}]{fb}"
    )


def _log_html(lines: List[str]) -> str:
    if not lines:
        return '<div class="r2-live-log"><div class="line">No backend events yet.</div></div>'
    chunks: List[str] = ['<div class="r2-live-log">']
    for ln in lines:
        cls = "line"
        prefix = "• "
        if ln.startswith("day "):
            cls += " day"
            prefix = "🧠 "
        elif ln.startswith("summary"):
            cls += " summary"
            prefix = "✅ "
        elif ln.startswith("grader"):
            prefix = "📊 "
        elif ln.startswith("reset"):
            prefix = "🔄 "
        safe = html.escape(prefix + ln)
        chunks.append(f'<div class="{cls}">{safe}</div>')
    chunks.append("</div>")
    return "".join(chunks)


# ---------------------------------------------------------------------------
# The streaming generator (the entry point used by demo/app.py)
# ---------------------------------------------------------------------------

def stream_live_episode(
    policy_name: str,
    seed: int,
    business_id: str,
    max_steps: int = DEMO_MAX_STEPS,
    distribution_view: str = "bar",
) -> Iterable[Tuple[str, str, str, "plt.Figure", "plt.Figure", "plt.Figure", "go.Figure", str]]:
    """Yield tuples of (head_html, step_card_html, log_html, bank_fig, action_fig, success_fig, flow_fig, scorecard_html)."""
    seed = int(seed)
    business_id = (business_id or "").strip()
    backend = BackendClient()
    head = _theater_head_html(policy_name, seed, business_id, backend.base_url, 0, max_steps, "starting...")
    _pat = patience_box(
        "Please be patient",
        "Configuring the live OpenEnv session. Loading the Qwen policy on first use can take 30s–2m; later runs are faster.",
    )
    yield (
        head,
        _pat
        + '<div class="r2-banner is-warn"><h3>Initialising</h3><p>Calling <code>/config</code> (if needed) and <code>/reset</code> on the OpenEnv backend…</p></div>',
        _log_html([]),
        _bank_chart([], starting_bank=0.0),
        _action_dist_chart(Counter()),
        _action_success_chart(Counter(), Counter()),
        go.Figure(),
        "",
    )

    if business_id:
        try:
            backend.config(business_id=business_id, seed=seed)
        except BackendError as exc:
            head = _theater_head_html(policy_name, seed, business_id, backend.base_url, 0, max_steps, f"config failed: {exc.kind}")
            yield (
                head,
                banner("Backend /config failed", html.escape(str(exc)), kind="danger"),
                _log_html([]),
                _bank_chart([], starting_bank=0.0),
                _action_dist_chart(Counter()),
                _action_success_chart(Counter(), Counter()),
                go.Figure(),
                "",
            )
            return

    try:
        reset_payload = backend.reset(seed=seed)
    except BackendError as exc:
        head = _theater_head_html(policy_name, seed, business_id, backend.base_url, 0, max_steps, f"reset failed: {exc.kind}")
        yield (
            head,
            banner("Backend /reset failed", html.escape(str(exc)), kind="danger"),
            _log_html([]),
            _bank_chart([], starting_bank=0.0),
            _action_dist_chart(Counter()),
            _action_success_chart(Counter(), Counter()),
            go.Figure(),
            "",
        )
        return

    obs = reset_payload.get("observation", {}) or {}
    starting_bank = float(obs.get("bank_balance", 0.0))
    banks: List[float] = [starting_bank]
    action_counts: Counter = Counter()
    success_counts: Counter = Counter()
    log_lines: List[str] = [
        f"reset       seed={seed:<8d} starting_bank={starting_bank:>12,.2f}",
    ]
    handle = get_policy(policy_name)
    if not handle.available:
        log_lines.append(f"# policy '{policy_name}' unavailable: {handle.reason}; emitting wait + recording fallback")

    rewards: List[float] = []
    fallback_count = 0
    bankrupt = False
    n_steps = 0

    head = _theater_head_html(policy_name, seed, business_id, backend.base_url, 0, max_steps, "running")
    yield (
        head,
        '<div class="r2-banner is-good"><h3>Episode started</h3><p>Bank balance and charts update each simulated day. Observe, reason, act, and react stream below.</p></div>',
        _log_html(log_lines),
        _bank_chart(banks, starting_bank),
        _action_dist_chart(action_counts, mode=distribution_view),
        _action_success_chart(action_counts, success_counts),
        go.Figure(),
        "",
    )

    obs_before = obs
    last_chart_mono = 0.0
    for step in range(1, max_steps + 1):
        action = infer_action(handle, obs_before)
        if action.get("_fallback_reason"):
            fallback_count += 1
        clean_action = {k: v for k, v in action.items() if not k.startswith("_")}
        try:
            r = backend.step(clean_action)
        except BackendError as exc:
            head = _theater_head_html(policy_name, seed, business_id, backend.base_url, step - 1, max_steps, f"step error: {exc.kind}")
            yield (
                head,
                banner("Backend /step failed", html.escape(str(exc)), kind="danger"),
                _log_html(log_lines),
                _bank_chart(banks, starting_bank),
                _action_dist_chart(action_counts, mode=distribution_view),
                _action_success_chart(action_counts, success_counts),
                go.Figure(),
                "",
            )
            return

        obs_after = r.get("observation", {}) or {}
        reward = float(r.get("reward", 0.0))
        info = r.get("info", {}) or {}
        success = _action_success(clean_action, info, obs_before, obs_after)
        record = {
            "step": step,
            "action": clean_action,
            "reward": round(reward, 6),
            "bank_balance": round(float(obs_after.get("bank_balance", 0.0)), 2),
            "intent": info.get("intent"),
            "action_quality": info.get("action_quality"),
            "confidence": info.get("confidence"),
            "info_error": info.get("error"),
            "success": bool(success),
            "fallback": action.get("_fallback_reason"),
        }
        rewards.append(reward)
        banks.append(record["bank_balance"])
        a_type = clean_action.get("action_type", "wait")
        action_counts[a_type] += 1
        if success:
            success_counts[a_type] += 1
        log_lines.append(_format_log_line(record))
        if record["bank_balance"] <= 0:
            bankrupt = True
        n_steps = step

        tmono = time.monotonic()
        # Charts dominate per-step time; head/step/log stay in sync every step.
        need_charts = (
            bool(r.get("done"))
            or (step == max_steps)
            or ((step - 1) % LIVE_CHART_EVERY_N == 0)
            or (tmono - last_chart_mono) >= LIVE_MIN_REDRAW_SEC
        )
        if need_charts:
            last_chart_mono = tmono
            bnk_f = _bank_chart(banks, starting_bank)
            act_f = _action_dist_chart(action_counts, mode=distribution_view)
            pol_f = _action_success_chart(action_counts, success_counts)
        else:
            bnk_f = gr.update()  # type: ignore[assignment]
            act_f = gr.update()  # type: ignore[assignment]
            pol_f = gr.update()  # type: ignore[assignment]

        head = _theater_head_html(policy_name, seed, business_id, backend.base_url, step, max_steps, "running")
        yield (
            head,
            _step_card_html(step, max_steps, obs_after, clean_action, record, policy_name),
            _log_html(log_lines),
            bnk_f,
            act_f,
            pol_f,
            _flow_graph(obs_after, clean_action, reward),
            "",
        )

        obs_before = obs_after
        if r.get("done"):
            break

        # Cosmetic pacing so the four phase cards have time to animate in
        # (CSS keyframes total ~330ms) and the judge can read each Day's
        # decision before the next one replaces it. No effect on backend
        # behaviour - just paces the yield loop.
        if STEP_PACE_SECONDS > 0:
            time.sleep(STEP_PACE_SECONDS)

    grader_scores: Dict[str, float] = {}
    try:
        grader_payload = backend.grader()
        for entry in grader_payload.get("scores", []):
            grader_scores[entry.get("task_id", "")] = float(entry.get("score", 0.0))
    except BackendError as exc:
        grader_scores["__error__"] = str(exc)

    total = sum(action_counts.values()) or 1
    distribution = {a: c / total for a, c in action_counts.items()}
    final_record = {
        "starting_bank": starting_bank,
        "final_bank": banks[-1] if banks else starting_bank,
        "total_reward": sum(rewards),
        "n_steps": n_steps,
        "fallback_count": fallback_count,
        "bankrupt": bankrupt,
        "entropy": _entropy(distribution),
        "grader_scores": grader_scores,
    }
    log_lines.append("---")
    log_lines.append(
        f"summary    final_bank={final_record['final_bank']:,.2f}  "
        f"total_reward={final_record['total_reward']:+.3f}  "
        f"steps={n_steps}  fallback={fallback_count}  bankrupt={'yes' if bankrupt else 'no'}"
    )
    if grader_scores and "__error__" not in grader_scores:
        log_lines.append(
            "grader     " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(grader_scores.items()))
        )
    elif grader_scores.get("__error__"):
        log_lines.append(f"grader_err {grader_scores['__error__']}")

    head = _theater_head_html(policy_name, seed, business_id, backend.base_url, n_steps, max_steps, "complete")
    yield (
        head,
        '<div class="r2-banner is-good"><h3>Episode complete</h3><p>Backend calls finished. Scorecard below.</p></div>',
        _log_html(log_lines),
        _bank_chart(banks, starting_bank),
        _action_dist_chart(action_counts, mode=distribution_view),
        _action_success_chart(action_counts, success_counts),
        go.Figure(),
        _scorecard_html(final_record),
    )


def _comparison_html(comparison: Dict[str, Any]) -> str:
    a = comparison.get("summary", {}).get("a", {}) or {}
    b = comparison.get("summary", {}).get("b", {}) or {}
    return (
        '<div class="r2-scorecard">'
        '<h3>Policy transition (same seed)</h3>'
        '<p style="margin:0 0 10px 0;color:var(--ink-soft);font-size:13px;">'
        'Both policies ran on the exact same seed + business so the gap reflects policy behaviour, not environment luck.'
        '</p>'
        '<table class="r2-table"><thead><tr><th>Metric</th><th>Baseline</th><th>Trained</th></tr></thead><tbody>'
        f'<tr><td>Final bank</td><td>{a.get("final_bank")}</td><td>{b.get("final_bank")}</td></tr>'
        f'<tr><td>Total reward</td><td>{a.get("total_reward")}</td><td>{b.get("total_reward")}</td></tr>'
        f'<tr><td>Bankrupt</td><td>{a.get("bankrupt")}</td><td>{b.get("bankrupt")}</td></tr>'
        f'<tr><td>Action entropy</td><td>{a.get("entropy")}</td><td>{b.get("entropy")}</td></tr>'
        '</tbody></table>'
        '</div>'
    )


def _comparison_bank_chart(comparison: Dict[str, Any]):
    ta = comparison.get("trace_a", {}) or {}
    tb = comparison.get("trace_b", {}) or {}
    a_series = [float(ta.get("starting_bank", 0.0))] + [float(s.get("bank_balance", 0.0)) for s in ta.get("steps", [])]
    b_series = [float(tb.get("starting_bank", 0.0))] + [float(s.get("bank_balance", 0.0)) for s in tb.get("steps", [])]
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE, dpi=CHART_DPI)
    fig.patch.set_facecolor(PAL_BG)
    _style_axis(ax)
    if a_series:
        ax.plot(range(len(a_series)), a_series, color=PAL_DANGER, linewidth=1.8, label="baseline_zero_shot")
    if b_series:
        ax.plot(range(len(b_series)), b_series, color=PAL_ACCENT2, linewidth=2.0, label="trained")
    ax.axhline(y=0.0, color=PAL_DANGER, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_title("Bank trajectory - same seed comparison", fontsize=11, pad=8)
    _legend_if_any(ax)
    fig.tight_layout()
    return fig


def _comparison_success_chart(comparison: Dict[str, Any]):
    ta = comparison.get("trace_a", {}) or {}
    tb = comparison.get("trace_b", {}) or {}
    a_by = ((ta.get("action_summary") or {}).get("by_action") or {})
    b_by = ((tb.get("action_summary") or {}).get("by_action") or {})
    actions = sorted(set(a_by.keys()) | set(b_by.keys()))

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE, dpi=CHART_DPI)
    fig.patch.set_facecolor(PAL_BG)
    _style_axis(ax)
    if not actions:
        ax.text(0.5, 0.5, "comparison unavailable", color=PAL_INK_MUTED, ha="center", va="center", transform=ax.transAxes, fontsize=10)
    else:
        import numpy as np

        x = np.arange(len(actions))
        width = 0.38
        baseline_rates = [float((a_by.get(a) or {}).get("success_rate") or 0.0) for a in actions]
        trained_rates = [float((b_by.get(a) or {}).get("success_rate") or 0.0) for a in actions]
        ax.bar(x - width / 2, baseline_rates, width, label="baseline", color="#c89b8a", edgecolor=PAL_BORDER, linewidth=0.8)
        ax.bar(x + width / 2, trained_rates, width, label="trained", color=PAL_ACCENT2, edgecolor=PAL_BORDER, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(actions, rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        _legend_if_any(ax)
    ax.set_title("Action success rate: baseline vs trained", fontsize=11, pad=8)
    ax.set_ylabel("success rate", color=PAL_INK_MUTED, fontsize=9)
    fig.tight_layout()
    return fig


def _comparison_bank_chart_partial(a_series: List[float], b_series: List[float]):
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE, dpi=CHART_DPI)
    fig.patch.set_facecolor(PAL_BG)
    _style_axis(ax)
    if a_series:
        ax.plot(range(len(a_series)), a_series, color=PAL_DANGER, linewidth=1.8, label="baseline_zero_shot")
    if b_series:
        ax.plot(range(len(b_series)), b_series, color=PAL_ACCENT2, linewidth=2.0, label="trained")
    ax.axhline(y=0.0, color=PAL_DANGER, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_title("Bank trajectory - same seed comparison (live)", fontsize=11, pad=8)
    _legend_if_any(ax)
    fig.tight_layout()
    return fig


def _comparison_success_chart_partial(
    baseline_attempts: Counter,
    baseline_successes: Counter,
    trained_attempts: Counter,
    trained_successes: Counter,
):
    actions = sorted(set(baseline_attempts.keys()) | set(trained_attempts.keys()))
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE, dpi=CHART_DPI)
    fig.patch.set_facecolor(PAL_BG)
    _style_axis(ax)
    if not actions:
        ax.text(0.5, 0.5, "comparison unavailable", color=PAL_INK_MUTED, ha="center", va="center", transform=ax.transAxes, fontsize=10)
    else:
        import numpy as np

        x = np.arange(len(actions))
        width = 0.38
        baseline_rates = [
            (baseline_successes.get(a, 0) / baseline_attempts[a]) if baseline_attempts[a] else 0.0
            for a in actions
        ]
        trained_rates = [
            (trained_successes.get(a, 0) / trained_attempts[a]) if trained_attempts[a] else 0.0
            for a in actions
        ]
        ax.bar(x - width / 2, baseline_rates, width, label="baseline", color="#c89b8a", edgecolor=PAL_BORDER, linewidth=0.8)
        ax.bar(x + width / 2, trained_rates, width, label="trained", color=PAL_ACCENT2, edgecolor=PAL_BORDER, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(actions, rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        _legend_if_any(ax)
    ax.set_title("Action success rate: baseline vs trained (live)", fontsize=11, pad=8)
    ax.set_ylabel("success rate", color=PAL_INK_MUTED, fontsize=9)
    fig.tight_layout()
    return fig


def _policy_transition_outputs(
    comparison: Dict[str, Any],
    seed: int,
    business_id: str,
    max_steps: int,
    distribution_view: str,
) -> Tuple[str, str, str, "plt.Figure", "plt.Figure", "plt.Figure", "go.Figure", str]:
    a = comparison.get("summary", {}).get("a", {}) or {}
    b = comparison.get("summary", {}).get("b", {}) or {}
    lines = [
        f"comparison seed={seed} business={business_id or 'default'}",
        f"baseline final_bank={a.get('final_bank')} total_reward={a.get('total_reward')} bankrupt={a.get('bankrupt')}",
        f"trained  final_bank={b.get('final_bank')} total_reward={b.get('total_reward')} bankrupt={b.get('bankrupt')}",
    ]
    trained_dist = Counter()
    dist = ((comparison.get("trace_b", {}).get("action_summary") or {}).get("distribution") or {})
    for action, pct in dist.items():
        trained_dist[action] = int(round(float(pct) * 100))
    head = _theater_head_html("baseline -> trained", seed, business_id, BackendClient().base_url, int(max_steps), int(max_steps), "comparison complete")
    step_card = banner(
        "Policy transition complete",
        "Ran baseline_zero_shot then trained on the same seed. Compare trajectories and action-level success.",
        kind="good",
    )
    return (
        head,
        step_card,
        _log_html(lines),
        _comparison_bank_chart(comparison),
        _comparison_success_chart(comparison),
        _action_dist_chart(trained_dist, mode=distribution_view, title="Trained action distribution (%)"),
        go.Figure(),
        _comparison_html(comparison),
    )


def run_policy_transition(
    seed: int,
    business_id: str,
    max_steps: int = DEMO_MAX_STEPS,
    distribution_view: str = "bar",
) -> Tuple[str, str, str, "plt.Figure", "plt.Figure", "plt.Figure", "go.Figure", str]:
    """Run baseline->trained on same seed and summarize policy transition."""
    seed = int(seed)
    business_id = (business_id or "").strip()
    comparison = run_ab_comparison(
        seed=seed,
        policy_a=POLICY_BASELINE_ZERO_SHOT,
        policy_b=POLICY_TRAINED,
        business_id=business_id or None,
        max_steps=int(max_steps),
    )
    return _policy_transition_outputs(
        comparison,
        seed=seed,
        business_id=business_id,
        max_steps=max_steps,
        distribution_view=distribution_view,
    )


def stream_policy_transition(
    seed: int,
    business_id: str,
    max_steps: int = DEMO_MAX_STEPS,
    distribution_view: str = "bar",
) -> Iterable[Tuple[str, str, str, "plt.Figure", "plt.Figure", "plt.Figure", "go.Figure", str]]:
    """Stream baseline->trained same-seed comparison with live per-policy progress."""
    seed = int(seed)
    business_id = (business_id or "").strip()
    step_budget = int(max_steps)
    backend_url = BackendClient().base_url
    total_steps_planned = max(1, step_budget) * 2

    baseline_series: List[float] = []
    trained_series: List[float] = []
    baseline_attempts: Counter = Counter()
    baseline_successes: Counter = Counter()
    trained_attempts: Counter = Counter()
    trained_successes: Counter = Counter()
    baseline_days = 0
    trained_days = 0
    log_lines: List[str] = [
        f"comparison seed={seed} business={business_id or 'default'}",
        "starting baseline_zero_shot...",
    ]

    events: "queue.Queue[Tuple[str, Any]]" = queue.Queue()

    def _on_step(policy: str, record: Dict[str, Any]) -> None:
        events.put(("step", {"policy": policy, "record": record}))

    def _worker() -> None:
        try:
            comparison = run_ab_comparison(
                seed=seed,
                policy_a=POLICY_BASELINE_ZERO_SHOT,
                policy_b=POLICY_TRAINED,
                business_id=business_id or None,
                max_steps=step_budget,
                on_step=_on_step,
            )
            events.put(("done", comparison))
        except Exception as exc:  # pragma: no cover
            events.put(("error", str(exc)))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    _cp = patience_box(
        "Please be patient",
        "Baseline-versus-trained runs two full episodes and two grader calls. The first run may be slow while the model loads; charts refresh as days complete.",
    )
    yield (
        _theater_head_html("baseline -> trained", seed, business_id, backend_url, 0, total_steps_planned, "preparing same-seed comparison"),
        _cp
        + banner(
            "Same-seed comparison is running",
            "Order: baseline zero-shot, then the trained LoRA policy. The step log and charts fill in as the backend returns each day.",
            kind="warn",
        ),
        _log_html(log_lines),
        _comparison_bank_chart_partial(baseline_series, trained_series),
        _comparison_success_chart_partial(baseline_attempts, baseline_successes, trained_attempts, trained_successes),
        _action_dist_chart(Counter(), mode=distribution_view, title="Current policy action mix"),
        go.Figure(),
        "",
    )

    last_yield_at = 0.0
    pending_label = "baseline"
    pending_day = 0

    def _absorb_step(payload: Dict[str, Any]) -> Tuple[str, int]:
        """Apply one step event to the running aggregates. Returns (label, day)."""
        nonlocal baseline_days, trained_days
        policy = payload.get("policy")
        record = payload.get("record", {}) or {}
        action = (record.get("action") or {}).get("action_type", "wait")
        success = bool(record.get("success"))
        bank = float(record.get("bank_balance", 0.0))
        reward = float(record.get("reward", 0.0))
        step = int(record.get("step", 0))

        if policy == POLICY_BASELINE_ZERO_SHOT:
            baseline_days = max(baseline_days, step)
            baseline_series.append(bank)
            baseline_attempts[action] += 1
            if success:
                baseline_successes[action] += 1
            label = "baseline"
            day = baseline_days
        else:
            if trained_days == 0 and not any("starting trained" in ln.lower() for ln in log_lines):
                log_lines.append("starting trained...")
            trained_days = max(trained_days, step)
            trained_series.append(bank)
            trained_attempts[action] += 1
            if success:
                trained_successes[action] += 1
            label = "trained"
            day = trained_days

        log_lines.append(
            f"{label:8s} day={step:02d} action={action:<10s} reward={reward:+.3f} bank={bank:,.2f} success={'yes' if success else 'no'}"
        )
        return label, day

    while True:
        kind, payload = events.get()
        if kind == "step":
            pending_label, pending_day = _absorb_step(payload)
            # Coalesce burst events: drain anything else already queued
            # before re-rendering the heavy matplotlib figures.
            try:
                while True:
                    extra_kind, extra_payload = events.get_nowait()
                    if extra_kind != "step":
                        events.put((extra_kind, extra_payload))
                        break
                    pending_label, pending_day = _absorb_step(extra_payload)
            except queue.Empty:
                pass

            now = time.monotonic()
            if (now - last_yield_at) < COMPARE_MIN_REDRAW_SEC:
                # Skip this redraw - we've folded the data in already and
                # the next event will pick it up. This caps redraw cost.
                continue
            last_yield_at = now

            completed = baseline_days + trained_days
            status = f"running {pending_label} day {pending_day}/{step_budget}"
            current_counts = baseline_attempts if pending_label == "baseline" else trained_attempts
            yield (
                _theater_head_html("baseline -> trained", seed, business_id, backend_url, completed, total_steps_planned, status),
                banner(
                    "Comparison in progress",
                    f"{pending_label} day {pending_day}/{step_budget} complete.",
                    kind="warn",
                ),
                _log_html(log_lines[-24:]),
                _comparison_bank_chart_partial(baseline_series, trained_series),
                _comparison_success_chart_partial(baseline_attempts, baseline_successes, trained_attempts, trained_successes),
                _action_dist_chart(current_counts, mode=distribution_view, title=f"{pending_label} action distribution (live)"),
                go.Figure(),
                "",
            )
            continue

        if kind == "error":
            yield (
                _theater_head_html("baseline -> trained", seed, business_id, backend_url, baseline_days + trained_days, total_steps_planned, "comparison failed"),
                banner("Comparison failed", html.escape(str(payload)), kind="danger"),
                _log_html(log_lines),
                _comparison_bank_chart_partial(baseline_series, trained_series),
                _comparison_success_chart_partial(baseline_attempts, baseline_successes, trained_attempts, trained_successes),
                _action_dist_chart(Counter(), mode=distribution_view, title="Current policy action mix"),
                go.Figure(),
                "",
            )
            return

        if kind == "done":
            comparison = payload
            yield _policy_transition_outputs(
                comparison,
                seed=seed,
                business_id=business_id,
                max_steps=step_budget,
                distribution_view=distribution_view,
            )
            return
