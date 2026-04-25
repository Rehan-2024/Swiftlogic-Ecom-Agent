"""Round-2 Gradio dashboard.

Composition only - all logic lives in the modules:
  backend_client, policy, episode_runner, artifact_loader, components, story_tab.

Tabs:
  1. Story         - 4-chapter narrative with real photos.
  2. Live Run      - real /reset + /step + /grader against the OpenEnv backend.
  3. Training Proof- artifacts + action-success comparison (Phase 3 + 4).
  4. Generalisation- multi-config performance evidence (Phase 6.3).
"""

from __future__ import annotations

import html
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.artifact_loader import (  # noqa: E402
    artifact_image_path,
    freshness_summary,
    generalization_covers_unseen_configs,
    judge_readiness,
    load_action_success,
    load_after_metrics,
    load_before_metrics,
    load_composite_score,
    load_failure_vs_recovery,
    load_generalization,
    load_pipeline_manifest,
    load_policy_signature,
    policy_signatures_distinct,
)
from demo.backend_client import BackendClient, BackendError  # noqa: E402
from demo.components import (  # noqa: E402
    banner,
    evidence_unavailable,
    fmt_currency,
    fmt_delta,
    fmt_pct,
    metric_card,
    metric_row,
    pill,
    table,
)
from demo.episode_runner import LIVE_RUNS_DIR, run_ab_comparison, run_episode  # noqa: E402
from demo.policy import (  # noqa: E402
    ALL_POLICIES,
    POLICY_BASELINE_WAIT,
    POLICY_BASELINE_ZERO_SHOT,
    POLICY_TRAINED,
)
from demo.story_tab import render_story_html  # noqa: E402

logger = logging.getLogger("commerceops.demo.app")

THEME_CSS = (Path(__file__).parent / "theme.css").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Header / startup banner
# ---------------------------------------------------------------------------

def _header_html() -> str:
    readiness = judge_readiness()
    pill_html = pill("JUDGE-READY", kind="ready") if readiness.ready else pill("PRE-TRAINING PREVIEW", kind="pre")
    return (
        '<div class="r2-card" style="margin-bottom:18px;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">'
        '<div>'
        '<h1 style="margin:0;font-size:24px;">Siyaani Commerce - AI CEO</h1>'
        f'<p style="margin:4px 0 0 0;color:var(--ink-soft);font-size:13.5px;">'
        f'Autonomous storefront operator trained with GRPO on the OpenEnv contract. '
        f'Pipeline provenance: <strong>{html.escape(readiness.provenance.value)}</strong>, '
        f'adapter: <strong>{html.escape(readiness.adapter_status)}</strong>.'
        '</p>'
        '</div>'
        f'<div>{pill_html}</div>'
        '</div>'
    )


def _startup_diagnostics_html() -> str:
    bc = BackendClient()
    health = bc.quick_self_check()
    readiness = judge_readiness()
    gen_check = generalization_covers_unseen_configs()
    sig_check = policy_signatures_distinct()
    rows: List[str] = []

    if health["ok"]:
        rows.append(banner(
            "Backend reachable",
            f"<code>{html.escape(health['base_url'])}</code> - <strong>{health.get('tasks_count','?')}</strong> tasks exposed.",
            kind="good",
        ))
    else:
        details = "; ".join(html.escape(e) for e in health["errors"])
        rows.append(banner(
            "Backend unreachable",
            f"<code>{html.escape(health['base_url'])}</code> - {details}. "
            "The Live Run tab will be disabled until the backend responds.",
            kind="danger",
        ))

    if not readiness.ready:
        why = "; ".join(html.escape(r) for r in readiness.reasons)
        rows.append(banner(
            "Pre-training preview - not judge-ready",
            f"Provenance gate failed: {why}. "
            "Run the GRPO pipeline (grpo_single_cell_colab_v5.py) to flip provenance to grpo_trained.",
            kind="warn",
        ))
    else:
        rows.append(banner(
            "Authenticity gate green",
            "All artifacts trained, adapter present, signatures distinct.",
            kind="good",
        ))

    if not gen_check["ok"]:
        rows.append(banner(
            "Generalisation evidence incomplete",
            f"Missing unseen configs: <code>{', '.join(gen_check['missing']) or 'none'}</code>. "
            "Re-run the generalisation pass over medplus_pharmacy and stackbase_saas.",
            kind="warn",
        ))

    if not sig_check["ok"] and sig_check["hashes"]:
        rows.append(banner(
            "Policy signature collision",
            "Trained-policy hash equals heuristic-policy hash - the dashboard will show identical distributions. "
            "Re-train and re-emit policy_signature.json.",
            kind="warn",
        ))

    return "".join(rows)


# ---------------------------------------------------------------------------
# Live tab
# ---------------------------------------------------------------------------

def _format_step_log(records: List[Dict[str, Any]]) -> str:
    lines = []
    for r in records:
        a = r["action"]
        a_type = a.get("action_type", "wait")
        params = {k: v for k, v in a.items() if k != "action_type"}
        params_str = json.dumps(params, separators=(",", ":")) if params else ""
        intent = r.get("intent") or "-"
        success_flag = "ok" if r.get("success") else "fail"
        fb = f" [fallback:{r.get('fallback')}]" if r.get("fallback") else ""
        err = f" err={r.get('info_error')}" if r.get("info_error") else ""
        lines.append(
            f"day {r['step']:02d}  {a_type:<10} {params_str:<48}  "
            f"reward={r['reward']:>+7.3f}  bank={r['bank_balance']:>10.2f}  "
            f"intent={intent}  [{success_flag}]{fb}{err}"
        )
    return "\n".join(lines)


def _final_cards_html(trace: Dict[str, Any]) -> str:
    bank_tone = "good"
    if trace.get("bankrupt"):
        bank_tone = "bad"
    elif trace.get("final_bank", 0) < trace.get("starting_bank", 0):
        bank_tone = "warn"
    return metric_row([
        metric_card("Final bank", fmt_currency(trace.get("final_bank", 0.0)), tone=bank_tone),
        metric_card("Total reward", f"{trace.get('total_reward', 0.0):+.2f}"),
        metric_card("Steps", str(trace.get("n_steps", 0))),
        metric_card("Fallbacks", str(trace.get("fallback_count", 0)), tone="warn" if trace.get("fallback_count", 0) > 0 else "neutral"),
        metric_card("Policy", trace.get("policy_type", "-")),
    ])


def _grader_card_html(trace: Dict[str, Any]) -> str:
    scores = trace.get("grader_scores", {}) or {}
    if "__error__" in scores:
        return banner("Grader unavailable", html.escape(scores["__error__"]), kind="warn")
    rows = [[k, f"{float(v):.4f}"] for k, v in sorted(scores.items()) if k != "__error__"]
    return table(["Task", "Score"], rows, empty_msg="grader returned no scores")


def _ab_table_html(comparison: Dict[str, Any]) -> str:
    a = comparison["summary"]["a"]
    b = comparison["summary"]["b"]
    rows = [
        ["Final bank",     fmt_currency(a.get("final_bank", 0.0)), fmt_currency(b.get("final_bank", 0.0))],
        ["Total reward",   f"{a.get('total_reward', 0.0):+.2f}",   f"{b.get('total_reward', 0.0):+.2f}"],
        ["Bankrupt",       "yes" if a.get("bankrupt") else "no",   "yes" if b.get("bankrupt") else "no"],
        ["Steps",          str(a.get("n_steps", 0)),               str(b.get("n_steps", 0))],
        ["Fallbacks",      str(a.get("fallback_count", 0)),        str(b.get("fallback_count", 0))],
        ["Action entropy", f"{a.get('entropy', 0.0):.3f}",         f"{b.get('entropy', 0.0):.3f}"],
    ]
    headers = ["Metric", a.get("policy", "A"), b.get("policy", "B")]
    return table(headers, rows)


def _do_run_single(policy_name: str, seed: int, business_id: str):
    """Streaming generator for the Live tab single-run button."""
    bc = BackendClient()
    log_lines: List[str] = []
    final_metrics_html = ""
    grader_html = ""
    summary_banner = banner("Running...", f"Policy <strong>{html.escape(policy_name)}</strong> on seed {int(seed)}.", kind="neutral")

    def on_step(rec: Dict[str, Any]) -> None:
        log_lines.append(_format_step_log([rec]))

    yield summary_banner, "\n".join(log_lines), final_metrics_html, grader_html, ""

    bid = business_id.strip() or None
    trace = run_episode(policy_name, int(seed), backend=bc, business_id=bid, on_step=on_step)

    if "error" in trace:
        err_banner = banner("Run failed", html.escape(trace["error"]), kind="danger")
        yield err_banner, "\n".join(log_lines), "", "", json.dumps(trace, indent=2, default=str)
        return

    final_metrics_html = _final_cards_html(trace)
    grader_html = _grader_card_html(trace)
    summary_banner = banner(
        "Run complete",
        f"run_id <code>{html.escape(trace['run_id'])}</code> persisted to "
        f"<code>{html.escape(str(trace.get('_path', '')))}</code>.",
        kind="good",
    )
    yield summary_banner, "\n".join(log_lines), final_metrics_html, grader_html, json.dumps({
        "run_id": trace["run_id"],
        "endpoint_call_counts": trace["endpoint_call_counts"],
        "starting_bank": trace.get("starting_bank"),
        "final_bank": trace.get("final_bank"),
        "total_reward": trace.get("total_reward"),
        "fallback_count": trace.get("fallback_count"),
        "bankrupt": trace.get("bankrupt"),
        "grader_scores": trace.get("grader_scores"),
        "git_sha": trace.get("git_sha"),
        "adapter_sha": trace.get("adapter_sha"),
    }, indent=2, default=str)


def _do_run_ab(seed: int, business_id: str):
    bid = business_id.strip() or None
    summary = banner(
        "Running A/B comparison",
        f"Same seed {int(seed)} for <strong>{POLICY_BASELINE_ZERO_SHOT}</strong> and <strong>{POLICY_TRAINED}</strong>.",
        kind="neutral",
    )
    yield summary, "", "", ""
    cmp_obj = run_ab_comparison(int(seed), POLICY_BASELINE_ZERO_SHOT, POLICY_TRAINED, business_id=bid)
    table_html = _ab_table_html(cmp_obj)
    summary = banner(
        "A/B comparison complete",
        f"comparison_id <code>{html.escape(cmp_obj['comparison_id'])}</code> at "
        f"<code>{html.escape(str(cmp_obj.get('_path', '')))}</code>. "
        f"Same seed: <strong>{int(seed)}</strong>.",
        kind="good",
    )
    yield summary, table_html, json.dumps(cmp_obj["summary"], indent=2, default=str), json.dumps({
        "comparison_id": cmp_obj["comparison_id"],
        "run_a_id": cmp_obj["run_a"]["run_id"],
        "run_b_id": cmp_obj["run_b"]["run_id"],
        "seed": cmp_obj["seed"],
        "business_id": cmp_obj["business_id"],
    }, indent=2, default=str)


# ---------------------------------------------------------------------------
# Training Proof tab
# ---------------------------------------------------------------------------

def _proof_metric_table_html() -> str:
    before = load_before_metrics()
    after = load_after_metrics()
    composite = load_composite_score()
    if not before and not after and not composite:
        return evidence_unavailable("before_metrics.json / after_metrics.json / composite_score.json")
    rows: List[List[Any]] = []
    if composite:
        b_score = composite.get("before", {}).get("score")
        a_score = composite.get("after", {}).get("score")
        if isinstance(b_score, (int, float)) and isinstance(a_score, (int, float)):
            rows.append(["Composite score", f"{b_score:.4f}", f"{a_score:.4f}", fmt_delta(b_score, a_score)])
    if before and after:
        for key, label in [
            ("avg_reward", "Avg reward"),
            ("avg_profit", "Avg profit"),
            ("stockout_rate", "Stockout rate"),
        ]:
            bv = before.get(key)
            av = after.get(key)
            if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                rows.append([label, f"{bv:.4f}", f"{av:.4f}", fmt_delta(bv, av)])
    fvr = load_failure_vs_recovery()
    if isinstance(fvr, dict):
        b_rate = fvr.get("baseline_bankruptcy_rate")
        a_rate = fvr.get("trained_bankruptcy_rate")
        if isinstance(b_rate, (int, float)) and isinstance(a_rate, (int, float)):
            rows.append(["Bankruptcy rate", f"{b_rate*100:.1f}%", f"{a_rate*100:.1f}%", fmt_delta(b_rate, a_rate)])
    if not rows:
        return evidence_unavailable("metrics keys not found in before/after JSONs")
    return table(["Metric", "Baseline", "Trained", "Delta"], rows)


def _action_success_table_html() -> str:
    base = load_action_success("baseline_zero_shot")
    trained = load_action_success("trained")
    if not base and not trained:
        return evidence_unavailable("action_success_baseline_zero_shot.json / action_success_trained.json")
    actions = sorted(set((base.get("by_action") or {}).keys()) | set((trained.get("by_action") or {}).keys()))
    rows: List[List[Any]] = []
    for a in actions:
        b = (base.get("by_action") or {}).get(a, {})
        t = (trained.get("by_action") or {}).get(a, {})
        rows.append([
            a,
            f"{b.get('attempts', 0)} / {b.get('successes', 0)}",
            f"{(b.get('success_rate') or 0.0):.2f}",
            f"{t.get('attempts', 0)} / {t.get('successes', 0)}",
            f"{(t.get('success_rate') or 0.0):.2f}",
        ])
    return table(["Action", "Baseline atts/succ", "Baseline rate", "Trained atts/succ", "Trained rate"], rows)


# ---------------------------------------------------------------------------
# Generalisation tab
# ---------------------------------------------------------------------------

def _generalization_table_html() -> str:
    gen = load_generalization()
    episodes = gen.get("episodes", []) if isinstance(gen, dict) else []
    if not episodes:
        return evidence_unavailable("generalization.json (no episodes)")
    rows: List[List[Any]] = []
    for ep in episodes:
        cfg = (ep.get("config") or "").replace("\\", "/").rsplit("/", 1)[-1].replace(".json", "")
        scores = ep.get("grader_scores", {}) or {}
        rows.append([
            cfg,
            ep.get("seed"),
            ep.get("steps"),
            f"{ep.get('final_bank', 0):.2f}",
            f"{(ep.get('format_compliance') or 0.0):.2f}",
            f"{scores.get('profit_task', 0):.2f}",
            f"{scores.get('inventory_task', 0):.2f}",
            f"{scores.get('competitor_response_task', 0):.2f}",
            f"{scores.get('crisis_recovery_task', 0):.2f}",
        ])
    return table(
        ["Config", "Seed", "Steps", "Final bank", "Format", "Profit", "Inventory", "Competitor", "Crisis"],
        rows,
    )


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Siyaani Commerce - AI CEO", analytics_enabled=False) as demo:
        gr.HTML(f"<style>{THEME_CSS}</style>")
        header_html = gr.HTML()
        diag_html = gr.HTML()

        with gr.Tabs():
            with gr.TabItem("Story"):
                gr.HTML(render_story_html())

            with gr.TabItem("Live Run"):
                with gr.Row():
                    with gr.Column(scale=1):
                        policy_radio = gr.Radio(
                            choices=list(ALL_POLICIES),
                            value=POLICY_TRAINED,
                            label="Policy",
                        )
                        seed_in = gr.Number(value=2026, label="Seed", precision=0)
                        business_in = gr.Textbox(
                            value="",
                            label="business_id (optional)",
                            placeholder="leave empty for default; e.g. medplus_pharmacy",
                        )
                        run_btn = gr.Button("Run single episode", variant="primary")
                        ab_btn = gr.Button("Run A/B comparison (same seed, baseline_zero_shot vs trained)", variant="secondary")
                    with gr.Column(scale=2):
                        run_summary = gr.HTML(banner("Idle", "No run yet.", kind="neutral"))
                        log_output = gr.Code(label="Step log", language="markdown", lines=20, elem_classes=["r2-log"])
                        final_cards = gr.HTML("")
                        grader_html = gr.HTML("")
                        run_meta = gr.Code(label="Run metadata (run_id, endpoint counts, hashes)", language="json", lines=12)

                run_btn.click(
                    fn=_do_run_single,
                    inputs=[policy_radio, seed_in, business_in],
                    outputs=[run_summary, log_output, final_cards, grader_html, run_meta],
                )

                gr.HTML('<hr style="border:none;border-top:1px solid var(--border);margin:24px 0;" />')

                ab_summary = gr.HTML(banner("A/B idle", "Click the A/B button to compare baseline vs trained on the same seed.", kind="neutral"))
                ab_table = gr.HTML("")
                ab_summary_json = gr.Code(label="A/B summary", language="json", lines=14)
                ab_meta = gr.Code(label="A/B metadata", language="json", lines=8)

                ab_btn.click(
                    fn=_do_run_ab,
                    inputs=[seed_in, business_in],
                    outputs=[ab_summary, ab_table, ab_summary_json, ab_meta],
                )

            with gr.TabItem("Training Proof"):
                gr.HTML('<h2 style="margin:0 0 12px 0;">Composite metrics: baseline vs trained</h2>')
                gr.HTML(_proof_metric_table_html())
                gr.HTML('<h3 style="margin:24px 0 12px 0;">Reward curve</h3>')
                p = artifact_image_path("reward_curve.png")
                if p:
                    gr.Image(value=p, show_label=False)
                else:
                    gr.HTML(evidence_unavailable("reward_curve.png"))
                gr.HTML('<h3 style="margin:24px 0 12px 0;">Exploration / entropy decay</h3>')
                p = artifact_image_path("exploration_curve.png")
                if p:
                    gr.Image(value=p, show_label=False)
                else:
                    gr.HTML(evidence_unavailable("exploration_curve.png"))
                gr.HTML('<h3 style="margin:24px 0 12px 0;">Action distribution shift</h3>')
                p = artifact_image_path("policy_evolution.png")
                if p:
                    gr.Image(value=p, show_label=False)
                else:
                    gr.HTML(evidence_unavailable("policy_evolution.png"))
                gr.HTML('<h3 style="margin:24px 0 12px 0;">Composite score lift</h3>')
                p = artifact_image_path("before_after_comparison.png")
                if p:
                    gr.Image(value=p, show_label=False)
                else:
                    gr.HTML(evidence_unavailable("before_after_comparison.png"))
                gr.HTML('<h3 style="margin:24px 0 12px 0;">Action success rates (per action_type)</h3>')
                gr.HTML(_action_success_table_html())

            with gr.TabItem("Generalisation"):
                gen_check = generalization_covers_unseen_configs()
                if not gen_check["ok"]:
                    gr.HTML(banner(
                        "Unseen-config evidence missing",
                        f"Need rows for: <code>{', '.join(gen_check['missing'])}</code>. "
                        "Re-run scripts/run_full_pipeline.py with --generalize-configs medplus_pharmacy stackbase_saas.",
                        kind="warn",
                    ))
                else:
                    gr.HTML(banner(
                        "Unseen-config evidence present",
                        f"Configs covered: <code>{', '.join(gen_check['covered'])}</code>.",
                        kind="good",
                    ))
                gr.HTML('<h3 style="margin:18px 0 12px 0;">Per-episode performance across configs</h3>')
                gr.HTML(_generalization_table_html())
                gr.HTML('<h3 style="margin:18px 0 12px 0;">Generalisation chart</h3>')
                p = artifact_image_path("generalization.png")
                if p:
                    gr.Image(value=p, show_label=False)
                else:
                    gr.HTML(evidence_unavailable("generalization.png"))
                gr.HTML('<h3 style="margin:18px 0 12px 0;">Failure vs recovery (same seed, two policies)</h3>')
                p = artifact_image_path("failure_vs_recovery.png")
                if p:
                    gr.Image(value=p, show_label=False)
                else:
                    gr.HTML(evidence_unavailable("failure_vs_recovery.png"))

        gr.HTML(
            '<div class="r2-footer">'
            'Trained with <strong>GRPO</strong> on <strong>Qwen2.5-1.5B-Instruct</strong>. '
            'Backend follows the <strong>OpenEnv</strong> contract. '
            'Built for the Round-2 hackathon judging panel - all numbers above come from real endpoint calls or files in <code>artifacts/</code>.'
            '</div>'
        )

        # Compute header + diagnostics on each page load so the banner reflects the
        # LIVE runtime state of the backend, not the pre-bind state at module import.
        demo.load(fn=lambda: (_header_html(), _startup_diagnostics_html()),
                  inputs=None, outputs=[header_html, diag_html])
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        show_error=True,
    )
