"""Round-2 Gradio dashboard - long-form scrollable storytelling layout.

Composition only - all the heavy lifting lives in:
  * demo.sections      - static HTML for hero / chapters / proof / generalisation / tech / footer
  * demo.live_theater  - streaming generator for the interactive Run-Demo theatre
  * demo.artifact_loader / backend_client / policy / episode_runner

Layout (top to bottom on a single scrollable page):
  1. Hero (real photo + brand title + status pills + jump-nav)
  2. Story  - 4 chapter blocks tied to real photos and a live metric each
  3. Live Demo Theater - controls + per-step Observe/Reason/Act/React + live charts
  4. Training Proof - composite metrics, reward + exploration + action-shift figures
  5. Generalisation - per-config table + figures
  6. Tech stack / How it works - 4 cards
  7. Footer
"""

from __future__ import annotations

import functools
import html as html_lib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.backend_client import BackendClient, BackendError  # noqa: E402
from demo.components import banner, patience_box  # noqa: E402
from demo.live_theater import build_pipeline_html, DEMO_MAX_STEPS, stream_live_episode, stream_policy_transition  # noqa: E402
from demo.policy import (  # noqa: E402
    ALL_POLICIES,
    POLICY_TRAINED,
)
from demo.sections import (  # noqa: E402
    generalization_artifact_paths,
    generalization_table_rows,
    render_authenticity_strip,
    render_footer,
    render_hero,
    render_impact_section,
    render_problem_solution_flow,
    render_sdg_section,
    render_story_section,
    render_autonomous_section,
    render_techstack_section,
    render_theater_intro,
    training_artifact_paths,
    training_metrics_table,
)

logger = logging.getLogger("commerceops.demo.app")

THEME_CSS = (Path(__file__).parent / "theme.css").read_text(encoding="utf-8")
STORE_CSS = (Path(__file__).parent / "store_theme.css").read_text(encoding="utf-8")


def _dispatch_live_run(
    run_mode: str,
    policy_name: str,
    seed: int,
    business_id: str,
    max_steps: int,
    distribution_view: str,
):
    if run_mode == "baseline_vs_trained_same_seed":
        yield from stream_policy_transition(seed, business_id, max_steps, distribution_view)
        return
    yield from stream_live_episode(policy_name, seed, business_id, max_steps, distribution_view)


CONFIGS_DIR = ROOT / "configs"


@functools.lru_cache(maxsize=8)
def _load_business_config(business_id: str) -> Dict[str, Any]:
    """Read configs/<business_id>.json once and cache forever (read-only)."""
    if not business_id:
        return {}
    path = CONFIGS_DIR / f"{business_id}.json"
    if not path.exists():
        return {}
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _product_metadata(business_id: str) -> Dict[str, Dict[str, Any]]:
    """Return a sku -> {display_name, sell_price, unit_cost} map from config."""
    cfg = _load_business_config(business_id) or _load_business_config("siyaani_fashion")
    products = cfg.get("products") or []
    out: Dict[str, Dict[str, Any]] = {}
    for p in products:
        sku = p.get("sku")
        if not sku:
            continue
        out[sku] = {
            "display_name": p.get("display_name") or sku.replace("_", " ").title(),
            "sell_price": float(p.get("sell_price", 0.0)),
            "unit_cost": float(p.get("unit_cost", 0.0)),
            "competitor_price": float(p.get("competitor_price", 0.0)),
            "initial_stock": int(p.get("initial_stock", 0)),
        }
    return out


# A small fixed accent palette so the product tiles look like real
# product cards without any external image fetches.
_PRODUCT_TILE_COLORS = [
    ("#f3e1d2", "#7a3f1f"),  # warm clay
    ("#e3ecde", "#3e6342"),  # sage
    ("#e8e1ef", "#5a4079"),  # lavender ink
    ("#f7e9c8", "#7a5f1f"),  # mustard
    ("#dde6ef", "#34557a"),  # slate blue
    ("#f0d8d6", "#923a3a"),  # terracotta
]


def _static_photo_href(filename: str) -> str:
    p = Path(__file__).parent / "assets" / "photos" / filename
    if not p.exists():
        return ""
    return f"/static/demo/photos/{filename}"


# Local thumbnails for Siyaani SKUs (ethnic wear — downloaded once into demo/assets/photos).
_SKU_THUMB: Dict[str, str] = {
    "silk_saree": "prd_silk_saree.jpg",
    "silk_kurta": "prd_silk_kurta.jpg",
    "cotton_set": "prd_cotton_set.jpg",
    "linen_dupatta": "prd_linen_dupatta.jpg",
}


def _product_tile_html(idx: int, sku: str, qty: int, meta: Dict[str, Any], competitor_price: float) -> str:
    bg, fg = _PRODUCT_TILE_COLORS[idx % len(_PRODUCT_TILE_COLORS)]
    name = meta.get("display_name") or sku.replace("_", " ").title()
    sell = float(meta.get("sell_price", 0.0))
    initials = "".join(part[0] for part in name.split()[:2]).upper() or sku[:2].upper()
    thumb = _static_photo_href(_SKU_THUMB.get(sku, ""))
    thumb_block = (
        f'<div class="r2-product-thumb is-photo"><img src="{html_lib.escape(thumb)}" alt="{html_lib.escape(name)}" loading="lazy" /></div>'
        if thumb
        else f'<div class="r2-product-thumb" style="background:{bg};color:{fg};">{html_lib.escape(initials)}</div>'
    )
    margin = (sell - float(meta.get("unit_cost", 0.0)))
    margin_pct = (margin / sell * 100.0) if sell else 0.0
    stock_cls = "is-low" if qty <= 5 else ("is-mid" if qty <= 15 else "is-ok")
    comp = competitor_price or float(meta.get("competitor_price", 0.0))
    diff = (sell - comp) if comp else 0.0
    diff_label = (
        f"-{abs(diff):,.0f} vs comp" if diff < 0 else
        (f"+{diff:,.0f} vs comp" if diff > 0 else "= comp")
    )
    return (
        '<div class="r2-product-card r2-fade-in-text">'
        f"{thumb_block}"
        '<div class="r2-product-body">'
        f'<div class="r2-product-name">{html_lib.escape(name)}</div>'
        f'<div class="r2-product-sku">{html_lib.escape(sku)}</div>'
        '<div class="r2-product-row">'
        f'<span class="r2-product-price">INR {sell:,.0f}</span>'
        f'<span class="r2-product-margin">{margin_pct:.0f}% margin</span>'
        '</div>'
        '<div class="r2-product-row">'
        f'<span class="r2-product-stock {stock_cls}">{int(qty)} in stock</span>'
        f'<span class="r2-product-comp">{html_lib.escape(diff_label)}</span>'
        '</div>'
        '</div>'
        '</div>'
    )


def _store_fetch_panel(show: bool, business_id: str = ""):
    """Build retail ops panel HTML; single round-trip to /state."""
    if not show:
        return gr.update(visible=False), ""
    bc = BackendClient()
    obs: Dict[str, Any] = {}
    err: Optional[str] = None
    try:
        state = bc.state()
        obs = state.get("observation", {}) or {}
    except BackendError as exc:
        err = str(exc)

    bank = float(obs.get("bank_balance", 0.0))
    bank_cls = "is-profit" if bank >= 0 else "is-loss"
    inventory = obs.get("inventory", {}) or {}
    competitor = obs.get("competitor_prices", {}) or {}
    tickets = obs.get("active_tickets", []) or []

    biz = (business_id or "").strip() or "siyaani_fashion"
    products_meta = _product_metadata(biz)

    # Always show every configured product, even if /state didn't return it,
    # so the grid mirrors the actual catalogue you compare against.
    skus = list(products_meta.keys())
    for sku in inventory.keys():
        if sku not in products_meta:
            products_meta[sku] = {
                "display_name": sku.replace("_", " ").title(),
                "sell_price": 0.0,
                "unit_cost": 0.0,
                "competitor_price": float(competitor.get(sku, 0.0)),
                "initial_stock": 0,
            }
            skus.append(sku)

    if not skus:
        product_cards = '<div class="r2-product-empty">No products in this business config.</div>'
    else:
        product_cards = "".join(
            _product_tile_html(
                i, sku,
                int(inventory.get(sku, products_meta[sku].get("initial_stock", 0))),
                products_meta[sku],
                float(competitor.get(sku, 0.0)),
            )
            for i, sku in enumerate(skus)
        )

    ticket_badges: List[str] = []
    for i, t in enumerate(tickets[:8]):
        tid = str((t or {}).get("ticket_id", f"T{i+1}"))
        text = str(t).lower()
        if "urgent" in text or "high" in text:
            cls = "is-high"
        elif "low" in text:
            cls = "is-low"
        else:
            cls = "is-mid"
        ticket_badges.append(f'<span class="r2-ticket-badge {cls}">{html_lib.escape(tid)}</span>')
    if not ticket_badges:
        ticket_badges = ['<span class="r2-ticket-badge is-low">No active tickets</span>']

    error_block = (
        banner("Live store snapshot unavailable", html_lib.escape(err), kind="warn")
        if err else ""
    )

    html_payload = (
        f'{error_block}'
        '<div class="r2-store-panel">'
        f'<h4>{html_lib.escape(biz.replace("_", " ").title())} - Live store view</h4>'
        '<div class="r2-store-grid">'
        '<div class="r2-store-kpi"><div class="k">Cash register</div>'
        f'<div class="v {bank_cls}">INR {bank:,.2f}</div></div>'
        f'<div class="r2-store-kpi"><div class="k">Catalogue SKUs</div><div class="v">{len(skus)}</div></div>'
        f'<div class="r2-store-kpi"><div class="k">Competitor tracked</div><div class="v">{len(competitor)}</div></div>'
        f'<div class="r2-store-kpi"><div class="k">Open tickets</div><div class="v">{len(tickets)}</div></div>'
        '</div>'
        '<h4 style="margin-top:14px;">Our products</h4>'
        '<div class="r2-product-grid">'
        f'{product_cards}'
        '</div>'
        '<h4 style="margin-top:14px;">Ticket queue urgency</h4>'
        + "".join(ticket_badges) +
        '</div>'
    )
    return gr.update(visible=True), html_payload


def _render_store_view_stream(show: bool, business_id: str = ""):
    """Two-step: patience callout, then live snapshot (smooth UX for /state)."""
    if not show:
        yield gr.update(visible=False), ""
        return
    yield gr.update(visible=True), (
        patience_box(
            "Please be patient",
            "We are reading the live OpenEnv /state for this business. Large inventories or a cold server can add a few seconds.",
        )
        + '<p class="r2-store-pending r2-fade-in-text" style="margin:0;">Preparing the Siyaani product grid…</p>'
    )
    yield _store_fetch_panel(True, business_id)


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Siyaani Commerce - The Self-Driving Brand",
        analytics_enabled=False,
        fill_width=True,
    ) as demo:
        gr.HTML(f"<style>{THEME_CSS}\n{STORE_CSS}</style>")
        hero_html = gr.HTML()
        story_html = gr.HTML()
        problem_html = gr.HTML()
        sdg_html = gr.HTML()
        impact_html = gr.HTML()
        autonomous_html = gr.HTML()
        authenticity_html = gr.HTML()
        tech_html = gr.HTML()

        gr.HTML(
            '<section class="r2-section r2-section-tight" id="proof-tabs">'
            '<p class="r2-section-eyebrow r2-fade-in-text">Siyaani · live proof</p>'
            '<h2 class="r2-section-title r2-fade-in-text">CEO command center, learning ledger, generalisation</h2>'
            "</section>"
        )
        with gr.Tabs():
            with gr.Tab("CEO Command Center"):
                gr.HTML(render_theater_intro())
                gr.Markdown(
                    "**Controls** — *Launch autonomous shift* / *Replay same seed* use the mode + policy you selected. "
                    "*Run baseline vs trained* always runs baseline zero-shot then trained (same seed). "
                    "*Pause run* cancels an in‑flight stream. *Retail ops view* + *Refresh* load **/state** in two frames (patience, then the SKU grid). "
                    "Graphs in single-policy mode refresh on a **stride** to stay fast; step cards and the log still update every day."
                )
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        run_mode = gr.Radio(
                            choices=["single_policy", "baseline_vs_trained_same_seed"],
                            value="single_policy",
                            label="Execution mode",
                            info="Single policy: day-by-day stream. Baseline vs trained: two episodes on the same seed (longer; charts coalesce for smooth UI).",
                        )
                        policy_radio = gr.Radio(
                            choices=list(ALL_POLICIES),
                            value=POLICY_TRAINED,
                            label="Policy profile",
                            info="trained: LoRA on Qwen2.5-1.5B. baseline_zero_shot: base model, T=0. baseline_wait_only: always wait (no LLM).",
                        )
                        seed_in = gr.Number(
                            value=20260425,
                            label="Seed",
                            precision=0,
                            info="Identical seed + business_id = comparable trajectories for judges.",
                        )
                        business_in = gr.Textbox(
                            value="siyaani_fashion",
                            label="business_id (optional)",
                            placeholder="siyaani_fashion · medplus_pharmacy · stackbase_saas",
                        )
                        steps_in = gr.Slider(
                            minimum=5,
                            maximum=DEMO_MAX_STEPS,
                            value=min(10, DEMO_MAX_STEPS),
                            step=1,
                            label="Shift length (days)",
                            info="Shorter shifts finish sooner. Chart refresh is throttled (every 2 days by default) for speed.",
                        )
                        dist_mode = gr.Radio(
                            choices=["bar", "pie"],
                            value="bar",
                            label="Action mix visual",
                            info="Bar counts or pie share for the current policy’s action distribution.",
                        )
                        with gr.Row():
                            run_btn = gr.Button("Launch autonomous shift", variant="primary")
                            rerun_btn = gr.Button("Replay same seed", variant="secondary")
                        with gr.Row():
                            compare_btn = gr.Button("Run baseline vs trained (same seed)", variant="secondary")
                            stop_btn = gr.Button("Pause run", variant="secondary")
                        store_toggle = gr.Checkbox(
                            value=False,
                            label="Retail ops view (live /state)",
                            info="Siyaani SKU grid from config + current stock; ticket urgency below.",
                        )
                        refresh_store_btn = gr.Button("Refresh retail snapshot", variant="secondary")

                    with gr.Column(scale=2):
                        with gr.Accordion("View System Architecture & Process Flow", open=False):
                            gr.Markdown("""
```mermaid
graph LR
    A[Environment Observation] -->|State: Inventory, Tickets| B(CEO Agent)
    B -->|Reasoning Engine| C{Decision Module}
    C -->|Wait| D[No Action]
    C -->|Restock / Negotiate| E[Supplier API]
    C -->|Refund| F[Support Queue]
    C -->|Ad Spend| G[Marketing Engine]
    D & E & F & G --> H[OpenEnv Backend]
    H -->|Reward & Next State| A
```
""")
                        with gr.Group(elem_classes=["r2-theater"]):
                            head_html = gr.HTML(
                                '<div class="r2-theater-head"><div>'
                                '<h3 class="episode-title r2-fade-in-text">Ready</h3>'
                                '<div class="episode-meta r2-fade-in-text">Pick execution mode, policy, seed, and shift length, then <strong>Launch</strong> or <strong>Run baseline vs trained</strong>.'
                                "</div></div></div>"
                                f"{build_pipeline_html(0.0)}"
                            )
                            step_card = gr.HTML(
                                '<div class="r2-banner is-warn r2-fade-in-text"><h3>Nothing running yet</h3>'
                                "<p>When you launch, a <em>Please be patient</em> callout may appear first while the backend resets and the policy loads. "
                                "Each day then shows Observe → Reason → Act → React.</p></div>"
                            )
                            with gr.Row():
                                bank_plot = gr.Plot(label="Cash trajectory")
                                action_plot = gr.Plot(label="Action mix / comparison")
                            policy_plot = gr.Plot(label="Execution quality")
                            log_output = gr.HTML('<div class="r2-live-log"><div class="line">Step log will appear here.</div></div>')
                            with gr.Group(visible=False) as store_group:
                                store_html = gr.HTML()
                            scorecard_html = gr.HTML("")

            with gr.Tab("Learning Ledger"):
                gr.Markdown("### Real artifacts from latest pipeline run")
                with gr.Row():
                    reward_img = gr.Image(label="reward_curve.png", interactive=False, type="filepath")
                    explore_img = gr.Image(label="exploration_curve.png", interactive=False, type="filepath")
                with gr.Row():
                    evolution_img = gr.Image(label="policy_evolution.png", interactive=False, type="filepath")
                    before_after_img = gr.Image(label="before_after_comparison.png", interactive=False, type="filepath")
                gr.Markdown("### Baseline vs Trained metrics (traceable to raw artifacts)")
                training_table = gr.Dataframe(
                    headers=["Metric", "Baseline", "Trained", "Improvement"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                )
                normalize_note = gr.Markdown("Normalized presentation uses raw-sourced values; no fabricated metrics.")

            with gr.Tab("Scale Proof"):
                with gr.Row():
                    generalization_img = gr.Image(label="generalization.png", interactive=False, type="filepath")
                    failure_img = gr.Image(label="failure_vs_recovery.png", interactive=False, type="filepath")
                generalization_table = gr.Dataframe(
                    headers=["Config", "Seed", "Final bank", "Format", "Profit task"],
                    datatype=["str", "number", "str", "str", "str"],
                    interactive=False,
                )

        footer_html = gr.HTML()

        store_event = store_toggle.change(
            fn=_render_store_view_stream,
            inputs=[store_toggle, business_in],
            outputs=[store_group, store_html],
        )
        refresh_store_event = refresh_store_btn.click(
            fn=_render_store_view_stream,
            inputs=[store_toggle, business_in],
            outputs=[store_group, store_html],
        )
        run_event = run_btn.click(
            fn=_dispatch_live_run,
            inputs=[run_mode, policy_radio, seed_in, business_in, steps_in, dist_mode],
            outputs=[head_html, step_card, log_output, bank_plot, action_plot, policy_plot, scorecard_html],
        )
        rerun_event = rerun_btn.click(
            fn=_dispatch_live_run,
            inputs=[run_mode, policy_radio, seed_in, business_in, steps_in, dist_mode],
            outputs=[head_html, step_card, log_output, bank_plot, action_plot, policy_plot, scorecard_html],
        )
        compare_event = compare_btn.click(
            fn=stream_policy_transition,
            inputs=[seed_in, business_in, steps_in, dist_mode],
            outputs=[head_html, step_card, log_output, bank_plot, action_plot, policy_plot, scorecard_html],
        )
        stop_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[run_event, rerun_event, compare_event, store_event, refresh_store_event],
        )

        demo.load(
            fn=lambda: (
                render_hero(),
                render_story_section(),
                render_problem_solution_flow(),
                render_sdg_section(),
                render_impact_section(),
                render_autonomous_section(),
                render_authenticity_strip(),
                render_techstack_section(),
                training_artifact_paths().get("reward_curve"),
                training_artifact_paths().get("exploration_curve"),
                training_artifact_paths().get("policy_evolution"),
                training_artifact_paths().get("before_after"),
                training_metrics_table(normalized=True),
                "Normalized presentation uses raw-sourced values; no fabricated metrics.",
                generalization_artifact_paths().get("generalization"),
                generalization_artifact_paths().get("failure_vs_recovery"),
                generalization_table_rows(),
                render_footer(),
            ),
            inputs=None,
            outputs=[
                hero_html,
                story_html,
                problem_html,
                sdg_html,
                impact_html,
                autonomous_html,
                authenticity_html,
                tech_html,
                reward_img,
                explore_img,
                evolution_img,
                before_after_img,
                training_table,
                normalize_note,
                generalization_img,
                failure_img,
                generalization_table,
                footer_html,
            ],
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        show_error=True,
    )
