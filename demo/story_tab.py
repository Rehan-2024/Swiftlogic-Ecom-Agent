"""Story tab - the 4-chapter narrative tied to real photos.

Reads demo/assets/photos/story.json so the team can edit copy or swap
photos without touching code.

Metrics for each chapter are sourced live from the artifacts the
training pipeline produces; if a metric isn't available yet (e.g. the
adapter hasn't been trained) the chapter still renders with the photo
and copy, and the metric block shows 'pending'.
"""

from __future__ import annotations

import base64
import html
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from demo.artifact_loader import (
    load_after_metrics,
    load_before_metrics,
    load_composite_score,
    load_generalization,
    load_pipeline_manifest,
    load_policy_signature,
)

PHOTOS_DIR = Path(__file__).resolve().parent / "assets" / "photos"
STORY_JSON = PHOTOS_DIR / "story.json"


def _photo_data_uri(filename: str) -> str:
    """Embed photos as data URIs so the rendered HTML doesn't depend on Gradio static routing."""
    p = PHOTOS_DIR / filename
    if not p.exists():
        return ""
    payload = p.read_bytes()
    mime = "image/jpeg"
    if filename.lower().endswith(".png"):
        mime = "image/png"
    b64 = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _resolve_metric(metric_key: str) -> str:
    """Return a short, judge-friendly string for the chapter metric block."""
    if metric_key == "starting_bank":
        gen = load_generalization()
        episodes = gen.get("episodes", []) if isinstance(gen, dict) else []
        if episodes:
            initial_bank = None
            for ep in episodes:
                fb = ep.get("final_bank")
                if isinstance(fb, (int, float)):
                    initial_bank = float(fb)
                    break
            if initial_bank is not None:
                return f"INR {initial_bank:,.0f}"
        return "INR 50,000"
    if metric_key == "format_compliance":
        gen = load_generalization()
        summary = gen.get("summary", {}) if isinstance(gen, dict) else {}
        v = summary.get("mean_format_compliance")
        if isinstance(v, (int, float)):
            return f"{v * 100:.1f}%"
        return "pending"
    if metric_key == "exploration_entropy_delta":
        sig = load_policy_signature().get("signatures", {}) if isinstance(load_policy_signature(), dict) else {}
        before = (sig.get("random") or {}).get("entropy")
        after = (sig.get("trained") or {}).get("entropy") or (sig.get("trained_fallback") or {}).get("entropy")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            return f"{before:.2f} -> {after:.2f} nats"
        return "pending"
    if metric_key == "generalization_composite":
        gen = load_generalization()
        summary = gen.get("summary", {}) if isinstance(gen, dict) else {}
        v = summary.get("composite_all_mean")
        if isinstance(v, (int, float)):
            return f"{v:.3f}"
        return "pending"
    return "pending"


def render_story_html() -> str:
    if not STORY_JSON.exists():
        return (
            '<div class="r2-banner is-warn">'
            '<h3>Story manifest missing</h3>'
            '<p>Run <code>python scripts/fetch_demo_photos.py</code> to set up the story tab.</p>'
            '</div>'
        )
    data = json.loads(STORY_JSON.read_text(encoding="utf-8"))
    title = html.escape(data.get("title", "Story"))
    subtitle = html.escape(data.get("subtitle", ""))

    chapters: List[Dict[str, Any]] = sorted(
        data.get("chapters", []),
        key=lambda c: int(c.get("order", 0)),
    )
    if not chapters:
        return banner_empty()

    hero = chapters[0]
    hero_uri = _photo_data_uri(hero.get("photo", ""))
    hero_html = ""
    if hero_uri:
        hero_html = (
            '<div class="r2-story-hero">'
            f'<img src="{hero_uri}" alt="{html.escape(hero.get("alt",""))}" />'
            '<div class="r2-story-hero-caption">'
            f'<p class="title">{title}</p>'
            f'<p class="subtitle">{subtitle}</p>'
            '</div>'
            '</div>'
        )

    chapter_blocks: List[str] = []
    for ch in chapters:
        photo_uri = _photo_data_uri(ch.get("photo", ""))
        if not photo_uri:
            continue
        metric_value = _resolve_metric(ch.get("metric_key", ""))
        chapter_blocks.append(
            '<div class="r2-story-chapter">'
            '<div class="photo">'
            f'<img src="{photo_uri}" alt="{html.escape(ch.get("alt",""))}" />'
            f'<span class="tag">{html.escape(ch.get("tag",""))}</span>'
            '</div>'
            '<div class="body">'
            f'<h3 class="chapter-title">{html.escape(ch.get("title",""))}</h3>'
            f'<p class="chapter-text">{html.escape(ch.get("story",""))}</p>'
            '<div class="chapter-metric">'
            f'<span class="label">{html.escape(ch.get("metric_label",""))}</span>'
            f'<span class="value">{html.escape(metric_value)}</span>'
            '</div>'
            '</div>'
            '</div>'
        )

    manifest = load_pipeline_manifest()
    provenance = manifest.get("provenance", "unknown")
    footer = (
        '<div class="r2-footer">'
        f'Story manifest version {data.get("version","?")}, '
        f'last edited {html.escape(str(data.get("last_edited","?")))} - '
        f'pipeline provenance: <strong>{html.escape(str(provenance))}</strong>'
        '</div>'
    )
    return hero_html + "".join(chapter_blocks) + footer


def banner_empty() -> str:
    return (
        '<div class="r2-banner is-warn">'
        '<h3>No chapters defined</h3>'
        '<p>Edit <code>demo/assets/photos/story.json</code> to add chapters.</p>'
        '</div>'
    )
