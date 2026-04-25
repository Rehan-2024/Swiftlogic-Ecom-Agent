"""Reusable UI fragments. All styling comes from theme.css; no inline color/glow."""

from __future__ import annotations

import html
from typing import Any, Dict, Iterable, List, Optional


def banner(title: str, body: str, *, kind: str = "neutral") -> str:
    """kind in {'neutral','warn','danger','good'}."""
    cls = "r2-banner"
    if kind in {"warn", "danger", "good"}:
        cls += f" is-{kind}"
    return (
        f'<div class="{cls}">'
        f'<h3>{html.escape(title)}</h3>'
        f'<p>{body}</p>'
        f'</div>'
    )


def pill(label: str, *, kind: str = "neutral") -> str:
    """kind in {'neutral','ready','pre','down'}."""
    cls = "r2-pill"
    if kind in {"ready", "pre", "down"}:
        cls += f" is-{kind}"
    return f'<span class="{cls}">{html.escape(label)}</span>'


def metric_card(label: str, value: str, *, sub: str = "", tone: str = "neutral") -> str:
    """tone in {'neutral','good','bad','warn'}."""
    tone_cls = ""
    if tone == "good":
        tone_cls = " is-good"
    elif tone == "bad":
        tone_cls = " is-bad"
    elif tone == "warn":
        tone_cls = " is-warn"
    sub_html = f'<div class="r2-metric-delta">{html.escape(sub)}</div>' if sub else ""
    return (
        '<div class="r2-card">'
        f'<div class="r2-metric-label">{html.escape(label)}</div>'
        f'<div class="r2-metric-value{tone_cls}">{html.escape(value)}</div>'
        f'{sub_html}'
        '</div>'
    )


def metric_row(cards: Iterable[str]) -> str:
    return '<div class="r2-card-row">' + "".join(cards) + '</div>'


def table(headers: List[str], rows: List[List[Any]], *, empty_msg: str = "evidence unavailable") -> str:
    if not rows:
        return banner("No data", empty_msg, kind="warn")
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_lines: List[str] = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(c))}</td>" for c in row)
        body_lines.append(f"<tr>{cells}</tr>")
    return f'<table class="r2-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body_lines)}</tbody></table>'


def evidence_unavailable(file_name: str) -> str:
    return banner(
        "Evidence unavailable",
        f"<code>{html.escape(file_name)}</code> not found in <code>artifacts/</code>. "
        "Run the training pipeline to produce it.",
        kind="warn",
    )


def fmt_currency(value: float, *, currency: str = "INR") -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{currency} -"
    sign = "-" if v < 0 else ""
    abs_v = abs(v)
    return f"{sign}{currency} {abs_v:,.2f}"


def fmt_pct(numerator: float, denominator: float) -> str:
    try:
        if not denominator:
            return "-"
        return f"{(numerator / denominator) * 100:.1f}%"
    except (TypeError, ValueError, ZeroDivisionError):
        return "-"


def fmt_delta(before: float, after: float) -> str:
    try:
        b = float(before)
        a = float(after)
        if b == 0:
            return "-"
        return f"{((a - b) / abs(b)) * 100:+.1f}%"
    except (TypeError, ValueError):
        return "-"
