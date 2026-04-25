"""Phase C2 — landing page + scripted-demo SSE route tests.

These guard the only behavioural changes added by Phase C2:

  * ``GET /`` still returns JSON for anything that is not a browser
    (Accept: */*, application/json, missing Accept) so every existing
    OpenEnv contract test keeps passing.
  * ``GET /?format=html`` and a browser-style ``Accept: text/html``
    return an ``HTMLResponse`` containing the "Run Demo" button +
    pinned headline.
  * ``GET /demo`` streams ``text/event-stream`` with at least one
    ``event: step`` line and exactly one ``event: summary`` line.
"""

from __future__ import annotations

import json


def test_root_default_still_returns_json(fresh_app):
    """Existing contract: TestClient default Accept must keep returning JSON."""
    r = fresh_app.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body["status"] == "online"
    assert "/demo" in body["endpoints"]


def test_root_html_when_browser_asks_for_it(fresh_app):
    r = fresh_app.get("/", headers={"Accept": "text/html"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    text = r.text
    assert "Run Demo" in text
    assert "Swiftlogic CommerceOps" in text
    assert "/demo" in text


def test_root_format_query_forces_html(fresh_app):
    r = fresh_app.get("/?format=html")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")


def test_demo_sse_emits_steps_and_summary(fresh_app):
    """The SSE endpoint should produce >=1 step event and exactly 1 summary."""
    with fresh_app.stream("GET", "/demo?steps=5&seed=20260425") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in response.iter_text())

    blocks = [b for b in body.split("\n\n") if b.strip()]
    step_blocks = [b for b in blocks if b.startswith("event: step")]
    summary_blocks = [b for b in blocks if b.startswith("event: summary")]
    assert len(step_blocks) >= 1, body
    assert len(summary_blocks) == 1, body

    first_step_data = step_blocks[0].split("data: ", 1)[1]
    parsed = json.loads(first_step_data)
    assert {"step", "day", "action_type", "reward", "bank_balance"} <= set(parsed)
    assert isinstance(parsed["reward"], (int, float))

    summary_data = summary_blocks[0].split("data: ", 1)[1]
    parsed_summary = json.loads(summary_data)
    assert parsed_summary["seed"] == 20260425
    assert parsed_summary["steps_executed"] >= 1
