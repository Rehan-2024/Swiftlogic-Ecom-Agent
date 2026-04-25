"""HTML landing page + SSE demo helpers for the OpenEnv server.

Phase C2 of the Rank-1 plan: keep the existing JSON-only ``GET /`` route
backwards-compatible (every contract test relies on that), but when a
browser hits the HF Space root we serve a compact landing page with a
"Run Demo" button that streams 30 deterministic steps via Server-Sent
Events. This is purely additive — no existing endpoint or schema changes.

The scripted tape is imported from :mod:`scripted_demo` so the demo a
judge sees in the browser is the *same* tape they can replay locally
with ``python scripted_demo.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

ROOT = Path(__file__).resolve().parents[1]


def _load_scripted_tape() -> List[Dict[str, Any]]:
    """Lazy import the scripted tape so the server import has no side effects."""
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripted_demo import _SCRIPTED_TAPE, _resolve_action  # noqa: WPS433

    return _SCRIPTED_TAPE, _resolve_action  # type: ignore[return-value]


def render_landing(state: Dict[str, Any]) -> str:
    """Return the landing page HTML for the OpenEnv server.

    Reads only inert fields off ``state`` (display name, business id,
    composite headline if available); never mutates env state.
    """
    env = state.get("env")
    display_name = "CommerceOps v2"
    business_id = "siyaani_fashion"
    if env is not None:
        cfg = env.world_engine.config
        display_name = cfg.get("display_name", display_name) or display_name
        business_id = cfg.get("business_id", business_id) or business_id

    composite_path = ROOT / "artifacts" / "composite_score.json"
    headline = "trained model improves over baseline"
    provenance = ""
    if composite_path.exists():
        try:
            data = json.loads(composite_path.read_text(encoding="utf-8"))
            headline = data.get("headline", headline)
            provenance = data.get("provenance", "")
        except (OSError, ValueError):
            pass

    badge = ""
    if provenance:
        badge = (
            f'<span class="prov" title="composite score provenance">'
            f"provenance: {provenance}</span>"
        )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
<title>Swiftlogic CommerceOps v2 — OpenEnv RL Demo</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: -apple-system, Segoe UI, Roboto, Inter, sans-serif;
       max-width: 920px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
header {{ display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; }}
h1 {{ margin: 0; font-size: 1.6rem; }}
.subtitle {{ opacity: 0.7; }}
.headline {{ font-size: 1.1rem; padding: 0.5rem 0.8rem; background: #1f2d3d10;
             border-radius: 0.4rem; display: inline-block; margin: 0.6rem 0; }}
.prov {{ font-size: 0.75rem; opacity: 0.6; margin-left: 0.4rem; }}
button {{ font-size: 1rem; padding: 0.5rem 1rem; border-radius: 0.4rem;
          border: 1px solid #888; cursor: pointer; }}
button:hover {{ background: #1f2d3d10; }}
section {{ margin-top: 1.5rem; }}
ul.endpoints {{ list-style: none; padding: 0; columns: 2; }}
ul.endpoints li {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                   padding: 0.15rem 0; }}
#log {{ background: #0d1117; color: #d1d5db; padding: 1rem; border-radius: 0.4rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        white-space: pre-wrap; height: 22rem; overflow: auto; font-size: 0.85rem; }}
.status {{ margin: 0.4rem 0; font-size: 0.9rem; opacity: 0.8; }}
footer {{ margin-top: 2rem; font-size: 0.8rem; opacity: 0.65; }}
a {{ color: inherit; }}
</style>
</head>
<body>
<header>
  <h1>Swiftlogic CommerceOps v2</h1>
  <span class=\"subtitle\">OpenEnv v0.2.3 · {display_name} ({business_id})</span>
</header>

<div class=\"headline\">{headline}{badge}</div>

<section>
  <p>This is a live <a href=\"https://github.com/meta-pytorch/OpenEnv\">OpenEnv</a>
  environment for an Indian e-commerce SMB. Click <strong>Run Demo</strong> to
  stream a deterministic 30-step scripted episode (seed 20260425) from the
  server in real time.</p>
  <button id=\"run\" type=\"button\">▶ Run Demo</button>
  <button id=\"stop\" type=\"button\" disabled>■ Stop</button>
  <div id=\"status\" class=\"status\">idle</div>
</section>

<section>
  <pre id=\"log\">// Demo output will stream here…
</pre>
</section>

<section>
  <h3>OpenEnv endpoints</h3>
  <ul class=\"endpoints\">
    <li>POST /reset</li><li>POST /step</li>
    <li>GET  /state</li><li>GET  /tasks</li>
    <li>POST /grader</li><li>POST /config</li>
    <li>GET  /health</li><li>GET  /demo  (SSE)</li>
  </ul>
</section>

<footer>
  Frozen at <code>release/env-frozen-v2.3</code> · trained adapter via GRPO+Unsloth on Colab T4 ·
  <a href=\"https://huggingface.co/spaces/Swiftlogic/E-commerce-agent\">HF Space</a>
</footer>

<script>
(function () {{
  const runBtn = document.getElementById('run');
  const stopBtn = document.getElementById('stop');
  const log = document.getElementById('log');
  const status = document.getElementById('status');
  let evt = null;

  function append(line) {{
    log.textContent += line + "\\n";
    log.scrollTop = log.scrollHeight;
  }}

  function setRunning(running) {{
    runBtn.disabled = running;
    stopBtn.disabled = !running;
    status.textContent = running ? 'streaming…' : 'idle';
  }}

  runBtn.addEventListener('click', () => {{
    log.textContent = '';
    setRunning(true);
    append('// connecting to /demo (SSE)…');
    evt = new EventSource('/demo?steps=30&seed=20260425');
    evt.addEventListener('step', (e) => {{
      try {{
        const data = JSON.parse(e.data);
        const tag = data.action_quality ? '[' + data.action_quality + ']' : '';
        append('Day ' + data.day + ' · ' + data.action_type
               + ' · reward=' + data.reward.toFixed(3)
               + ' · bank=₹' + Math.round(data.bank_balance)
               + ' ' + tag);
      }} catch (err) {{ append('// parse error: ' + err.message); }}
    }});
    evt.addEventListener('summary', (e) => {{
      append('--');
      append('summary: ' + e.data);
      setRunning(false);
      evt.close();
    }});
    evt.addEventListener('error', (e) => {{
      append('// stream error');
      setRunning(false);
      if (evt) evt.close();
    }});
  }});

  stopBtn.addEventListener('click', () => {{
    if (evt) evt.close();
    setRunning(false);
    append('// stopped by user');
  }});
}})();
</script>
</body>
</html>
"""


def stream_scripted_demo(
    env: Any,
    *,
    seed: int = 20260425,
    steps: int = 30,
) -> Iterator[bytes]:
    """Yield SSE-formatted events for a deterministic scripted demo run.

    Note: the caller MUST hold the env lock for the duration of this
    generator — every iteration calls ``env.step`` which mutates state.
    """
    tape, resolve = _load_scripted_tape()
    obs = env.reset(seed=seed)
    initial_bank = float(getattr(obs, "bank_balance", 0.0))
    last_reward = 0.0
    total_reward = 0.0
    step_count = min(steps, len(tape))

    for i in range(step_count):
        action = resolve(tape[i % len(tape)], obs)
        a_type = action.get("action_type", "wait")
        try:
            from ecom_env import EcomAction  # local import to avoid cycle

            validated = EcomAction.model_validate(action).model_dump()
        except Exception:
            validated = {"action_type": "wait"}
            a_type = "wait"

        result = env.step(validated)

        def _to_float(value: Any) -> float:
            if hasattr(value, "value"):
                value = value.value
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        if hasattr(result, "observation"):
            obs = result.observation
            reward = _to_float(getattr(result, "reward", 0.0))
            done = bool(getattr(result, "done", False))
            info = getattr(result, "info", {}) or {}
        elif isinstance(result, tuple) and len(result) >= 3:
            obs = result[0]
            reward = _to_float(result[1])
            done = bool(result[2])
            info = result[3] if len(result) > 3 else {}
        else:
            obs, reward, done, info = result, 0.0, False, {}

        total_reward += reward
        last_reward = reward
        payload = {
            "step": i + 1,
            "day": int(getattr(obs, "current_day", i + 1)),
            "action_type": a_type,
            "reward": round(reward, 4),
            "bank_balance": round(float(getattr(obs, "bank_balance", 0.0)), 2),
            "action_quality": (info or {}).get("action_quality"),
            "strategy_phase": (info or {}).get("strategy_phase"),
        }
        yield f"event: step\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
        if done:
            break

    summary = {
        "steps_executed": step_count,
        "total_reward": round(total_reward, 4),
        "last_reward": round(last_reward, 4),
        "final_bank": round(float(getattr(obs, "bank_balance", 0.0)), 2),
        "delta_bank": round(float(getattr(obs, "bank_balance", 0.0)) - initial_bank, 2),
        "seed": seed,
    }
    yield f"event: summary\ndata: {json.dumps(summary)}\n\n".encode("utf-8")
