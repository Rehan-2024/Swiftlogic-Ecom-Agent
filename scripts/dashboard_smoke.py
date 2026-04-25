"""End-to-end smoke test for the mounted FastAPI + Gradio dashboard.

What it does:
  1. Spawns `uvicorn demo.entry:app` on a free port in a child process.
  2. Polls /health until it responds (or 30s timeout).
  3. Hits /tasks to confirm the OpenEnv contract is alive.
  4. Confirms /demo/ returns HTML (Gradio mount).
  5. Runs ONE real episode through demo.episode_runner.run_episode
     using the wait-only baseline (deterministic, no model loading).
  6. Verifies a live_runs/<id>.json was persisted.
  7. Kills the server and prints a structured report.

Exit code 0 = green. Anything else = a real failure to investigate.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demo.backend_client import BackendClient  # noqa: E402
from demo.episode_runner import LIVE_RUNS_DIR, run_episode  # noqa: E402
from demo.policy import POLICY_BASELINE_WAIT  # noqa: E402


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_for_health(base_url: str, timeout_s: float = 60.0) -> bool:
    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return True
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last_err = exc.__class__.__name__
        time.sleep(0.4)
    print(f"  ! /health never came up: last_err={last_err}", file=sys.stderr)
    return False


def main() -> int:
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["DEMO_PATH"] = "/demo"
    env["ENV_URL"] = base_url
    env["PYTHONPATH"] = str(ROOT)

    print(f"[smoke] starting uvicorn on {base_url}")
    log_path = ROOT / "artifacts" / f"smoke_uvicorn_{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "demo.entry:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "info"],
        cwd=str(ROOT),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    failures = []

    try:
        if not wait_for_health(base_url, timeout_s=90.0):
            failures.append("server never reached /health")
            return 2

        print("[smoke] /health OK")

        try:
            tasks = requests.get(f"{base_url}/tasks", timeout=10).json()
        except Exception as exc:
            failures.append(f"/tasks call failed: {exc}")
            tasks = []
        if not isinstance(tasks, list) or len(tasks) != 6:
            failures.append(f"/tasks returned wrong shape: type={type(tasks).__name__} len={len(tasks) if hasattr(tasks,'__len__') else '?'}")
        else:
            print(f"[smoke] /tasks returned {len(tasks)} entries")

        try:
            r = requests.get(f"{base_url}/demo/", timeout=10)
            if r.status_code != 200:
                failures.append(f"/demo/ returned HTTP {r.status_code}")
            elif "<html" not in r.text.lower():
                failures.append("/demo/ did not return HTML")
            else:
                print(f"[smoke] /demo/ returned {len(r.text)} bytes of HTML")
        except Exception as exc:
            failures.append(f"/demo/ probe failed: {exc}")

        print("[smoke] running 1 wait-only episode (max 5 steps for speed)")
        bc = BackendClient(base_url=base_url)
        trace = run_episode(POLICY_BASELINE_WAIT, seed=2026, backend=bc, max_steps=5)
        if "error" in trace:
            failures.append(f"episode errored: {trace['error']}")
        else:
            run_id = trace["run_id"]
            print(f"[smoke] episode run_id={run_id}, steps={trace['n_steps']}, "
                  f"final_bank={trace['final_bank']}, total_reward={trace['total_reward']}")
            persisted = LIVE_RUNS_DIR / f"{run_id}.json"
            if not persisted.exists():
                failures.append(f"episode trace not persisted at {persisted}")
            else:
                payload = json.loads(persisted.read_text(encoding="utf-8"))
                ec = payload.get("endpoint_call_counts", {})
                if (ec.get("reset", 0) < 1) or (ec.get("step", 0) < 1):
                    failures.append(f"endpoint_call_counts insufficient: {ec}")
                else:
                    print(f"[smoke] live_runs trace persisted at {persisted}")
                    print(f"        endpoint_call_counts: {ec}")

    finally:
        print("[smoke] stopping uvicorn")
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_fh.close()

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        print(f"server log saved to {log_path}")
        return 1
    print("dashboard smoke: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
