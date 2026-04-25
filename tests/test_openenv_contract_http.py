"""OpenEnv wire-level HTTP contract test.

Spawns a real Uvicorn process on a random port and exercises every contract
endpoint (``/reset``, ``/step``, ``/state``, ``/tasks``, ``/grader``,
``/health``) through ``requests`` — i.e. the exact surface the OpenEnv
validator and any downstream trainer talks to. Complements the in-process
``TestClient`` tests in ``tests/test_api_contract.py``.

Requires ``requests`` and a free TCP port on ``127.0.0.1``; skips if Uvicorn
binding fails.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_ready(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.25)
    return False


@pytest.fixture(scope="module")
def live_server():
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["COMMERCEOPS_CONFIG"] = str(ROOT / "configs" / "siyaani_fashion.json")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        if not _wait_ready(url):
            proc.terminate()
            out, _ = proc.communicate(timeout=5)
            pytest.skip(
                f"uvicorn did not come up within 30s. Tail:\n{(out or b'').decode('utf-8', 'ignore')[-2000:]}"
            )
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _log(records: list, line: str) -> None:
    records.append(line)


def test_full_contract_roundtrip(live_server):
    url = live_server
    records: list[str] = []
    _log(records, f"# OpenEnv contract HTTP test — {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    _log(records, f"target={url}")

    # /health
    r = requests.get(f"{url}/health", timeout=5)
    _log(records, f"GET /health -> {r.status_code} {r.text.strip()}")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

    # /tasks (should be at least 3; becomes 6 after A3)
    r = requests.get(f"{url}/tasks", timeout=5)
    _log(records, f"GET /tasks -> {r.status_code} count={len(r.json())}")
    assert r.status_code == 200
    tasks = r.json()
    assert isinstance(tasks, list)
    assert len(tasks) >= 3
    task_ids = {t["id"] for t in tasks}
    assert {"triage_task", "inventory_task", "profit_task"}.issubset(task_ids)

    # /reset
    r = requests.post(f"{url}/reset", json={"seed": 2026}, timeout=10)
    _log(records, f"POST /reset seed=2026 -> {r.status_code}")
    assert r.status_code == 200
    body = r.json()
    assert "observation" in body
    obs = body["observation"]
    # Must contain the documented keys
    for key in ("current_day", "bank_balance", "inventory", "active_tickets"):
        assert key in obs, f"missing obs key {key}"

    # /state (read-only)
    r = requests.get(f"{url}/state", timeout=5)
    _log(records, f"GET /state -> {r.status_code}")
    assert r.status_code == 200
    obs_again = r.json()["observation"]
    assert obs_again["bank_balance"] == obs["bank_balance"]

    # /step — drive each of 6 action types
    actions = [
        {"action_type": "wait"},
        {"action_type": "restock", "sku": "cotton_set", "quantity": 5},
        {"action_type": "ad_spend", "sku": "silk_kurta", "budget": 200.0},
        {"action_type": "negotiate", "sku": "cotton_set", "quantity": 10},
        {"action_type": "set_price", "sku": "cotton_set", "price": 1100.0},
        {"action_type": "refund", "ticket_id": "__nonexistent__"},
    ]
    for action in actions:
        r = requests.post(f"{url}/step", json=action, timeout=10)
        _log(records, f"POST /step {action['action_type']} -> {r.status_code}")
        assert r.status_code == 200, r.text
        body = r.json()
        for key in ("observation", "reward", "done", "info"):
            assert key in body, f"missing /step key {key}"
        # info must carry reward_breakdown
        assert "reward_breakdown" in body["info"], "reward_breakdown missing from info"

    # /grader
    r = requests.post(f"{url}/grader", timeout=10)
    _log(records, f"POST /grader -> {r.status_code}")
    assert r.status_code == 200
    scores = r.json()["scores"]
    assert len(scores) >= 3
    for entry in scores:
        score = float(entry["score"])
        assert 0.01 <= score <= 0.99, f"score {entry['task_id']}={score} out of clamp"

    # bogus action — should 400, not crash
    r = requests.post(f"{url}/step", json={"action_type": "hack_the_gibson"}, timeout=5)
    _log(records, f"POST /step hack_the_gibson -> {r.status_code}")
    assert r.status_code == 400

    # Persist the trace
    log_path = ROOT / "artifacts" / "openenv_contract.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(records) + "\n", encoding="utf-8")
