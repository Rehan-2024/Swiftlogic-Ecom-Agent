"""Smoke-test the project Dockerfile end-to-end.

Steps performed (all gated on the local docker daemon being reachable):
  1. ``docker build -t swiftlogic-openenv:smoke .``
  2. ``docker run --rm -d -p 7860:7860 --name swiftlogic-openenv-smoke swiftlogic-openenv:smoke``
  3. Poll ``http://127.0.0.1:7860/health`` until 200 (max ~30s).
  4. Hit ``/tasks`` to ensure the OpenEnv contract is live (≥6 tasks).
  5. Tear the container down.

Writes ``artifacts/docker_smoke.log`` with the full transcript so we have a
checked-in record. The script returns non-zero if any stage fails so it can be
wired into CI / the run_full_pipeline orchestrator.

This script is intentionally Python (no shell-specific syntax) so it works on
both Windows PowerShell and Linux/macOS.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
LOG_PATH = ARTIFACTS / "docker_smoke.log"

IMAGE_TAG = "swiftlogic-openenv:smoke"
CONTAINER_NAME = "swiftlogic-openenv-smoke"


def _log(handle, msg: str) -> None:
    print(msg)
    handle.write(msg + "\n")
    handle.flush()


def _run(handle, cmd: list[str], *, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
    _log(handle, f"$ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=False,
    )
    if proc.stdout:
        handle.write(proc.stdout)
    if proc.stderr:
        handle.write(proc.stderr)
    handle.flush()
    if check and proc.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)} (exit={proc.returncode})")
    return proc


def _docker_available(handle) -> bool:
    if shutil.which("docker") is None:
        _log(handle, "docker CLI not on PATH; skipping smoke test.")
        return False
    proc = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        _log(handle, "docker daemon is not reachable; skipping live build.")
        if proc.stderr:
            handle.write(proc.stderr)
        return False
    _log(handle, f"docker daemon ready: server={proc.stdout.strip()}")
    return True


def _wait_health(handle, port: int, timeout_s: float = 30.0) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    _log(handle, f"GET /health -> 200 :: {body.strip()[:200]}")
                    return True
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = exc
        time.sleep(1.0)
    _log(handle, f"/health did not return 200 within {timeout_s}s; last_err={last_err}")
    return False


def _verify_tasks(handle, port: int) -> bool:
    url = f"http://127.0.0.1:{port}/tasks"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        _log(handle, f"/tasks request failed: {exc}")
        return False
    tasks = payload.get("tasks", []) if isinstance(payload, dict) else payload
    _log(handle, f"GET /tasks -> {len(tasks)} task descriptors")
    if len(tasks) < 6:
        _log(handle, "expected at least 6 tasks (3 training + 3 evaluation_only)")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--keep", action="store_true", help="don't tear the container down on success")
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as handle:
        handle.write("# docker_smoke run\n")
        handle.write(f"timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        handle.write(f"image_tag: {IMAGE_TAG}\n")
        handle.write(f"container: {CONTAINER_NAME}\n\n")

        if not _docker_available(handle):
            handle.write("\nresult: SKIPPED (no docker daemon)\n")
            return 0

        with contextlib.suppress(subprocess.CalledProcessError):
            subprocess.run(
                ["docker", "rm", "-f", CONTAINER_NAME],
                capture_output=True,
                text=True,
            )

        _run(handle, ["docker", "build", "-t", IMAGE_TAG, "."])

        run_cmd = [
            "docker",
            "run",
            "--rm",
            "-d",
            "-p",
            f"{args.port}:7860",
            "--name",
            CONTAINER_NAME,
            IMAGE_TAG,
        ]
        _run(handle, run_cmd)

        ok_health = _wait_health(handle, args.port)
        ok_tasks = ok_health and _verify_tasks(handle, args.port)

        if not args.keep:
            with contextlib.suppress(subprocess.CalledProcessError):
                _run(handle, ["docker", "stop", CONTAINER_NAME], check=False)

        success = ok_health and ok_tasks
        handle.write(f"\nresult: {'OK' if success else 'FAIL'}\n")
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
