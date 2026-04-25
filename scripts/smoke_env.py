"""Live-server smoke check against a deployed CommerceOps v2 endpoint.

This script intentionally lives under ``scripts/`` (not ``tests/`` and not the
repo root) so ``pytest`` never tries to import it at collection time -- a
previous version under the ``test_env.py`` name would run the network calls
below during collection and could hang the test suite.

Run manually::

    python scripts/smoke_env.py [https://your-space-url]
"""

from __future__ import annotations

import sys

import requests


DEFAULT_URL = "https://swiftlogic-e-commerce-agent.hf.space"


def main(url: str = DEFAULT_URL) -> int:
    print("RESET:", requests.post(f"{url}/reset", json={}).status_code)
    print("STEP:", requests.post(
        f"{url}/step",
        json={"action_type": "wait"},
    ).status_code)
    print("STATE:", requests.get(f"{url}/state").status_code)
    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    sys.exit(main(target))
