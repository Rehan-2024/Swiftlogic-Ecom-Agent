"""HTTP client for the OpenEnv FastAPI backend.

Single source of truth for every dashboard call to /reset, /step, /state,
/tasks, /grader, /config, /health. Wraps timeout, retry, and structured
error reporting so the UI never sees a raw exception.

Per the Round-2 plan section 1, every visible value in the UI must come
from these calls or from artifacts/. There is no mocked-mode fallback.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("commerceops.demo.backend_client")

DEFAULT_ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
DEFAULT_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15"))
DEFAULT_RETRIES = int(os.getenv("REQUEST_RETRIES", "1"))


@dataclass
class BackendError(Exception):
    """Structured backend failure reported into the UI as a card, not a stack trace."""
    kind: str            # 'unreachable' | 'timeout' | 'http_error' | 'malformed'
    detail: str
    status_code: Optional[int] = None
    endpoint: str = ""

    def __str__(self) -> str:
        return f"[{self.kind}] {self.endpoint} -> {self.detail}"


@dataclass
class BackendClient:
    base_url: str = DEFAULT_ENV_URL
    timeout: float = DEFAULT_TIMEOUT
    retries: int = DEFAULT_RETRIES
    call_counts: Dict[str, int] = field(default_factory=lambda: {
        "reset": 0, "step": 0, "state": 0, "grader": 0,
        "tasks": 0, "config": 0, "health": 0,
    })

    def _bump(self, name: str) -> None:
        self.call_counts[name] = self.call_counts.get(name, 0) + 1

    def _request(self, method: str, path: str, *, json_body: Any = None) -> Any:
        url = f"{self.base_url.rstrip('/')}{path}"
        attempts = max(1, self.retries + 1)
        last_exc: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                resp = requests.request(
                    method,
                    url,
                    json=json_body,
                    timeout=self.timeout,
                )
            except requests.Timeout as exc:
                last_exc = exc
                if attempt < attempts:
                    time.sleep(0.4 * attempt)
                    continue
                raise BackendError(
                    kind="timeout",
                    detail=f"timeout after {self.timeout}s",
                    endpoint=path,
                ) from exc
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < attempts:
                    time.sleep(0.4 * attempt)
                    continue
                raise BackendError(
                    kind="unreachable",
                    detail=exc.__class__.__name__,
                    endpoint=path,
                ) from exc

            if resp.status_code >= 400:
                raise BackendError(
                    kind="http_error",
                    detail=f"{resp.status_code} {resp.reason}: {resp.text[:240]}",
                    status_code=resp.status_code,
                    endpoint=path,
                )
            try:
                return resp.json()
            except ValueError as exc:
                raise BackendError(
                    kind="malformed",
                    detail="response was not JSON",
                    endpoint=path,
                ) from exc
        raise BackendError(
            kind="unreachable",
            detail=str(last_exc) if last_exc else "exhausted retries",
            endpoint=path,
        )

    # Endpoints ----------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        self._bump("health")
        return self._request("GET", "/health")

    def tasks(self) -> Any:
        self._bump("tasks")
        return self._request("GET", "/tasks")

    def reset(self, seed: int = 42) -> Dict[str, Any]:
        self._bump("reset")
        return self._request("POST", "/reset", json_body={"seed": int(seed)})

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._bump("step")
        return self._request("POST", "/step", json_body=action)

    def state(self) -> Dict[str, Any]:
        self._bump("state")
        return self._request("GET", "/state")

    def grader(self) -> Dict[str, Any]:
        self._bump("grader")
        return self._request("POST", "/grader", json_body={})

    def config(self, business_id: str, seed: int = 42) -> Dict[str, Any]:
        self._bump("config")
        return self._request("POST", "/config", json_body={"business_id": business_id, "seed": int(seed)})

    # Composite checks ---------------------------------------------------

    def quick_self_check(self) -> Dict[str, Any]:
        """Return a structured snapshot used by the UI startup banner."""
        report: Dict[str, Any] = {
            "base_url": self.base_url,
            "ok": False,
            "errors": [],
            "tasks_count": None,
            "version": None,
        }
        try:
            self.health()
        except BackendError as exc:
            report["errors"].append(str(exc))
            return report
        try:
            tasks = self.tasks()
            report["tasks_count"] = len(tasks) if isinstance(tasks, list) else None
        except BackendError as exc:
            report["errors"].append(str(exc))
        report["ok"] = not report["errors"]
        return report
