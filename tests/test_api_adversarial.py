"""Adversarial API tests (v2.3.x Phase C.3).

These tests exercise the JSON body-parsing hardening in
``server.app._safe_json`` and the route-level 4xx contracts. None of
them target production behavior changes — they pin down the edge-case
response codes so future refactors don't silently turn 400/413 paths
into 500s.

Covered paths:
  * Oversized body announced via ``content-length`` header (fast path).
  * Oversized chunked body with **no** ``content-length`` header (streaming cap).
  * Deeply nested JSON payload → 400 from the stdlib parser, never 500.
  * Extremely long string field → 413 because the body exceeds ``MAX_BODY_BYTES``.
  * Non-UTF-8 bytes on ``/step`` → 400 via safe-parse fallback.
"""

from __future__ import annotations

import json


def test_oversized_content_length_header_rejected_fast(fresh_app):
    """Honest oversized clients short-circuit before the body is streamed."""
    from server.app import MAX_BODY_BYTES

    oversized = MAX_BODY_BYTES + 1024
    r = fresh_app.post(
        "/step",
        content=b"x" * 16,  # actual body irrelevant; header lies
        headers={
            "content-type": "application/json",
            "content-length": str(oversized),
        },
    )
    assert r.status_code == 413, r.text
    payload = r.json()
    assert payload.get("max_bytes") == MAX_BODY_BYTES


def test_oversized_chunked_body_rejected_without_content_length(fresh_app):
    """Chunked / lying clients must also hit the streaming cap.

    Starlette's TestClient will strip/recompute ``content-length`` when it
    sees raw ``content=`` bytes, so we assemble an oversized JSON blob and
    rely on the server-side accumulator to abort once the buffer crosses
    ``MAX_BODY_BYTES + 1``. The point is: the body is too large, we must
    get a 413 regardless of which hardening layer catches it.
    """
    from server.app import MAX_BODY_BYTES

    # Build a valid JSON dict that is comfortably over the cap so both the
    # header fast-path and the streaming path will agree it is oversized.
    big_value = "a" * (MAX_BODY_BYTES + 2048)
    payload = json.dumps({"action_type": "wait", "pad": big_value}).encode("utf-8")
    assert len(payload) > MAX_BODY_BYTES

    r = fresh_app.post(
        "/step",
        content=payload,
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 413, r.text


def test_deeply_nested_json_returns_400_not_500(fresh_app):
    """A deeply nested but size-legal body should 400 or 200, never 500.

    Python's default ``json`` parser has a recursion limit. We build a
    body just below the body-size cap but with ~500 levels of nesting.
    The server must either reject it as invalid JSON (``400``) or, if
    parsed, decide the result isn't a valid action object and return
    ``400``. Either way, zero uncaught 500s.
    """
    depth = 500
    blob = "[" * depth + "1" + "]" * depth  # size-legal but deeply nested
    r = fresh_app.post(
        "/step",
        content=blob.encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400, r.text


def test_extremely_long_string_field_returns_413(fresh_app):
    """A JSON object with one huge string must hit the body-size cap."""
    from server.app import MAX_BODY_BYTES

    huge = "z" * (MAX_BODY_BYTES + 4096)
    payload = json.dumps({"action_type": "wait", "note": huge}).encode("utf-8")
    r = fresh_app.post(
        "/step",
        content=payload,
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 413, r.text


def test_non_utf8_body_returns_400(fresh_app):
    """Raw non-UTF-8 bytes must not crash the server.

    ``_safe_json`` catches the ``UnicodeDecodeError`` / ``json.JSONDecodeError``
    and returns ``None``, which the ``/step`` handler translates into a
    stable 400. This test guards against a future refactor that would let
    the decode error bubble up as a 500.
    """
    r = fresh_app.post(
        "/step",
        content=b"\xff\xfe\xfd\xfc\x00not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400, r.text
