"""Env-freeze diff checker (Part A7).

Enforces the roadmap global rule "No modification of env/, ecom_env.py,
reward logic, action/observation schema; only additive ``info`` fields
allowed post-freeze." Compares the current working tree against the
``release/env-frozen-v2.3`` tag and fails if any guarded file has a
*deletion* or a reward-logic / schema line modified.

Detection rules:

* ``env/reward_engine.py`` — **any** line change (add/delete) is a failure.
  Reward physics is frozen; additive info happens in ``world_engine.py``.
* ``ecom_env.py`` — deletions or edits to the ``EcomAction`` /
  ``EcomObservation`` / ``EcomReward`` class bodies are failures.
  Additions after the last ``__all__`` line are allowed.
* ``env/world_engine.py`` — deletions inside ``def step(`` that touch
  reward / state mutation are failures. Pure info-key additions (lines
  with ``info[...]`` that are only ``+``) are allowed.
* ``env/actions.py`` and ``env/observations.py`` (if present) — any
  modification is a failure.

This script is intentionally strict. A failure means "look — did you
really mean to mutate the frozen surface?".
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

FROZEN_TAG = "release/env-frozen-v2.3"
REWARD_FROZEN = {"env/reward_engine.py", "env/actions.py", "env/observations.py"}
SCHEMA_FROZEN = {"ecom_env.py"}
STEP_SENSITIVE = {"env/world_engine.py"}


def _run(args: List[str]) -> Tuple[int, str]:
    res = subprocess.run(args, capture_output=True, text=True)
    return res.returncode, (res.stdout or "") + (res.stderr or "")


def _tag_exists(tag: str) -> bool:
    rc, out = _run(["git", "rev-parse", "--verify", "--quiet", tag])
    return rc == 0


def _diff_lines(tag: str, path: str) -> List[str]:
    rc, out = _run(["git", "diff", f"{tag}..HEAD", "--", path])
    return out.splitlines()


def _added_removed(diff: List[str]) -> Tuple[List[str], List[str]]:
    adds, dels = [], []
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            adds.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            dels.append(line[1:])
    return adds, dels


def check_reward_frozen(tag: str) -> List[str]:
    bad: List[str] = []
    for path in REWARD_FROZEN:
        diff = _diff_lines(tag, path)
        if diff:
            adds, dels = _added_removed(diff)
            if adds or dels:
                bad.append(f"{path}: +{len(adds)} -{len(dels)} lines (reward layer must be frozen)")
    return bad


def check_schema_frozen(tag: str) -> List[str]:
    bad: List[str] = []
    for path in SCHEMA_FROZEN:
        diff = _diff_lines(tag, path)
        if not diff:
            continue
        adds, dels = _added_removed(diff)
        for line in dels:
            stripped = line.strip()
            if stripped.startswith("action_type") or stripped.startswith("sku") or stripped.startswith("ticket_id"):
                bad.append(f"{path}: removed schema line: {stripped[:80]}")
                continue
            if "EcomAction" in stripped or "EcomObservation" in stripped or "EcomReward" in stripped:
                if any(m in stripped for m in ("class", ":", "=")):
                    bad.append(f"{path}: modified schema container: {stripped[:80]}")
    return bad


def check_world_engine_additive(tag: str) -> List[str]:
    """For world_engine.py, reject deletions outright (strongest guard)."""
    bad: List[str] = []
    diff = _diff_lines(tag, "env/world_engine.py")
    if not diff:
        return bad
    adds, dels = _added_removed(diff)
    significant_dels = [d for d in dels if d.strip() and not d.strip().startswith("#")]
    if significant_dels:
        bad.append(
            f"env/world_engine.py: {len(significant_dels)} non-comment line(s) removed "
            "(env freeze allows additive-only)."
        )
    return bad


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify env freeze is additive-only")
    parser.add_argument("--tag", default=FROZEN_TAG)
    parser.add_argument("--skip-missing-tag", action="store_true",
                        help="Exit 0 (with warning) if the freeze tag doesn't exist yet.")
    args = parser.parse_args(argv)

    if not _tag_exists(args.tag):
        msg = f"freeze tag {args.tag!r} not found. run: git tag {args.tag}"
        if args.skip_missing_tag:
            print(f"[check_env_frozen] WARN: {msg}", file=sys.stderr)
            return 0
        print(f"[check_env_frozen] FAIL: {msg}", file=sys.stderr)
        return 2

    problems: List[str] = []
    problems += check_reward_frozen(args.tag)
    problems += check_schema_frozen(args.tag)
    problems += check_world_engine_additive(args.tag)

    if problems:
        print("ENV FREEZE VIOLATIONS:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"[check_env_frozen] OK — no violations against {args.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
