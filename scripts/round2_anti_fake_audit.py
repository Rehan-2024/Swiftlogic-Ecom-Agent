"""
round2_anti_fake_audit.py - Final authenticity gate for the Round-2 dashboard.

Implements the 10 checks in the Round-2 plan section 10.7. Exit code 0
means every check passed (merge to main is allowed). Non-zero means at
least one authenticity invariant is violated.

Usage:
    python scripts/round2_anti_fake_audit.py [--env-url URL] [--require-trained]

Flags:
    --env-url URL       Probe a live backend (default: http://localhost:7860).
                        Pass --no-live to skip live probes.
    --require-trained   Treat heuristic-fallback artifacts as a hard failure
                        (use for the merge gate). Default: warn-only.
    --no-live           Skip the live-backend probe (use for offline checks).

Designed to be safe to run from CI - reads files, makes a few HTTP calls,
prints a structured report, and never modifies state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DEMO = ROOT / "demo"


@dataclass
class Check:
    id: str
    title: str
    passed: bool = False
    detail: str = ""
    skipped: bool = False
    severity: str = "blocker"   # 'blocker' or 'warn'


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)
    base_url: Optional[str] = None
    require_trained: bool = False
    started_at: str = ""

    def add(self, c: Check) -> None:
        self.checks.append(c)

    def summary(self) -> Dict[str, Any]:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        skipped = sum(1 for c in self.checks if c.skipped)
        failed_blockers = sum(1 for c in self.checks if not c.passed and not c.skipped and c.severity == "blocker")
        failed_warns = sum(1 for c in self.checks if not c.passed and not c.skipped and c.severity == "warn")
        return {
            "total": total,
            "passed": passed,
            "skipped": skipped,
            "failed_blockers": failed_blockers,
            "failed_warns": failed_warns,
            "exit_ok": failed_blockers == 0,
        }


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _commit_ts() -> Optional[float]:
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%ct", "HEAD"],
            cwd=ROOT,
            timeout=5,
        )
        return float(out.decode("ascii").strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_provenance_grpo(report: Report) -> None:
    c = Check(
        id="C1_provenance",
        title="pipeline_manifest.json provenance == grpo_trained",
        severity="blocker" if report.require_trained else "warn",
    )
    manifest = _read_json(ARTIFACTS / "pipeline_manifest.json")
    if manifest is None:
        c.detail = "pipeline_manifest.json missing"
    else:
        prov = (manifest.get("provenance") or "unknown").lower()
        c.passed = prov == "grpo_trained"
        c.detail = f"provenance={prov}"
    report.add(c)


def check_adapter_present(report: Report) -> None:
    c = Check(
        id="C2_adapter",
        title="adapter_status == available AND adapter_config.json exists",
        severity="blocker" if report.require_trained else "warn",
    )
    manifest = _read_json(ARTIFACTS / "pipeline_manifest.json") or {}
    status = (manifest.get("adapter_status") or "unknown").lower()
    cfg = ARTIFACTS / "swiftlogic_grpo_adapter" / "adapter_config.json"
    c.passed = (status == "available") and cfg.exists()
    c.detail = f"adapter_status={status}, adapter_config.json={'present' if cfg.exists() else 'missing'}"
    report.add(c)


def check_artifact_freshness(report: Report) -> None:
    c = Check(
        id="C3_freshness",
        title="every artifact PNG mtime >= manifest.pipeline_run_at - 60s",
        severity="warn",
    )
    manifest = _read_json(ARTIFACTS / "pipeline_manifest.json") or {}
    run_at = manifest.get("pipeline_run_at")
    if not run_at:
        c.detail = "pipeline_run_at not in manifest"
        c.skipped = True
        report.add(c)
        return
    try:
        ts = time.mktime(time.strptime(run_at, "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        c.detail = f"unparseable pipeline_run_at={run_at}"
        c.skipped = True
        report.add(c)
        return
    pngs = [
        "reward_curve.png", "exploration_curve.png", "policy_evolution.png",
        "before_after_comparison.png", "failure_vs_recovery.png", "generalization.png",
    ]
    stale: List[str] = []
    missing: List[str] = []
    for name in pngs:
        p = ARTIFACTS / name
        if not p.exists():
            missing.append(name)
            continue
        if p.stat().st_mtime + 60 < ts:
            stale.append(f"{name}(mtime={time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(p.stat().st_mtime))})")
    if missing or stale:
        bits = []
        if missing:
            bits.append(f"missing={missing}")
        if stale:
            bits.append(f"stale={stale}")
        c.detail = "; ".join(bits)
    else:
        c.passed = True
        c.detail = "all PNGs newer than pipeline_run_at"
    report.add(c)


def check_signature_distinct(report: Report) -> None:
    c = Check(
        id="C4_sig_distinct",
        title="trained policy signature hash distinct from heuristic/wait/random",
        severity="blocker" if report.require_trained else "warn",
    )
    sig = (_read_json(ARTIFACTS / "policy_signature.json") or {}).get("signatures", {})
    if not isinstance(sig, dict) or not sig:
        c.detail = "policy_signature.json missing or empty"
        report.add(c)
        return
    hashes = {name: body.get("hash") for name, body in sig.items() if isinstance(body, dict)}
    distinct = len(set(hashes.values())) == len(hashes)
    trained_hash = hashes.get("trained") or hashes.get("trained_fallback")
    other_hashes = [h for k, h in hashes.items() if k not in {"trained", "trained_fallback"}]
    collision = trained_hash and trained_hash in other_hashes
    c.passed = bool(distinct and not collision and trained_hash)
    c.detail = f"hashes={hashes} distinct={distinct} collision_with_trained={bool(collision)}"
    report.add(c)


def check_composite_improved(report: Report) -> None:
    c = Check(
        id="C5_composite_lift",
        title="composite_score.json after.score > before.score",
        severity="warn",
    )
    cs = _read_json(ARTIFACTS / "composite_score.json") or {}
    before = (cs.get("before") or {}).get("score")
    after = (cs.get("after") or {}).get("score")
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        c.detail = "before/after score not numeric or absent"
    else:
        c.passed = after > before
        c.detail = f"before={before:.4f} after={after:.4f} delta={after-before:+.4f}"
    report.add(c)


def check_generalization_unseen(report: Report) -> None:
    c = Check(
        id="C6_generalization_unseen",
        title="generalization.json includes medplus_pharmacy and stackbase_saas",
        severity="blocker",
    )
    gen = _read_json(ARTIFACTS / "generalization.json") or {}
    eps = gen.get("episodes", []) if isinstance(gen, dict) else []
    cfgs = set()
    for ep in eps:
        cfg = (ep.get("config") or "").replace("\\", "/").lower()
        if "medplus_pharmacy" in cfg:
            cfgs.add("medplus_pharmacy")
        if "stackbase_saas" in cfg:
            cfgs.add("stackbase_saas")
    required = {"medplus_pharmacy", "stackbase_saas"}
    missing = sorted(required - cfgs)
    c.passed = not missing
    c.detail = f"covered={sorted(cfgs)} missing={missing}"
    report.add(c)


def check_action_success_files(report: Report) -> None:
    c = Check(
        id="C7_action_success",
        title="trained restock success_rate > baseline_zero_shot restock success_rate",
        severity="warn",
    )
    base = _read_json(ARTIFACTS / "action_success_baseline_zero_shot.json")
    trained = _read_json(ARTIFACTS / "action_success_trained.json")
    if not base or not trained:
        c.detail = "action_success_baseline_zero_shot.json or action_success_trained.json missing"
        report.add(c)
        return
    b_rate = (base.get("by_action") or {}).get("restock", {}).get("success_rate")
    t_rate = (trained.get("by_action") or {}).get("restock", {}).get("success_rate")
    if not isinstance(b_rate, (int, float)) or not isinstance(t_rate, (int, float)):
        c.detail = "restock success_rate missing in one of the files"
    else:
        c.passed = t_rate > b_rate
        c.detail = f"baseline_restock={b_rate:.3f} trained_restock={t_rate:.3f}"
    report.add(c)


def _is_comment_line(line: str, ext: str) -> bool:
    s = line.lstrip()
    if ext == ".css":
        return s.startswith("/*") or s.startswith("*") or s.startswith("//")
    if ext in {".py"}:
        return s.startswith("#")
    return False


def check_no_neon_css(report: Report) -> None:
    """Forbid neon/glow ONLY in active style/code, not in comments or docstrings."""
    c = Check(
        id="C8_no_neon",
        title="demo/* contains no linear-gradient / text-shadow / glowing box-shadow (excludes comments)",
        severity="blocker",
    )
    pattern_gradient = re.compile(r"\b(linear|radial)-gradient\s*\(", re.IGNORECASE)
    pattern_text_shadow = re.compile(r"text-shadow\s*:[^;]+", re.IGNORECASE)
    pattern_box_shadow = re.compile(r"box-shadow\s*:[^;]+", re.IGNORECASE)
    pattern_glow_value = re.compile(r"\b0\s+0\s+\d+(?:\.\d+)?px\s+(?!0\b)\S+", re.IGNORECASE)

    hits: List[str] = []
    for path in DEMO.rglob("*"):
        if path.suffix.lower() not in {".css", ".py", ".html"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        ext = path.suffix.lower()
        in_block_comment = False
        in_python_string = False
        py_string_quote = ""
        for line_idx, raw in enumerate(text.splitlines(), start=1):
            line = raw
            if ext == ".css":
                if "/*" in line and "*/" not in line:
                    in_block_comment = True
                    continue
                if in_block_comment:
                    if "*/" in line:
                        in_block_comment = False
                    continue
            if _is_comment_line(line, ext):
                continue
            if ext == ".py":
                stripped = line.lstrip()
                if not in_python_string and (stripped.startswith('"""') or stripped.startswith("'''")):
                    in_python_string = True
                    py_string_quote = stripped[:3]
                    if stripped.count(py_string_quote) >= 2:
                        in_python_string = False
                    continue
                if in_python_string:
                    if py_string_quote in line:
                        in_python_string = False
                    continue
            if pattern_gradient.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{line_idx}: gradient -> {line.strip()[:120]}")
            ts = pattern_text_shadow.search(line)
            if ts and "none" not in ts.group(0).lower():
                hits.append(f"{path.relative_to(ROOT)}:{line_idx}: text-shadow -> {ts.group(0).strip()[:120]}")
            bs = pattern_box_shadow.search(line)
            if bs:
                inner = bs.group(0)
                if "none" in inner.lower():
                    continue
                glow = pattern_glow_value.search(inner)
                if glow:
                    hits.append(f"{path.relative_to(ROOT)}:{line_idx}: box-shadow glow -> {bs.group(0).strip()[:120]}")
    c.passed = not hits
    c.detail = f"violations={len(hits)}; first={hits[:5]}" if hits else "no neon/glow patterns found"
    report.add(c)


def check_no_hardcoded_metrics(report: Report) -> None:
    """Reject the specific hardcoded metric strings the old dashboard had baked in."""
    c = Check(
        id="C9_no_hardcoded",
        title="demo/app.py + demo/components.py contain no hardcoded fake metric numbers",
        severity="blocker",
    )
    forbidden = [
        re.compile(r"composite_score_baseline.*0\.35"),  # old default in app.py:67
        re.compile(r"composite_score_trained.*0\.81"),
        re.compile(r"improvement_pct.*\+131%"),
        re.compile(r"\bdummy_score\b"),
        re.compile(r"\bfake_metric\b"),
    ]
    hits: List[str] = []
    for name in ["app.py", "components.py"]:
        path = DEMO / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for pat in forbidden:
            for m in pat.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                hits.append(f"{path.relative_to(ROOT)}:{line_no}: {m.group(0)[:80]}")
    c.passed = not hits
    c.detail = f"violations={len(hits)}; first={hits[:5]}" if hits else "no hardcoded metric strings found"
    report.add(c)


def check_live_run_trace(report: Report) -> None:
    c = Check(
        id="C10_live_run_trace",
        title="at least one live_runs/<id>.json has reset>=1, step>=1, endpoint_base_url set",
        severity="warn",
    )
    runs_dir = ARTIFACTS / "live_runs"
    if not runs_dir.exists():
        c.detail = "artifacts/live_runs/ missing"
        report.add(c)
        return
    files = sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    files = [f for f in files if not f.name.startswith("comparison_")]
    if not files:
        c.detail = "no live_runs/<id>.json present"
        report.add(c)
        return
    latest = files[0]
    trace = _read_json(latest) or {}
    counts = trace.get("endpoint_call_counts", {}) or {}
    base_url = trace.get("endpoint_base_url")
    ok = (
        isinstance(counts.get("reset"), int) and counts.get("reset", 0) >= 1
        and isinstance(counts.get("step"), int) and counts.get("step", 0) >= 1
        and isinstance(base_url, str) and base_url.startswith(("http://", "https://"))
    )
    c.passed = ok
    c.detail = f"latest={latest.name} reset={counts.get('reset')} step={counts.get('step')} base_url={base_url}"
    report.add(c)


def check_live_backend(report: Report, base_url: str) -> None:
    c = Check(
        id="L1_live_backend",
        title=f"live backend at {base_url} responds healthy + 6 tasks",
        severity="blocker",
    )
    try:
        import requests
        h = requests.get(f"{base_url.rstrip('/')}/health", timeout=10)
        if h.status_code != 200:
            c.detail = f"/health -> {h.status_code}"
            report.add(c)
            return
        t = requests.get(f"{base_url.rstrip('/')}/tasks", timeout=10)
        if t.status_code != 200:
            c.detail = f"/tasks -> {t.status_code}"
            report.add(c)
            return
        payload = t.json()
        n = len(payload) if isinstance(payload, list) else None
        if n != 6:
            c.detail = f"/tasks returned {n} entries, expected 6"
        else:
            c.passed = True
            c.detail = f"healthy, /tasks returned 6 entries"
    except Exception as exc:
        c.detail = f"{exc.__class__.__name__}: {exc}"
    report.add(c)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(report: Report, *, with_live: bool, base_url: str) -> int:
    check_provenance_grpo(report)
    check_adapter_present(report)
    check_artifact_freshness(report)
    check_signature_distinct(report)
    check_composite_improved(report)
    check_generalization_unseen(report)
    check_action_success_files(report)
    check_no_neon_css(report)
    check_no_hardcoded_metrics(report)
    check_live_run_trace(report)
    if with_live:
        check_live_backend(report, base_url)

    print()
    print("Round-2 anti-fake audit")
    print("=" * 64)
    print(f"started_at:   {report.started_at}")
    print(f"base_url:     {report.base_url or '(no live probe)'}")
    print(f"strict mode:  {'yes (heuristic == FAIL)' if report.require_trained else 'no (heuristic == WARN)'}")
    print()
    for c in report.checks:
        if c.skipped:
            mark = "SKIP"
        elif c.passed:
            mark = "PASS"
        elif c.severity == "warn":
            mark = "WARN"
        else:
            mark = "FAIL"
        print(f"  [{mark}] {c.id}  {c.title}")
        if c.detail:
            print(f"         -> {c.detail}")
    print()
    summary = report.summary()
    print(f"summary: passed={summary['passed']}/{summary['total']} "
          f"warnings={summary['failed_warns']} blocking_failures={summary['failed_blockers']} "
          f"skipped={summary['skipped']}")
    print(f"merge gate: {'OPEN' if summary['exit_ok'] else 'CLOSED'}")
    return 0 if summary["exit_ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default=os.getenv("ENV_URL", "http://localhost:7860"))
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument("--require-trained", action="store_true")
    args = parser.parse_args()
    report = Report(
        base_url=args.env_url if not args.no_live else None,
        require_trained=bool(args.require_trained),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    return run(report, with_live=not args.no_live, base_url=args.env_url)


if __name__ == "__main__":
    sys.exit(main())
