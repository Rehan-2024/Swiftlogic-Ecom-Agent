"""Refresh the composite-score headline in README.md from artifacts/composite_score.json.

Phase C4 helper: keeps the README headline in sync with whatever
``scripts/run_full_pipeline.py`` (or the Colab notebook) last wrote, so
contributors cannot accidentally publish a stale "0.XX -> 0.YY (+Z%)"
in the top section.

Looks for the line that starts with ``**Composite score`` and rewrites
the inline backticked value plus the ``provenance:`` token in the
italicised note that follows. Other lines (badges, theme paragraph,
artifact previews) are left untouched.

Usage:
    python scripts/refresh_readme_headline.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
COMPOSITE = ROOT / "artifacts" / "composite_score.json"


def main() -> int:
    if not COMPOSITE.exists():
        print(f"composite_score.json not found at {COMPOSITE}; nothing to refresh.")
        return 1
    if not README.exists():
        print(f"README.md not found at {README}.")
        return 1

    data = json.loads(COMPOSITE.read_text(encoding="utf-8"))
    headline = data.get("headline", "0.00 -> 0.00 (+0%)")
    provenance = data.get("provenance", "unknown")

    text = README.read_text(encoding="utf-8")

    headline_pattern = re.compile(
        r"\*\*Composite score \(per \[`artifacts/composite_score\.json`\]\(artifacts/composite_score\.json\)\): `[^`]+`\*\*"
    )
    headline_replacement = (
        f"**Composite score (per [`artifacts/composite_score.json`](artifacts/composite_score.json)): "
        f"`{headline}`**"
    )
    if not headline_pattern.search(text):
        print("Could not locate the composite-score headline line in README.md.")
        return 1
    text = headline_pattern.sub(headline_replacement, text)

    provenance_pattern = re.compile(r"provenance: `[^`]+`")
    text = provenance_pattern.sub(f"provenance: `{provenance}`", text, count=1)

    README.write_text(text, encoding="utf-8")
    print(f"README headline pinned -> {headline}  (provenance={provenance})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
