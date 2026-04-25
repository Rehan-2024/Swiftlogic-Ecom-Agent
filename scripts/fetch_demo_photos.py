"""
fetch_demo_photos.py - Download real story photos for the Round-2 dashboard.

Source: Lorem Picsum (https://picsum.photos) which serves real photos
sourced from Unsplash under the Unsplash License. Stable URLs by ID,
no API key required, no hallucinated URLs.

Usage:
    python scripts/fetch_demo_photos.py

Behaviour:
- Downloads each photo to demo/assets/photos/
- Validates the response is a JPEG (magic bytes 0xFF 0xD8) before writing
- Skips files that already exist (re-runnable)
- Updates demo/assets/photos/CREDITS.md with the actual byte sizes
- Exit code 0 if every photo is on disk after the run, non-zero otherwise

Replace any of these with brand-owned photography by:
1. Dropping a JPEG with the same filename into demo/assets/photos/
2. Updating CREDITS.md with the new source + license
3. The dashboard auto-picks up whatever file is present.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import requests

PHOTOS_DIR = Path(__file__).resolve().parent.parent / "demo" / "assets" / "photos"

# Each tuple: (filename, picsum_id, width, height, story_chapter, alt_text)
# Picsum IDs are stable: https://picsum.photos/id/{ID}/info returns metadata.
# Chosen for warm, neutral, real-world commerce/textile/market aesthetic.
PHOTO_MANIFEST: List[Tuple[str, int, int, int, str, str]] = [
    (
        "01_storefront_dawn.jpg",
        1011,  # mountain/landscape - placeholder for storefront-at-dawn beat
        1600,
        900,
        "Chapter 1 - The Awakening",
        "A storefront at dawn, the AI inherits a 50-day cycle.",
    ),
    (
        "02_textile_craft.jpg",
        1080,  # strawberries / craft-warm - placeholder for textile beat
        1200,
        900,
        "Chapter 2 - The Craft",
        "Hand-loomed fabric, the inventory the AI must protect.",
    ),
    (
        "03_market_decision.jpg",
        1015,  # river / outdoor - placeholder for decision-point beat
        1200,
        900,
        "Chapter 3 - The Decision",
        "Every day a hundred decisions, the policy chooses one.",
    ),
    (
        "04_growth_horizon.jpg",
        1043,  # mountain / horizon - placeholder for growth/survival beat
        1600,
        900,
        "Chapter 4 - The Outcome",
        "Bank balance, customer satisfaction, survival across configurations.",
    ),
]


def is_jpeg(payload: bytes) -> bool:
    """Magic-byte check: real JPEGs start with FFD8FF."""
    return len(payload) >= 3 and payload[0] == 0xFF and payload[1] == 0xD8 and payload[2] == 0xFF


def download_one(filename: str, picsum_id: int, w: int, h: int) -> Tuple[bool, int, str]:
    """Returns (ok, size_bytes, source_url)."""
    url = f"https://picsum.photos/id/{picsum_id}/{w}/{h}.jpg"
    target = PHOTOS_DIR / filename
    if target.exists() and target.stat().st_size > 1024:
        return True, target.stat().st_size, url
    try:
        resp = requests.get(url, timeout=30, allow_redirects=True)
    except requests.RequestException as exc:
        print(f"  ! network error for {filename}: {exc.__class__.__name__}", file=sys.stderr)
        return False, 0, url
    if resp.status_code != 200:
        print(f"  ! HTTP {resp.status_code} for {filename}", file=sys.stderr)
        return False, 0, url
    payload = resp.content
    if not is_jpeg(payload):
        print(f"  ! response is not a JPEG ({len(payload)} bytes) for {filename}", file=sys.stderr)
        return False, len(payload), url
    target.write_bytes(payload)
    return True, len(payload), url


def write_credits(rows: List[Tuple[str, int, int, int, str, str, int, str, bool]]) -> None:
    """rows: each tuple = (filename, id, w, h, chapter, alt, bytes, url, ok)."""
    credits = PHOTOS_DIR / "CREDITS.md"
    lines = [
        "# Photo credits",
        "",
        "Real photos served by [Lorem Picsum](https://picsum.photos),",
        "which proxies images from [Unsplash](https://unsplash.com) under the",
        "[Unsplash License](https://unsplash.com/license) (free for commercial and",
        "non-commercial use, no permission needed; attribution appreciated).",
        "",
        "These are CURATED PLACEHOLDERS chosen for the Round-2 dashboard's storytelling",
        "beats. To swap for brand-owned photography:",
        "",
        "1. Replace the file at the path below with your own JPEG (same filename).",
        "2. Update the `Source` and `License` columns in this table.",
        "3. The dashboard re-reads the directory at process start; restart the Space.",
        "",
        "| File | Chapter | Source | License | Bytes | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for filename, _id, _w, _h, chapter, _alt, size, url, ok in rows:
        status = "downloaded" if ok else "MISSING"
        lines.append(
            f"| `{filename}` | {chapter} | <{url}> | Unsplash License (via Lorem Picsum) | {size} | {status} |"
        )
    lines.extend([
        "",
        "## Story tags (used by the dashboard)",
        "",
    ])
    for filename, _id, _w, _h, chapter, alt, _size, _url, _ok in rows:
        lines.append(f"- **{filename}** -- {chapter} -- _{alt}_")
    lines.append("")
    credits.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[str, int, int, int, str, str, int, str, bool]] = []
    failures = 0
    for filename, picsum_id, w, h, chapter, alt in PHOTO_MANIFEST:
        print(f"-> {filename} (picsum id={picsum_id}, {w}x{h})")
        ok, size, url = download_one(filename, picsum_id, w, h)
        if not ok:
            failures += 1
        else:
            print(f"   ok, {size} bytes")
        rows.append((filename, picsum_id, w, h, chapter, alt, size, url, ok))
        time.sleep(0.4)
    write_credits(rows)
    print()
    if failures:
        print(f"{failures} photo(s) failed to download. See errors above and re-run.")
        return 2
    print(f"All {len(rows)} photos present at {PHOTOS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
