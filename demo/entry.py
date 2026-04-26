"""HF Spaces entrypoint - mounts the Gradio dashboard under FastAPI on a single port.

HF Spaces exposes exactly one port. We mount the Gradio Blocks at /demo
so the existing FastAPI endpoints (/, /health, /reset, /step, /state,
/tasks, /grader, /config) keep working unchanged at the same port.

We also mount StaticFiles for /static/demo/photos and /static/demo/artifacts
so the dashboard can reference brand photos and pipeline figures by URL
instead of base64-encoding them on every render. This keeps page loads
snappy and avoids any "live fetch" feel - all assets are local.

Local run:
    uvicorn demo.entry:app --host 0.0.0.0 --port 7860
HF Dockerfile CMD does the same thing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from server.app import app as fastapi_app  # noqa: E402
from demo.app import demo as gradio_blocks  # noqa: E402

DEMO_PATH = os.getenv("DEMO_PATH", "/demo")

_PHOTOS_DIR = ROOT / "demo" / "assets" / "photos"
_ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))

if _PHOTOS_DIR.exists():
    fastapi_app.mount(
        "/static/demo/photos",
        StaticFiles(directory=str(_PHOTOS_DIR), check_dir=False),
        name="demo_photos",
    )
if _ARTIFACTS_DIR.exists():
    fastapi_app.mount(
        "/static/demo/artifacts",
        StaticFiles(directory=str(_ARTIFACTS_DIR), check_dir=False),
        name="demo_artifacts",
    )

app = gr.mount_gradio_app(fastapi_app, gradio_blocks, path=DEMO_PATH)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
