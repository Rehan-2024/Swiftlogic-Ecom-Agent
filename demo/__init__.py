"""Round-2 Gradio dashboard for Swiftlogic CommerceOps v2.

Modules:
  app             - Gradio Blocks composition (Tabs only).
  backend_client  - HTTP wrappers for /reset|/step|/state|/grader|/tasks|/config.
  policy          - Model loading + action inference (reuses inference.build_step_trace).
  episode_runner  - Orchestrates baseline_wait_only | baseline_zero_shot | trained.
  artifact_loader - Provenance + freshness checks + JSON loaders.
  components      - Reusable UI fragments (cards, tables, banners).
  story_tab       - Storytelling tab driven by demo/assets/photos/story.json.
  entry           - HF Spaces entrypoint that mounts Gradio under FastAPI.
"""
