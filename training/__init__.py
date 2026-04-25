"""Training-support package for Part B / B+ of the OpenEnv hackathon.

Each module is import-safe without ``torch`` / ``trl`` / ``unsloth`` so
the pipeline / smoke tests can exercise the environment-facing logic
without an LLM stack. Heavy dependencies (``transformers``, ``peft``,
``unsloth``) are imported lazily inside the functions that need them.
"""
