"""Episode rollout with strict action validation and safe fallback (Part B2).

The trainer's ``reward_fn`` calls ``rollout_episode`` once per sampled
completion. It parses the model output into a list of candidate
``EcomAction`` dicts, validates each one with ``EcomAction.model_validate``,
substitutes a ``{"action_type": "wait"}`` on any parse / validation
failure, and runs the full episode against a fresh ``EcomEnv`` so the
grader scores see a realistic final state.

Format compliance is tracked explicitly — every step reports whether
the model emitted a valid JSON action, whether schema validation passed,
and whether the fallback fired. These signals feed the format-reward
term (Part B3).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import ValidationError

from ecom_env import EcomAction, EcomEnv, EcomObservation


# ---------------------------------------------------------------------------
# Action extraction
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def extract_action_json(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Extract the first plausible JSON action dict from ``text``.

    Returns ``(parsed_or_None, raw_snippet)`` — the snippet is whatever
    substring we attempted to parse, useful for logging failures.
    """
    if not text:
        return None, ""
    # Prefer an explicit "Action:" marker to tolerate Thought+Action format.
    marker = text.rfind("Action:")
    if marker >= 0:
        text = text[marker + len("Action:") :]
    for match in _JSON_OBJECT_RE.finditer(text):
        snippet = match.group(0)
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict) and "action_type" in obj:
                return obj, snippet
        except (json.JSONDecodeError, ValueError):
            continue
    return None, text[:200]


def validate_action(candidate: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Validate ``candidate`` through ``EcomAction``; fall back to wait.

    Returns ``(action_dict, was_valid, error)``.
    ``action_dict`` is always a dict safely executable by ``EcomEnv.step``.
    """
    if candidate is None:
        return {"action_type": "wait"}, False, "no_action_found"
    try:
        validated = EcomAction.model_validate(candidate)
        return validated.model_dump(), True, None
    except ValidationError as exc:
        return {"action_type": "wait"}, False, str(exc.errors()[:1])
    except Exception as exc:  # noqa: BLE001
        return {"action_type": "wait"}, False, f"unexpected: {exc!r}"


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    action: Dict[str, Any]
    reward: float
    was_valid: bool
    was_fallback: bool
    error: Optional[str]
    raw_output: Optional[str]
    info_keys: Dict[str, Any]


@dataclass
class EpisodeRecord:
    seed: int
    config_path: str
    steps: List[StepRecord] = field(default_factory=list)
    initial_obs: Optional[EcomObservation] = None
    final_obs: Optional[EcomObservation] = None
    grader_scores: Dict[str, float] = field(default_factory=dict)
    total_reward: float = 0.0

    @property
    def format_compliance(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1.0 for s in self.steps if s.was_valid) / len(self.steps)

    @property
    def fallback_rate(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1.0 for s in self.steps if s.was_fallback) / len(self.steps)


ActionProducer = Callable[[EcomObservation, Dict[str, Any]], Tuple[Dict[str, Any], Optional[str]]]
"""Signature: (obs, episode_state) -> (candidate_action_dict_or_None, raw_text)."""


def rollout_episode(
    producer: ActionProducer,
    seed: int,
    *,
    config_path: str = "configs/siyaani_fashion.json",
    max_steps: Optional[int] = None,
) -> EpisodeRecord:
    """Run one episode from ``seed`` driven by ``producer``.

    ``producer`` returns ``(candidate_action, raw_text)``. If ``candidate_action``
    is None or fails schema validation, we substitute ``wait`` and mark
    the step as a fallback. The full reward signal from ``EcomEnv.step``
    is retained so GRPO sees authentic physics.
    """
    from ecom_env import (
        grade_inventory_task,
        grade_profit_task,
        grade_triage_task,
    )

    env = EcomEnv(config_path=config_path)
    obs = env.reset(seed=seed)
    rec = EpisodeRecord(seed=seed, config_path=config_path, initial_obs=obs)
    limit = max_steps or env.world_engine.config["episode"]["max_steps"]
    episode_state: Dict[str, Any] = {"seed": seed, "step": 0}
    done = False
    step_idx = 0
    while not done and step_idx < limit:
        candidate, raw = producer(obs, episode_state)
        action, valid, err = validate_action(candidate)
        fallback = (not valid) or (action.get("action_type") == "wait" and candidate is not None and candidate.get("action_type") != "wait")
        obs, reward, done, info = env.step(action)
        r = float(reward.value) if hasattr(reward, "value") else float(reward)
        rec.steps.append(
            StepRecord(
                step=step_idx + 1,
                action=action,
                reward=round(r, 6),
                was_valid=valid,
                was_fallback=not valid,
                error=err,
                raw_output=raw[:500] if raw else None,
                info_keys={
                    "action_quality": info.get("action_quality"),
                    "strategy_phase": info.get("strategy_phase"),
                    "intent": info.get("intent"),
                },
            )
        )
        rec.total_reward += r
        step_idx += 1
        episode_state["step"] = step_idx
    rec.final_obs = obs

    # 3 training graders (A3 evaluation-only graders are computed by caller
    # if they want the full scoreboard — we keep rollout lean).
    ctx = env.grader_context
    rec.grader_scores = {
        "triage_task": float(grade_triage_task(rec.initial_obs, rec.final_obs, context=ctx)),
        "inventory_task": float(grade_inventory_task(rec.initial_obs, rec.final_obs, context=ctx)),
        "profit_task": float(grade_profit_task(rec.initial_obs, rec.final_obs, context=ctx)),
    }
    return rec


def full_grader_scores(rec: EpisodeRecord) -> Dict[str, float]:
    """Compute all 6 grader scores (training + evaluation-only)."""
    from ecom_env import (
        EcomEnv,
        grade_competitor_response_task,
        grade_crisis_recovery_task,
        grade_inventory_task,
        grade_profit_task,
        grade_stability_task,
        grade_triage_task,
    )

    # Re-derive context from a scratch env of the same config — context
    # is config-only, not seed-dependent.
    env = EcomEnv(config_path=rec.config_path)
    env.reset(seed=rec.seed)
    ctx = env.grader_context
    i, f = rec.initial_obs, rec.final_obs
    return {
        "triage_task": float(grade_triage_task(i, f, context=ctx)),
        "inventory_task": float(grade_inventory_task(i, f, context=ctx)),
        "profit_task": float(grade_profit_task(i, f, context=ctx)),
        "stability_task": float(grade_stability_task(i, f, context=ctx)),
        "competitor_response_task": float(grade_competitor_response_task(i, f, context=ctx)),
        "crisis_recovery_task": float(grade_crisis_recovery_task(i, f, context=ctx)),
    }
