"""3-stage curriculum progression (Part B4).

Stages progress the config file used for training, not the reward. The
notebook polls ``next_stage`` once per training batch; once the rolling
mean of the episode reward crosses ``advance_threshold`` for ``streak``
consecutive evaluations, we advance and reset the streak counter.

Stage list (ordered):

1. ``configs/siyaani_fashion_easy.json`` — fewer tickets, cheaper stock,
   smaller max_steps; converges within ~30 episodes.
2. ``configs/siyaani_fashion.json`` — the production configuration.
3. ``configs/siyaani_fashion_demo.json`` (if present) or the production
   config again with seed-shifted starts — stress test.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Stage:
    name: str
    config_path: str
    advance_threshold: float
    streak: int = 3


@dataclass
class CurriculumState:
    stages: List[Stage]
    index: int = 0
    streak_count: int = 0
    history: List[dict] = field(default_factory=list)

    @property
    def current(self) -> Stage:
        return self.stages[self.index]

    @property
    def is_final(self) -> bool:
        return self.index >= len(self.stages) - 1

    def observe(self, rolling_mean_reward: float) -> Optional[Stage]:
        """Record a rolling mean reward; return the new stage if advanced."""
        cur = self.current
        if rolling_mean_reward >= cur.advance_threshold:
            self.streak_count += 1
        else:
            self.streak_count = 0
        self.history.append(
            {"stage": cur.name, "rolling_mean": rolling_mean_reward, "streak": self.streak_count}
        )
        if self.streak_count >= cur.streak and not self.is_final:
            self.index += 1
            self.streak_count = 0
            return self.current
        return None


def default_curriculum(root: str = ".") -> CurriculumState:
    root_path = Path(root)
    stages: List[Stage] = []
    easy = root_path / "configs" / "siyaani_fashion_easy.json"
    prod = root_path / "configs" / "siyaani_fashion.json"
    demo = root_path / "configs" / "siyaani_fashion_demo.json"

    if easy.exists():
        stages.append(
            Stage(name="easy", config_path=str(easy), advance_threshold=0.8, streak=3)
        )
    stages.append(
        Stage(name="production", config_path=str(prod), advance_threshold=0.9, streak=3)
    )
    if demo.exists():
        stages.append(
            Stage(name="demo", config_path=str(demo), advance_threshold=999.0, streak=999)
        )
    return CurriculumState(stages=stages)


def rolling_mean(rewards: List[float], window: int = 10) -> float:
    if not rewards:
        return 0.0
    tail = rewards[-window:]
    return statistics.mean(tail) if tail else 0.0
