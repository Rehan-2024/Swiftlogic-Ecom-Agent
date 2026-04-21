"""
ticket_system.py — Dynamic customer support ticket generator for CommerceOps v2.

Supports both initial episode seeding (``generate_episode_tickets``) and
per-day stochastic spawning (``spawn_daily_tickets``). All tickets are stored
as plain dicts so the WorldEngine can keep a JSON-serializable state; the
Pydantic adapter in ``ecom_env.py`` coerces them into ``Ticket`` instances on
the way out.
"""

from __future__ import annotations

import random
from typing import List, Optional

TICKET_TYPES = ["refund", "damage", "delay", "wrong_item"]
URGENCY_LEVELS = ["normal", "urgent", "critical"]
DEFAULT_URGENCY_WEIGHTS = [0.5, 0.35, 0.15]


def _new_ticket(
    index: int,
    created_day: int,
    issue_types: Optional[List[str]] = None,
    urgency_levels: Optional[List[str]] = None,
    urgency_weights: Optional[List[float]] = None,
    rng: Optional[random.Random] = None,
) -> dict:
    issues = issue_types or TICKET_TYPES
    levels = urgency_levels or URGENCY_LEVELS
    weights = urgency_weights or DEFAULT_URGENCY_WEIGHTS
    rnd = rng if rng is not None else random
    return {
        "ticket_id": f"TKT-{index:03d}",
        "issue_type": rnd.choice(issues),
        "urgency": rnd.choices(levels, weights=weights, k=1)[0],
        "status": "open",
        "created_day": created_day,
    }


def generate_episode_tickets(
    num: Optional[int] = None,
    current_day: int = 1,
    min_count: int = 3,
    max_count: int = 5,
    issue_types: Optional[List[str]] = None,
    urgency_levels: Optional[List[str]] = None,
    urgency_weights: Optional[List[float]] = None,
    rng: Optional[random.Random] = None,
) -> List[dict]:
    """Seed a fresh episode with 3-5 randomized open tickets.

    v2.3 Phase 5.1 — accepts an optional ``rng`` so every WorldEngine owns
    its own ``random.Random`` instance and tickets don't leak between two
    envs sharing the process-wide ``random`` module.
    """
    rnd = rng if rng is not None else random
    if num is None:
        lo = max(0, int(min_count))
        hi = max(lo, int(max_count))
        n = rnd.randint(lo, hi) if hi > 0 else 0
    else:
        n = max(0, int(num))
    return [
        _new_ticket(
            index=i + 1,
            created_day=current_day,
            issue_types=issue_types,
            urgency_levels=urgency_levels,
            urgency_weights=urgency_weights,
            rng=rng,
        )
        for i in range(n)
    ]


def spawn_daily_tickets(
    active_tickets: List[dict],
    current_day: int,
    spawn_rate_per_day: float = 0.0,
    issue_types: Optional[List[str]] = None,
    urgency_levels: Optional[List[str]] = None,
    urgency_weights: Optional[List[float]] = None,
    rng: Optional[random.Random] = None,
    max_active: Optional[int] = None,
) -> List[dict]:
    """Spawn new tickets in-place with probability ``spawn_rate_per_day`` per day.

    The rate can exceed 1.0 — in which case ``floor(rate)`` tickets are spawned
    deterministically and the fractional remainder becomes the probability for
    one additional ticket. Returns the list of newly spawned tickets.

    Post-audit B.9 — when ``max_active`` is provided, spawning stops once the
    number of currently-open tickets reaches that cap. This prevents
    pathological long-horizon episodes from ballooning the ticket queue
    beyond what a policy could ever triage. ``None`` means "unbounded"
    and preserves the pre-v2.3 behaviour.
    """
    if spawn_rate_per_day <= 0:
        return []

    rnd = rng if rng is not None else random
    full = int(spawn_rate_per_day)
    frac = spawn_rate_per_day - full
    n_spawn = full + (1 if rnd.random() < frac else 0)
    if n_spawn <= 0:
        return []

    existing_ids = {t.get("ticket_id") for t in active_tickets}
    # Determine next index based on the largest existing TKT-### id we can parse.
    next_idx = 1
    for tid in existing_ids:
        if not isinstance(tid, str) or not tid.startswith("TKT-"):
            continue
        try:
            next_idx = max(next_idx, int(tid.split("-", 1)[1]) + 1)
        except (ValueError, IndexError):
            continue

    # Count currently-open tickets once; subsequent additions bump the
    # counter locally so the cap is respected without re-scanning the list.
    open_count = sum(1 for t in active_tickets if t.get("status") == "open")
    cap = int(max_active) if max_active is not None else None

    new_tickets: List[dict] = []
    for _ in range(n_spawn):
        if cap is not None and open_count >= cap:
            break
        ticket = _new_ticket(
            index=next_idx,
            created_day=current_day,
            issue_types=issue_types,
            urgency_levels=urgency_levels,
            urgency_weights=urgency_weights,
            rng=rng,
        )
        new_tickets.append(ticket)
        active_tickets.append(ticket)
        open_count += 1
        next_idx += 1
    return new_tickets
