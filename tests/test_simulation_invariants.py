"""Phase 6.2 — invariants that must hold for every action/day transition."""

from __future__ import annotations

import random


def _run_random_rollout(world, steps: int, seed: int = 17):
    rng = random.Random(seed)
    skus = list(world.state["inventory"].keys())
    allowed = list(world.config.get("actions", {}).get("allowed", []))
    for _ in range(steps):
        atype = rng.choice(allowed)
        if atype == "restock":
            action = {"action_type": "restock", "sku": rng.choice(skus), "quantity": rng.randint(1, 5)}
        elif atype == "ad_spend":
            action = {"action_type": "ad_spend", "sku": rng.choice(skus), "budget": 100.0}
        elif atype == "negotiate":
            action = {"action_type": "negotiate", "sku": rng.choice(skus), "quantity": rng.randint(1, 10)}
        elif atype == "refund":
            tickets = world.state.get("active_tickets") or []
            if not tickets:
                action = {"action_type": "wait"}
            else:
                action = {"action_type": "refund", "ticket_id": tickets[0].get("ticket_id")}
        else:
            action = {"action_type": "wait"}

        state, reward, done, info = world.step(action)

        for sku, qty in state["inventory"].items():
            assert qty >= 0, f"inventory went negative for {sku}: {qty}"
        assert state["bank_balance"] >= float(
            world.config["financials"].get("bankruptcy_threshold", 0.0)
        ) or done
        if done:
            break


def test_inventory_never_negative(world):
    _run_random_rollout(world, steps=50, seed=3)


def test_bank_balance_transitions_are_consistent(world):
    """A pure WaitAction shouldn't change bank outside daily revenue."""
    before = world.state["bank_balance"]
    state, _r, _d, _i = world.step({"action_type": "wait"})
    # Wait should never debit funds.
    revenue = sum(
        int(state["daily_sales"].get(sku, 0)) * float(state["prices"].get(sku, 0.0))
        for sku in state["daily_sales"]
    )
    assert state["bank_balance"] == before + revenue


def test_done_on_horizon(world):
    max_steps = int(world.config["episode"].get("max_steps", 50))
    # Use waits to reach horizon without risking bankruptcy.
    done = False
    for _ in range(max_steps):
        _s, _r, done, _i = world.step({"action_type": "wait"})
    assert done is True


def test_rolling_daily_sales_history_kept_at_three(world):
    # Run 10 waits so some sales accumulate.
    for _ in range(10):
        world.step({"action_type": "wait"})
    for sku, hist in world.state["daily_sales_history"].items():
        assert isinstance(hist, list)
        assert len(hist) <= 3, f"{sku} history exceeded 3: {hist}"


def test_resolved_tickets_pruned_after_retention_window(world):
    """v2.3 Phase 5.5 — resolved tickets older than the configured retention
    window must be removed so long rollouts don't inflate the active list.
    """
    retention = int(
        (world.config.get("tickets", {}) or {}).get("resolved_retention_days", 7)
    )
    # Inject a stale "resolved" ticket with created_day in the far past.
    today = int(world.state.get("current_day", 0))
    stale_id = "T_STALE_RESOLVED"
    world.state["active_tickets"].append(
        {
            "ticket_id": stale_id,
            "issue_type": "refund",
            "urgency": "normal",
            "status": "resolved",
            "created_day": today - (retention + 3),
        }
    )
    # One more wait to trigger the next _simulate_day's prune step.
    world.step({"action_type": "wait"})
    remaining_ids = {t.get("ticket_id") for t in world.state["active_tickets"]}
    assert stale_id not in remaining_ids


def test_reset_does_not_mutate_global_random():
    """Post-audit M-2 — ``WorldEngine.reset(seed=...)`` must no longer
    reseed the process-wide ``random`` / ``numpy.random`` globals. Two
    engines resetting in the same process must each own their RNG without
    perturbing anyone else's state.
    """
    import numpy as np
    from env.world_engine import WorldEngine

    py_state_before = random.getstate()
    np_state_before = np.random.get_state()
    w = WorldEngine("configs/siyaani_fashion.json")
    w.reset(seed=7)
    assert random.getstate() == py_state_before, (
        "reset(seed=) mutated the global random state"
    )
    # np.random.get_state returns a tuple whose arrays have to match via
    # numpy comparison; compare field by field for a sharper error message.
    after = np.random.get_state()
    assert py_state_before == random.getstate()
    assert np_state_before[0] == after[0]
    assert np.array_equal(np_state_before[1], after[1])
    assert np_state_before[2:] == after[2:]


def test_snapshot_does_not_share_pending_deliveries_lists(world):
    """Post-audit M-1 — ``_snapshot_state`` must deep-copy the inner lists
    of ``pending_deliveries`` so a snapshot cannot be mutated into the live
    state via a shared list reference. Mirrors the existing guarantee for
    ``daily_sales_history``.
    """
    # Seed a pending delivery via a normal restock with lead_days > 0. The
    # shipped Siyaani config sets lead_days=2 on silk_kurta.
    sku = "silk_kurta"
    # Make sure we have budget to cover the restock.
    world.state["bank_balance"] = 100000.0
    world.step({"action_type": "restock", "sku": sku, "quantity": 3})
    assert world.state["pending_deliveries"].get(sku), (
        "precondition failed: no pending delivery scheduled"
    )
    snap = world._snapshot_state()
    # Mutating the snapshot's inner list must not affect live state.
    snap["pending_deliveries"][sku].append((99999, 777))
    assert (99999, 777) not in world.state["pending_deliveries"][sku], (
        "snapshot leaked pending_deliveries list mutation into live state"
    )


def test_observation_exposes_pending_delivery_schedule(world):
    """Post-audit m-10 — an SKU with ``restock_lead_days > 0`` must show
    up on the observation's ``pending_orders_schedule`` as a list of
    ``[delivery_day, qty]`` pairs. The aggregate ``pending_orders`` is
    preserved for back-compat.
    """
    from ecom_env import EcomEnv

    env = EcomEnv("configs/siyaani_fashion.json")
    env.reset(seed=42)
    # Fund the restock generously so the order is accepted.
    env.world_engine.state["bank_balance"] = 100_000.0
    env.step({"action_type": "restock", "sku": "silk_kurta", "quantity": 4})
    obs = env.state()
    schedule = obs.pending_orders_schedule
    assert "silk_kurta" in schedule, schedule
    entries = schedule["silk_kurta"]
    assert entries and isinstance(entries, list)
    day, qty = entries[0]
    assert isinstance(day, int) and isinstance(qty, int)
    assert qty == 4
    # The aggregate counter still tracks the same quantity.
    assert obs.pending_orders.get("silk_kurta", 0) == 4


def test_spawn_respects_max_active_cap():
    """Post-audit B.9 — ``tickets.max_active`` caps the number of open
    tickets ``spawn_daily_tickets`` will create in a single call.
    """
    from env.ticket_system import spawn_daily_tickets

    rnd = random.Random(1234)
    active = [
        {"ticket_id": f"TKT-{i:03d}", "status": "open"} for i in range(1, 4)
    ]  # 3 open tickets already
    spawned = spawn_daily_tickets(
        active_tickets=active,
        current_day=1,
        spawn_rate_per_day=50.0,  # wildly high — would explode without a cap
        rng=rnd,
        max_active=5,
    )
    open_count = sum(1 for t in active if t.get("status") == "open")
    assert open_count <= 5, open_count
    assert len(spawned) == open_count - 3  # only 2 could actually spawn


def test_spawn_without_max_active_is_unbounded():
    """Back-compat: omitting ``max_active`` (or passing ``None``) preserves
    the pre-v2.3 unbounded behaviour so existing configs keep working.
    """
    from env.ticket_system import spawn_daily_tickets

    rnd = random.Random(1234)
    active = []
    spawn_daily_tickets(
        active_tickets=active,
        current_day=1,
        spawn_rate_per_day=10.0,
        rng=rnd,
        max_active=None,
    )
    assert len(active) == 10


def test_reward_breakdown_sum_matches_total(world):
    """Post-audit D.3 — per-term reward breakdown must sum to the scalar
    ``reward`` field in ``info`` (modulo the ``daily_revenue`` passthrough).
    Protects downstream loggers that reconcile the two.
    """
    _s, reward, _d, info = world.step({"action_type": "wait"})
    bd = info.get("reward_breakdown", {})
    # ``daily_revenue`` is a passthrough scalar, not a shaping term; exclude
    # it from the sum so we only compare actual reward components.
    term_sum = sum(v for k, v in bd.items() if k != "daily_revenue")
    assert abs(round(term_sum, 4) - reward) < 1e-3, (term_sum, reward, bd)


def test_inventory_target_bonus_visible_in_reward_breakdown(world):
    """v2.3 Phase 6.2 — when the target SKU is at/above target_units, the
    ``inventory_target_bonus`` term should surface in
    ``info['reward_breakdown']``.
    """
    bonus = float(world.config.get("rewards", {}).get("inventory_target_bonus", 0.0))
    if bonus <= 0:
        # Stackbase pins this to 0.0 — nothing to assert there.
        return
    _s, _r, _d, info = world.step({"action_type": "wait"})
    bd = info.get("reward_breakdown", {})
    assert "inventory_target_bonus" in bd
    assert bd["inventory_target_bonus"] == round(bonus, 4) or bd["inventory_target_bonus"] == 0.0
