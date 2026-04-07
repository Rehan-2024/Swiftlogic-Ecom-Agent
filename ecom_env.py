"""
ecom_env.py — AI Autonomous Startup Operator Environment
Pydantic models (Phase 2), EcomEnv class (Phase 3), and grading functions (Phase 4).
"""

import random
from typing import Annotated, Dict, List, Literal, Union

from pydantic import BaseModel, Field, RootModel


# ---------------------------------------------------------------------------
# Phase 2: Pydantic Models
# ---------------------------------------------------------------------------

class Ticket(BaseModel):
    ticket_id: str
    issue_type: str
    status: str


class EcomObservation(BaseModel):
    current_day: int
    step_count: int
    bank_balance: float
    inventory: Dict[str, int]
    pending_orders: Dict[str, int]
    active_tickets: List[Ticket]
    daily_sales: Dict[str, int]
    active_ad_spend: Dict[str, float]
    reward: float = 0.0
    done: bool = False


class RestockAction(BaseModel):
    action_type: Literal["restock"] = "restock"
    sku: str
    quantity: int


class RefundAction(BaseModel):
    action_type: Literal["refund"] = "refund"
    ticket_id: str


class WaitAction(BaseModel):
    action_type: Literal["wait"] = "wait"


class EcomAction(RootModel):
    root: Annotated[
        Union[RestockAction, RefundAction, WaitAction],
        Field(discriminator="action_type"),
    ]


class EcomReward(BaseModel):
    value: float


# ---------------------------------------------------------------------------
# Phase 3: EcomEnv Class
# ---------------------------------------------------------------------------

class EcomEnv:
    """OpenEnv-compatible e-commerce storefront environment."""

    def __init__(self):
        self._current_state: EcomObservation = None
        self._step_count: int = 0
        self.reset()  # Auto-initialize to a valid state

    # -- Seeding --------------------------------------------------------
    def seed(self, seed: int) -> None:
        """Set the global random seed for deterministic runs."""
        random.seed(seed)

    # -- Reset ----------------------------------------------------------
    def reset(self, seed: int = None) -> EcomObservation:
        """Initialize (or re-initialize) the storefront to a known state."""
        if seed is not None:
            self.seed(seed)

        self._step_count = 0
        self._current_state = EcomObservation(
            current_day=1,
            step_count=0,
            bank_balance=1000.0,
            inventory={"silk_kurta": 50, "cotton_set": 30},
            pending_orders={"silk_kurta": 5, "cotton_set": 3},
            active_tickets=[
                Ticket(
                    ticket_id="TKT-001",
                    issue_type="refund",
                    status="open",
                )
            ],
            daily_sales={"silk_kurta": 0, "cotton_set": 0},
            active_ad_spend={"silk_kurta": 0.0, "cotton_set": 0.0},
            reward=0.0,
            done=False,
        )
        return self._current_state

    # -- Step -----------------------------------------------------------
    def step(self, action: EcomAction):
        """
        Execute one action and return (observation, reward, done, info).
        """
        # Unwrap RootModel if the server passed us an EcomAction wrapper
        if hasattr(action, 'root'):
            action = action.root

        self._step_count += 1
        self._current_state.step_count = self._step_count
        self._current_state.reward = 0.0
        reward_val: float = 0.0
        info: dict = {}

        # --- Restock logic ---
        if isinstance(action, RestockAction):
            sku_valid = action.sku in self._current_state.inventory
            qty_valid = action.quantity > 0
            if not sku_valid or not qty_valid:
                reward_val = -0.2
                info["error"] = "invalid_restock"
            else:
                self._current_state.inventory[action.sku] += action.quantity
                reward_val = 0.1
                info["restock"] = {"sku": action.sku, "quantity": action.quantity}

        # --- Refund logic ---
        elif isinstance(action, RefundAction):
            matched = False
            for ticket in self._current_state.active_tickets:
                if ticket.ticket_id == action.ticket_id:
                    ticket.status = "resolved"
                    matched = True
                    break
            if matched:
                reward_val = 0.3
                info["refund"] = {"ticket_id": action.ticket_id, "resolved": True}
            else:
                reward_val = -0.2
                info["error"] = "ticket_not_found"

        # --- Wait logic ---
        elif isinstance(action, WaitAction):
            reward_val = 0.0

        done = self._step_count >= 10

        self._current_state.reward = reward_val
        self._current_state.done = done

        return (
            self._current_state,
            EcomReward(value=float(reward_val)),
            done,
            info,
        )

    # -- State ----------------------------------------------------------
    def state(self) -> EcomObservation:
        """Return the current observation state."""
        return self._current_state

    # -- Framework compatibility (OpenEnv v0.2.3) -----------------------
    async def reset_async(self, seed=None, **kwargs):
        """Framework requirement: Async version of reset."""
        return self.reset(seed=seed)

    async def step_async(self, action, **kwargs):
        """Framework requirement: Async version of step."""
        obs, reward, done, info = self.step(action)
        return obs

    def close(self):
        """Framework requirement: Cleanup method."""
        pass


# ---------------------------------------------------------------------------
# Phase 4: Deterministic Grading Functions
# ---------------------------------------------------------------------------

def grade_triage_task(initial_state: EcomObservation, final_state: EcomObservation) -> float:
    """Calculate the ratio of resolved tickets to total tickets, clamped to (0.01, 0.99)."""
    if not final_state.active_tickets:
        return 0.99
    resolved = sum(1 for t in final_state.active_tickets if t.status == "resolved")
    ratio = resolved / len(final_state.active_tickets)
    return max(0.01, min(0.99, ratio))


def grade_inventory_task(initial_state: EcomObservation, final_state: EcomObservation) -> float:
    """Ratio of current 'cotton_set' stock vs a target of 10 units, clamped to (0.01, 0.99)."""
    stock = final_state.inventory.get("cotton_set", 0)
    ratio = stock / 10.0
    return max(0.01, min(0.99, ratio))


def grade_profit_task(initial_state: EcomObservation, final_state: EcomObservation) -> float:
    """Normalize bank_balance around starting 1000.0 -> 0.5, clamped to (0.01, 0.99)."""
    profit = final_state.bank_balance - 1000.0
    score = 0.5 + (profit / 400.0)
    return max(0.01, min(0.99, score))
