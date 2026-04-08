import os
import sys
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Ensure ecom_env is importable from both local and Docker contexts
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from ecom_env import (
        EcomEnv, EcomAction, EcomObservation, EcomReward,
        RestockAction, RefundAction, WaitAction,
        grade_triage_task, grade_inventory_task, grade_profit_task,
    )
except ImportError as e:
    raise RuntimeError(f"Cannot import ecom_env — check PYTHONPATH. Error: {e}")

app = FastAPI(title="Swiftlogic E-commerce Agent")

env = EcomEnv()

# --- Store initial state for grading ---
_initial_state = None


TASKS = [
    {
        "id": "triage_task",
        "name": "Customer Ticket Triage",
        "description": "Resolve all open customer support tickets by issuing refund actions.",
        "difficulty": "easy",
        "grader": {
            "module": "ecom_env",
            "function": "grade_triage_task"
        }
    },
    {
        "id": "inventory_task",
        "name": "Inventory Management",
        "description": "Maintain cotton_set stock above zero while keeping bank balance positive.",
        "difficulty": "medium",
        "grader": {
            "module": "ecom_env",
            "function": "grade_inventory_task"
        }
    },
    {
        "id": "profit_task",
        "name": "Profit Maximization",
        "description": "Grow the bank balance beyond the initial $1000 seed capital.",
        "difficulty": "hard",
        "grader": {
            "module": "ecom_env",
            "function": "grade_profit_task"
        }
    },
]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {
        "status": "online",
        "environment": "Swiftlogic E-commerce Agent",
        "version": "1.0.0",
        "framework": "OpenEnv v0.2.3",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/health"],
    }


@app.post("/reset")
async def reset_env(request: Request):
    global _initial_state
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    seed = body.get("seed", 42) if isinstance(body, dict) else 42
    obs = env.reset(seed=seed)
    _initial_state = obs.model_copy(deep=True)
    return {"observation": obs.model_dump(), "reward": 0.0, "done": False}


@app.post("/step")
async def step_env(request: Request):
    body = await request.json()
    action_data = body.get("action", body) if isinstance(body, dict) else body

    a_type = action_data.get("action_type")
    if a_type == "restock":
        action = RestockAction(**action_data)
    elif a_type == "refund":
        action = RefundAction(**action_data)
    elif a_type == "wait":
        action = WaitAction()
    else:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Unknown action_type: {a_type}"},
        )

    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.value,
        "done": done,
        "info": info,
    }


@app.get("/state")
async def get_state():
    obs = env.state()
    return {"observation": obs.model_dump()}


@app.get("/tasks")
async def get_tasks():
    return TASKS


@app.post("/grader")
async def run_grader(request: Request):
    """Run all graders against current state and return scores."""
    global _initial_state
    final_state = env.state()

    if _initial_state is None:
        # fallback: reset and use current as both
        env.reset(seed=42)
        _initial_state = env.state().model_copy(deep=True)

    scores = {
        "triage_task":    grade_triage_task(_initial_state, final_state),
        "inventory_task": grade_inventory_task(_initial_state, final_state),
        "profit_task":    grade_profit_task(_initial_state, final_state),
    }

    results = []
    for task in TASKS:
        tid = task["id"]
        score = scores.get(tid, 0.0)
        results.append({
            "task_id": tid,
            "score": round(score, 4),
            "grader": task["grader"],
        })

    return {"scores": results}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()