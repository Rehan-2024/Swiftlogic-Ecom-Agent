import os
import sys
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Add the parent directory to the path so it can find ecom_env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecom_env import (
    EcomEnv, EcomAction, EcomObservation, EcomReward,
    RestockAction, RefundAction, WaitAction,
)

# --- Create the app manually for full control over the /step body format ---
app = FastAPI(title="Swiftlogic E-commerce Agent")

# Shared environment instance
env = EcomEnv()


@app.get("/")
async def root():
    return {
        "status": "online",
        "environment": "Swiftlogic E-commerce Agent",
        "version": "1.0.0",
        "framework": "OpenEnv v0.2.3",
        "endpoints": ["/reset", "/step", "/state"],
    }


@app.post("/reset")
async def reset_env(request: Request):
    body = await request.json() if await request.body() else {}
    seed = body.get("seed", 42)
    obs = env.reset(seed=seed)
    return {"observation": obs.model_dump(), "reward": 0.0, "done": False}


@app.post("/step")
async def step_env(request: Request):
    """
    Accepts FLAT JSON like {"action_type": "wait"}
    OR wrapped JSON like {"action": {"action_type": "wait"}}.
    """
    body = await request.json()

    # Support both flat and wrapped formats
    action_data = body.get("action", body) if isinstance(body, dict) else body

    # Map action_type to the correct Pydantic model
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


def main():
    # Bind to the port Hugging Face expects
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()