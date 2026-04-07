"""
inference.py — Hardcoded mock inference loop for OpenEnv evaluation.
Prints strictly formatted [START], [STEP], and [END] lines to stdout.
"""

import os
import json

from openai import OpenAI

from ecom_env import EcomEnv, WaitAction, RestockAction, RefundAction


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def main():
    # --- Environment config ---
    task_name = "commerce_ops_v1"
    benchmark = "commerce_ops_v1"
    api_base = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

    # --- Initialize OpenAI client ---
    client = OpenAI(base_url=api_base, api_key=api_key, timeout=20)

    # --- Initialize environment ---
    env = EcomEnv()
    env.reset(seed=42)

    # --- [START] line ---
    log_start(task_name, benchmark, model_name)

    system_prompt = """You are an autonomous digital storefront operator.
Your goal is to manage inventory, handle customer support tickets, and maximize profit.
You must return only raw JSON matching one of these Action schemas, depending on your choice:
1. {"action_type": "restock", "sku": "<string>", "quantity": <int>}
2. {"action_type": "refund", "ticket_id": "<string>"}
3. {"action_type": "wait"}
Do not output any markdown formatting or explanations, just the JSON object.
"""

    # --- Active Inference Loop ---
    rewards: list[float] = []
    total_steps = 5
    done = False
    step_num = 0

    for step_num in range(1, total_steps + 1):
        obs_state = env.state()
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Current Observation:\n{obs_state.model_dump_json()}"}
                ],
                response_format={"type": "json_object"}
            )
            raw_text = response.choices[0].message.content
            action_data = json.loads(raw_text)
            
            a_type = action_data.get("action_type")
            if a_type == "restock":
                action = RestockAction(**action_data)
            elif a_type == "refund":
                action = RefundAction(**action_data)
            else:
                action = WaitAction()
        except Exception:
            action = WaitAction()

        obs, reward_obj, done, info = env.step(action)

        reward_val = reward_obj.value
        rewards.append(reward_val)

        error_str = info.get("error")
        log_step(step_num, action.action_type, reward_val, done, error_str)

        if done:
            break

    # --- [END] line ---
    success = not done or step_num == total_steps
    raw_score = sum(rewards)
    score = max(0.01, min(0.99, raw_score))
    log_end(success, len(rewards), score, rewards)


if __name__ == "__main__":
    main()
