"""
inference.py — Hardcoded mock inference loop for OpenEnv evaluation.
Prints strictly formatted [START], [STEP], and [END] lines to stdout.
"""

import os

from openai import OpenAI

from ecom_env import EcomEnv, WaitAction


def main():
    # --- Environment config ---
    task_name = "commerce_ops_v1"
    benchmark = "commerce_ops_v1"
    api_base = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
    api_key = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", ""))

    # --- Initialize OpenAI client (unused in mock loop, but required) ---
    client = OpenAI(base_url=api_base, api_key=api_key)  # noqa: F841

    # --- Initialize environment ---
    env = EcomEnv()
    env.reset(seed=42)

    # --- [START] line ---
    print(f"[START] task={task_name} env={benchmark} model={model_name}")

    # --- Mock loop: 5 steps of WaitAction ---
    rewards: list[float] = []
    total_steps = 5
    done = False

    for step_num in range(1, total_steps + 1):
        action = WaitAction()
        obs, reward_obj, done, info = env.step(action)

        reward_val = reward_obj.value
        rewards.append(reward_val)

        error_str = info.get("error", "null") if info.get("error") else "null"
        done_str = "true" if done else "false"

        print(
            f"[STEP] step={step_num} "
            f"action={action.action_type} "
            f"reward={reward_val:.2f} "
            f"done={done_str} "
            f"error={error_str}"
        )

        if done:
            break

    # --- [END] line ---
    success = not done or step_num == total_steps
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(f"[END] success={success_str} steps={len(rewards)} rewards={rewards_str}")


if __name__ == "__main__":
    main()
