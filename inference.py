"""
inference.py — Swiftlogic CommerceOps v2 inference loop for OpenEnv evaluation.
Prints strictly formatted [START], [STEP], and [END] lines to stdout.

Runs the full 50-step business cycle per task. Supports all five action types:
restock, refund, ad_spend, negotiate, wait. Falls back to WaitAction on any
parsing error. Also emits a per-task diagnostic line summarising negotiate
usage and quote-to-restock conversion to aid policy debugging.
"""

import json
import logging
import os

from openai import OpenAI


logger = logging.getLogger("commerceops.inference")

from ecom_env import (
    EcomEnv,
    WaitAction,
    RestockAction,
    RefundAction,
    AdSpendAction,
    NegotiateAction,
    SetPriceAction,
    grade_triage_task,
    grade_inventory_task,
    grade_profit_task,
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float], graders_str: str = "") -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    graders_part = f" graders={graders_str}" if graders_str else ""
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}{graders_part}",
        flush=True,
    )


def log_diagnostics(task: str, negotiate_count: int, restock_count: int,
                    negotiated_restock_count: int, total_steps: int) -> None:
    """Emit a single structured diagnostics line per task.

    Metrics:
      * negotiate_rate       - fraction of steps that used NegotiateAction
      * quote_conversion     - fraction of restocks that consumed a quote
    """
    neg_rate = (negotiate_count / total_steps) if total_steps else 0.0
    conv_rate = (negotiated_restock_count / restock_count) if restock_count else 0.0
    print(
        f"[DIAG] task={task} negotiate_count={negotiate_count} restock_count={restock_count} "
        f"negotiated_restock_count={negotiated_restock_count} "
        f"negotiate_rate={neg_rate:.3f} quote_conversion={conv_rate:.3f}",
        flush=True,
    )


def _build_action(action_data: dict):
    a_type = action_data.get("action_type")
    if a_type == "restock":
        return RestockAction(**action_data)
    if a_type == "refund":
        return RefundAction(**action_data)
    if a_type == "ad_spend":
        return AdSpendAction(**action_data)
    if a_type == "negotiate":
        return NegotiateAction(**action_data)
    if a_type == "set_price":
        return SetPriceAction(**action_data)
    return WaitAction()


def main():
    benchmark = "commerce_ops_v2"
    api_base = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME", "default-model")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

    client = OpenAI(base_url=api_base, api_key=api_key, timeout=20)

    env = EcomEnv()

    system_prompt = """You are an autonomous digital storefront operator for an Indian ethnic wear brand (Siyaani).
Your goal is to manage inventory, resolve customer support tickets, allocate ad spend,
negotiate supplier prices, and maximize profit over a 50-day business cycle without going bankrupt.

You must return ONLY raw JSON matching exactly one of these Action schemas:
1. {"action_type": "restock", "sku": "<string>", "quantity": <int>}
2. {"action_type": "refund", "ticket_id": "<string>"}
3. {"action_type": "ad_spend", "sku": "<string>", "budget": <float>}
4. {"action_type": "negotiate", "sku": "<string>", "quantity": <int>}
5. {"action_type": "wait"}
6. {"action_type": "set_price", "sku": "<string>", "price": <float>}

Negotiate requests a supplier unit-price quote for a future restock on the same
SKU. Quotes expire after 3 steps if not consumed by a restock. Small-volume
negotiated orders unlock a supplier volume discount; un-negotiated restocks
pay a spot-market premium over list cost.

set_price directly mutates the sell price for a SKU. It must stay within the
configured [price_min_mult_competitor, price_max_mult_competitor] band vs the
competitor's price; out-of-band prices are rejected with invalid_action.

The observation exposes ``pending_orders`` (aggregate in-flight quantity per
SKU) and ``pending_orders_schedule`` (per-SKU list of [delivery_day, qty]
pairs). Use the schedule to avoid over-restocking when a previous order is
already in transit.

Do not output any markdown formatting or explanations, just the JSON object.
"""

    tasks = ["triage_task", "inventory_task", "profit_task"]

    for task_name in tasks:
        env.reset(seed=42)
        initial_state = env.state().model_copy(deep=True)

        log_start(task_name, benchmark, model_name)

        rewards: list[float] = []
        total_steps = 50
        done = False
        step_num = 0

        negotiate_count = 0
        restock_count = 0
        negotiated_restock_count = 0

        for step_num in range(1, total_steps + 1):
            obs_state = env.state()

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Current Observation:\n{obs_state.model_dump_json()}"},
                    ],
                    response_format={"type": "json_object"},
                )
                raw_text = response.choices[0].message.content
                action_data = json.loads(raw_text)
                action = _build_action(action_data)
            except Exception as exc:
                # Phase G.2 — surface parse failures so a run of all-``wait``
                # steps is distinguishable from an offline policy. We log at
                # WARNING with the step number and exception class, but do
                # NOT emit the raw model output because it can be large and
                # may contain PII in some eval harnesses.
                logger.warning(
                    "action_parse_failed step=%s task=%s exc=%s",
                    step_num,
                    task_name,
                    exc.__class__.__name__,
                )
                action = WaitAction()

            obs, reward_obj, done, info = env.step(action)

            reward_val = reward_obj.value
            rewards.append(reward_val)

            # Diagnostics bookkeeping -------------------------------------------------
            if action.action_type == "negotiate" and not info.get("error"):
                negotiate_count += 1
            if action.action_type == "restock" and not info.get("error"):
                restock_count += 1
                if info.get("restock", {}).get("negotiated"):
                    negotiated_restock_count += 1

            error_str = info.get("error")
            log_step(step_num, action.action_type, reward_val, done, error_str)

            if done:
                break

        success = not done or step_num == total_steps
        raw_score = sum(rewards)
        score = max(0.01, min(0.99, raw_score))

        final_state = env.state()

        # Post-audit m-6 — pass the env-local grader context explicitly so
        # we bypass the module mirror (and its DeprecationWarning).
        grader_ctx = getattr(env, "grader_context", None)
        if task_name == "triage_task":
            grader_score = grade_triage_task(initial_state, final_state)
        elif task_name == "inventory_task":
            grader_score = grade_inventory_task(
                initial_state, final_state, context=grader_ctx
            )
        elif task_name == "profit_task":
            grader_score = grade_profit_task(
                initial_state, final_state, context=grader_ctx
            )
        else:
            grader_score = 0.0

        graders_str = f"{task_name}:{grader_score:.2f}"

        log_end(success, len(rewards), score, rewards, graders_str)
        log_diagnostics(
            task=task_name,
            negotiate_count=negotiate_count,
            restock_count=restock_count,
            negotiated_restock_count=negotiated_restock_count,
            total_steps=len(rewards) or 1,
        )


if __name__ == "__main__":
    main()
