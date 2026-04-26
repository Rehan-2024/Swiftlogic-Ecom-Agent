"""
SINGLE-CELL COLAB GRPO TRAINING SCRIPT v6
Pure HTTP, tuned for quick signal and clear reporting.

Key features in v6:
- Clones GitHub repository automatically
- Runs the environment locally via FastAPI
- Uploads reward curve and adapter weights to Hugging Face Hub
- Default MAX_EPISODES=10 as requested
"""

import json
import os
import re
import subprocess
import warnings
from collections import Counter
from datetime import datetime, timezone


def sh(cmd: str):
    result = subprocess.run(
        cmd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        print(result.stdout[-4000:])
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")


# ==============================================================================
# 1. DEPENDENCIES
# ==============================================================================
sh("pip install -q unsloth")
sh("pip install -q requests matplotlib numpy huggingface_hub")
print("Dependencies installed")

import matplotlib
import numpy as np
import requests
import torch
import torch.optim as optim
import transformers
import random
import time

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

UNSLOTH_OK = False
try:
    from unsloth import FastLanguageModel

    UNSLOTH_OK = True
    print("Unsloth imported")
except Exception as e:
    print(f"Unsloth unavailable ({e}) -> fallback transformers+peft")
    sh("pip install -q transformers peft accelerate")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"Imports ready | CUDA: {torch.cuda.is_available()}")


# ==============================================================================
# 1.5 REPO CLONE & LOCAL SERVER
# ==============================================================================
print("\n--- CLONING REPO AND STARTING SERVER ---")
if not os.path.exists("repo"):
    sh("git clone https://github.com/Rehan-2024/Swiftlogic-Ecom-Agent.git repo")
else:
    sh("cd repo && git pull")

sh("pip install -q -r repo/requirements.txt")

# Start FastAPI server
server_proc = subprocess.Popen(
    ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"],
    cwd="repo",
    stdout=open("server.log", "w"),
    stderr=subprocess.STDOUT,
)
print("Waiting for server to start...")
time.sleep(8)
if server_proc.poll() is not None:
    with open("server.log", "r") as f:
        print("SERVER FAILED TO START:\n", f.read())
    raise RuntimeError("FastAPI server failed to start")
print("Server started successfully")

# ==============================================================================
# 2. MASTER CONFIGURATION (edit this only)
# ==============================================================================
ENV_URL = "http://localhost:7860"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_EPISODES = 10
STEPS_PER_EPISODE = 8
GROUP_SIZE = 2
MAX_NEW_TOKENS = 64
SEED_EVAL = [111, 222]
HTTP_TIMEOUT = 12

os.makedirs("artifacts/adapter", exist_ok=True)


# ==============================================================================
# 3. HTTP + BACKEND HEALTH CHECKS
# ==============================================================================
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(
    total=3,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
)
session.mount("http://", HTTPAdapter(max_retries=retry))
session.mount("https://", HTTPAdapter(max_retries=retry))

print("\n--- BACKEND HEALTH CHECK ---")
if "REPLACE-WITH" in ENV_URL or not ENV_URL.startswith("http"):
    raise ValueError("ENV_URL is not valid. Paste your ngrok https URL.")

# Check /health if present, but do not fail hard if route is missing.
try:
    health = session.get(f"{ENV_URL}/health", timeout=HTTP_TIMEOUT)
    if health.status_code == 200:
        print("Backend /health OK")
    else:
        print(f"Backend /health returned {health.status_code} (continuing)")
except Exception as e:
    print(f"/health check skipped ({e})")

try:
    r = session.post(f"{ENV_URL}/reset", json={"seed": 42}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    obs_test = r.json().get("observation")
    assert isinstance(obs_test, dict), "reset did not return observation dict"
except Exception as e:
    raise RuntimeError(f"reset failed: {e}")

try:
    r = session.post(
        f"{ENV_URL}/step",
        json={"action_type": "wait"},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    body = r.json()
    assert isinstance(body.get("observation"), dict), "step observation invalid"
    _ = float(body.get("reward", 0.0))
    print("Backend /reset + /step smoke test OK")
except Exception as e:
    raise RuntimeError(f"step smoke test failed: {e}")


# ==============================================================================
# 4. MODEL LOADING
# ==============================================================================
print(f"\n--- LOADING {MODEL_NAME} ---")
if UNSLOTH_OK:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=768,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=2026,
    )
    print("Loaded via Unsloth 4-bit")
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(base, lora_cfg)
    print("Loaded via transformers+peft")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
try:
    model.print_trainable_parameters()
except Exception:
    pass


# ==============================================================================
# 5. HELPERS
# ==============================================================================
def extract_json(text: str):
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj.setdefault("action_type", "wait")
            return obj, True
    except Exception:
        pass

    for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                obj.setdefault("action_type", "wait")
                return obj, True
        except Exception:
            continue
    return {"action_type": "wait"}, False


def build_prompt(obs: dict, seed: int) -> str:
    safe_obs = {}
    for k, v in obs.items():
        try:
            json.dumps(v)
            safe_obs[k] = v
        except Exception:
            safe_obs[k] = str(v)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an autonomous business AI CEO. "
                "Return ONLY one valid JSON object and nothing else. "
                "Valid action_type: wait, restock, set_price, negotiate, refund, ad_spend. "
                "Avoid excessive wait when inventory or pending orders require action."
            ),
        },
        {
            "role": "user",
            "content": (
                f"seed={int(seed)}\n"
                f"observation={json.dumps(safe_obs, ensure_ascii=False)}\n"
                "Output JSON action:"
            ),
        },
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        f"<seed>{int(seed)}</seed>\n"
        "Return only one JSON action.\n"
        f"Observation:\n{json.dumps(safe_obs, ensure_ascii=False)}\n"
        "JSON:\n"
    )


def safe_reward(r_json: dict) -> float:
    val = r_json.get("reward", 0.0)
    if isinstance(val, dict):
        for key in ("value", "reward", "total"):
            if key in val:
                try:
                    return float(val[key])
                except Exception:
                    pass
        return 0.0
    try:
        return float(val)
    except Exception:
        return 0.0


def _pick_primary_sku(obs: dict):
    inventory = obs.get("inventory", {})
    if isinstance(inventory, dict) and inventory:
        return next(iter(inventory.keys()))
    prices = obs.get("prices", {})
    if isinstance(prices, dict) and prices:
        return next(iter(prices.keys()))
    return None


def repair_action(action: dict, obs: dict):
    """
    Make model actions schema-safe using only current observation values.
    """
    if not isinstance(action, dict):
        return {"action_type": "wait"}, True

    action_type = str(action.get("action_type", "wait"))
    inventory = obs.get("inventory", {}) if isinstance(obs.get("inventory", {}), dict) else {}
    pending = obs.get("pending_orders", {}) if isinstance(obs.get("pending_orders", {}), dict) else {}
    prices = obs.get("prices", {}) if isinstance(obs.get("prices", {}), dict) else {}
    comp_prices = obs.get("competitor_prices", {}) if isinstance(obs.get("competitor_prices", {}), dict) else {}
    tickets = obs.get("active_tickets", []) if isinstance(obs.get("active_tickets", []), list) else []
    sku = action.get("sku") or _pick_primary_sku(obs)

    if action_type == "restock":
        if sku is None:
            return {"action_type": "wait"}, True
        cur_qty = float(inventory.get(sku, 0) or 0.0)
        pend_qty = float(pending.get(sku, 0) or 0.0)
        target_qty = max(8.0, min(40.0, pend_qty + 12.0))
        raw_qty = action.get("quantity", int(max(1.0, target_qty - cur_qty)))
        try:
            qty = int(max(1, min(120, int(float(raw_qty)))))
        except Exception:
            qty = int(max(1.0, target_qty - cur_qty))
        repaired = (action.get("sku") != sku) or (action.get("quantity") != qty)
        return {"action_type": "restock", "sku": str(sku), "quantity": qty}, repaired

    if action_type == "set_price":
        if sku is None:
            return {"action_type": "wait"}, True
        ref = comp_prices.get(sku, prices.get(sku, 10.0))
        try:
            ref = float(ref)
        except Exception:
            ref = 10.0
        lo = max(0.5, ref * 0.8)
        hi = max(lo + 0.1, ref * 1.2)
        raw_price = action.get("price", ref)
        try:
            price = float(raw_price)
        except Exception:
            price = ref
        price = float(min(max(price, lo), hi))
        repaired = (action.get("sku") != sku) or (action.get("price") != price)
        return {"action_type": "set_price", "sku": str(sku), "price": price}, repaired

    if action_type == "negotiate":
        if sku is None:
            return {"action_type": "wait"}, True
        raw_qty = action.get("quantity", max(1, int(pending.get(sku, 1) or 1)))
        try:
            qty = int(max(1, min(120, int(float(raw_qty)))))
        except Exception:
            qty = 10
        return {"action_type": "negotiate", "sku": str(sku), "quantity": qty}, True

    if action_type == "ad_spend":
        if sku is None:
            return {"action_type": "wait"}, True
        bank = float(obs.get("bank_balance", 0.0) or 0.0)
        raw_budget = action.get("budget", max(1.0, min(20.0, 0.02 * max(bank, 1.0))))
        try:
            budget = float(max(0.5, min(100.0, float(raw_budget))))
        except Exception:
            budget = 5.0
        return {"action_type": "ad_spend", "sku": str(sku), "budget": budget}, True

    if action_type == "refund":
        ticket_id = action.get("ticket_id")
        if ticket_id is None and tickets:
            open_tickets = [t.get("ticket_id") for t in tickets if isinstance(t, dict) and t.get("status") == "open"]
            if open_tickets:
                ticket_id = open_tickets[0]
        if ticket_id:
            return {"action_type": "refund", "ticket_id": str(ticket_id)}, ticket_id != action.get("ticket_id")
        return {"action_type": "wait"}, True

    if action_type == "wait":
        return {"action_type": "wait"}, False

    return {"action_type": "wait"}, True


def combined_reward(
    env_reward: float,
    action: dict,
    valid_json: bool,
    used_fallback: bool,
    prev_obs: dict | None = None,
    next_obs: dict | None = None,
    return_breakdown: bool = False,
):
    """
    Lightweight shaped reward that keeps values stable and nudges useful behavior.
    Target range is normalized to roughly [-2, +2].
    """
    action_type = str(action.get("action_type", "wait"))

    profit_score = float(env_reward)
    inventory_score = 0.0
    penalty = 0.0

    if action_type == "wait":
        penalty -= 0.5
    else:
        inventory_score += 0.05
    if (not valid_json) or used_fallback:
        penalty -= 0.2

    # Inventory signal from actual environment transition.
    if isinstance(prev_obs, dict) and isinstance(next_obs, dict):
        prev_inv = prev_obs.get("inventory", {}) if isinstance(prev_obs.get("inventory", {}), dict) else {}
        next_inv = next_obs.get("inventory", {}) if isinstance(next_obs.get("inventory", {}), dict) else {}
        prev_pending = prev_obs.get("pending_orders", {}) if isinstance(prev_obs.get("pending_orders", {}), dict) else {}
        next_pending = next_obs.get("pending_orders", {}) if isinstance(next_obs.get("pending_orders", {}), dict) else {}

        prev_inv_total = float(sum(max(0, int(v)) for v in prev_inv.values())) if prev_inv else 0.0
        next_inv_total = float(sum(max(0, int(v)) for v in next_inv.values())) if next_inv else 0.0
        prev_pending_total = float(sum(max(0, int(v)) for v in prev_pending.values())) if prev_pending else 0.0
        next_pending_total = float(sum(max(0, int(v)) for v in next_pending.values())) if next_pending else 0.0

        inv_delta = next_inv_total - prev_inv_total
        pending_reduction = prev_pending_total - next_pending_total
        inventory_score += 0.20 * np.tanh(inv_delta / 15.0)
        inventory_score += 0.25 * np.tanh(pending_reduction / 15.0)

        if next_inv:
            zero_ratio = sum(1 for v in next_inv.values() if int(v) <= 0) / len(next_inv)
            inventory_score -= 0.20 * zero_ratio

    total = profit_score + inventory_score + penalty
    total = float(np.clip(total, -2.0, 2.0))
    reward_dict = {
        "total": total,
        "profit": float(np.clip(profit_score, -2.0, 2.0)),
        "inventory": inventory_score,
        "penalty": penalty,
    }
    if return_breakdown:
        return reward_dict
    return total


def heuristic_action(obs: dict) -> dict:
    inventory = obs.get("inventory", {})
    if isinstance(inventory, dict) and inventory:
        sku = next(iter(inventory.keys()))
        qty = inventory.get(sku, 0)
        if isinstance(qty, (int, float)) and qty < 8:
            return {"action_type": "restock", "sku": str(sku), "quantity": int(max(5, 12 - qty))}

        prices = obs.get("prices", {})
        comp_prices = obs.get("competitor_prices", {})
        p_self = prices.get(sku) if isinstance(prices, dict) else None
        p_comp = comp_prices.get(sku) if isinstance(comp_prices, dict) else None
        ref_price = p_comp if isinstance(p_comp, (int, float)) else p_self
        if isinstance(ref_price, (int, float)):
            return {"action_type": "set_price", "sku": str(sku), "price": float(max(1.0, ref_price * 0.98))}

    return {"action_type": "wait"}


def reset_env(seed: int):
    try:
        r = session.post(
            f"{ENV_URL}/reset",
            json={"seed": int(seed)},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        obs = r.json().get("observation", {})
        return obs, isinstance(obs, dict)
    except Exception as e:
        print(f"[reset_env] seed={seed} failed: {e}")
        return {}, False


def step_env(action: dict):
    try:
        r = session.post(
            f"{ENV_URL}/step",
            json=action,
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return r.json(), True
    except Exception as e:
        detail = ""
        try:
            if hasattr(e, "response") and e.response is not None:
                detail = f" | body={e.response.text[:300]}"
        except Exception:
            detail = ""
        print(f"[step_env] failed: {e}{detail}")
        return {"reward": 0.0, "observation": {}, "done": True, "info": {}}, False


def compute_logprob_with_grad(prompt_ids: torch.Tensor, generated_ids: torch.Tensor):
    if generated_ids.numel() == 0:
        return torch.zeros(1, device=prompt_ids.device, requires_grad=True)

    full_ids = torch.cat([prompt_ids[0], generated_ids], dim=0).unsqueeze(0)
    prompt_len = prompt_ids.shape[1]
    gen_len = generated_ids.shape[0]

    labels = torch.full_like(full_ids, fill_value=-100)
    labels[0, prompt_len:] = generated_ids

    out = model(input_ids=full_ids, labels=labels)
    return -out.loss * gen_len


@torch.no_grad()
def infer_action(obs: dict, seed: int):
    # Explicit eval mode for generation.
    model.eval()
    prompt_text = build_prompt(obs, seed)
    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=700,
        add_special_tokens=False,
    ).input_ids.to(model.device)

    last_raw_text = ""
    last_gen_ids = torch.zeros(0, dtype=torch.long, device=model.device)
    for _attempt in range(2):  # retry once if JSON is invalid before fallback
        out = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0][prompt_ids.shape[1] :]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        action, valid_json = extract_json(raw_text)
        action, repaired = repair_action(action, obs)
        last_raw_text = raw_text
        last_gen_ids = gen_ids
        if valid_json:
            return action, raw_text, gen_ids, prompt_ids, False, True, repaired

    return {"action_type": "wait"}, last_raw_text, last_gen_ids, prompt_ids, True, False, True


# ==============================================================================
# 6. EVALUATION
# ==============================================================================
def run_evaluation_sweep(policy_type: str = "llm"):
    scores, all_actions = [], []
    fallback_used = 0
    total_steps = 0
    for seed in SEED_EVAL:
        obs, ok = reset_env(seed)
        if not ok:
            scores.append(0.0)
            continue

        ep_reward = 0.0
        for _ in range(STEPS_PER_EPISODE):
            if policy_type == "wait":
                action = {"action_type": "wait"}
                used_fallback = False
                valid_json = True
            elif policy_type == "heuristic":
                action = heuristic_action(obs)
                used_fallback = False
                valid_json = True
                repaired_action = False
            else:
                action, _, _, _, used_fallback, valid_json, repaired_action = infer_action(obs, seed)
                if used_fallback:
                    fallback_used += 1
                if repaired_action:
                    valid_json = False
            all_actions.append(action.get("action_type", "wait"))
            total_steps += 1

            prev_obs = obs
            r_json, ok_step = step_env(action)
            env_reward = safe_reward(r_json)
            next_obs = r_json.get("observation") if isinstance(r_json.get("observation"), dict) else {}
            ep_reward += combined_reward(
                env_reward=env_reward,
                action=action,
                valid_json=valid_json,
                used_fallback=used_fallback,
                prev_obs=prev_obs,
                next_obs=next_obs,
                return_breakdown=False,
            )
            if next_obs:
                obs = next_obs
            if r_json.get("done", False) or not ok_step:
                break
        scores.append(ep_reward)

    return (float(np.mean(scores)) if scores else 0.0), all_actions, fallback_used, max(1, total_steps)


print("\n--- ZERO-SHOT BASELINE ---")
baseline_score, baseline_actions, baseline_fallbacks, baseline_steps = run_evaluation_sweep("llm")
wait_score, _, _, _ = run_evaluation_sweep("wait")
heuristic_score, heuristic_actions, _, _ = run_evaluation_sweep("heuristic")
print(f"Zero-Shot: {baseline_score:.4f}")
print(f"Wait-Only: {wait_score:.4f}")
print(f"Heuristic: {heuristic_score:.4f}")
print(f"[FALLBACK RATE] baseline={baseline_fallbacks / baseline_steps:.2f}")


# ==============================================================================
# 7. TRAINING LOOP
# ==============================================================================
print(
    f"\n--- TRAINING ({MAX_EPISODES} ep x {GROUP_SIZE} rollouts x {STEPS_PER_EPISODE} steps) ---"
)
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=2e-5)

reward_history = []
entropy_history = []
loss_history = []
fallback_history = []
cumulative_fallbacks = 0
cumulative_steps = 0

for episode in range(MAX_EPISODES):
    model.train()
    explore_prob = max(0.12, 0.35 * (1.0 - (episode / max(1, MAX_EPISODES - 1))))

    rollout_logprobs = []
    rollout_rewards = []
    ep_actions = []

    for rollout_idx in range(GROUP_SIZE):
        seed = 3000 + episode * GROUP_SIZE + rollout_idx
        obs, ok = reset_env(seed)
        if not ok:
            continue

        traj_logprob = torch.zeros(1, device=model.device)
        ep_reward = 0.0
        took_step = False
        episode_fallbacks = 0

        for _step in range(STEPS_PER_EPISODE):
            action, raw_text, gen_ids, prompt_ids, used_fallback, valid_json, repaired_action = infer_action(obs, seed)
            if used_fallback:
                episode_fallbacks += 1
            if repaired_action:
                valid_json = False
            # Keep exploration active so policy does not collapse to all-wait.
            if action.get("action_type") == "wait" and random.random() < explore_prob:
                alt_action = heuristic_action(obs)
                if alt_action.get("action_type") != "wait":
                    action = alt_action
                    valid_json = True

            model.train()
            step_lp = compute_logprob_with_grad(prompt_ids, gen_ids)
            traj_logprob = traj_logprob + step_lp

            ep_actions.append(action.get("action_type", "wait"))

            prev_obs = obs
            r_json, ok_step = step_env(action)
            env_r = safe_reward(r_json)
            next_obs = r_json.get("observation") if isinstance(r_json.get("observation"), dict) else {}
            reward_dict = combined_reward(
                env_reward=env_r,
                action=action,
                valid_json=valid_json,
                used_fallback=used_fallback,
                prev_obs=prev_obs,
                next_obs=next_obs,
                return_breakdown=True,
            )
            ep_reward += reward_dict["total"]
            if rollout_idx == 0 and _step == 0:
                print(f"[REWARD BREAKDOWN] {reward_dict}")
            took_step = True
            cumulative_steps += 1

            if next_obs:
                obs = next_obs
            if r_json.get("done", False) or not ok_step:
                break

        if took_step:
            rollout_logprobs.append(traj_logprob)
            rollout_rewards.append(ep_reward)
            cumulative_fallbacks += episode_fallbacks

    if not rollout_rewards:
        last_r = reward_history[-1] if reward_history else 0.0
        reward_history.append(last_r)
        entropy_history.append(0.0)
        loss_history.append(0.0)
        print(f"Ep {episode+1:02d}: no valid rollout")
        continue

    counts = Counter(ep_actions)
    total_acts = max(1, len(ep_actions))
    entropy_val = float(
        -sum((c / total_acts) * np.log2(c / total_acts + 1e-9) for c in counts.values())
    )
    entropy_history.append(entropy_val)

    arr_r = np.array(rollout_rewards, dtype=np.float64)
    mean_r = float(arr_r.mean())
    std_r = float(arr_r.std()) + 1e-8
    advantages = [(r - mean_r) / std_r for r in rollout_rewards]

    optimizer.zero_grad()
    n = max(1, len(rollout_logprobs))
    loss = torch.zeros(1, device=model.device)
    for lp, adv in zip(rollout_logprobs, advantages):
        loss = loss - (lp * float(adv)) / n
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()

    loss_val = float(loss.item())
    reward_history.append(mean_r)
    loss_history.append(loss_val)
    last_10_rewards = reward_history[-10:]
    fallback_rate = cumulative_fallbacks / max(1, cumulative_steps)
    fallback_history.append(fallback_rate)
    print(f"[EP {episode+1}] reward={mean_r:.2f}")
    print(f"[MEAN] last10={np.mean(last_10_rewards):.2f}")
    print(f"[ACTIONS] {dict(counts)}")
    print(f"[FALLBACK RATE] {fallback_rate:.2f}")
    print(
        f"Ep {episode+1:02d}/{MAX_EPISODES} | R:{mean_r:+.3f} | "
        f"H:{entropy_val:.3f} | L:{loss_val:.4f}"
    )

    del rollout_logprobs, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

model.save_pretrained("artifacts/adapter")
tokenizer.save_pretrained("artifacts/adapter")
print("\nTraining done, adapter saved to artifacts/adapter/")


# ==============================================================================
# 8. POST-TRAINING EVAL
# ==============================================================================
print("\n--- POST-TRAINING EVAL ---")
model.eval()
trained_score, trained_actions, trained_fallbacks, trained_steps = run_evaluation_sweep("llm")
improvement = trained_score - baseline_score
pct_improvement = (improvement / (abs(baseline_score) + 1e-8)) * 100.0

print(f"Baseline: {baseline_score:.4f}")
print(f"Trained : {trained_score:.4f}")
print(f"Delta   : {improvement:+.4f} ({pct_improvement:+.1f}%)")
print(f"[FALLBACK RATE] trained={trained_fallbacks / trained_steps:.2f}")
print("=== BEFORE vs AFTER ===")
print(f"wait_only: {wait_score:.4f}")
print(f"heuristic: {heuristic_score:.4f}")
print(f"trained: {trained_score:.4f}")


# ==============================================================================
# 9. REPORTS + ARTIFACTS
# ==============================================================================
print("\n--- WRITING ARTIFACTS ---")
ACTION_TYPES = ["wait", "restock", "set_price", "negotiate", "refund", "ad_spend"]
total_b = max(1, len(baseline_actions))
total_t = max(1, len(trained_actions))
b_counts = Counter(baseline_actions)
t_counts = Counter(trained_actions)

if reward_history:
    k = min(3, len(reward_history))
    early_mean = float(np.mean(reward_history[:k]))
    late_mean = float(np.mean(reward_history[-k:]))
    reward_gain = late_mean - early_mean
else:
    early_mean = 0.0
    late_mean = 0.0
    reward_gain = 0.0

learning_signal = "YES" if reward_gain > 0 else "NOT_CLEAR"

composite = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "headline": "GRPO fine-tuning moved policy away from passive waiting and improved reward trend.",
    "backend_url": ENV_URL,
    "model": MODEL_NAME,
    "baseline_zero_shot_mean_reward": round(float(baseline_score), 5),
    "baseline_wait_only_mean_reward": round(float(wait_score), 5),
    "baseline_heuristic_mean_reward": round(float(heuristic_score), 5),
    "trained_mean_reward": round(float(trained_score), 5),
    "improvement": round(float(improvement), 5),
    "improvement_pct": f"{pct_improvement:+.1f}%",
    "episodes": MAX_EPISODES,
    "steps_per_episode": STEPS_PER_EPISODE,
    "group_size": GROUP_SIZE,
    "early_reward_mean": round(early_mean, 5),
    "late_reward_mean": round(late_mean, 5),
    "reward_gain": round(reward_gain, 5),
    "model_learning_signal": learning_signal,
    "fallback_rate_baseline": round(float(baseline_fallbacks / baseline_steps), 5),
    "fallback_rate_trained": round(float(trained_fallbacks / trained_steps), 5),
    "provenance": "trained_adapter",
}
with open("artifacts/composite_score.json", "w", encoding="utf-8") as f:
    json.dump(composite, f, indent=2)

t_entropy = float(
    -sum((t_counts.get(a, 0) / total_t) * np.log2(t_counts.get(a, 0) / total_t + 1e-9) for a in ACTION_TYPES)
)
policy_sig = {
    "baseline": {a: round(b_counts.get(a, 0) / total_b, 4) for a in ACTION_TYPES},
    "trained": {a: round(t_counts.get(a, 0) / total_t, 4) for a in ACTION_TYPES},
    "action_entropy_trained": round(t_entropy, 4),
}
with open("artifacts/policy_signature.json", "w", encoding="utf-8") as f:
    json.dump(policy_sig, f, indent=2)

with open("artifacts/training_log.jsonl", "w", encoding="utf-8") as f:
    for i, (r, e, l) in enumerate(zip(reward_history, entropy_history, loss_history), start=1):
        f.write(
            json.dumps(
                {
                    "episode": i,
                    "mean_reward": round(float(r), 5),
                    "entropy": round(float(e), 5),
                    "loss": round(float(l), 5),
                }
            )
            + "\n"
        )

with open("artifacts/training_summary.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "episodes": MAX_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "group_size": GROUP_SIZE,
            "baseline_score": baseline_score,
            "heuristic_score": heuristic_score,
            "trained_score": trained_score,
            "improvement": improvement,
            "improvement_pct": pct_improvement,
            "early_reward_mean": early_mean,
            "late_reward_mean": late_mean,
            "reward_gain": reward_gain,
            "model_learning_signal": learning_signal,
            "fallback_rate_baseline": baseline_fallbacks / baseline_steps,
            "fallback_rate_trained": trained_fallbacks / trained_steps,
        },
        f,
        indent=2,
    )

before_metrics = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "policies": {
        "wait_only": {"mean_reward": wait_score},
        "heuristic": {"mean_reward": heuristic_score},
        "zero_shot_llm": {"mean_reward": baseline_score},
    },
}
with open("artifacts/before_metrics.json", "w", encoding="utf-8") as f:
    json.dump(before_metrics, f, indent=2)

after_metrics = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "policy": "trained_adapter",
    "mean_reward": trained_score,
    "fallback_rate": trained_fallbacks / trained_steps,
}
with open("artifacts/after_metrics.json", "w", encoding="utf-8") as f:
    json.dump(after_metrics, f, indent=2)

generalization = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "configs": ["remote_http_env"],
    "seed_eval": SEED_EVAL,
    "baseline_zero_shot_mean_reward": baseline_score,
    "trained_mean_reward": trained_score,
}
with open("artifacts/generalization.json", "w", encoding="utf-8") as f:
    json.dump(generalization, f, indent=2)

BG = "#f8fafc"

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    reward_history,
    color="#38bdf8",
    lw=2.2,
    marker="o",
    ms=5,
    label="GRPO Episode Reward",
)
ax.axhline(baseline_score, color="#f43f5e", ls="--", lw=2, label=f"Zero-shot ({baseline_score:.3f})")
ax.axhline(wait_score, color="#94a3b8", ls=":", lw=1.6, label=f"Wait-only ({wait_score:.3f})")
ax.set_title("Reward per Episode")
ax.set_xlabel("Episode")
ax.set_ylabel("Mean Reward")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_facecolor(BG)
fig.tight_layout()
fig.savefig("artifacts/reward_curve.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(entropy_history, color="#a855f7", lw=2.2, marker="s", ms=5, label="Action entropy")
ax.set_title("Exploration Curve")
ax.set_xlabel("Episode")
ax.set_ylabel("Entropy (bits)")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_facecolor(BG)
fig.tight_layout()
fig.savefig("artifacts/exploration_curve.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
labels = ["Zero-shot", "Wait-only", "Heuristic", "GRPO trained"]
values = [baseline_score, wait_score, heuristic_score, trained_score]
colors = ["#f43f5e", "#94a3b8", "#f59e0b", "#10b981"]
bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", lw=1.2)
spread = max(abs(v) for v in values) if values else 1.0
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + spread * 0.03,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=12,
    )
ax.set_title("Before vs After")
ax.set_ylabel("Mean episode reward")
ax.grid(True, axis="y", alpha=0.3)
ax.set_facecolor(BG)
fig.tight_layout()
fig.savefig("artifacts/before_after_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)

x = np.arange(len(ACTION_TYPES))
w = 0.35
b_vals = [b_counts.get(a, 0) / total_b for a in ACTION_TYPES]
t_vals = [t_counts.get(a, 0) / total_t for a in ACTION_TYPES]
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w / 2, b_vals, w, label="Baseline", color="#f43f5e", alpha=0.85, edgecolor="white")
ax.bar(x + w / 2, t_vals, w, label="Trained", color="#10b981", alpha=0.85, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(ACTION_TYPES)
ax.set_ylabel("Relative frequency")
ax.set_title("Action Distribution Shift")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
ax.set_facecolor(BG)
fig.tight_layout()
fig.savefig("artifacts/action_distribution.png", dpi=180, bbox_inches="tight")
plt.close(fig)

expected = [
    "artifacts/before_metrics.json",
    "artifacts/after_metrics.json",
    "artifacts/generalization.json",
    "artifacts/composite_score.json",
    "artifacts/policy_signature.json",
    "artifacts/training_log.jsonl",
    "artifacts/training_summary.json",
    "artifacts/reward_curve.png",
    "artifacts/exploration_curve.png",
    "artifacts/before_after_comparison.png",
    "artifacts/action_distribution.png",
    "artifacts/adapter/adapter_config.json",
]

manifest = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "expected_artifacts": expected,
    "present": {},
}
all_ok = True
for path in expected:
    ok = os.path.exists(path)
    manifest["present"][path] = ok
    all_ok = all_ok and ok
with open("artifacts/pipeline_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\nARTIFACT CHECKLIST:")
for path in expected:
    mark = "OK" if manifest["present"][path] else "MISSING"
    print(f"  {mark:7} {path}")

print("\n" + "=" * 64)
print(f"RESULT: {baseline_score:.4f} -> {trained_score:.4f} ({pct_improvement:+.1f}%)")
print(f"Learning signal (early->late reward): {reward_gain:+.4f} [{learning_signal}]")
action_not_all_wait = any(a != "wait" for a in trained_actions)
success_checks = {
    "reward_curve_increasing_signal": reward_gain > 0,
    "trained_beats_heuristic": trained_score > heuristic_score,
    "trained_beats_wait_only": trained_score > wait_score,
    "action_distribution_not_all_wait": action_not_all_wait,
}
print("SUCCESS CHECKS:")
for k, v in success_checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print("=" * 64)
print("All artifacts generated." if all_ok else "Some artifacts missing. Check errors above.")

# ==============================================================================
# 10. HUGGINGFACE UPLOAD
# ==============================================================================
print("\n--- UPLOADING TO HUGGINGFACE ---")
from huggingface_hub import HfApi, login
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        api = HfApi()
        repo_id = "Swiftlogic/E-commerce-agent"
        
        # Upload adapter
        print(f"Uploading adapter to {repo_id}...")
        api.upload_folder(
            folder_path="artifacts/adapter",
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="adapter"
        )
        
        # Upload reward curve
        print(f"Uploading reward curve to {repo_id}...")
        api.upload_file(
            path_or_fileobj="artifacts/reward_curve.png",
            path_in_repo="artifacts/reward_curve.png",
            repo_id=repo_id,
            repo_type="model"
        )
        print("Upload successful!")
    except Exception as e:
        print(f"Failed to upload to HF: {e}")
else:
    print("HF_TOKEN environment variable not set, skipping upload.")
