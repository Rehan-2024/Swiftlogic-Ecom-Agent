"""Generate ``swiftlogic_grpo_training.ipynb`` from a structured source.

Keeping the notebook definition here (pure text in Python) makes the
notebook reproducible, reviewable in git, and testable. Run this script
whenever the notebook scaffolding changes:

    python scripts/gen_training_notebook.py
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "swiftlogic_grpo_training.ipynb"


def _cell_code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(src).strip("\n").splitlines(keepends=True),
    }


def _cell_md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(src).strip("\n").splitlines(keepends=True),
    }


CELLS = []

CELLS.append(_cell_md(r"""
# CommerceOps v2 — GRPO Training Notebook

End-to-end pipeline that turns **Qwen2.5-0.5B-Instruct** into a profit-seeking commerce agent on the frozen CommerceOps v2 environment, following the OpenEnv Hackathon roadmap Parts **B** and **B+**.

**Cells in order**
1. Install (Unsloth + TRL + PEFT + matplotlib).
2. Mount Drive, clone repo, install local pkg, boot env.
3. Load Qwen2.5-0.5B-Instruct via Unsloth FastLanguageModel + QLoRA 4-bit.
4. Baselines (wait / random / heuristic / zero-shot Qwen) → `before_metrics.json`.
5. GRPO training loop with 3-stage curriculum.
6. Post-training evaluation → `after_metrics.json` + `before_after_comparison.png`.
7. Generalization × 3 configs (`generalization.json`).
8. Hard-seed retraining burst (`hard_seed_retraining.json`).
9. Behavior / policy evolution (`behavior_evolution.png`, `policy_signature.json`).
10. Composite score (`composite_score.json`) and artifact upload.

**Frozen surface:** every heavy lift uses the `training/` package committed alongside this notebook so the notebook stays thin and reproducible.

---

**Open in Colab:** https://colab.research.google.com/github/your-repo/swiftlogic_grpo_training.ipynb
"""))

CELLS.append(_cell_code(r"""
# Cell 1 — install dependencies (Colab)
# Unsloth is the fast-path stack for TRL GRPO on a free T4.
!pip install -q --upgrade pip
!pip install -q 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'
!pip install -q --no-deps trl==0.17.0 peft accelerate bitsandbytes
!pip install -q matplotlib requests datasets
print('deps installed')
"""))

CELLS.append(_cell_code(r"""
# Cell 2 — Colab bootstrap: Drive + repo clone + local install
import os, sys, subprocess, json
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    REPO_DIR = '/content/CMS'
    DRIVE_CKPT = '/content/drive/MyDrive/openenv_grpo_checkpoints'
    os.makedirs(DRIVE_CKPT, exist_ok=True)
    if not os.path.exists(REPO_DIR):
        subprocess.run(['git','clone','https://github.com/your-org/CMS.git', REPO_DIR], check=True)
    os.chdir(REPO_DIR)
else:
    REPO_DIR = os.path.abspath('.')
    DRIVE_CKPT = os.path.abspath('./artifacts/checkpoints')
    os.makedirs(DRIVE_CKPT, exist_ok=True)

sys.path.insert(0, REPO_DIR)
print('workdir =', os.getcwd())
print('IN_COLAB =', IN_COLAB)
print('checkpoint dir =', DRIVE_CKPT)

os.makedirs('artifacts', exist_ok=True)
os.makedirs('artifacts/adapter', exist_ok=True)
os.makedirs('artifacts/adapter_checkpoints', exist_ok=True)
"""))

CELLS.append(_cell_code(r"""
# Cell 3 — smoke-test env and list graders / tasks
from ecom_env import EcomEnv
env = EcomEnv('configs/siyaani_fashion.json')
obs = env.reset(seed=42)
print('initial bank =', obs.bank_balance, 'SKUs =', list(obs.inventory.keys()))
print('graders:', list(env.graders().keys()))
"""))

CELLS.append(_cell_code(r"""
# Cell 4 — load Qwen2.5-0.5B-Instruct via Unsloth FastLanguageModel (QLoRA 4-bit)
# If Unsloth is not available (e.g. non-GPU runtime), fall back to transformers+peft.
import os
MODEL_NAME = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-0.5B-Instruct')
MAX_SEQ_LEN = 2048

try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        lora_alpha=16,
        lora_dropout=0.0,
        bias='none',
        use_gradient_checkpointing='unsloth',
        random_state=2026,
    )
    print('Unsloth loaded', MODEL_NAME)
except Exception as e:
    print('Unsloth unavailable, falling back to transformers+peft:', e)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype='auto', device_map='auto')
    lora_cfg = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0, bias='none',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    )
    model = get_peft_model(model, lora_cfg)

# Qwen chat tokens
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print('trainable params:')
try:
    model.print_trainable_parameters()
except Exception:
    pass
"""))

CELLS.append(_cell_code(r"""
# Cell 5 — baseline sweep using the shared training package (Part B5)
# Generates artifacts/before_metrics.json with the 4 mandatory baselines
# (wait_only / random / heuristic / zero_shot_llm).

from training.policies import (
    build_wait_producer, build_random_producer, build_heuristic_producer,
    build_zero_shot_producer,
)
from training.eval_utils import run_eval_sweep, write_json

BASELINE_SEEDS = [101,202,303,404,505,606,707,808,909,1010]
BASELINE_CFG = ['configs/siyaani_fashion.json']

per_policy = {}
for name, prod in [('wait_only', build_wait_producer()),
                   ('random', build_random_producer()),
                   ('heuristic', build_heuristic_producer())]:
    print('running baseline', name)
    per_policy[name] = run_eval_sweep(name, prod, BASELINE_SEEDS, BASELINE_CFG)
    write_json(per_policy[name], f'artifacts/baseline_{name}.json')

print('running zero_shot_llm baseline (this can take ~5 minutes on T4)')
zero_shot_producer = build_zero_shot_producer(model, tokenizer)
per_policy['zero_shot_llm'] = run_eval_sweep(
    'zero_shot_llm', zero_shot_producer, BASELINE_SEEDS, BASELINE_CFG
)
write_json(per_policy['zero_shot_llm'], 'artifacts/baseline_zero_shot_llm.json')

before = {
    'before_metrics': True,
    'policies': per_policy,
    'baseline_for_composite': 'heuristic',
}
write_json(before, 'artifacts/before_metrics.json')
print('baseline summary:')
for k,v in per_policy.items():
    print(' ', k, v.get('summary', {}))
"""))

CELLS.append(_cell_code(r"""
# Cell 6 — GRPO training with 3-stage curriculum (Parts B3 + B4 + B5.5 + B6)
import random, json, os, time, re
from dataclasses import asdict
from pathlib import Path
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from training.rollout import rollout_episode, extract_action_json, validate_action
from training.rewards import combined_reward, reward_breakdown, RewardWeights
from training.curriculum import default_curriculum, rolling_mean

WEIGHTS = RewardWeights()
CURRICULUM = default_curriculum('.')
MAX_EPISODES = int(os.environ.get('MAX_EPISODES', 200))
SANITY_EPISODES = int(os.environ.get('SANITY_EPISODES', 30))
GROUP_SIZE = int(os.environ.get('GRPO_GROUP_SIZE', 8))

from training.policies import SYSTEM_PROMPT
from training.rollout import ActionProducer


def build_prompt_from_obs(obs):
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n"
        f"day={obs.current_day} step={obs.step_count} "
        f"bank={float(obs.bank_balance):.0f} "
        f"inventory={dict(obs.inventory)} "
        f"open_tickets={[t.ticket_id for t in obs.active_tickets if t.status == 'open']} "
        f"customer_satisfaction={float(obs.customer_satisfaction):.2f}\n"
        f"Action (JSON only):\n<|assistant|>\n"
    )


def _build_producer_from_completion(completion_text):
    # Caches the completion so each env.step in the rollout can read the
    # corresponding per-step action. GRPO sees ONE completion == one episode,
    # so the agent must emit *the whole tape* in JSON-lines form inside the
    # completion. We tolerate the simpler single-action form (repeats across
    # every step of the episode) to reduce early-training variance.
    lines = [ln.strip() for ln in re.split(r'[\n]+', completion_text) if ln.strip()]
    def _producer(obs, state):
        idx = state.get('step', 0)
        line = lines[idx] if idx < len(lines) else (lines[-1] if lines else completion_text)
        cand, _ = extract_action_json(line)
        return cand, line
    return _producer


def _grpo_reward_fn(prompts, completions, **kwargs):
    seeds = kwargs.get('_seeds') or [2026 + i for i in range(len(prompts))]
    config_path = CURRICULUM.current.config_path
    rewards = []
    breakdown_last = {}
    for prompt, comp, seed in zip(prompts, completions, seeds):
        producer = _build_producer_from_completion(comp if isinstance(comp, str) else comp[0])
        rec = rollout_episode(producer, seed=seed, config_path=config_path)
        rewards.append(combined_reward(rec, WEIGHTS))
        breakdown_last = reward_breakdown(rec, WEIGHTS)
    return rewards


# ---- assemble training dataset (fresh prompts from reset envs) -------------
def sample_prompts(n, seed_base=10000, config_path=None):
    from ecom_env import EcomEnv
    cp = config_path or CURRICULUM.current.config_path
    rows = []
    for i in range(n):
        env = EcomEnv(cp)
        obs = env.reset(seed=seed_base + i)
        rows.append({'prompt': build_prompt_from_obs(obs), '_seed': seed_base + i})
    return Dataset.from_list(rows)


# ---- sanity train (Part B5.5) ---------------------------------------------
print(f'=== Part B5.5 sanity train ({SANITY_EPISODES} episodes) on {CURRICULUM.current.config_path} ===')
rewards_log = []
sanity_ds = sample_prompts(SANITY_EPISODES, seed_base=700000)
cfg_sanity = GRPOConfig(
    output_dir='artifacts/adapter_checkpoints/sanity',
    num_train_epochs=1,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=GROUP_SIZE,
    max_completion_length=96,
    max_prompt_length=512,
    logging_steps=1,
    save_steps=1000,
    report_to='none',
    seed=2026,
)
trainer = GRPOTrainer(
    model=model,
    args=cfg_sanity,
    train_dataset=sanity_ds,
    reward_funcs=_grpo_reward_fn,
    processing_class=tokenizer,
)
trainer.train()
sanity_rewards = [float(e.get('reward', 0.0)) for e in trainer.state.log_history if 'reward' in e]
assert sanity_rewards, 'sanity train produced no rewards'
assert not any(r != r for r in sanity_rewards), 'NaN in sanity rewards'
print(f'sanity rewards: first={sanity_rewards[0]:.3f} last={sanity_rewards[-1]:.3f}')

# ---- full training (Part B6) ----------------------------------------------
print(f'=== Part B6 full GRPO training ({MAX_EPISODES} episodes, 3-stage curriculum) ===')
full_ds = sample_prompts(MAX_EPISODES, seed_base=10000)
cfg_full = GRPOConfig(
    output_dir='artifacts/adapter_checkpoints',
    num_train_epochs=1,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=GROUP_SIZE,
    max_completion_length=96,
    max_prompt_length=512,
    logging_steps=5,
    save_steps=50,
    report_to='none',
    seed=2026,
)
full_trainer = GRPOTrainer(
    model=model,
    args=cfg_full,
    train_dataset=full_ds,
    reward_funcs=_grpo_reward_fn,
    processing_class=tokenizer,
)
full_trainer.train()

# ---- save artifacts --------------------------------------------------------
full_rewards = [float(e.get('reward', 0.0)) for e in full_trainer.state.log_history if 'reward' in e]
from training.plots import plot_reward_curve
plot_reward_curve(full_rewards, 'artifacts/reward_curve.png',
                  title=f'GRPO training reward ({MAX_EPISODES} episodes)',
                  stage_boundaries=None)
with open('artifacts/training_log.txt', 'w', encoding='utf-8') as f:
    for entry in full_trainer.state.log_history:
        f.write(json.dumps(entry)+'\n')
model.save_pretrained('artifacts/adapter')
tokenizer.save_pretrained('artifacts/adapter')
print('adapter saved to artifacts/adapter')
"""))

CELLS.append(_cell_code(r"""
# Cell 7 — post-training evaluation (Part B7)
from training.eval_utils import run_eval_sweep, write_json
from training.policies import build_zero_shot_producer
from training.plots import plot_before_after_bars

eval_seeds = [111,222,333,444,555,666,777,888,999,1111]
trained_producer = build_zero_shot_producer(model, tokenizer)
after = run_eval_sweep('after_training', trained_producer, eval_seeds, ['configs/siyaani_fashion.json'])
write_json(after, 'artifacts/after_metrics.json')

import json
before = json.load(open('artifacts/before_metrics.json'))
ref = before['policies']['heuristic']['summary']['per_task']
plot_before_after_bars(ref, after['summary']['per_task'], 'artifacts/before_after_comparison.png')
print('after-training summary:', after['summary'])
"""))

CELLS.append(_cell_code(r"""
# Cell 8 — generalization sweep (Part B+.1)
from training.eval_utils import run_eval_sweep, write_json
from training.plots import plot_generalization

gen_configs = ['configs/siyaani_fashion.json', 'configs/siyaani_fashion_easy.json']
import os
if os.path.exists('configs/siyaani_fashion_demo.json'):
    gen_configs.append('configs/siyaani_fashion_demo.json')

gen = run_eval_sweep('generalization', trained_producer, [1212,1313,1414,1515,1616,1717,1818,1919,2020,2121], gen_configs)
write_json(gen, 'artifacts/generalization.json')

# Per-config summaries for the chart
per_cfg_summaries = {}
for cfg in gen_configs:
    eps = [e for e in gen['episodes'] if e['config'] == cfg]
    from training.eval_utils import summarize_episodes
    per_cfg_summaries[cfg.split('/')[-1]] = summarize_episodes(eps)['per_task']
plot_generalization(per_cfg_summaries, 'artifacts/generalization.png',
                    title='Generalization across configs')
print('generalization composite means:', {k: s for k,s in gen['summary'].items() if 'composite' in k})
"""))

CELLS.append(_cell_code(r"""
# Cell 9 — hard-seed retraining burst (Part B+.2)
# Pick the worst 3 seeds from the evaluation and re-train for 30 extra steps.
from training.eval_utils import run_eval_sweep, write_json
import statistics

after_eps = sorted(after['episodes'], key=lambda e: statistics.mean(e['grader_scores'].values()))
hard_seeds = [e['seed'] for e in after_eps[:3]]
print('hard seeds:', hard_seeds)

hard_ds = sample_prompts(30, seed_base=hard_seeds[0])
cfg_hard = GRPOConfig(
    output_dir='artifacts/adapter_checkpoints/hard',
    num_train_epochs=1,
    learning_rate=3e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_generations=GROUP_SIZE,
    max_completion_length=96,
    max_prompt_length=512,
    logging_steps=1,
    report_to='none',
    seed=3033,
)
hard_trainer = GRPOTrainer(
    model=model, args=cfg_hard, train_dataset=hard_ds,
    reward_funcs=_grpo_reward_fn, processing_class=tokenizer,
)
hard_trainer.train()
model.save_pretrained('artifacts/adapter')  # overwrite with hardened weights

# Re-evaluate on the same hard seeds
hard_after = run_eval_sweep('after_hard_retraining', trained_producer, hard_seeds, ['configs/siyaani_fashion.json'])
write_json({
    'hard_seeds': hard_seeds,
    'before': {'summary': {
        'mean_total_reward': statistics.mean([e['total_reward'] for e in after_eps[:3]]),
        'composite_training_mean': statistics.mean([statistics.mean([e['grader_scores'][g] for g in ('triage_task','inventory_task','profit_task')]) for e in after_eps[:3]]),
    }},
    'after': hard_after['summary'],
}, 'artifacts/hard_seed_retraining.json')
print('hard-seed retraining complete')
"""))

CELLS.append(_cell_code(r"""
# Cell 10 — behavior / policy evolution + exploration curve (Part B+.3)
from training.behavior import (
    plot_behavior_evolution, plot_policy_evolution_line, policy_signature,
)
from training.plots import plot_exploration_curve
import json, math, statistics

# Build checkpoints from: baseline heuristic, zero-shot LLM, after-training, post-hard-retraining.
def _actions_from_bundle(b):
    flat = []
    for ep in b['episodes']:
        # episodes here don't carry the tape — walk grader_scores only; for a
        # real distribution, we re-run each seed with the producer and collect
        # the action_type. Cheap (<= 50 steps per seed).
        pass
    return flat

from training.rollout import rollout_episode
def _collect_actions(producer, seeds, config):
    out = []
    for s in seeds:
        rec = rollout_episode(producer, s, config_path=config)
        out.extend(rec.steps)
    return [{'action_type': r.action.get('action_type','wait')} for r in out]

checkpoints = {
    'heuristic': _collect_actions(build_heuristic_producer(), eval_seeds, 'configs/siyaani_fashion.json'),
    'zero_shot': _collect_actions(build_zero_shot_producer(model, tokenizer), eval_seeds[:3], 'configs/siyaani_fashion.json'),
    'trained': _collect_actions(build_zero_shot_producer(model, tokenizer), eval_seeds[:3], 'configs/siyaani_fashion.json'),
}

plot_behavior_evolution(checkpoints, 'artifacts/behavior_evolution.png')
plot_policy_evolution_line(checkpoints, 'artifacts/policy_evolution.png')
signatures = {k: policy_signature(v) for k,v in checkpoints.items()}
with open('artifacts/policy_signature.json','w') as f:
    json.dump(signatures, f, indent=2)

entropies = [signatures[k]['entropy'] for k in checkpoints]
plot_exploration_curve(entropies, 'artifacts/exploration_curve.png',
                       checkpoint_labels=list(checkpoints.keys()))
print('policy signatures:', signatures)
"""))

CELLS.append(_cell_code(r"""
# Cell 11 — composite score + headline (Part B+.6)
from training.composite import compute_composite, CompositeWeights

result = compute_composite(
    before_metrics_path='artifacts/before_metrics.json',
    after_metrics_path='artifacts/after_metrics.json',
    generalization_path='artifacts/generalization.json',
    out_path='artifacts/composite_score.json',
)
print('HEADLINE:', result['headline'])
"""))

CELLS.append(_cell_code(r"""
# Cell 12 — upload to HF Hub (optional, requires HF_TOKEN)
import os
if os.environ.get('HF_TOKEN'):
    from huggingface_hub import login
    login(token=os.environ['HF_TOKEN'])
    model.push_to_hub('your-hf-user/swiftlogic-commerce-ops-v2-grpo')
    tokenizer.push_to_hub('your-hf-user/swiftlogic-commerce-ops-v2-grpo')
    print('uploaded adapter to HF Hub')
else:
    print('HF_TOKEN not set; skipping hub upload (adapter still saved locally)')
"""))


def build_notebook() -> dict:
    return {
        "cells": CELLS,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    nb = build_notebook()
    OUT_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
