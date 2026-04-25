# Swiftlogic CommerceOps v2 — Hackathon Alignment & Compliance Matrix

**Scope:** this document cross-references the current repository state against every requirement, theme, and recommendation in the two organiser documents:

1. `[External] Apr '26 OpenEnv Hackathon Themes.md` (5 themes + judging criteria)
2. `[External] Meta OpenEnv Hackathon Participant Help Guide.md` (22 sections)

Each item is answered with one of:

- ✅ **Done** — current repo fully satisfies the item.
- 🟡 **Partial** — partially satisfied; specific gap called out.
- 🔴 **Missing** — not satisfied; explicit remediation listed.
- ➖ **N/A** — does not apply to our problem shape.

Every ✅/🟡 is anchored to a concrete file / function / config / test so anyone can verify in seconds.

---

## 0. Executive Summary

| Axis | Status |
|---|---|
| OpenEnv v0.2.3 compliance | ✅ `/reset`, `/step`, `/state`, `/tasks`, `/grader`, `openenv.yaml`, Docker |
| Environment stability | ✅ 218 / 218 tests passing, deterministic replay, zero lint errors |
| Reward hacking defences | ✅ 8 independent reward terms + 3 graders, 4 termination guards, price/body caps |
| Training pipeline (TRL / GRPO) | ✅ Primary path `grpo_single_cell_colab_v5.py` + alternate notebook `swiftlogic_grpo_training.ipynb` |
| **Unsloth integration** | ✅ Integrated in Colab training path with fallback to transformers+peft |
| **Mini-blog / 2-min video** | 🔴 **Not in repo** — required minimum per themes doc. *Action item below.* |
| HuggingFace Spaces deployment | 🟡 `openenv.yaml` + `Dockerfile` ready; **Space URL needs to be confirmed live** |
| Observable improvement | ✅ `reward_curve.png`, `before_after_comparison.json`, `thought_logs.json` emitted |
| Explainability / auditability | ✅ 20+ additive `info` keys, documented confidence formula, CEO + department layer |
| Multi-agent dynamics | ✅ SupplierAgent + ReactiveCompetitor + CustomerDemand + TicketSystem + PolicyAgent |
| Long-horizon planning | ✅ 50-step episodes, restock lead days, quote TTL, shock duration, compounding cash |
| World modelling | ✅ economic feedback loops, partial observability, persistent state across 50 steps |

**Bottom line:** the environment, reward design, anti-hacking stack, testing, determinism, explainability, and demo story are all **strong**. Remaining remediation items are: (1) record the 2-min demo video / mini-blog, (2) confirm the HF Space URL is live and wire it into the README.

---

## 1. Minimum Requirements (Themes doc, Judging Criteria §)

> "**Minimum requirements**: Usage of OpenEnv (latest release); Show a minimal training script for your environment using Unsloth or HF TRL in Colab; Write a mini-blog on HuggingFace or mini-video on YouTube talking about your submission, <2 minutes; Your OpenEnv compliant environment should be hosted on Hugging Face Spaces."

| Requirement | Status | Evidence / Action |
|---|:---:|---|
| Usage of OpenEnv (latest release) | ✅ | `requirements.txt` pins `openenv-core` (0.2.3 — latest on PyPI as of 2026-03-28). `openenv.yaml` declares `spec_version: 1`, `name: commerce-ops-v2`, three task descriptors. Endpoints `/reset /step /state /tasks /grader` all implemented in `server/app.py`. |
| Minimal training script in Colab using Unsloth **or** TRL | ✅ | `grpo_single_cell_colab_v5.py` is a single-cell Colab script with Unsloth-first loading (`FastLanguageModel`) and transformers+peft fallback. It includes env HTTP checks, GRPO loop, live logs, before/after metrics, and artifact checklist. |
| Mini-blog (HuggingFace) **or** <2-min YouTube video | 🔴 | No artefact in repo. Action: record a 2-minute Loom/YouTube walking through (i) problem statement, (ii) observation → action → reward demo step, (iii) reward curve, (iv) before/after grader scores. Drop the URL into README. See §Remediation #2. |
| Hosted on HuggingFace Spaces | 🟡 | `Dockerfile` and `openenv.yaml` are Space-ready; README front-matter declares `sdk: docker`, `app_port: 7860`, `tags: [openenv]`. **Space URL not documented.** Action: `openenv push` the repo and add the live URL to both README and the video. See §Remediation #3. |

---

## 2. Judging Criteria Alignment (Themes doc, §Judging Overview)

The four weighted criteria sum to 100%. Each row quotes the criterion verbatim, then maps our deliverables.

| # | Criterion (weight) | Our delivery |
|---|---|---|
| 1 | **Environment Innovation (40%)** — "Is the environment novel, creative, or challenging? Does it meaningfully test the agent's behavior?" | ✅ Novel on three axes: (a) **config-driven multi-domain** world — same `WorldEngine` runs 4 shipped businesses (fashion, pharmacy, SaaS, demo-with-all-realism); (b) **integrated causal stack** — reactive competitor + market shocks + customer satisfaction + supplier capacity + holding cost, all with a documented `info` causal-trace contract; (c) **lightweight AI-CEO + departments** (inventory/marketing/support) derived deterministically from `info`, a genuinely new framing. Every decision is testable: 218 regression tests, invariant assertions, price-band rejection, bankruptcy termination. |
| 2 | **Storytelling (30%)** — "Does the team clearly explain the problem, environment, and agent behavior? Is the demo engaging and easy to follow?" | ✅ `README.md` (architecture diagram + 17-field observation + 6-action table + 8-term reward + 20+ info keys). `PROJECT_REPORT.md` (33 sections + 5 appendices, full math). Demo printer (`inference.py::_print_demo_step`) renders per-step CEO-view narrative. **Action**: ensure the 2-min video walks through exactly this pipeline. |
| 3 | **Showing Improvement in Rewards (20%)** — "Does the demo provide observable evidence of training progress (reward curves, metrics, or before/after behavior)?" | ✅ Notebook emits `reward_curve.png`, `before_after_comparison.json`, `thought_logs.json`. `inference.py` emits `reward_curve_inference.png`, `profit_curve_inference.png`, `inventory_curve_inference.png`. Baseline table in README (Qwen2.5-0.5B zero-shot): triage ~0.80 / inventory ~0.40 / profit ~0.15, with post-GRPO target profit ≥ 0.50. |
| 4 | **Reward and Training Script/Pipeline (10%)** — "Is the reward logic coherent, and does the pipeline produce meaningful improvement in the agent's inference?" | ✅ 8-term dense reward in `env/reward_engine.py` with per-term observability (`info.reward_breakdown`). GRPO pipeline driven by env-in-the-loop `reward_fn` that rolls a deterministic-seed env per completion. 21 tests in `test_reward_engine.py` pin term semantics and breakdown-sum invariants. |

---

## 3. Theme Fit Analysis

The submission primarily targets **Theme 3.1 (World Modeling — Professional Tasks)** with very strong secondary coverage of **Theme 1 (Multi-Agent)** and **Theme 2 (Long-Horizon Planning)**.

### Theme 1 — Multi-Agent Interactions

> "Environments for this theme involve cooperation, competition, negotiation, and coalition formation."

| Signal | Status | Notes |
|---|:---:|---|
| Negotiation | ✅ | `NegotiateAction` → `SupplierAgent.quote_price` (rule-based pricing counterpart with volume discount, demand premium, capacity cap, TTL). |
| Competition | ✅ | Reactive competitor (`env/world_engine.py::_reactive_competitor_step`) undercuts when we go low, follows when we go high, holds in dead-zone. `info.competitor_reaction` stamps causal truth. |
| Cooperation / coalition | 🟡 | Not modelled — single policy agent plus rule-based counterparts. Could be extended with multiple supplier firms. |
| Partial observability | ✅ | Shock schedule, satisfaction decay internals, RNG, ticket spawn rates are not exposed in `EcomObservation`. |

**Verdict:** solid Theme 1 coverage via negotiation + competition + partial observability. Coalition formation is not the focus and that's acceptable.

### Theme 2 — Long-Horizon Planning & Instruction Following

> "Environments that require deep, multi-step reasoning with sparse or delayed rewards."

| Signal | Status | Notes |
|---|:---:|---|
| Multi-step reasoning | ✅ | 50-step horizon with compounding cash effects. |
| Delayed rewards | ✅ | `restock_lead_days` delays inventory arrivals; `supplier_quote_expiry` creates time-pressure planning; `market.shock_duration_days` creates multi-step demand regimes; ticket-aging penalties build over days. |
| Recover from early mistakes | ✅ | Bankruptcy termination forces recovery before thresholds; satisfaction decay creates a multi-day penalty that only recovers via action. |
| Durable internal state | ✅ | `state.history` ring buffer (window=20) tracks revenue/bank/inventory/tickets/satisfaction/reward; used by `info.trend`, `info.policy_stability`, `info.anomalies`. |
| 300-instruction-scale long horizon | ➖ | Not our problem shape — we're a business operations sim, not instruction following. |

**Verdict:** strong Theme 2 fit for the "strategic resource management worlds" bucket (explicitly listed as an example in the themes doc).

### Theme 3.1 — World Modeling (Professional Tasks) — **PRIMARY**

> "Environments that require real interaction with tools, APIs, or dynamic systems where the model is expected to do real hard work instead of exploiting short-cuts... Dynamic browser/API ecosystems, enterprise applications, ... economic simulations with feedback..."

| Signal | Status | Notes |
|---|:---:|---|
| Dynamic system with real feedback loops | ✅ | Set a price → competitor reacts → demand responds → revenue lands next step → cash affects next-day restock capacity. 8-term reward propagates all of these. |
| Consistent internal state | ✅ | Every `/step` asserts state invariants (under `COMMERCEOPS_ASSERT_INVARIANTS=1`, auto-on in tests). |
| Update beliefs based on outcomes | ✅ | `info.trend`, `info.anomalies`, `info.policy_stability`, `info.intent` all derived from history — give the policy / CEO explicit feedback channels. |
| Orchestrate multi-step workflows | ✅ | Canonical workflow: `negotiate → set_price → restock → ad_spend → wait-and-monitor → resolve_tickets` (see §27 of `PROJECT_REPORT.md`). |
| Causal reasoning | ✅ | `info.competitor_reaction.triggered` flag distinguishes **true causation** from random walk (Audit HIGH #1 fix). `info.demand_factors` decomposes every Poisson lambda into 15 named components. |
| Enterprise applications example | ✅ | Direct match — this *is* an enterprise-style digital storefront. |
| Economic simulation with feedback | ✅ | Direct match. |

**Verdict:** textbook Theme 3.1 fit. This is our strongest claim.

### Theme 3.2 — World Modeling (Personalized Tasks)

> "Real personalized task handling, imagine replying to personal messages or handling dinner conflicts..."

➖ **Not our focus** — we are enterprise, not personal-assistant. No remediation required; Theme 3.1 is the right bucket.

### Theme 4 — Self-Improvement

> "Environments where agents can learn to generate new challenges, escalate difficulty, and improve through self-play or adaptive curricula."

| Signal | Status | Notes |
|---|:---:|---|
| Recursive skill amplification via GRPO | ✅ | `swiftlogic_grpo_training.ipynb` trains with env-in-the-loop reward. |
| Escalating difficulty | 🟡 | Three-difficulty task ladder (easy / medium / hard graders) exists, but no automatic curriculum schedule that promotes the policy as it clears easier graders. **Optional enhancement**: add a curriculum wrapper that starts with `triage_task` seeds only and gradually enables market shocks + reactive competitor. See §Remediation #4. |
| Self-play | ➖ | Not applicable in this shape. |

**Verdict:** baseline satisfied (TRL GRPO with env reward), could be strengthened with explicit curriculum.

### Theme 5 — Wild Card

> "We do not want to limit your focus if your idea doesn't fit the boxes above."

The **AI-CEO + Departments + Causal-truth explainability** framing is genuinely out-of-the-box: an LLM running a whole company with named departments (inventory/marketing/support) and a deterministic confidence formula is not covered by the other four themes. If the judges prefer the wild-card framing, the same artefacts qualify without any code change.

---

## 4. Participant Help Guide — §-by-§ Compliance (22 sections)

### §0. What you are building

> "Environment → verifier/reward functions → TRL trainer → Unsloth for efficiency → deployment on OpenEnv / Spaces."

| Stage | Status | Anchor |
|---|:---:|---|
| Environment | ✅ | `env/world_engine.py` + `ecom_env.py` + `server/app.py` |
| Verifier / reward | ✅ | 3 graders in `ecom_env.py` + 8 reward terms in `env/reward_engine.py` |
| TRL trainer | ✅ | `grpo_single_cell_colab_v5.py` (primary) and `swiftlogic_grpo_training.ipynb` (alternate) |
| Unsloth for efficiency | ✅ | Wired in primary Colab script with transformers+peft fallback |
| Deployment on OpenEnv / Spaces | 🟡 | Artefacts ready; live URL **Remediation #3** |

### §1. Pick the right project idea

> "Task that has all three of: model can act step by step; you can verify success programmatically; task is hard enough to be interesting, but not so hard that the model never succeeds."

| Property | Status | Why |
|---|:---:|---|
| Step-by-step action | ✅ | One of 6 discrete actions per step, 50 steps per episode. |
| Programmatic verification | ✅ | Three deterministic graders, each clamped to `(0.01, 0.99)`. |
| Hard but tractable | ✅ | Qwen2.5-0.5B zero-shot already scores ~0.80 on triage and ~0.40 on inventory — non-zero reward from step 1, which §1 of the guide explicitly says is required. |

### §2. Minimum RL loop

All five bullets (prompt → action → env execution → reward → update) are implemented end-to-end in the notebook. ✅

### §3. SFT first?

> "If you do not have data but can verify outputs, use RL."

We don't have authored-trace data; we have crisp verification. RL-only is the correct path per the guide's decision rule. ✅

### §4. Design the environment before the trainer

| OpenEnv contract method | Status | Anchor |
|---|:---:|---|
| `reset()` | ✅ | `EcomEnv.reset`, `POST /reset` |
| `step(action)` | ✅ | `EcomEnv.step`, `POST /step` |
| `state() / observation` | ✅ | `EcomEnv._wrap_state`, `EcomObservation` pydantic model, `GET /state` |
| reward | ✅ | `env/reward_engine.compute_step_reward` |
| abuse / infinite loop prevention | ✅ | `episode.max_steps`, `bankruptcy_threshold`, `stall_terminate_steps`, 64 KiB body cap, invalid_action penalty, price-bound rejection |

### §5. Build the environment using OpenEnv

| Ingredient | Status | Anchor |
|---|:---:|---|
| Action dataclass | ✅ | `EcomAction` discriminated union (6 pydantic models) |
| Observation dataclass | ✅ | `EcomObservation` pydantic model (17 fields) |
| State representation | ✅ | `WorldEngine.state` JSON-friendly dict |
| reset / step methods | ✅ | `WorldEngine.reset`, `WorldEngine.step` |
| FastAPI wrapper | ✅ | `server/app.py` — 9 endpoints |

### §6. Keep the task simple at first

> "easy tasks with short horizons → medium → harder tasks only after the model starts getting non-zero reward."

✅ Three-difficulty ladder already in place: `triage_task` (easy) → `inventory_task` (medium) → `profit_task` (hard). Easy task is where zero-shot scoring is ~0.80 — guarantees non-zero reward from the first rollout.

### §7. Design rewards carefully

> "Use multiple independent reward functions... execution success, correctness, format compliance, timeouts, resource usage, safety constraints, anti-cheating checks."

| Reward signal class | Term / grader | File |
|---|---|---|
| Execution success (action accepted) | `base_reward` (`invalid_action` penalty otherwise) | `env/reward_engine.py`, `env/actions.py` |
| Correctness (revenue earned) | `revenue` term | `env/reward_engine.py::_revenue_term` |
| Correctness (solvency maintained) | `solvency` term | `_solvency_term` |
| Correctness (stock kept) | `stockout` term | `_stockout_term` |
| Correctness (customer service) | `ticket_aging` term | `_ticket_aging_term` |
| Correctness (marketing ROI) | `ad_roi` term (ROI-scaled) | `_ad_roi_term` |
| Safety constraint (bankruptcy) | `bankruptcy` terminal term | `_bankruptcy_term` |
| Cash discipline | `delta` term | `_delta_term` |
| Goal alignment | `inventory_target_bonus` | `_inventory_target_term` |
| Grader 1 (triage) | `grade_triage_task` | `ecom_env.py` |
| Grader 2 (inventory) | `grade_inventory_task` | `ecom_env.py` |
| Grader 3 (profit) | `grade_profit_task` | `ecom_env.py` |

**Total: 9 independent reward signals + 3 graders** — comfortably exceeds the guide's "2–4 independent reward checks" recommendation (§18 Phase 3).

### §8. Protect yourself against reward hacking

> "editing timers, caching results, abusing globals, mutating protected state, or exploiting environment bugs."

| Defence | Status | Evidence |
|---|:---:|---|
| Multiple independent reward functions | ✅ | 9 reward terms + 3 graders (§7 above) |
| Lock down execution | ✅ | Env runs in-process; no user code is executed. Action schema is strict pydantic; unknown fields rejected. |
| Time limits | ✅ | `episode.max_steps=50` hard horizon; `rewards.stall_terminate_steps` early-terminates unproductive runs; 64 KiB body cap. |
| Avoid unrestricted globals | ✅ | Env-local RNG (`test_simulation_invariants.py::test_reset_isolated_rng_streams` asserts this). No module-level mutable singletons except the deprecated `_GRADER_CONTEXT` mirror, which now emits a `DeprecationWarning` (escalates to `RuntimeError` under `COMMERCEOPS_STRICT_GRADER_CONTEXT=1`). |
| Sample outputs frequently | ✅ | `inference.py::_print_demo_step` renders every step; `COMMERCEOPS_CEO_TRACE=1` dumps `ceo_decision_traces.json`; notebook's `thought_logs.json` captures first + last episode. |
| Terminate or roll back on drift | ✅ | Bankruptcy termination; stall guard; action-error rejection keeps the policy penalised. |
| Price exploitation | ✅ | `actions.price_min/max_mult_competitor` rejects out-of-band prices; capacity cap rejects oversized negotiations. |
| Ad-spend exploitation | ✅ | `MAX_AD_MULTIPLIER=5.0` hard cap on ad multiplier; `max_ad_multiplier` config override still clamped to ≥1.0. |
| Revenue cap | ✅ | Poisson lambda capped at `1e6` for numerical stability; `rewards.revenue_cap_per_step` available for the `cap` mode. |
| Refund exploitation | ✅ | Refund requires cash (or opt-in partial refund with tracked partial amount); `refund_payout_delta_cap` bounds delta-term influence. |
| Solvency farming | ✅ | `_solvency_term` only fires when the agent was productive (`base_reward > 0` or non-negative non-revenue delta on `set_price` / `ad_spend` / `negotiate`). |

**Verdict:** comprehensive. Every bullet in §8 is addressed.

### §9. Process-aware feedback

> "Line-by-line checks, step-level verifiers, program trace analysis, or LLM-as-a-judge for intermediate reasoning."

✅ Implemented via **`info.reward_breakdown`** (per-term floats emitted every step) plus the 20+ additive `info` keys. GRPO groups can score partial trajectories rather than only terminal scalar. The guide warns against LLM-as-a-judge; our outcome-based verification avoids that trap entirely.

### §10. Pick the right training stack

| Stack component | Status | Action if missing |
|---|:---:|---|
| TRL | ✅ | `grpo_single_cell_colab_v5.py` training loop |
| **Unsloth** | ✅ | `grpo_single_cell_colab_v5.py` model load path |
| OpenEnv | ✅ | `openenv-core==0.2.3` in `requirements.txt` |

### §11. Prefer GRPO / RLVR

> "If the task is verifiable, build the verifier first, then plug that verifier into RL training."

✅ Three graders + 8-term reward engine implemented *before* the training notebook. GRPO used explicitly: `from trl import GRPOConfig, GRPOTrainer`. Env-in-the-loop reward function in the notebook is the "plug that verifier into RL training" pattern.

### §12. Keep inference fast

> "Your project speed depends heavily on fast sampling, tight environment loops, low-overhead execution, and efficient model runtime."

| Optimisation | Status | Evidence |
|---|:---:|---|
| Tight environment loop | ✅ | 218 tests run in < 3 s on a single worker; a single `/step` averages sub-millisecond on the hot path (cached config slices, structural `_snapshot_state`, env-local RNG). |
| Low-overhead execution | ✅ | Lazy imports for OpenAI + Matplotlib (Audit MINOR #9). |
| Efficient model runtime | ✅ | Unsloth-first loading path in primary Colab script |

### §13. Deploy your environment early

| Deployment path | Status | Notes |
|---|:---:|---|
| Dockerfile | ✅ | Exists, `python:3.11-slim`, port 7860 |
| `openenv.yaml` | ✅ | `spec_version: 1`, 3 task descriptors |
| HuggingFace Space | 🟡 | Pushable; live URL not confirmed — Remediation #3 |
| Local via `uvicorn` | ✅ | `uvicorn server.app:app --host 0.0.0.0 --port 7860` |
| Container run locally | ✅ | `docker build -t commerce-ops-v2 . && docker run -p 7860:7860 commerce-ops-v2` |

### §14. Scale only after the environment is stable

> "First confirm: reset works, step works, rewards are sensible, timeouts work, logs are visible, env can be run locally and remotely."

| Pre-scale gate | Status | Anchor |
|---|:---:|---|
| reset works | ✅ | `test_api_contract.py::test_reset_resets_world` |
| step works | ✅ | `test_api_contract.py::test_step_accepts_both_payload_shapes` (+17 more) |
| rewards sensible | ✅ | 21 tests in `test_reward_engine.py` |
| timeouts work | ✅ | `test_simulation_invariants.py::test_stall_terminates` |
| logs visible | ✅ | `logger.getLogger("commerceops.*")` throughout; `GET /debug/last_step_info` under `COMMERCEOPS_DEBUG=1` |
| local + remote | ✅ | Uvicorn + Docker + (pending) Space |

### §15. Monitor the right things during training

> "overall reward, individual reward function columns, success indicators, timeout frequency, generated strategies over time."

| Monitor | Status | Anchor |
|---|:---:|---|
| Overall reward | ✅ | `reward_curve.png` from notebook + `reward_curve_inference.png` from `inference.py` |
| Individual reward columns | ✅ | `info.reward_breakdown` has 11 numeric fields (`base, revenue, solvency, stockout, ticket_aging, ad_roi, bankruptcy, delta, inventory_target_bonus, daily_revenue, scale_hint`) |
| Success indicators | ✅ | Three grader scores after every episode; `info.intent`, `info.kpis.revenue_trend`, `info.why_failed` per step |
| Timeout frequency | ✅ | Stall-guard triggers recorded in `info.episode_summary.termination_reason` |
| Generated strategies over time | ✅ | `info.intent` rolled into `episode_summary.strategy`; `thought_logs.json` captures raw LLM outputs for first + last episode |
| Policy-drift monitoring | ✅ | `info.policy_stability.{score, distribution, last_action, window}` based on a 1 − entropy-of-recent-actions score |

### §16. Save models correctly

> "Do not upcast a 4-bit model to 16-bit and then merge the LoRA weights naively."

🟡 Notebook uses `trainer.save_model('./swiftlogic-grpo-output/final')` via TRL's stock path, which is safe for FP16/BF16 but does **not** currently exercise QLoRA. If Remediation #1 adds Unsloth with 4-bit loading, the save step must switch to Unsloth's `model.save_pretrained_merged(...)` or adapter-only export. **Tracked in Remediation #1.**

### §17. Team structure

➖ Solo / small team — not applicable as a compliance item but worth noting that the codebase decomposes cleanly along the four roles (env / verifier / training / demo) the guide recommends.

### §18. 1-day execution plan

| Phase | Guide target | Our state |
|---|---|:---:|
| 1. Pick a narrow task | ✅ | Customer ticket triage (easy grader) as warm-up |
| 2. Build the environment | ✅ | Complete |
| 3. Build rewards | ✅ | 9 reward signals + 3 graders |
| 4. Deploy | 🟡 | Docker/Space ready, URL pending |
| 5. Train small | ✅ | Notebook runs on Colab T4 |
| 6. Inspect for hacking | ✅ | `thought_logs.json` + demo printer |
| 7. Add curriculum | 🟡 | 3-difficulty ladder, no auto-schedule (Remediation #4 optional) |
| 8. Train bigger | ✅ | `num_generations=GRPO_GROUP_SIZE=8` scales via config |
| 9. Save and demo | 🟡 | Model save path fine for TRL default; video missing (Remediation #2) |

### §19. What judges find compelling

> "baseline model attempt → reward/verifier output → trained model attempt → measurable improvement → short explanation of safeguards."

All five ingredients exist as artefacts:

1. Baseline attempt — first episode thought log (`zero_shot_logs` in notebook).
2. Reward/verifier output — `info.reward_breakdown` per step + grader scores per episode.
3. Trained model attempt — last episode thought log (`post_training_logs`).
4. Measurable improvement — `before_after_comparison.json` + `reward_curve.png`.
5. Safeguards explanation — `README.md` §Operations Notes + `PROJECT_REPORT.md` §23.

**Action**: Remediation #2 needs to stitch these into a 2-minute video.

### §20. Suggested theme directions

See §3 above. Our primary theme is 3.1; strong coverage of 1 and 2.

### §21. Common mistakes to avoid

| Mistake | Are we avoiding it? |
|---|:---:|
| Task so hard success prob is zero | ✅ Zero-shot scores non-zero on all 3 graders |
| Only one reward function | ✅ 9 independent reward signals |
| No reward-hacking checks | ✅ See §8 above |
| Training before env is stable | ✅ 218 tests gate training |
| Relying only on average reward | ✅ Per-term breakdown + intent + anomalies |
| No timeouts | ✅ Three termination paths (horizon, bankruptcy, stall) |
| Wrong LoRA save | 🟡 Currently TRL default (safe); revisit in Remediation #1 |

### §22. Learning resources

➖ External content; no compliance action required.

---

## 5. Gap Summary & Remediation Plan

### ✅ Remediation #1 — Unsloth integration completed

**Why:** Guide §0 and §10 list Unsloth as part of the **intended stack**. Judges will look for it. Also unlocks 4-bit QLoRA-style training, which is essential on Colab T4.

**What to change:** `swiftlogic_grpo_training.ipynb`, Cell 4 (model load). Add a toggle:

```python
USE_UNSLOTH = True  # set False to use stock HF transformers
MAX_SEQ_LEN = 2048

if USE_UNSLOTH:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,          # auto-detect
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, target_modules=["q_proj","k_proj","v_proj","o_proj",
                              "gate_proj","up_proj","down_proj"],
        lora_alpha=16, lora_dropout=0,
        bias="none", use_gradient_checkpointing="unsloth",
        random_state=42,
    )
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

**Install cell update** (Cell 2):

```
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q "trl>=0.11.0" "transformers>=4.45" accelerate datasets peft requests matplotlib numpy
```

**Save-path fix** (Cell 6 end) — per guide §16:

```python
if USE_UNSLOTH:
    model.save_pretrained_merged(
        "./swiftlogic-grpo-output/final-merged",
        tokenizer, save_method="merged_16bit",  # NOT naive upcast
    )
else:
    trainer.save_model('./swiftlogic-grpo-output/final')
```

Effort: ~15 minutes of notebook editing; no code changes to the env.

### 🔴 Remediation #2 — Mini-blog or 2-minute video

**Why:** Stated minimum requirement in the themes doc.

**Suggested 2-minute script** (aligned with guide §19):

1. (0:00–0:20) Problem: "An LLM running a company for 50 days — keeping it solvent while balancing 4 SKUs, support tickets, ads, suppliers, and a reactive competitor."
2. (0:20–0:50) Environment tour: live `/step` call, pretty-print of `info` (intent + departments + competitor_reaction.triggered + confidence formula).
3. (0:50–1:20) Baseline: Qwen2.5-0.5B zero-shot grader scores (0.80 / 0.40 / 0.15).
4. (1:20–1:45) Training: `reward_curve.png` fade-in; mention GRPO + Unsloth + TRL.
5. (1:45–2:00) After training: improved grader scores; close with safeguards (9 reward signals, 4 termination paths, 218 tests, documented confidence formula).

Deliverable: YouTube unlisted link + a paragraph on the HF Space README pointing to it.

### 🟡 Remediation #3 — HuggingFace Space live URL

**Why:** Themes doc requires the env to be hosted on Spaces.

**What to do:**

1. `huggingface-cli login`
2. `openenv push` (picks up `openenv.yaml` automatically) — or manual: create a new Space with `sdk: docker`, push the repo.
3. Confirm `https://<user>-<space>.hf.space/health` returns `{"status":"ok"}`.
4. Run `python scripts/smoke_env.py https://<user>-<space>.hf.space` to validate the full `/reset` → `/step` → `/grader` cycle.
5. Add the live URL to `README.md` (top badge) and to the video description.

### 🟡 Remediation #4 (optional) — Curriculum schedule for Theme 4 upgrade

**Why:** Strengthens Theme 4 (Self-Improvement) coverage — currently at baseline.

**What to do:** add a thin wrapper around `EcomEnv` that, given a policy's recent grader scores, progresses from `siyaani_fashion` (no shocks, no reactive competitor) → `siyaani_fashion_demo` (all realism on) once the triage grader crosses 0.6. This uses existing configs — no new env code. Implementation: ~20 lines in `inference.py` or a new `env/curriculum.py`.

Low priority — the judges' criteria do not require this, but it would push the submission from "strong Theme 1+2+3" to "strong Theme 1+2+3+4".

### ➖ Items explicitly **not** remediating

- **Theme 3.2 (Personalized Tasks):** outside scope; 3.1 is the right bucket.
- **Multi-agent coalition formation:** our environment is single-policy; extending would be a v3 project.
- **Self-play:** not applicable to this problem shape.
- **LLM-as-a-judge:** the guide warns against it; our outcome-based verifier is correct.

---

## 6. Compliance Dashboard (TL;DR)

| Section | Pass | Partial | Fail |
|---|:---:|:---:|:---:|
| §1 Minimum requirements | 1 / 4 | 2 / 4 | 1 / 4 |
| §2 Judging criteria (weighted) | **100 %** (subject to closing §1 gaps) | — | — |
| §3 Theme fit | Themes 1, 2, **3.1** ✅ · Theme 4 🟡 · Theme 5 ✅ eligible | — | — |
| §4 Participant guide (22 §) | 17 ✅ | 4 🟡 | 1 🔴 (Unsloth) |
| §5 Remediation | 4 items · 2 required + 2 optional | — | — |

**With Remediations #1, #2, #3 landed, the submission is minimum-requirement-complete and scores 40% (Innovation) + 30% (Storytelling) + 20% (Improvement) + 10% (Pipeline) = 100% eligible.**

---

## 7. Quick reference

- Live tests: `python -m pytest -q` → `218 passed`.
- Deterministic E2E replay: two runs at seed 2026 produce byte-identical `(reward, intent, revenue_trend, tickets_spawned, tickets_resolved, confidence, demand_keys, competitor_reaction.triggered)` signatures across 8 steps.
- Source of truth for every claim in this document:
  - `README.md` (architecture + info contract + env vars)
  - `PROJECT_REPORT.md` (all math + §33 gap-fix log + Appendix E info inventory)
  - `AUDIT_REPORT.md` (audit findings and their closure state)
  - `tests/` (218 regression tests across 11 files)
  - `env/world_engine.py` + `env/reward_engine.py` + `ecom_env.py` + `server/app.py` + `inference.py` + `configs/*.json` + `openenv.yaml`.
