# ROADMAP — OpenEnv RL environment: path to 100/100

Status update (2026-04-26): active Colab training execution path is `grpo_single_cell_colab_v5.py` (single-cell HTTP GRPO). Notebook path remains optional.

**What this is (top line):** A **reproducible OpenEnv RL environment** (frozen dynamics + verifiable reward + graders) with a **GRPO training loop** that produces **evidence of learned behavior**—not a moving target where “intelligence” is simulated by changing the world. Judges see: same env, better policy.

**Target:** Raise total judging score from **73.5/100 → 100/100** and land in the winning tier of the Apr '26 OpenEnv Hackathon, with **strong, defensible alignment to Theme 2, Theme 3, and Theme 4** without breaking the system.

---

## North star (lock this first)

```text
ENVIRONMENT = FROZEN after the freeze tag (no new physics, no hidden state, no reward refactor)

LEARNING   = happens in the training loop (GRPO, curriculum via config paths, hard-seed retrain)

INTELLIGENCE = proven via measurable behavior change + generalization + baselines (not via new env tricks)
```

This resolves the main contradiction: **the env stays auditable; improvement is in weights and rollouts.** “Real intelligence” here means **learned control** that beats **wait-only / random / heuristic / zero-shot** on the **same** frozen simulator, and **transfers** to held-out configs where applicable—not hardcoded CEO narration posing as learning.

## Learning Definition

Learning is defined as:
- Policy improvement over baselines
- Behavior change (`policy_signature`)
- Generalization to unseen environments
- Improvement after retraining (hard seeds)

NOT learning:
- Derived labels (`intent`, `strategy_phase`)
- Explainability outputs
- Narrative signals

**Hackathon doc anchors (FAQ-style):**

| Source | What to use |
|--------|-------------|
| `[External] Apr '26 OpenEnv Hackathon Themes.md` | **Theme 2** (long-horizon / delayed effects / recovery), **Theme 3** (world modeling / tools & dynamics / economic feedback), **Theme 4** (curricula / self-improvement), **Minimum requirements** (OpenEnv, TRL or Unsloth script, <2 min blog/video, HF Space), **Judging criteria** (40/30/20/10) |
| `[External] Meta OpenEnv Hackathon Participant Help Guide.md` | **§4** design env before trainer, **§6** curriculum (success early), **§7** multiple reward signals, **§8** anti–reward hacking, **§10–11** TRL/GRPO for verifiable tasks, **§16** adapter save/merge safety |

**Reference docs (do not re-derive; trust these):**
- `FINAL_ALIGNMENT_AUDIT.md` — per-question scoring and verified gaps
- `HACKATHON_ALIGNMENT.md` — theme/criteria alignment map
- `[External] Apr '26 OpenEnv Hackathon Themes.md` — 5 themes + judging rubric
- `[External] Meta OpenEnv Hackathon Participant Help Guide.md` — 22-section self-serve guide

**Hard constraints that apply to all parts:**
- DO NOT modify OpenEnv endpoints (`/reset`, `/step`, `/state`, `/info`, `/health`, `/tasks`, `/grade`).
- DO NOT change `EcomAction` or `EcomObservation` schemas.
- DO NOT alter existing reward physics or add new reward **terms** to the env (no “helpful” extra shaping after freeze).
- All new `info` / explainability outputs MUST be **additive** and **derivable** from observable state, history, and action (no fake learned labels).
- All 218 existing tests MUST keep passing.
- Determinism under fixed seed MUST be preserved.

**Execution order is strict:** Part A → Part B → Part B+ → Part C. Do not begin the next part until every checkbox in the previous part is green.

---

## Theme 2 / 3 / 4 — how we earn 10/10 without env churn

| Theme | What judges look for (themes + guide) | Our proof (no new physics) |
|-------|----------------------------------------|----------------------------|
| **2 — Long-horizon** | Multi-step effects, delayed consequences, recovery from early mistakes | Existing dynamics (lead times, shocks, compounding cash). **Expose** with trajectory logs, day 1–N sequence, **failure → recovery** scripted contrast (A.7, B+.10), delayed-effect narration from `demand_factors` / `action_effect` / `episode_summary` |
| **3 — World modeling** | Interaction with a dynamic system, causal structure, not shortcuts | **Already strong:** tools = pricing / inventory / supplier negotiation / support actions; **expose** `causal_chain`, `competitor_reaction`, `demand_factors` as **ground-truth** structure the policy must exploit |
| **4 — Self-improvement** | Curriculum, harder challenges, improved capability | **Not** new env modules: **B.5** training curriculum (config path swap) + **B+.2** hard-seed retraining + reward/composite **iteration** on frozen env |

**Theme 1 (multi-entity competition/negotiation)** remains a **supporting** story (competitor + supplier + tickets)—cite honestly as rule-based counterparts, not multi-policy RL.

**Strict no (keeps the system honest):** Do **not** add hidden state, partial-observability rewrites, new randomness in the env, new reward terms after freeze, reward “refactors,” multi-agent RL for supplier/competitor, or async backend changes for “smarter” behavior. Do **not** claim autonomous cross-agent “collaboration” or that the **environment** “learned strategy.” **Curriculum** = training config only (guide §6).

---

## POST-REVIEW PATCH LOG (applied to this roadmap)

After four rounds of external review (internal audit → Gemini → final no-compromise pass → **final Theme 2/3/4 consolidation**), the following surgical edits were applied. Recorded here so the diff is auditable.

**CONSOLIDATION (Theme 2/3/4 + RL-first — no system break):**
- Roadmap now leads with **OpenEnv RL environment** + **freeze / train / prove** north star; **Themes 2, 3, 4** are the primary submission narrative; Theme 1 is supporting.
- **A.2.3** is **exposure-only** (no new env reward term) to honor “no new reward terms after freeze” while still scoring Theme 2 in judging.
- **Interpretability** (`intent`, `confidence`, `strategy_phase`, CEO trace) explicitly framed as **derived / deterministic** signals—**not** “the model learned strategy” in the env.

**REMOVED (risk-reducing — protects env-freeze):**
- `env/curriculum.py` module — curriculum now lives ONLY in the training loop (B.5 + B+.2).
- `info["curriculum_stage"]` — not added to env; stage tracked in training logs.
- Reward-engine refactor (A.3.1 was "expose internals") — kept as documentation + regression test only.
- `asyncio.wait_for` FastAPI step-timeout wrapper (was A.5.2) — no concurrency changes.

**CLARIFIED (no net new work, just scope boundaries):**
- A.4 graders: 3 core graders drive training reward; 3 new graders are **evaluation-only**, tagged `"evaluation_only": true`.
- B.5 curriculum: explicitly training-loop only, via config-path swap, zero env-side logic.
- A.3: reuses existing `info["reward_breakdown"]` instead of refactoring.

**ADDED (already in Part B+ / B from prior rounds — confirmed present, no dup):**
- Generalization (B+.1), hard-seed retrain (B+.2), behavior plots (B+.3), action_quality (B+.4), strategy_phase (B+.5), composite score (B+.6), ablation (B+.7), run_config (B+.8), confidence demo (B+.9), failure-vs-recovery (B+.10), run_full_pipeline (B+.11), mandatory heuristic baseline (B.7.2).

**NEWLY ADDED in this patch:**
- `--fast-mode` flag on `run_full_pipeline.py` — 30-min judge-friendly run (B+.11.4–5).
- HF Space landing page with "Run Demo" button — makes the Space demo-able without a client (C.2.9–10).
- 4-beat symmetric video timeline (0–10 / 10–30 / 30–90 / 90–120s) replacing the 5-beat script (C.3.1a).
- `configs/siyaani_fashion_easy.json` — the stage-1 easy-mode config for training curriculum (B.5.0).

---

# PART A — PRE-TRAINING: Validation, compliance, and exposure (env frozen at gate)

**Goal of Part A:** Every endpoint, grader, reward wiring, and explainability contract is **correct and honest**. After **A.9.6**, the env is **frozen**: Part A is **not** where “intelligence” is invented—it is where the **RL problem** is verified and **Theme 2/3 hooks** are **surfaced** (logs, demo, documentation), not re-shaped. Part B cannot fail due to environment instability.

**Exit criteria for Part A (all must be true):**
- [ ] 218+ tests passing, `pytest -q` green in CI-equivalent run.
- [ ] `COMMERCEOPS_ASSERT_INVARIANTS=1` run of a 300-step episode with no warnings.
- [ ] Determinism script: same seed, same actions → identical reward sequence (bit-exact).
- [ ] All 5 themes listed; **Themes 2, 3, and 4** scored with evidence; 1/3.2/5 as applicable in `HACKATHON_ALIGNMENT.md`.
- [ ] At least 5 independent reward signals documented + tested (guide §7).
- [ ] At least 5 graders/tasks with non-zero and non-saturating success probability across 10 seeds.
- [ ] Anti-reward-hacking tests committed.
- [ ] One deterministic `scripted_demo.py` producing a reproducible wow moment.
- [ ] README pins the **Theme 2 / 3 / 4** story (see A.2.1) plus **Theme 1** as supporting.
- [ ] Every claim in README/PROJECT_REPORT either (a) matches code, or (b) is deleted.

---

## A.1 — OpenEnv compliance audit (re-verify, don't assume)

**Status today:** Compliant. Do a final proof pass.

- [ ] **A.1.1** — Run `/health` → expect 200 `{"status":"ok"}`.
- [ ] **A.1.2** — Run `/reset` with a seed → observation JSON validates against `EcomObservation`.
- [ ] **A.1.3** — Run `/step` with each of the 6 action types → 6×6 matrix of (action, observation, reward, done, info) validated.
- [ ] **A.1.4** — Run `/tasks` → exactly 3 tasks today (`triage_task`, `inventory_task`, `profit_task`). *See A.5 — we will add more.*
- [ ] **A.1.5** — Run `/grade` with each task id → grade ∈ (0.01, 0.99).
- [ ] **A.1.6** — Confirm `/state` is read-only (never mutates env).
- [ ] **A.1.7** — Confirm `/info` returns all 20+ documented keys on a fresh reset episode.
- [ ] **A.1.8** — Confirm multi-worker warning fires when `uvicorn --workers 2` is attempted.
- [ ] **A.1.9** — Run the OpenEnv contract test (write one) that POSTs `/reset` → N × `/step` → `/grade` via HTTP, not in-process, to prove wire-level compliance.

**Deliverable:** `tests/test_openenv_contract_http.py` — spins up Uvicorn on a random port, runs the full loop via `requests`, asserts schemas with Pydantic. Commit output log to `artifacts/openenv_contract.log`.

---

## A.2 — Theme alignment (lead with 2, 3, 4; keep 1 honest)

The hackathon has 5 themes. This submission **targets Theme 2, Theme 3, and Theme 4** with evidence (see table above). **Theme 1** is **supported** (competitor, supplier, tickets) but framed accurately: **rule-based entities**, not learned multi-agent RL. **Theme 3.2 / 5** are out of scope; optional acknowledgment only.

- [ ] **A.2.1** — Pick and pin in README header (single short paragraph):
  - **Primary narrative:** **Theme 3.1** — World Modeling / Professional Tasks (dynamic economic system + tool-like actions).
  - **Co-primary evidence:** **Theme 2** — long-horizon / delayed effects / recovery (via traces + existing dynamics, not new reward hacks).
  - **Co-primary evidence:** **Theme 4** — self-improvement through **training** (B.5 curriculum + B+.2 hard-seed retrain), not env self-play.
  - **Supporting:** **Theme 1** — multi-entity competition & negotiation (reactive competitor + supplier rules).
- [ ] **A.2.2 (Theme 1 — optional, freeze-safe)** — Prefer **varying business configs in training** (B.5) over new competitor modes in code. If a competitor tier is still desired, it MUST be **pure `configs/*.json` parameters** already consumed by the engine—**no** new `world_engine` branches after A.9.6. If it would add env logic, **skip** and document competitor behavior via existing `competitor_reaction` + demo.
- [ ] **A.2.3 (Theme 2 — exposure-only, no new env reward term)** — **Do NOT** add a sparse terminal bonus or any new term to `env/reward_engine.py` (conflicts with env-freeze + “no new reward terms”). **Instead**, prove long-horizon for judges by:
  - Logging **day 1–N trajectories** (restock today → stock arrives later; price change → demand lag; shock duration) from existing state/`info`.
  - Using **`episode_summary`**, `info["trend"]`, `info["anomalies"]`, and scripted **failure → recovery** (A.7, B+.10) as the **Theme 2** story.
  - One paragraph in README / `HACKATHON_ALIGNMENT.md` quoting Theme 2 language from the themes doc and pointing to these artifacts.
- [ ] **A.2.4 (Theme 4 — SELF-IMPROVING SYSTEM)** — **REVISED after final review: curriculum lives in the TRAINING LOOP, NOT in the env.** The env stays frozen per A.9.6. The Theme-4 signal comes from:
  - The 3-stage training-loop curriculum in **B.5** (easy config → medium config → hard config via existing `configs/*.json` swap, no new env code).
  - The hard-seed retraining loop in **B+.2** (failure-driven self-improvement — the stronger Theme-4 pattern).
  - These two together fully cover Theme 4 without introducing new `env/` modules.
  - **Do NOT add `env/curriculum.py`** — that would break the env-freeze contract and introduce a new code surface to test.
  - **Do NOT add `info["curriculum_stage"]`** to the env. If a stage indicator is useful, the training loop writes it to training logs, not to env `info`.
- [ ] **A.2.5** — Update `HACKATHON_ALIGNMENT.md` theme table: cite B.5 + B+.2 as Theme-4 evidence (not an env module).

---

## A.3 — Reward sufficiency (guide §7: multiple independent reward functions)

Guide §7 explicitly warns: **"use multiple independent reward functions, not just one."** Today the reward engine combines 8 terms inside one scalar. That is *one* function with multiple *components* — judges may read it as one function. We'll make the independence explicit — **WITHOUT refactoring the reward engine** (post-review hardening: do not touch the reward physics).

Current reward components (in `env/reward_engine.py`, already internally decomposed):
1. revenue term
2. profit delta term
3. stockout penalty
4. holding cost penalty
5. overstock penalty
6. service level reward (ticket SLA)
7. solvency term
8. bankruptcy penalty

- [ ] **A.3.1** — **DO NOT refactor `env/reward_engine.py`.** The existing `info["reward_breakdown"]` already exposes each named component per step. Verify the keys are present and documented in README. No code change — just documentation + a test.
- [ ] **A.3.2** — Add `tests/test_reward_breakdown_keys.py` — fixture-based regression test asserting that every `info["reward_breakdown"]` call contains all 8 named components. Prevents silent drift. No env code touched.
- [ ] **A.3.3** — (Optional, low priority) Add an **anti-exploit monitor** flag read from the existing history: if the agent repeats the same action >40 steps in a row, set `info["anti_exploit_flag"] = true`. Additive `info` key only. Skip if it requires any new state in `world_engine`.
- [ ] **A.3.4** — **Format compliance reward** (LLM-side only — lives in training notebook, NOT in env): reward valid JSON action parse, penalize malformed. Implemented in B.4.3. Nothing to do here.
- [ ] **A.3.5** — Update README reward section to cite the 8 existing independent components from `info["reward_breakdown"]`. Frame them explicitly as "multiple independent reward signals" for judge-readability.

**Net effect:** same reward physics, zero risk, clear framing as multi-signal.

---

## A.4 — Grader sufficiency (do we have enough tasks?)

Today: **3 graders** (`triage`, `inventory`, `profit`). Guide recommends multiple independent verifiers. 3 is the floor; 5-6 is a comfortable ceiling without losing focus.

**Post-review boundary (IMPORTANT):**
- The **3 existing graders** (`triage`, `inventory`, `profit`) remain the **ONLY graders used during training** (via the B.4.2 reward formula). This keeps the training signal stable and low-variance.
- The **3 new graders** (`stability`, `competitor_response`, `crisis_recovery`) are **evaluation-only** — they show up in `/tasks`, `/grade`, `before_metrics.json`, `after_metrics.json`, and the composite score (B+.6), but they do NOT feed into the GRPO reward during training.
- Rationale: adding 3 new signals to the training reward at the same time as the policy is learning is a known source of instability. Evaluate on 6, train on 3.

- [ ] **A.4.1** — Add **`grade_stability_task`** (easy, **evaluation-only**): "Keep customer satisfaction ≥ target for 30 consecutive days." Clamped (0.01, 0.99).
- [ ] **A.4.2** — Add **`grade_competitor_response_task`** (medium, **evaluation-only**): "After a competitor undercut, agent matches or beats price within 3 steps." Verifiable via `info["competitor_reaction"]`.
- [ ] **A.4.3** — Add **`grade_crisis_recovery_task`** (hard, **evaluation-only**): "After a market shock, bank recovers to 90% of pre-shock value within 15 steps." Verifiable via `state["bank"]` history + `info["anomalies"]`.
- [ ] **A.4.4** — Register all 6 tasks in `server/app.py::TASKS`. Mark difficulty, add description. Tag the 3 new ones as `"evaluation_only": true` in the task metadata so reviewers see the boundary explicitly.
- [ ] **A.4.5** — Run a **success-probability sweep** over 10 seeds × random policy × each task → commit to `artifacts/task_baselines.json`. Expected: every task has success probability > 0 and < 1 (guide §1). If any task is 0% or 100% under random policy, **redesign it now**.
- [ ] **A.4.6** — Update README task table with all 6 tasks, their baseline success rates, and the training-vs-evaluation tag.

---

## A.5 — Anti-reward-hacking hardening (guide §8)

Guide §8 lists concrete failure modes. Audit ours:

- [ ] **A.5.1 — Locked-down execution.** Pydantic already locks action schema. Confirm no `exec()` / `eval()` anywhere in server or env. Add `tests/test_no_eval.py` — `ast.parse` the env/ and server/ modules, fail if `exec|eval|__import__|compile` appears.
- [ ] **A.5.2 — Time limits.** Episode terminates after `max_days` (already present). **DO NOT add `asyncio.wait_for` wrappers or any FastAPI concurrency tuning** — post-review hardening: avoid backend complexity that can introduce non-determinism or race conditions. The existing episode cap + env-step synchronous path is sufficient. If a step ever hangs in practice, debug the step, don't wrap it in a timeout.
- [ ] **A.5.3 — No unrestricted globals.** Env uses env-local RNG. Confirm no `random.random()` / `np.random.rand()` at module top-level. Add `tests/test_no_global_rng.py`.
- [ ] **A.5.4 — Periodic inspection.** Training pipeline must sample + log 5 generations every 50 steps. Wire into notebook (Part B).
- [ ] **A.5.5 — Mutation guard.** `/state` endpoint must be read-only. Add regression test: POST to `/state` → 405 Method Not Allowed. Pure read-side test — no server code change.
- [ ] **A.5.6 — Reward exploit test.** `tests/test_reward_exploit.py` — try known exploit patterns: (a) infinite refund loop, (b) ad-spend-until-bank-zero, (c) price at $0.01 to trigger service reward. Assert none of them give top-5% reward. Read-only probing — no env change.

---

## A.6 — Explainability tightening (interpretability ≠ “learned intelligence”)

**Wording contract (applies to README, demo, and `info`):** CEO trace, `intent`, department suggestions, and `confidence` are **deterministic interpretability** signals derived from **state + action + history** so humans and judges can audit behavior. They are **not** evidence that the **environment** “learned” a strategy. **Learned** capability is shown only by **Part B / B+** (weights, baselines, generalization, hard-seed retrain).

- [ ] **A.6.0** — Grep and remove/rewrite claims: “AI learned strategy,” “autonomous inter-agent collaboration,” or any phrasing that implies the **simulation** is learning. Replace with: **trained policy** (GRPO) vs baselines; **rule-based** competitor/supplier.
- [ ] **A.6.1** — Add `info["causal_graph"]` — structured edge list `[("action", "demand"), ("demand", "revenue"), ...]` with weights derived from the actual step. LLM-verifiable.
- [ ] **A.6.2** — Add `info["decision_context"]["attributed_department"]` — map each of 6 actions to {inventory, marketing, support, cross-functional}. Zero-risk narrative payoff.
- [ ] **A.6.3** — Audit every explainability builder for stale data: `build_market_reaction`, `build_reasoning`, `build_causal_chain`, `build_why_it_worked`, `build_state_summary`. Run a 10-step episode with `COMMERCEOPS_DEMO_VERBOSITY=full` and diff each output against hand-computed expected text.
- [ ] **A.6.4** — Document `confidence` formula in the info payload itself: `info["confidence_formula"] = "weighted_avg(data_completeness, action_success, anomaly_absence, policy_stability)"` — judges love in-band documentation.

---

## A.7 — Demo & storytelling polish (biggest ROI in audit)

- [ ] **A.7.1** — Create `scripted_demo.py` — fixed seed + fixed action sequence designed to land on:
  - day 5: competitor undercut
  - day 10: market shock
  - day 14: stockout cascade
  - day 18: agent recovery
  - day 25: profit peak
  This is the **deterministic wow moment**.
- [ ] **A.7.2** — Add `COMMERCEOPS_DEMO_VERBOSITY=compact|normal|full` env var. Compact mode = 3 lines per step (headline, action+effect, KPIs).
- [ ] **A.7.3** — Reorder `_print_demo_step` so `department_suggestions` print **before** the chosen action (CEO narrative: briefings first, decision second).
- [ ] **A.7.4** — Add a one-line "headline" at the top of each step in normal/full mode, e.g.:
  `"Day 12: Revenue -18% — competitor undercut ₹80 — intent shifted to avoid_stockout"`.
- [ ] **A.7.5** — Create a `DEMO.md` with embedded mermaid diagram + 6 annotated screenshots from `scripted_demo.py`. One page, scannable.
- [ ] **A.7.6** — Record a 90-second Loom (plan only here — actual recording is Part C).

---

## A.8 — Documentation credibility sweep

Docs currently claim Unsloth, training improvements, etc., before they're proven. Judges will notice. Fix honesty first, ship claims after Part B.

- [ ] **A.8.1** — Grep all `.md` for "Unsloth" — either (a) pin to "planned (Part B)" or (b) remove.
- [ ] **A.8.2** — Grep all `.md` for "before/after" — same treatment.
- [ ] **A.8.3** — Grep all `.md` for "HuggingFace Space" — replace with "Local demo today; HF Space in Part C."
- [ ] **A.8.4** — Add a "Status" badge block at the top of README:
  ```
  Status: Environment ✅  Graders ✅  Explainability ✅  Training [in progress]  HF Space [planned]
  ```

---

## A.9 — Final Part A verification gate

- [ ] **A.9.1** — `pytest -q` — all tests pass.
- [ ] **A.9.2** — `COMMERCEOPS_ASSERT_INVARIANTS=1 python scripted_demo.py` — no invariant violation.
- [ ] **A.9.3** — Run determinism check: two replays with same seed → identical `reward_breakdown` across 50 steps.
- [ ] **A.9.4** — `python -m server.app &` then hit `/health`, `/tasks`, `/info` via `curl` — all green.
- [ ] **A.9.5** — Commit `artifacts/part_a_verification.log`.
- [ ] **A.9.6 (ENV FREEZE — post-Gemini hardening)** — Formally freeze the environment before any Part B work:
  - Create annotated tag: `git tag -a release/env-frozen-v2.3 -m "Env locked for training; additive-only after this point"`.
  - Create working branch: `git checkout -b feature/training-demo`.
  - From this point on, any commit that touches `env/`, `ecom_env.py`, `server/app.py`, or any `configs/*.json` file MUST be rejected unless it is strictly additive (new keys only) and passes all existing tests unchanged.
  - Add a CI / pre-commit guard (or a `scripts/check_env_frozen.sh` called manually) that diffs against the tag and fails on schema-shape changes to action/observation models.
  - Why: prevents accidental reward-physics drift during training iteration. Judges inspecting the git log will see a clean freeze boundary — strong signal.

**Expected score impact after Part A:** 73.5 → **~82** (Innovation 35, Storytelling 26, Training 12.5, Engineering 9). Training gap still open — that's Part B.

---

# PART B — TRAINING (core RL): GRPO + Unsloth + env-in-the-loop

**Goal of Part B:** This is where **learning happens**: produce committed, reproducible evidence that a **trained** policy improves on the **frozen** env (reward curves, baselines, before/after). **Theme 4** “self-improvement” for judges is grounded here (GRPO + optional curriculum in the notebook), not in env code. This is the 20% "Reward Improvement" bucket — currently at 3/10. Target: 18/20.

**Exit criteria for Part B (all must be true):**
- [ ] **200–300 training episodes** (or equivalent GRPO steps totalling the same order of rollouts) on **Qwen2.5-0.5B** (or 1.5B fallback) — tune for Colab; document actual count in `run_config.json`.
- [ ] `artifacts/reward_curve.png` committed — shows upward trend.
- [ ] `artifacts/before_metrics.json` committed — zero-shot baseline across 10 seeds.
- [ ] `artifacts/after_metrics.json` committed — trained policy across same 10 seeds.
- [ ] `artifacts/training_log.txt` committed — TRL GRPO logs.
- [ ] `artifacts/thought_logs.json` — at least 5 Thought+Action traces from trained model.
- [ ] Trained LoRA adapter on HF Hub (or in `artifacts/adapter/`).
- [ ] Measurable win: `mean(after) > mean(before)` by ≥ 1 standard deviation.
- [ ] Notebook runs end-to-end on a fresh Colab T4 in < 2 hours.

---

## B.1 — Model selection

**Criteria (guide §1, §12):**
1. Can act step by step — ✅ all instruct models.
2. Fits in Colab free T4 (15 GB VRAM) with 4-bit load — pushes us to ≤ 1.5B params.
3. Reasonable base instruction-following — else RL won't get non-zero reward.

**Primary choice: `Qwen/Qwen2.5-0.5B-Instruct`**
- 0.5B params, fits easily in 4-bit (~0.5 GB VRAM).
- Strong JSON/tool-call behavior for its size.
- Fast rollout (guide §12: "inference can dominate total runtime").

**Fallback: `Qwen/Qwen2.5-1.5B-Instruct`**
- Better policy quality.
- Still fits Colab free T4 with QLoRA 4-bit.
- Use if 0.5B fails to learn.

**Hard reject:** Llama-3-8B, Qwen-7B — won't fit Colab free T4 under QLoRA with reasonable batch size for GRPO group sampling.

- [ ] **B.1.1** — Set `MODEL_NAME` default in notebook to `Qwen/Qwen2.5-0.5B-Instruct`.
- [ ] **B.1.2** — Document fallback chain in notebook markdown cell.

---

## B.2 — Is GRPO the right algorithm?

**Verdict: Yes. GRPO is correct for this env.**

**Reasoning (from guide §11):**
- Task is *verifiable* (we have programmatic graders + multi-signal reward).
- Guide explicitly recommends GRPO / RLVR for verifiable tasks.
- GRPO is "a more efficient evolution relative to older PPO-style setups, especially by simplifying away parts like the value model."
- Colab T4 cannot comfortably train PPO with a separate value model on even 1.5B params.

**No algorithm swap needed.** What needs improvement in our current setup:

- [ ] **B.2.1** — GRPO group size: start at 8 (notebook default). If Colab OOMs, drop to 4. Do not go below 4 (destroys GRPO's variance-reduction property).
- [ ] **B.2.2** — Rollout length: 50 steps per episode is correct for our env. Do not exceed 100 (context length + inference time concern).
- [ ] **B.2.3** — Prompts: use the current "CEO running a company, observe this state, pick one of 6 actions, output JSON" format. Keep it stable — do not change mid-training.
- [ ] **B.2.4** — Sampling temperature: 0.7 during training (explore), 0.0 for evaluation (reproducible).

---

## B.3 — Unsloth integration (required by hackathon minimum)

Hackathon minimum requirements (themes doc): **"Show a minimal training script for your environment using Unsloth or HF TRL in Colab."** Unsloth is listed first. Not using it is legal but costs Q19 points.

- [ ] **B.3.1** — Add Unsloth install cell at the top of the notebook:
  ```python
  !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  !pip install -q --no-deps trl peft accelerate bitsandbytes
  ```
- [ ] **B.3.2** — Replace the `AutoModelForCausalLM.from_pretrained(...)` call with:
  ```python
  from unsloth import FastLanguageModel
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name = MODEL_NAME,
      max_seq_length = 2048,
      dtype = None,              # auto
      load_in_4bit = True,       # QLoRA
  )
  model = FastLanguageModel.get_peft_model(
      model,
      r = 16,
      target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
      lora_alpha = 16,
      lora_dropout = 0,
      bias = "none",
      use_gradient_checkpointing = "unsloth",
      random_state = 3407,
  )
  ```
- [ ] **B.3.3** — Plug into `GRPOTrainer` directly — Unsloth returns a PEFT model compatible with TRL.
- [ ] **B.3.4** — Test inference-time speedup: Unsloth claims ~2× faster inference. Log baseline rollout speed before and after.
- [ ] **B.3.5** — Save path (guide §16 — **DO NOT upcast 4-bit then merge naively**):
  - Save adapters: `model.save_pretrained("swiftlogic_grpo_adapter")` → push to HF Hub.
  - Do NOT call `model.merge_and_unload()` on a 4-bit model.
  - For a merged 16-bit export, reload the base in 16-bit first, apply LoRA, then merge (guide §16). Provide both paths in the notebook.

---

## B.4 — Env-in-the-loop wiring (how GRPO talks to our environment)

This is the core of the training pipeline. The rollout function is the only place where the LLM meets our env.

```
For each training step:
  1. Sample GRPO_GROUP_SIZE prompts from task queue
  2. For each prompt (each is a fresh env episode):
     a. POST /reset?seed=... → observation
     b. Loop STEPS_PER_EPISODE times:
        - Build prompt from obs + info (system + user template)
        - LLM samples action (JSON string)
        - Parse + validate against EcomAction
        - POST /step with action → (next_obs, reward, done, info)
        - Accumulate reward
        - If done: break
     c. Call /grade → clamped grade scores (3-6 graders)
  3. Compute GRPO advantage from group rewards
  4. Update policy
```

- [ ] **B.4.1** — Notebook must have a `rollout_episode(env_client, model, tokenizer) -> (trajectory, total_reward, grade_vector)` function.
- [ ] **B.4.2** — Reward for GRPO = `α * total_env_reward + β * sum(grade_vector) + γ * format_compliance_reward`. Suggested: α=1.0, β=10.0, γ=2.0. Tune if rewards are dominated by one signal.
- [ ] **B.4.3** — Format compliance reward: +1 if action parses as valid `EcomAction`, 0 otherwise. Implements guide §7 recommendation.
- [ ] **B.4.4** — Connect to local env: set `ENV_URL=http://localhost:7860` and run `python -m server.app` in a background Colab cell using `%%bash --bg`. Or connect to HF Space after Part C.
- [ ] **B.4.5** — Sanity test before training: run one rollout with the untrained model, verify the notebook prints the full (prompt, response, parsed_action, reward) trace for 3 steps. Guide §8: "do not just let training run forever without checking generations."

---

## B.5 — Training curriculum (guide §6, §7) — **TRAINING-LOOP ONLY, NO ENV CHANGE**

Guide §6: **"Start with the easiest version of your environment that still proves the concept."** We have the config swap to do exactly this — AND per the post-review hardening, ALL curriculum logic lives in the training notebook. The env stays frozen (A.9.6).

**Mechanism:** the notebook rollout function takes a `config_path` argument. To advance a stage, the notebook calls `/config` with the next config path (or re-instantiates `EcomEnv` if running in-process). The env itself has no notion of "stage" — it just loads the config handed to it.

**Stage 1 (first 30% of training):** Easy config — `configs/siyaani_fashion_easy.json` (NEW — see B.5.0).
- Reactive competitor OFF, market shocks OFF, customer satisfaction OFF.
- Goal: agent learns JSON format + action semantics on a stable market.

**Stage 2 (middle 40%):** Medium config = existing `configs/siyaani_fashion.json`.
- Reactive competitor ON, shocks OFF.

**Stage 3 (final 30%):** Hard config = existing `configs/siyaani_fashion_demo.json`.
- All realism features ON.

- [ ] **B.5.0** — Create `configs/siyaani_fashion_easy.json` — purely a new JSON file, no code change. Copy of base config with `reactive_competitor.enabled: false`, `market_shocks.enabled: false`, `customer_satisfaction.enabled: false`.
- [ ] **B.5.1** — Notebook implements stage auto-advance in the **rollout loop** (not in env): when 20-episode rolling mean reward > threshold, swap the config path used for the next batch of episodes.
- [ ] **B.5.2** — Log stage transitions to `artifacts/training_log.txt`. Also dump the active config path into each checkpoint's metadata.
- [ ] **B.5.3** — Theme 4 — Self-Improvement evidence. Cite this section + B+.2 in `HACKATHON_ALIGNMENT.md`.

---

## B.6 — Monitoring during training (guide §15)

Guide §15: **"Do not watch only one scalar."** Log:

- [ ] **B.6.1** — Overall reward (per-episode and per-batch mean).
- [ ] **B.6.2** — Each independent reward component from `info["reward_components_independent"]`.
- [ ] **B.6.3** — Format compliance rate (% of actions that parse).
- [ ] **B.6.4** — Per-grader scores (triage / inventory / profit / stability / competitor / crisis).
- [ ] **B.6.5** — KL divergence from base model (sanity check).
- [ ] **B.6.6** — Sample 5 generations every 50 training steps, dump to `thought_logs.json`. Guide §8: "Periodic human inspection is still necessary."
- [ ] **B.6.7** — Plot live: reward curve, grade curve, KL curve.

---

## B.7 — Before/after evaluation (this closes the 20% gap)

This is the single highest-leverage deliverable in the entire roadmap. Without it, Q14 is stuck at 3/10.

- [ ] **B.7.1** — Define evaluation protocol in notebook:
  - 10 fixed seeds: `[1, 2, 3, 5, 7, 11, 13, 17, 19, 23]`.
  - Each seed: run full 50-step episode.
  - Record per-seed: total reward, all 6 graders, bank_final, stockout_days, satisfaction_final.
- [ ] **B.7.2** — Baselines to run (ALL FOUR MANDATORY — the contrast is the submission):
  - **Wait-only policy.** Always emits `WaitAction`. Defines the "do nothing" floor.
  - **Random policy.** Uniformly samples one of 6 action types with bounded random parameters (price in ±20% of current, budget in [0, 2000], quantity in [1, 20]). Defines the "no strategy" floor.
  - **Heuristic/rule-based policy (MANDATORY — not optional).** Concrete rules:
    - If `focus_inventory < reorder_threshold` → `restock` at 2× threshold.
    - Else if `competitor_price < our_price * 0.95` → `set_price` matching competitor -1₹.
    - Else if any open urgent ticket → `refund` the oldest.
    - Else → `wait`.
    This is the "strong contrast" baseline a judge will compare the trained LLM against — guide §19 explicitly says judges want observable improvement over a non-trivial baseline. Without this, "training improved things" is indistinguishable from "training learned the heuristic we could have hard-coded."
  - **Zero-shot LLM (untrained).** Same model as trained version, same prompts, temperature 0.0.
- [ ] **B.7.3** — Run each baseline → per-policy files: **`artifacts/baseline_wait.json`**, **`artifacts/baseline_random.json`**, plus heuristic and zero-shot as documented in `artifacts/baseline_metrics.json` (or separate JSONs for each), and aggregate into `artifacts/before_metrics.json`.
- [ ] **B.7.4** — After training, run trained LLM → `artifacts/after_metrics.json`.
- [ ] **B.7.5** — Generate comparison plot: box plot per-policy per-metric → `artifacts/before_after_comparison.png`.
- [ ] **B.7.6** — Write a short `artifacts/README.md` explaining what each file is.

**Success threshold:** trained LLM beats zero-shot LLM on **≥4 of 6 graders** and beats wait-only on `profit_task` by ≥25%. If not hit, iterate: re-check reward weights, increase training epochs, or swap to 1.5B model.

---

## B.8 — Colab practicalities

- [ ] **B.8.1** — Notebook must start with a `nvidia-smi` cell to confirm GPU availability.
- [ ] **B.8.2** — Add a `if COLAB: mount Google Drive for checkpoints` cell.
- [ ] **B.8.3** — Save checkpoint every 25 GRPO steps to Drive (Colab disconnects).
- [ ] **B.8.4** — Cell to resume from latest checkpoint.
- [ ] **B.8.5** — Full run fits in **< 2 hours on free T4**. If over, cut episodes per stage.
- [ ] **B.8.6** — At the end: `files.download('reward_curve.png')` convenience cell.

---

## B.9 — Connection back to this codebase

- [ ] **B.9.1** — Notebook imports `ecom_env.EcomAction` + validators directly from the repo (via `pip install -e .` in Colab if published, or `git clone` first).
- [ ] **B.9.2** — Notebook's rollout function calls our `/step` endpoint — identical behavior to our tests.
- [ ] **B.9.3** — Training reward uses `info["reward_components_independent"]` — the same keys our env produces. No reward duplication.
- [ ] **B.9.4** — Trained model outputs validated against live env — no mock graders.

---

## B.10 — Part B verification gate

- [ ] **B.10.1** — `artifacts/reward_curve.png` shows non-trivial improvement.
- [ ] **B.10.2** — `artifacts/before_metrics.json` and `artifacts/after_metrics.json` both committed.
- [ ] **B.10.3** — `artifacts/training_log.txt` > 100 lines.
- [ ] **B.10.4** — `artifacts/thought_logs.json` has ≥5 entries.
- [ ] **B.10.5** — LoRA adapter pushed to HF Hub; URL in README.
- [ ] **B.10.6** — README updated: "Status: Training ✅".

**Expected score impact after Part B:** 82 → **~94** (Innovation 38, Storytelling 26, Training 19, Engineering 10).

---

# PART B+ — INTELLIGENCE PROOF LAYER (post-review hardening)

**Goal of Part B+:** Turn "the system works" into "the **policy** measurably improved" with **generalization**, **hard-seed retraining**, **behavior evolution**, and **judge-readable** numbers—without conflating **interpretability** with **learning**.

All additions are **strictly additive** — zero schema changes, zero reward-physics changes, no new training graders beyond the agreed set. All run AFTER the frozen training run in Part B is complete. Nothing in Part B+ can invalidate Part B results.

**Scope control note:** Advanced analysis methods (for example clustering-heavy policy consistency systems, full causal credit-assignment infrastructure, learned quality proxies, and extra unseen-extreme environment branches) are intentionally excluded to keep signals clear, reproducible, and hackathon-executable.

**Exit criteria for Part B+:**
- [ ] `artifacts/generalization.json` committed — trained model tested on 3 configs without retraining.
- [ ] `artifacts/hard_seed_retraining.json` committed — focused retraining on failure seeds.
- [ ] `artifacts/behavior_evolution.png` committed — behavioral metrics over training.
- [ ] `artifacts/ablation_no_negotiation.json` committed — feature-importance proof.
- [ ] `artifacts/run_config.json` committed — full reproducibility manifest.
- [ ] `artifacts/composite_score.json` committed — single judge-friendly number (before/after).
- [ ] `scripts/run_full_pipeline.py` — one-command orchestrator runs end-to-end on a fresh machine.
- [ ] `info["action_quality"]` and `info["strategy_phase"]` wired and visible in demo.

---

## B+.1 — Generalization proof (the single biggest anti-overfit signal)

**Why:** Training on `siyaani_fashion` only means a judge's first question is "is this overfit?" Running the same trained adapter against `medplus_pharmacy` and `stackbase_saas` — **without retraining** — answers that question conclusively.

- [ ] **B+.1.1** — Load the trained LoRA adapter from Part B.
- [ ] **B+.1.2** — For each of the 3 business configs (`siyaani_fashion`, `medplus_pharmacy`, `stackbase_saas`):
  - Use `/config` endpoint (or re-instantiate `EcomEnv(config_path=...)`) to swap config.
  - Run 10 evaluation episodes × 50 steps with fixed seeds.
  - Collect: total reward, all grader scores, bank_final, anomalies count.
- [ ] **B+.1.3** — Also run the zero-shot LLM (same model, untrained) on the same 3 configs × 10 seeds for contrast.
- [ ] **B+.1.4** — Write `artifacts/generalization.json`:
  ```json
  {
    "trained": { "siyaani": {...}, "medplus": {...}, "stackbase": {...} },
    "zero_shot": { "siyaani": {...}, "medplus": {...}, "stackbase": {...} },
    "summary": {
      "mean_improvement_seen_env": 0.XX,
      "mean_improvement_unseen_envs": 0.YY,
      "generalization_ratio": "YY / XX"
    }
  }
  ```
- [ ] **B+.1.5** — Success threshold: mean improvement on unseen envs ≥ 60% of the improvement on the trained env. Below that → the model is overfit and we retrain with domain randomization.
- [ ] **B+.1.6** — Plot: 3-panel bar chart `artifacts/generalization.png` showing trained-vs-zero-shot across all 3 envs.

---

## B+.2 — Hard-seed retraining loop (TRUE Theme 4 — Self-Improvement)

**Why:** The curriculum in A.2.4 + B.5 is *externally scheduled* self-improvement. Hard-seed retraining is *failure-driven* self-improvement — the agent detects its own weak spots and retrains on them. This is the archetypal Theme 4 pattern.

- [ ] **B+.2.1** — From the Part B post-training evaluation across 10 seeds, identify the bottom-3 seeds where trained reward is lowest. Call these `hard_seeds`.
- [ ] **B+.2.2** — Run a short focused retraining burst (20–40 additional GRPO steps) using only `hard_seeds` in rollout sampling.
- [ ] **B+.2.3** — Save the resulting adapter as `swiftlogic_grpo_adapter_hardened`.
- [ ] **B+.2.4** — Re-evaluate on ALL 10 seeds (not just the hard ones) and compare:
  - Reward on the original hard seeds: expect ↑.
  - Reward on the easy seeds: expect same or ↑ (if it drops sharply, we overfit the hard set — stop and revert).
  - Overall mean reward: expect ↑.
- [ ] **B+.2.5** — Write `artifacts/hard_seed_retraining.json`:
  ```json
  {
    "hard_seeds": [...],
    "before_hardening": { "per_seed_reward": {...}, "mean": 0.XX },
    "after_hardening":  { "per_seed_reward": {...}, "mean": 0.YY },
    "delta_on_hard_seeds": 0.ZZ,
    "delta_on_easy_seeds": 0.WW,
    "regression_detected": false
  }
  ```
- [ ] **B+.2.6** — Reference this artifact in README under "Theme 4 — Self-Improvement" evidence.

---

## B+.3 — Behavior evolution visualization

**Why:** A reward curve proves the scalar went up. It does not prove *the behavior changed*. Judges watching the demo need to see that the trained model **acts differently**, not just scores higher.

- [ ] **B+.3.1** — During Part B training, at checkpoints every 25 GRPO steps, run 3 evaluation episodes and log per-step:
  - stockout days / episode
  - negotiate-action frequency
  - ad-spend-action frequency
  - set-price-action frequency
  - mean profit margin
  - mean customer satisfaction
  - action diversity entropy (Shannon entropy over action-type distribution)
- [ ] **B+.3.2** — Generate `artifacts/behavior_evolution.png` — 6-panel grid, x-axis = training step, one line per metric, with zero-shot baseline drawn as a flat dashed line.
- [ ] **B+.3.3** — Caption each panel with a single interpretation sentence (e.g., "Stockout days drop from 12 → 3 over training. Policy shifted toward inventory-preserving behavior.").
- [ ] **B+.3.4** — This plot is the **second most-screenshotted artifact in the video** after the reward curve. Make it clean.
- [ ] **B+.3.5 — Policy Signature (Core Learning Proof)**  
  Track policy-level behavior changes across training:
  - `avg_price_delta`
  - `restock_frequency`
  - `negotiation_usage`
  - `ticket_resolution_speed`  
  Log per checkpoint (every 25 GRPO steps).  
  Save: `artifacts/policy_signature.json`  
  Plot: `artifacts/policy_evolution.png`  
  Purpose: show that the **policy** itself changed, not just reward.
- [ ] **B+.3.6 — Exploration vs Exploitation Curve**  
  Compute action entropy over training and plot entropy vs training step (expected: high → medium → low).  
  Save: `artifacts/exploration_curve.png`.

---

## B+.4 — Action quality signal (`info["action_quality"]`)

**Why:** Currently we emit reward and KPIs but not a direct "was this a good call?" flag. Judges skim demos — one of `"good" | "neutral" | "bad"` per step is instantly legible.
**Clarification:** `action_quality` is a **heuristic interpretability signal**, not a learned evaluation function.

**Deterministic rule (no randomness, no LLM judge):**
```
Let step_reward = info["reward_breakdown"]["total"].
Let rolling_mean = mean of last 10 step_rewards (or 0 if < 10 steps).
Let rolling_std  = std  of last 10 step_rewards (or 1 if < 10 steps).

If step_reward > rolling_mean + 0.5 * rolling_std AND action_succeeded:
    action_quality = "good"
Elif step_reward < rolling_mean - 0.5 * rolling_std OR action_error:
    action_quality = "bad"
Else:
    action_quality = "neutral"
```

- [ ] **B+.4.1** — Implement `info["action_quality"]` in `world_engine.step` using the rule above. Additive to `info`, zero schema change.
- [ ] **B+.4.2** — Expose `info["action_quality_reason"]` — single sentence explaining the classification (e.g., `"reward +18% above rolling mean, action succeeded"`).
- [ ] **B+.4.3** — Surface in demo: each step's headline includes `[✓ good decision]` / `[~ neutral]` / `[✗ bad decision]` tag.
- [ ] **B+.4.4** — Write `tests/test_action_quality_rule.py` — deterministic fixture-based test proving the classification rule is stable.

---

## B+.5 — Strategy phase signal (`info["strategy_phase"]`)

**Why:** A **derived** label that helps **interpret** how chaotic vs stable the *observed* action stream is. It is **not** “the model learned a strategy” inside the env—it is a **post-hoc descriptor** of behavior evolution (complements the **trained** policy’s actual improvement in Part B).

**Deterministic rule based on history buffer (no ML, no heuristics the env can't verify):**
```
Let action_entropy = Shannon entropy over last 20 action types.
Let stability     = info["policy_stability"] (already computed).
Let intent_consistency = fraction of last 20 steps where info["intent"] stayed the same.

If action_entropy > 1.5 AND stability < 0.4:
    strategy_phase = "reactive"         # flailing, high action churn
Elif action_entropy in [0.8, 1.5] AND stability in [0.4, 0.7]:
    strategy_phase = "adaptive"         # settling into patterns
Elif action_entropy < 0.8 AND stability > 0.7 AND intent_consistency > 0.7:
    strategy_phase = "strategic"        # coherent multi-step plans
Else:
    strategy_phase = "reactive"         # default
```

- [ ] **B+.5.1** — Implement `info["strategy_phase"]` in `world_engine.step` using the rule above.
- [ ] **B+.5.2** — Track `info["strategy_phase_confidence"]` — how cleanly the thresholds were crossed (distance to nearest boundary).
- [ ] **B+.5.3** — In Part B training logs, plot strategy_phase transitions over training steps — this visualizes a **derived descriptor of behavior patterns** (interpretability signal), not learned intelligence.
- [ ] **B+.5.4** — Demo prints current strategy_phase as part of the step headline.
- [ ] **B+.5.5** — Honest caveat: **`strategy_phase` is a derived interpretability tag**, not learned intelligence. Document in the info payload: `info["strategy_phase_note"] = "rule-based descriptor of action/history patterns; not a learned classifier"`. **Intent** in `info` must remain **derivable** from state/action history, not free-form model output. **Confidence** = deterministic interpretability score (see A.6.4), not self-reported “model certainty.”
- [ ] **B+.5.6** — Add explicit note in README/demo:  
  `strategy_phase` is NOT learned intelligence; it is a deterministic interpretation of action patterns.  
  Learning is proven via reward improvement, policy signature, and generalization.

---

## B+.6 — Composite score (the single number a judge will remember)

**Why:** Judges remember one number. Not six. If we give them six graders, they will remember none.

- [ ] **B+.6.1** — Define weights (tunable, pinned in config):
  ```
  composite = 0.30 * profit_task
            + 0.20 * inventory_task
            + 0.15 * triage_task
            + 0.15 * stability_task
            + 0.10 * competitor_response_task
            + 0.10 * crisis_recovery_task
  ```
- [ ] **B+.6.2** — Compute composite for: random, wait-only, heuristic, zero-shot LLM, trained LLM, hardened LLM.
- [ ] **B+.6.3** — Write `artifacts/composite_score.json`:
  ```json
  {
    "weights": {...},
    "scores": {
      "random":     0.XX,
      "wait_only":  0.XX,
      "heuristic":  0.XX,
      "zero_shot":  0.XX,
      "trained":    0.YY,
      "hardened":   0.ZZ
    },
    "headline_before_vs_after": "0.42 → 0.81 (+93%)"
  }
  ```
- [ ] **B+.6.4** — This headline goes in: README top, video final slide, DEMO.md cover, HACKATHON_ALIGNMENT.md verdict. Everywhere. **One number, everywhere.**

---

## B+.7 — Ablation: trained model with negotiation disabled

**Why:** Proves that the trained model has actually learned to use a specific feature — not just "get more reward somehow." If disabling negotiation drops reward significantly, that is hard evidence of learned feature-use.

- [ ] **B+.7.1** — Copy the trained adapter. In rollout, remap any generated `NegotiateAction` → `WaitAction` before submitting to `/step`. This disables negotiation *without retraining*.
- [ ] **B+.7.2** — Run 10 seeds × 50 steps × 3 configs.
- [ ] **B+.7.3** — Write `artifacts/ablation_no_negotiation.json`:
  ```json
  {
    "full_model":        { "mean_reward": 0.YY, "per_seed": [...] },
    "no_negotiation":    { "mean_reward": 0.XX, "per_seed": [...] },
    "contribution_of_negotiation": "YY - XX = 0.ZZ"
  }
  ```
- [ ] **B+.7.4** — Expected: `no_negotiation` performs noticeably worse, especially under the `stackbase_saas` / supplier-heavy config. If NOT, we have evidence the model is ignoring negotiation — worth investigating.
- [ ] **B+.7.5** — Optional second ablation: ad-spend disabled. Only add if time permits.

---

## B+.8 — Reproducibility manifest (`run_config.json`)

**Why:** Judges who want to reproduce our results need a single pinned manifest.

- [ ] **B+.8.1** — At start of training run, dump `artifacts/run_config.json`:
  ```json
  {
    "git_commit": "<sha>",
    "env_frozen_tag": "release/env-frozen-v2.3",
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "trainer": "trl.GRPOTrainer",
    "unsloth_version": "...",
    "trl_version": "...",
    "torch_version": "...",
    "python_version": "...",
    "seeds": {
      "training": 2026,
      "evaluation": [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    },
    "hyperparameters": {
      "grpo_group_size": 8,
      "learning_rate": 5e-6,
      "steps_per_episode": 50,
      "num_training_episodes": 200,
      "lora_r": 16,
      "lora_alpha": 16,
      "temperature_train": 0.7,
      "temperature_eval": 0.0
    },
    "env_configs_used": {
      "stage_1_easy":   "configs/siyaani_fashion_easy.json",
      "stage_2_medium": "configs/siyaani_fashion.json",
      "stage_3_hard":   "configs/siyaani_fashion_demo.json"
    },
    "reward_weights": { "alpha": 1.0, "beta": 10.0, "gamma": 2.0 }
  }
  ```
- [ ] **B+.8.2** — Commit alongside all other training artifacts.
- [ ] **B+.8.3** — Link from README: "To reproduce, see `artifacts/run_config.json`."

---

## B+.9 — Confidence calibration in the demo

**Why:** We already emit `info["confidence"]` with a documented formula. The demo currently doesn't narrate the *calibration* — which is what impresses evaluators of agentic AI.

- [ ] **B+.9.1** — In `_print_demo_step`, after the chosen action + action_quality are printed, add a calibration line:
  - `confidence ≥ 0.75` + `action_quality == "good"` → `[✓ high confidence, validated]`
  - `confidence ≥ 0.75` + `action_quality == "bad"` → `[⚠ overconfident — model was wrong]`
  - `confidence < 0.50` + `action_quality == "good"` → `[~ lucky — low-confidence good call]`
  - `confidence < 0.50` + `action_quality == "bad"` → `[✓ model knew it was uncertain]`
- [ ] **B+.9.2** — Over an episode, aggregate these 4 buckets and print a calibration summary at the end:
  ```
  Calibration: 18 validated / 4 overconfident / 2 lucky / 6 uncertain-and-correct
  ECE-style score: 0.XX
  ```
- [ ] **B+.9.3** — Commit `artifacts/calibration_summary.json` for trained vs zero-shot — expect trained model to be *better calibrated* (fewer overconfident failures).

---

## B+.10 — Failure-case demo (baseline collapse → trained recovery)

**Why:** Contrast is the single most memorable storytelling device. A judge who sees "baseline went bankrupt on day 23; trained agent survived and ended with ₹X profit" will remember the submission.

- [ ] **B+.10.1** — Extend `scripted_demo.py` (from A.7.1) with a `--policy {baseline|trained}` flag.
- [ ] **B+.10.2** — Using the same seed and the same environmental shock schedule (deterministic):
  - `baseline`: runs the wait-only policy. Expected trajectory: stockout cascade → satisfaction collapse → bank decline → episode ends badly.
  - `trained`: runs the trained LLM policy on the identical seed. Expected: restock before shock, undercut competitor response, satisfaction recovery.
- [ ] **B+.10.3** — Save both runs to `artifacts/demo_baseline.log` and `artifacts/demo_trained.log`.
- [ ] **B+.10.4** — Generate `artifacts/failure_vs_recovery.png` — 2-panel time series (baseline bank vs trained bank, same x-axis). This is the video money shot.
- [ ] **B+.10.5** — 2-minute video must include the 15-second contrast cutaway between these two runs.
- [ ] **B+.10.6 — Manual Credit Assignment Example (Demo Only)**  
  Show one trajectory manually in narration:
  - Day 5: restock  
  - Day 9: inventory recovered → revenue up  
  Used only in demo narration; no system-level implementation.

---

## B+.11 — One-command pipeline (`scripts/run_full_pipeline.py`)

**Why:** A judge who can re-run the entire thing with one command is a judge who trusts the submission.

- [ ] **B+.11.1** — Create `scripts/run_full_pipeline.py` that:
  1. Verifies env frozen tag matches current commit.
  2. Runs `pytest -q` — hard fail if any test broken.
  3. Starts `server.app` on a random port.
  4. Runs 4 baselines (wait / random / heuristic / zero-shot) → `baseline_metrics.json`.
  5. Trains with GRPO (respects `--episodes N` flag; default 200).
  6. Runs after-training eval → `after_metrics.json`.
  7. Runs B+.1 generalization test.
  8. Runs B+.2 hard-seed retraining.
  9. Runs B+.7 ablation.
  10. Runs B+.10 failure-vs-recovery demo.
  11. Computes B+.6 composite.
  12. Generates all plots.
  13. Writes `artifacts/run_config.json`.
  14. Shuts down server cleanly.
- [ ] **B+.11.2** — Test on a fresh clone: `git clone <repo> && cd <repo> && pip install -r requirements.txt && python scripts/run_full_pipeline.py --smoke-test` should complete in < 15 minutes and produce every artifact (just with fewer training episodes).
- [ ] **B+.11.3** — Full mode: `python scripts/run_full_pipeline.py` produces the submission-ready artifact set. Target wall-clock: **≤ 90 minutes on free Colab T4**, hard cap 2 hours. If over 2 hours, reduce `MAX_EPISODES` or switch to 0.5B model.
- [ ] **B+.11.4** — **`--fast-mode` flag** (post-review hardening): a judge-friendly mode that must finish in **≤ 30 minutes** on any machine. Fast mode:
  - Forces `MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct` (no 1.5B fallback).
  - Caps training at 30 GRPO steps (vs. full 200+).
  - Uses only 5 evaluation seeds instead of 10.
  - **Skips** hard-seed retraining (B+.2), ablation (B+.7), and cross-env generalization (B+.1) — or runs each with 3 seeds only.
  - Still produces: `reward_curve.png`, `before_metrics.json`, `after_metrics.json`, `composite_score.json`, `run_config.json` (marked `"mode": "fast"`).
  - Use case: a judge clones the repo, hits `--fast-mode`, sees a real reward curve in 30 min.
- [ ] **B+.11.5** — Matrix of modes:
  - `--smoke-test` → <15 min, integration test only, tiny training burst
  - `--fast-mode` → <30 min, demonstration-quality artifacts
  - (default) full mode → ≤90 min, submission-quality artifacts
- [ ] **B+.11.6** — README: top-of-file one-liner: `Run the full pipeline: python scripts/run_full_pipeline.py` (with a note about `--fast-mode` for quick judging).

---

## B+.12 — Part B+ verification gate

- [ ] **B+.12.1** — All 8 artifact files from the exit-criteria list committed.
- [ ] **B+.12.2** — Composite score `0.XX → 0.YY (+Z%)` pinned in README.
- [ ] **B+.12.3** — `scripts/run_full_pipeline.py --smoke-test` completes from a fresh clone.
- [ ] **B+.12.4** — `info["action_quality"]` and `info["strategy_phase"]` visible in a live `/step` call.

## FINAL RL PROOF SUMMARY

The system demonstrates real RL learning via:
1. Reward improvement (`reward_curve.png`)
2. Behavior change (`policy_signature.json`)
3. Exploration reduction (`exploration_curve.png`)
4. Generalization across environments (`generalization.json`)
5. Self-improvement via retraining (`hard_seed_retraining.json`)
6. Baseline comparison (`before_metrics.json` / `after_metrics.json`)

Conclusion: the agent learns a policy, not just produces better outputs.

**Expected score impact after Part B+:** 94 → **~97-98** (Innovation 40, Storytelling 29, Training 20, Engineering 10; the final 2-3 pts come from Part C polish).

---

# PART C — DEPLOYMENT & FINAL: Docker, HF Space, and top-1 polish

**Goal of Part C:** Satisfy every minimum hackathon requirement, deploy publicly, record the demo, and run a final 40-point submission checklist.

**Hackathon minimum requirements** (from themes doc):
1. ✅ Usage of OpenEnv (latest release) — confirm version pin.
2. ✅ Minimal training script using Unsloth or HF TRL in Colab — done in Part B.
3. ☐ Mini-blog on HuggingFace OR ≤2-minute YouTube video.
4. ☐ Environment hosted on Hugging Face Spaces.

---

## C.1 — Dockerize the environment

OpenEnv environments deploy as HF Spaces, which use Docker containers.

- [ ] **C.1.1** — Create `Dockerfile` at repo root:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 7860
  CMD ["uvicorn", "server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
  ```
- [ ] **C.1.2** — Create `.dockerignore` — exclude `artifacts/`, `__pycache__/`, `.venv/`, `*.ipynb_checkpoints`, tests, transcripts.
- [ ] **C.1.3** — Local test: `docker build -t swiftlogic . && docker run -p 7860:7860 swiftlogic` → `curl localhost:7860/health` → 200.
- [ ] **C.1.4** — Confirm `COMMERCEOPS_DEMO_MODE=1` works inside the container via `docker run -e COMMERCEOPS_DEMO_MODE=1`.
- [ ] **C.1.5** — Image size < 1 GB preferred.

---

## C.2 — Hugging Face Space deployment

HF Spaces give you: a running server, a Git repo, and a container registry (guide §13).

- [ ] **C.2.1** — Install OpenEnv CLI: `pip install openenv` (or equivalent — follow guide §13).
- [ ] **C.2.2** — `openenv init` in a clean clone to align with current scaffold, OR manually create HF Space with SDK: Docker.
- [ ] **C.2.3** — Create HF Space: `swiftlogic-commerceops-env` (SDK: Docker, hardware: CPU basic is enough — env doesn't need GPU).
- [ ] **C.2.4** — Push repo to HF: `git remote add space https://huggingface.co/spaces/<user>/swiftlogic-commerceops-env && git push space main`.
- [ ] **C.2.5** — Wait for build. Check space logs. Verify `/health` from external URL.
- [ ] **C.2.6** — Add README badge: `[HF Space](https://huggingface.co/spaces/<user>/swiftlogic-commerceops-env)`.
- [ ] **C.2.7** — Update notebook Cell 2: `ENV_URL=https://<user>-swiftlogic-commerceops-env.hf.space`.
- [ ] **C.2.8** — Re-run a mini training burst pointing at the deployed Space to prove the loop works remotely — guide §13.
- [ ] **C.2.9 — Preloaded demo UX on the Space (post-review hardening).** A Space that only exposes `/step` and `/reset` is invisible to a judge without a client. Make the Space immediately demo-able:
  - Add a minimal landing page (FastAPI root `/`) that renders an HTML page with:
    - **"Run Demo"** button → triggers a fixed-seed 30-step scripted episode server-side and streams the demo output to the browser.
    - A small card listing the 3 configs with a dropdown to pick one before running.
    - A link to the OpenEnv endpoints (`/reset`, `/step`, `/tasks`, `/grade`) for developers.
    - A link to the Colab notebook and to the HF model adapter.
  - Zero schema change — this is a static landing + one extra non-API route.
  - Space README.md front-matter includes a preview description so the HF Space card looks clean.
- [ ] **C.2.10** — Smoke test: open the Space URL in an incognito browser → click "Run Demo" → see 30 steps of scripted output within 15 seconds. If not, fix before submission.

---

## C.3 — Mini-blog or video (required)

Pick one. Video is higher-impact for a 2-minute judge pass; blog is safer.

**Option A — 2-minute YouTube (recommended):**
- [ ] **C.3.1a** — Script the video with the **4-beat symmetric timeline** (post-review hardening):
  - **0–10s — PROBLEM.** "Running an e-commerce company means juggling pricing, inventory, competitors, and customer support simultaneously. Most AI agents can't handle this." Show the chaos.
  - **10–30s — SYSTEM.** Architecture diagram. "We built an OpenEnv environment where an AI CEO observes the market, decides actions through departments — inventory, marketing, support — and receives multi-signal rewards. Everything is explainable."
  - **30–90s — SIMULATION.** The scripted_demo cutaway (B+.10): side-by-side baseline vs trained on identical seed. Baseline goes bankrupt. Trained recovers. Narrate the causal chain: "Day 7 — competitor undercut. Day 10 — market shock. Day 14 — stockout. Day 18 — the trained agent restocks, cuts price, retains customers."
  - **90–120s — LEARNING PROOF.** Reward curve + behavior evolution plot + **composite score headline**: `"0.42 → 0.81 (+93%) across 3 business environments."` Final line: "Built on OpenEnv + TRL + Unsloth. Try it: [HF Space URL]."
- [ ] **C.3.2a** — Record with OBS / Loom. Unlisted YouTube upload.
- [ ] **C.3.3a** — Embed link in README top + HACKATHON_ALIGNMENT.md + DEMO.md.
- [ ] **C.3.4a** — The final frame must show the composite score line `0.42 → 0.81 (+93%)` or whatever the actual numbers are — this is the one line a judge will remember (B+.6.4).

**Option B — HuggingFace Mini-Blog:**
- [ ] **C.3.1b** — 800-word post. Structure: problem → environment → reward design → training → before/after → theme alignment → try it.
- [ ] **C.3.2b** — Include 3 figures: architecture diagram, reward curve, before/after comparison.
- [ ] **C.3.3b** — Publish on HF `posts` or personal blog; link from README.

---

## C.4 — Final audit checklist (top-1 polish)

Score every item honestly. Every ☐ is a point left on the table.

### Innovation (40 pts)
- [ ] **C.4.1** — Environment explicitly names **Theme 2, 3, and 4** evidence + **Theme 1** as supporting (A.2.1).
- [ ] **C.4.2** — At least 5 independent reward signals (A.3).
- [ ] **C.4.3** — At least 5 graders/tasks, each with 0 < success_rate < 1 across 10 seeds (A.4).
- [ ] **C.4.4** — Curriculum / self-improvement hook present (A.2.4).
- [ ] **C.4.5** — Multi-agent interactions documented with concrete feedback-loop example.
- [ ] **C.4.6** — Long-horizon **evidence** without new env reward: delayed-effect + trajectory + recovery story (A.2.3, B+.10, `episode_summary`).
- [ ] **C.4.7** — Anti-reward-hacking tests committed (A.5).

### Storytelling (30 pts)
- [ ] **C.4.8** — Primary theme pinned in README header.
- [ ] **C.4.9** — `DEMO.md` exists with mermaid diagram + screenshots.
- [ ] **C.4.10** — `scripted_demo.py` exists with deterministic wow moment.
- [ ] **C.4.11** — Video or blog link at top of README.
- [ ] **C.4.12** — `COMMERCEOPS_DEMO_VERBOSITY=compact` mode works.
- [ ] **C.4.13** — Department suggestions print **before** chosen action.
- [ ] **C.4.14** — Every step prints a one-line headline in normal mode.

### Training / Reward (20 pts)
- [ ] **C.4.15** — `artifacts/reward_curve.png` committed.
- [ ] **C.4.16** — `artifacts/before_metrics.json` + `artifacts/after_metrics.json` committed.
- [ ] **C.4.17** — `artifacts/thought_logs.json` with ≥5 traces committed.
- [ ] **C.4.18** — Trained adapter on HF Hub with URL in README.
- [ ] **C.4.19** — Notebook runs end-to-end on fresh Colab T4 in < 2 hours.
- [ ] **C.4.20** — `before` vs `after` shows statistically meaningful improvement (≥1 σ).

### Engineering / Pipeline (10 pts)
- [ ] **C.4.21** — 218+ tests pass locally and in a clean venv.
- [ ] **C.4.22** — Dockerfile builds and runs cleanly.
- [ ] **C.4.23** — HF Space live and responsive.
- [ ] **C.4.24** — README claims match code reality (no broken claims).
- [ ] **C.4.25** — License file present.
- [ ] **C.4.26** — CHANGELOG or version tag indicating submission state.

### Bonus / Intelligence
- [ ] **C.4.27** — CEO intent + department suggestions visible in demo and in `info`.
- [ ] **C.4.28** — `why_failed` + `anomalies` populate on actual failures.
- [ ] **C.4.29** — Confidence formula documented in-band.
- [ ] **C.4.30** — Episode summary rendered in human prose at termination.

### Part B+ Intelligence Proof artifacts (added post-review)
- [ ] **C.4.31** — `artifacts/generalization.json` + `artifacts/generalization.png` committed; mean-improvement ratio unseen/seen ≥ 0.6.
- [ ] **C.4.32** — `artifacts/hard_seed_retraining.json` committed; no easy-seed regression.
- [ ] **C.4.33** — `artifacts/behavior_evolution.png` committed; 6-panel behavior change over training.
- [ ] **C.4.34** — `artifacts/ablation_no_negotiation.json` committed; negotiation contribution > 0.
- [ ] **C.4.35** — `artifacts/composite_score.json` with headline `"0.XX → 0.YY (+Z%)"`; same headline pinned in README top, video final slide, DEMO.md.
- [ ] **C.4.36** — `artifacts/run_config.json` committed; judges can reproduce everything.
- [ ] **C.4.37** — `artifacts/calibration_summary.json` + demo narrates high-confidence-validated vs overconfident-failure buckets.
- [ ] **C.4.38** — `artifacts/failure_vs_recovery.png` committed; 15-second video cutaway uses it.
- [ ] **C.4.39** — `info["action_quality"]` and `info["strategy_phase"]` wired, visible in demo, documented.
- [ ] **C.4.40** — `scripts/run_full_pipeline.py` works end-to-end on a fresh clone in ≤ 90 min (smoke mode ≤ 15 min).
- [ ] **C.4.41** — `git tag release/env-frozen-v2.3` exists; env diff since tag is additive-only.

---

## C.5 — Submission package

- [ ] **C.5.1** — Repo: public GitHub URL.
- [ ] **C.5.2** — HF Space: live URL.
- [ ] **C.5.3** — HF Hub adapter: URL.
- [ ] **C.5.4** — Video / blog: URL.
- [ ] **C.5.5** — Notebook: accessible in Colab (`Open in Colab` badge in README).
- [ ] **C.5.6** — Theme statement: one sentence pinned in README.
- [ ] **C.5.7** — Reproducibility: `seed=2026 python scripted_demo.py` produces identical output for any judge who clones the repo.
- [ ] **C.5.8** — Final `pytest -q` run: committed as `artifacts/final_test_output.txt`.

---

## C.6 — Part C verification gate

- [ ] **C.6.1** — All C.4.1 through C.4.30 boxes green.
- [ ] **C.6.2** — A judge with zero context can:
  1. Click README video link → understand product in 90 seconds.
  2. Click HF Space → see live env.
  3. Click Colab link → run training on T4.
  4. Open `artifacts/` → see reward_curve.png + before/after JSONs.
  5. Run `scripted_demo.py` locally → see the wow moment.

**Expected score impact after Part C:** 94 → **~99–100** (Innovation 40, Storytelling 30, Training 20, Engineering 10).

---

# SUMMARY — The critical path

```
Part A  (validation + exposure; then freeze) → +8.5 pts  → score ~82
  A.2    Theme 2/3/4 narrative + A.2.3 exposure-only (no new env reward)
  A.2.4  Self-improvement = training-side (B.5 + B+.2), not env curriculum
  A.4    Add 3 more graders (now 6 total; 3 eval-only)
  A.5    Anti-reward-hacking tests
  A.7    Scripted demo + compact verbosity
  A.9.6  Env-freeze tag (post-Gemini hardening)

Part B  (training + proof)             → +12 pts   → score ~94
  B.3    Unsloth integration
  B.5    3-stage curriculum during training
  B.7    Mandatory 4-baseline contrast + reward curve + before/after JSON

Part B+ (intelligence proof layer)     → +3-4 pts  → score ~97-98
  B+.1   Generalization (3 configs, no retrain)
  B+.2   Hard-seed retraining (TRUE Theme 4)
  B+.3   Behavior evolution plots
  B+.4   info["action_quality"]  (additive)
  B+.5   info["strategy_phase"]  (additive)
  B+.6   Single composite score "0.XX → 0.YY"
  B+.7   Ablation (no-negotiation)
  B+.8   run_config.json reproducibility
  B+.9   Confidence calibration in demo
  B+.10  Failure vs recovery cutaway
  B+.11  One-command run_full_pipeline.py

Part C  (deploy + demo + final)        → +2-3 pts  → score ~99-100
  C.1    Docker
  C.2    HF Space live
  C.3    90-second video (includes B+.10 cutaway + B+.6 headline)
  C.4    41-item final checklist green
```

**Total estimated timeline:**
- Part A: **1 full day** of focused work.
- Part B: **1 day** (small Colab run) + 1 overnight (longer run) = ~24 hours of wall clock with ~4 hours active.
- Part B+: **6-8 hours** (most is re-running evals with the already-trained adapter; no new training from scratch).
- Part C: **4-6 hours** (Docker + Space + video + checklist).

**Total: ~3 calendar days.**

---

# What to do RIGHT NOW

1. Read this file end to end (north star + Theme 2/3/4 table first).
2. Confirm the **submission narrative**: **Themes 2, 3, and 4** as primary evidence; **Theme 1** supporting; **no** claims that the env “learns” or that rule-based entities are RL co-learners.
3. Approve **env freeze** before Part B (A.9.6): no new reward terms, no new physics—**learning only in training**.
4. Approve the Qwen 0.5B → 1.5B fallback strategy and **200–300 episode** target (or equivalent GRPO volume).
5. Approve video vs blog choice (default: video).
6. **Approve the Part B+ additions** (generalization, hard-seed retraining, action_quality, strategy_phase as **derived** labels, composite score, ablation, run_config, calibration demo, failure-vs-recovery, `run_full_pipeline`). These are *measurement and orchestration* on the frozen env; **baselines** include **wait** + **random** with committed JSON (`baseline_wait.json`, `baseline_random.json`).

Start Part A at A.1.1 and check boxes sequentially.

**Do not skip Part A to rush to Part B. Do not skip Part B+ to rush to Part C. The score model is strictly additive: Part B+ is what pushes 94 → 98.**
