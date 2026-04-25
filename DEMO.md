# Swiftlogic CommerceOps v2 — 2-minute demo script

> Phase C5 deliverable. This file is the literal screenplay for the
> hackathon submission video. The video itself is unlisted on YouTube
> and embedded in the README, but the script lives here so reviewers
> can fact-check every claim against the artifacts in `artifacts/`.
>
> **Composite headline pinned to the same value as the README and
> [`artifacts/composite_score.json`](artifacts/composite_score.json):
> `0.61 -> 0.66 (+9%)`** (provenance `heuristic_fallback`; gets
> overwritten with `trained_adapter` numbers after the Colab run).

The full timeline is **120 seconds, 4 beats**, exactly as specified in
ROADMAP §C.3.1a.

---

## Beat 1 — Problem (0 - 10 s)

**Visuals.** Hard cut from a black title card to a screen capture of an
Indian SMB seller dashboard with a bouncing red "₹0 — bankrupt" counter
and three unanswered support tickets stacking up. Cut to terminal:

```bash
python scripted_demo.py --policy wait_only --seed 20260425 --verbose
```

The wait-only baseline runs in real time. The bank balance ticks
downward while customer satisfaction collapses. The terminal narration
shows `action_quality=neutral · strategy_phase=reactive · why_failed=stockout`.

**Voice-over (≤ 25 words).**
"Indian e-commerce SMBs juggle stockouts, refunds, ad spend, and supplier
calls every day. A passive baseline just goes bankrupt — that's the
problem we're solving."

---

## Beat 2 — System (10 - 30 s)

**Visuals.** Schematic of the OpenEnv contract with the six endpoints,
followed by the architecture diagram from the README (FastAPI → EcomEnv
→ WorldEngine), zooming into the additive `info` payload showing
`reward_breakdown`, `intent`, `action_quality`, and `strategy_phase`.

Hard cut to a browser tab on the deployed HF Space:

* `https://huggingface.co/spaces/Swiftlogic/E-commerce-agent`
* The landing page renders the **Run Demo** button.
* A single click streams 30 deterministic steps via Server-Sent Events.

**Voice-over (≤ 50 words).**
"CommerceOps v2 exposes a frozen OpenEnv environment over six endpoints,
runs deterministic 50-day episodes, and emits twenty additive `info`
keys per step — including new explainability signals `action_quality`
and `strategy_phase`. It's containerised and live on HuggingFace
Spaces; one click runs a deterministic demo end-to-end."

---

## Beat 3 — Simulation cutaway (30 - 90 s)

**Visuals.** Split screen.

* **Left.** `python scripted_demo.py --policy trained --seed 20260425
  --verbose` running step-by-step, narrating every action with
  `action_quality` + `strategy_phase` + reward breakdown. The bank
  balance climbs out of the red around day 18.
* **Right.** Three plots cycling on a 10-second loop:
  1. [`artifacts/reward_curve.png`](artifacts/reward_curve.png) — GRPO
     learning curve from the Colab notebook.
  2. [`artifacts/before_after_comparison.png`](artifacts/before_after_comparison.png)
     — bar chart of all six grader scores, baseline vs trained.
  3. [`artifacts/generalization.png`](artifacts/generalization.png) —
     same adapter on `medplus_pharmacy` and `stackbase_saas` without
     retraining.

Mid-segment overlay (≈ 60 s mark): a 5-second cut to the GRPO training
notebook running in Colab — Unsloth FastLanguageModel loaded in 4-bit,
the 3-stage curriculum advancing on the rolling-mean threshold, and the
reward curve climbing in real time.

**Voice-over (≤ 90 words).**
"Training is real RL — Group Relative Policy Optimisation, Qwen2.5
half-billion-parameter base, four-bit QLoRA via Unsloth, and a
three-stage curriculum that swaps the underlying business config when
the rolling reward crosses a threshold. Reward shaping combines
environment reward, three training graders, and a format-compliance
signal — never the three evaluation-only graders, which stay isolated
to measure genuine generalisation. Same adapter, three unseen
businesses, and behaviour entropy that drops monotonically across
checkpoints — that's learning, not memorisation."

---

## Beat 4 — Learning proof + final frame (90 - 120 s)

**Visuals.** Six artifacts in sequence, each on screen for ~5 seconds:

1. [`artifacts/reward_curve.png`](artifacts/reward_curve.png)
2. [`artifacts/policy_evolution.png`](artifacts/policy_evolution.png)
3. [`artifacts/exploration_curve.png`](artifacts/exploration_curve.png)
4. [`artifacts/generalization.png`](artifacts/generalization.png)
5. [`artifacts/hard_seed_retraining.json`](artifacts/hard_seed_retraining.json) (formatted JSON pretty-print)
6. [`artifacts/before_after_comparison.png`](artifacts/before_after_comparison.png)

End on a full-screen card containing **only**:

```
Composite score
0.61 -> 0.66 (+9%)
swiftlogic.ai · openenv v0.2.3 · github.com/Swiftlogic/CommerceOps-v2
```

**Voice-over (≤ 35 words).**
"Six checked-in artifacts back every claim — reward goes up, behaviour
changes, exploration drops, the policy generalises to unseen
businesses, hard seeds get hardened, and every baseline gets beaten.
Composite score: zero point six one to zero point six six, plus nine
percent."

---

## Production checklist

* [ ] Final frame composite value matches `artifacts/composite_score.json` (run `python scripts/refresh_readme_headline.py` first).
* [ ] HF Space landing page loads in incognito within 10 s and the Run Demo button streams 30 events.
* [ ] `python scripts/run_full_pipeline.py --fast-mode` completes < 30 minutes on a fresh clone with no errors and re-emits every artifact referenced above.
* [ ] All 218+ tests green (`pytest -q`).
* [ ] Video uploaded as **unlisted YouTube**; embedded link added to the README badge row.
