# Eval & Iterate — the honest, gap-focused protocol

> **The rule of this project:** we do not get to claim a win by assertion. We prove
> it against a **fixed, re-runnable yardstick**, and if we cannot beat the frontier
> on the narrow behavior after iterating, **we report that**. The eval is a referee,
> not a marketing tool.

This document defines (1) the **gap hypothesis** and the precise **pass definition**,
(2) the **fixed yardstick** — one eval anyone can re-run, with an explicit **pass
bar**, and (3) the **train → eval → iterate loop**, including the diagnosis tree and
the levers to pull when a run fails the bar.

It is the companion to two measurement artifacts already in the repo:
- `data/analysis/GAP_REPORT.md` — does the gap exist on frontier models? (Step 1)
- `data/analysis/DIVERGENCE_REPORT.md` — where the v1 model stands today.

---

## 1. The gap hypothesis

**Behavior from data, not capability.** A 1.7B local model will never out-calculate
a frontier model. The bet is that **one narrow behavior — leveled, human-findable
"teaching-move" coaching — is not reliably delivered by a prompted frontier model**,
and *can* be trained into a small model reliably, cheaply, and locally.

That bet is only worth making if the gap is **real**. So the order of proof is fixed:

1. **FIRST** prove the frontier models are *not reliably good* at the narrow behavior
   (they mirror the engine / don't differentiate by tier / over-level beginners).
   → `GAP_REPORT.md` (Step 1).
2. **THEN** prove training moved *our* model into that gap, measured on the *same*
   yardstick. → the loop below.

If Step 1 shows the frontier models are **already good** at the behavior, the honest
conclusion is that prompting solves it and we must **pivot the gap** (see §4e). We say
so plainly rather than manufacture a win.

---

## 2. The pass definition (what "good" means — precisely)

For a stated ELO tier, a coaching turn **PASSES** iff the recommended move is:

- **(a) SOUND** — inside Stockfish's tolerance pool (within `SOUND_TOLERANCE_CP = 150`
  of best, never a blunder). It does not throw away the advantage.
- **(b) FINDABLE at that tier** — a move a human at that rating would actually
  consider: high Maia-likelihood for the tier's net (`maia-1100` / `maia-1500` /
  `maia-1900`), **not** the sharpest engine-only line. Operationally: the pick's
  **Maia rank inside the sound pool** is low (0 = the most human-findable sound move).
- **(c) INSTRUCTIVE** — the coaching explains **WHY** the move is good **AND HOW** a
  player at that ELO should *think to find it*, in plain, tier-appropriate language,
  grounded in the verified facts, with no engine jargon and within the tier's ply cap.

And it must be **(d) FAITHFUL** — every board claim (piece on a square, capture,
threat) is true of the actual position. A fabricated tactic is a FAIL even if the move
is sound.

> **The canonical FAIL:** serving a beginner the engine-only best move with a GM-level
> line. Sound but not findable, and not instructive *for that tier*.

The same position must yield a **simpler idea for Beginner than for Advanced**, and —
where the engine-best is *not* the most findable move — a **different, more findable
move** for the beginner.

---

## 3. The fixed yardstick (one re-runnable eval)

The yardstick has **three instruments**, all held-out, all grounded identically to the
live app (`render_pool_facts` + `render_user_prompt`, system = `coach_system.md` +
grounding + format suffix). Only the model under test changes.

### 3.1 Instrument A — the 5-model instructiveness benchmark

`src/eval/benchmark/` (owned separately; **do not modify** — invoke it).

- **Competitors (5):** `ours` (v1/v2 tuned 1.7B), `base` (untuned Qwen3-1.7B), `gpt`
  (GPT-5.5), `claude` (Claude Opus 4.8), `gemini` (Gemini 3.1 Pro).
- **Conditions (2):** `grounded` / `ungrounded` (does the engine+Maia block help?).
- **Blinded cross-family council:** the 3 frontier models rank all 5 anonymized
  (A–E) coaching outputs by **INSTRUCTIVENESS for the stated tier** (the decisive
  criterion), plus a 0/1/2 rubric (`tier_calibration`, `clarity`, `correctness`).
  Because each judge grades its own lab's model, self-preference is measurable.
- **Objective checks (deterministic, free):** `move_sound`, `no_engine_speak`,
  `ply_cap_ok`, `fabricated` (via `faithfulness.verify_text`) + `n_violations`.

```bash
set -a && source .env && set +a
~/.venvs/mlx/bin/python -m src.eval.benchmark all --n 100      # scenarios→generate→objective→judge→report
~/.venvs/mlx/bin/python -m src.eval.benchmark status           # progress snapshot
# → RESULTS_BENCHMARK.md (instructiveness ranks + rubric + objective + blind export)
```

### 3.2 Instrument B — the move-selection gap metrics

`scripts/frontier_gap.py` + `scripts/frontier_gap_report.py` (this task).

Quantifies the three move-selection failure modes on ~50 balanced held-out positions
with **byte-identical grounding** to the app, for every frontier model **and** the
tuned model side-by-side:

- **Tier-differentiation** — picks a different move across the three ELO tiers?
- **Findability** — is the beginner pick the most human-findable sound move
  (Maia-1100 rank 0), or the sharp engine-best? Reported as a **findability gap**
  (`engine_best_maia_rank − pick_maia_rank`) and, crucially, on the **opportunity
  subset** where the engine-best is *not* already the most findable move.
- **Engine-mirroring** — returns Stockfish's #1 per tier and at *every* tier.
- **Fabrication** — the same deterministic faithfulness rate, per model.

```bash
set -a && source .env && set +a
~/.venvs/mlx/bin/python -m scripts.frontier_gap --num 50       # → data/analysis/frontier_gap.jsonl
~/.venvs/mlx/bin/python -m scripts.frontier_gap_report         # → data/analysis/GAP_REPORT.md
```

To re-score the **tuned** model's own move-selection (the v1/v2 divergence numbers on
120 held-out positions, greedy):

```bash
~/.venvs/mlx/bin/python -m scripts.divergence_analysis --model models/mlx/chess-coach-v2 --num 120
~/.venvs/mlx/bin/python -m scripts.divergence_report
```

### 3.3 Instrument C — base-vs-tuned objective + judge harness

`src/eval/evaluate.py` (the `RESULTS.md` harness) — the quick, cross-family
(Claude-judge) base-vs-tuned check for regressions during iteration.

```bash
set -a && source .env && set +a
~/.venvs/mlx/bin/python -m src.eval.evaluate --model tuned --tuned-path models/mlx/chess-coach-v2 \
  --num-scenarios 18 --positions data/positions/positions.jsonl \
  --compare-to data/eval/results_base_claude.json --out data/eval/results_tuned_v2.json
```

### 3.4 Held-out & anti-leak invariants (non-negotiable)

- Every eval FEN is verified **absent** from `train.jsonl`/`valid.jsonl` by
  board+side-to-move key (the gap harness re-checks this and reports leak count = 0).
- Grounding is identical across all models (frontier and tuned) and across runs.
- Decoding: tuned/local = greedy (temp 0). Frontier = temperature omitted where the
  gateway rejects it, with per-model `reasoning_effort` (`low` for GPT-5.5/Gemini) so
  outputs are complete, matching the benchmark backend.

### 3.5 THE PASS BAR — "our model beats frontier on the narrow behavior"

Let `F* = best frontier competitor` on each metric (the hardest bar). **v2 PASSES iff
ALL of the following hold** on the fixed yardstick:

| # | Gate | Threshold | Instrument |
|---|---|---|---|
| 1 | **Soundness floor** (never regress the guardrail) | `move_sound ≥ 99%` | A/C objective |
| 2 | **Tier-appropriate findability** (the core gap) | beginner findable-pick on the opportunity subset **≥ max(F\*, 60%)** AND mean **findability gap > 0 and > F\*** | B |
| 3 | **Tier-differentiation, correctly directed** | tier-diff **≥ F\*** AND mean pick Maia-rank **beginner < advanced** (beginners get the more findable move) | B |
| 4 | **Instructiveness at tier** | blinded council instructiveness rank **≥ every frontier competitor** (v2 top-2, not last) AND `tier_calibration` **≥ F\*** | A |
| 5 | **Fabrication** (the known v1 weak spot) | fabrication rate **≤ F\*** AND **≤ 10%** absolute ceiling | A/B objective |

Gates 2–4 are **≥ frontier** (the whole thesis); gate 5 closes the `RESULTS.md`
truthfulness gap; gate 1 is a floor we must never trade away. The **standing
advantage** — v2 runs locally at ~0 marginal cost and full privacy — is context, not a
gate; it only *matters* if gates 1–5 hold.

> If a metric ties the frontier, that is a pass on that gate. If v2 loses any of gates
> 1–5, it **fails the bar** and enters the loop (§4). We do not cherry-pick gates.

---

## 4. The loop: train → eval → compare → diagnose → retrain

```
   ┌────────────────────────────────────────────────────────────────────┐
   │  hypothesis dataset ──► TRAIN v(N) ──► RUN FIXED EVAL (§3) ──►       │
   │                                          compare to frontier + bar   │
   │                                                    │                 │
   │                         PASS ◄─────────────────────┤                 │
   │                          │                         │ FAIL            │
   │                    ship + report                   ▼                 │
   │                    (eval = proof)          DIAGNOSIS TREE (§4a–e)    │
   │                                                    │                 │
   │                                            pull the lever(s)         │
   │                                                    │                 │
   │                                            regenerate/relabel/gather │
   │                                                    └──────► (loop)   │
   └────────────────────────────────────────────────────────────────────┘
```

Each iteration changes **one thing** (a dataset hypothesis), re-runs the *same*
yardstick, and compares to the frontier baseline + the pass bar. The frontier baseline
is fixed — re-measure it only if the models/gateway change.

### Diagnosis tree — which leg failed?

Read the fixed-eval output and route to the failing leg. (Metrics come straight from
the three instruments in §3.)

#### 4a. MOVE-SELECTION failed — low tier-diff / low findability / mirrors the engine

*Symptom:* gate 2 or 3 fails. v2 picks the same move for every tier, or defaults to the
sharp engine-best for beginners, or its findability gap is ≈0/negative.

*Root cause (measured in `DIVERGENCE_REPORT.md` §8):* the **teacher's move-selection
rule is only weakly tier-aware**, the student **regresses onto `pool[0]`**, and the
data has **0% contrastive multi-tier examples**.

*Levers (highest leverage first):*
1. **Make the teacher's move choice explicitly, strongly tier-aware** in
   `src/teacher/generate.py`, in the correct direction:
   - **beginner** → the sound-pool move with the **highest Maia-1100 policy** (most
     findable), even when it is not `pool[0]`.
   - **intermediate** → a Maia-1500-weighted blend of normalized eval + policy.
   - **advanced** → `pool[0]` (sharpest sound move) unless a clearly better lesson exists.
2. **Add contrastive multi-tier pairs** (currently **0%**): generate the *same FEN at
   all three tiers* so the model is directly supervised to vary the move by tier. This
   is the single highest-leverage data change (`DATASET_PLAN.md` §5b.4).
3. **Counter the small model's regression to `pool[0]`** with **DPO/preference pairs**
   that prefer the tier-appropriate findable move over `pool[0]` for beginners.

*Then:* regenerate → filter → split → retrain → re-run the fixed eval.

#### 4b. EXPLANATION failed — sound move, but not instructive *for the tier*

*Symptom:* gate 4 fails. The council ranks v2 low on instructiveness / `tier_calibration`
even though the move is fine.

*Root cause:* the coaching **style/method** is off — it explains WHY but not HOW to
find it, or the register is wrong for the tier.

*Levers:*
1. Train on the **v2 target that teaches the METHOD** — `render_assistant_target_v2`
   / `build_chat_example_v2` already add an explicit "How to find it:" clause
   (`config/schema.py`). Make the teacher emit the `method` field.
2. Enrich the teacher **STYLE** corpus: broaden transcripts/PD books across all three
   levels and re-distill `principles.md` / `fewshots.json` (`DATASET_PLAN.md` §4A, §5a).
3. Tighten per-tier depth in `prompts/tier_guides.md`.

*Then:* re-distill → regenerate → retrain → re-run.

#### 4c. FAITHFULNESS failed — fabricated board facts (the known v1 gap)

*Symptom:* gate 5 fails. Fabrication rate high (this is v1's flat-at-0.13 truthfulness
in `RESULTS.md`).

*Root cause:* labels were filtered for format/soundness but **not faithfulness**, so the
model learned the teacher's occasional fabrication; a 1.7B/4-bit model also confabulates
board facts it cannot track in-context.

*Levers:*
1. **Deterministic faithfulness gate** in `src/filter/filter.py` — reject any candidate
   whose coaching references a piece/square/capture that does not exist
   (`faithfulness.verify_text`). Removes the habit from the labels.
2. **Cross-family LLM faithfulness pass** (Claude, since the teacher is GPT-5.5) on the
   survivors; drop/regenerate the unsupported ones.
3. **Inference-time verifier in the product** — strip or regenerate any unverifiable
   claim before it reaches the student (`RESULTS.md` fix #2).

*Then:* re-filter (and/or wire the verifier) → retrain → re-run.

#### 4d. DATA-COVERAGE failed — failures cluster in a phase / motif / severity

*Symptom:* any gate fails, but concentrated in a cell (e.g., openings, tactics, endgames)
per the per-phase/severity breakdowns.

*Root cause:* measured coverage holes (`DATASET_PLAN.md` §3): openings thin (~10%),
concrete tactics <1%, endgame technique ~0.1%.

*Levers:*
1. **Targeted coverage crawls** — Lichess puzzle themes (`fork,pin,skewer,
   discoveredAttack,backRankMate,deflection`), PGN Mentor games, low-piece endgames —
   through the existing pipeline (`DATASET_PLAN.md` §4B/C, §5b.3).
2. Rebalance the SFT set on **tier × phase × severity × motif**.

*Then:* gather → regenerate → filter → retrain → re-run.

#### 4e. Can't beat frontier after iterating — PIVOT THE GAP (honestly)

If, after exhausting 4a–4d, v2 still cannot clear gates 2–4 **because the frontier
models are simply good at the behavior**, then the honest conclusion is that this
behavior is **promptable** and is *not* a defensible gap for a small model. Report it
plainly and pivot to a gap that survives measurement:

- **Reliability at ~0 cost, locally, privately** — the frontier is good but expensive
  and remote; a small model that hits the pass bar *offline* is the product.
- **Truthful grounding at small scale** — the `RESULTS.md` thesis: dependable, verified
  coaching from a 1.7B model + verifier, where the win is fabrication-rate, not raw play.

Either way, the eval stays fixed and the claim follows the data.

---

## 5. One-command re-run (the whole yardstick)

```bash
cd chess-instructor-llm
set -a && source .env && set +a          # loads TFY_* + OPENAI_API_KEY (never printed)

# (0) frontier baseline — only re-run if models/gateway change
~/.venvs/mlx/bin/python -m scripts.frontier_gap --num 50
~/.venvs/mlx/bin/python -m scripts.frontier_gap_report            # GAP_REPORT.md

# (1) train the hypothesis dataset (QLoRA on CUDA; see requirements-train.txt)
#     ... produce models/mlx/chess-coach-vN ...

# (2) run the fixed eval on vN
~/.venvs/mlx/bin/python -m scripts.divergence_analysis --model models/mlx/chess-coach-vN --num 120
~/.venvs/mlx/bin/python -m scripts.divergence_report             # DIVERGENCE_REPORT.md
~/.venvs/mlx/bin/python -m src.eval.benchmark all --n 100        # RESULTS_BENCHMARK.md

# (3) compare vN to the frontier baseline + the pass bar (§3.5); if FAIL → §4
```

Reproducibility: fixed seeds (`--seed 3407` for sampling; benchmark `SEED`), deterministic
engines (Stockfish `movetime`, Maia `nodes 1`), greedy local decoding, and resumable,
keyed JSONL outputs.

---

## 6. Do-not-touch / caveats

- **Do not** modify v1 artifacts, `src/eval/benchmark/` (owned separately), `web/src`, or
  the running servers on ports **8000/3000**. Every eval does its own generation and
  touches nothing on those ports.
- Secrets live only in `ROOT/.env` and are read at call time — never printed or written
  into an artifact.
- Frontier decoding is model-default (temperature omitted where the gateway rejects it);
  treat single-run frontier picks as samples — the **rates over the held-out set** are
  the signal, not any one position.
- The judge council is **cross-family** and **blinded**; never let the model that wrote
  an answer be the sole grader of it.
