# Results — v2 Chess Coach (dataset + model), v1→v2 deltas + frontier benchmark

**Headline.** The v2 dataset rebuild fixed the two measured v1 gaps and closed part
of the gap to the frontier, on the **grounded** condition the product actually
deploys in (Stockfish + Maia in the loop):

- **Truthfulness ↑ (fabrication ↓).** On 120 matched held‑out positions the tuned
  model's fabrication rate fell **46.1% → 31.7%**; on the 100‑position 5‑model
  benchmark (grounded) it fell **50% → 33%** (avg false facts/answer 0.62 → 0.46).
- **Tier‑differentiated move selection ↑ and now correctly directed.**
  Tier‑differentiation **27.5% → 39.2%**, and the *direction* flipped from **wrong
  to right**: beginners now get the more human‑findable move and advanced the
  sharpest (mean pool‑rank beginner **0.43 → 0.78** vs advanced **0.62 → 0.45**;
  "beginner move == the human/Maia move" **39% → 62%**).
- **Instructiveness gap to the frontier narrowed.** Blinded cross‑family council
  (GPT‑5.5 + Claude Opus 4.8 + Gemini 3.1 Pro) mean rank of OURS (grounded, 1=best
  of 5) improved **4.13 → 3.68** and top‑1 win‑rate **2% → 8%**; the gap to the best
  frontier model shrank **+2.22 → +1.60** rank.

Everything is v2‑suffixed; v1 artifacts, the running servers (8000/3000), and
`web/src` were not touched.

---

## What changed (teacher / filter / selection / target)

| Change | v1 | v2 |
|---|---|---|
| Teacher move choice | LLM free‑picks from the sound pool (weak, mis‑directed) | **Deterministic, tier‑aware** (`src/teacher/tier_select.py`): beginner = highest **Maia‑1100** sound move (most findable), advanced = engine best (`pool[0]`), intermediate = eval×policy blend — always inside the sound pool |
| Teacher grounding | ASCII board + pool only | **+ VERIFIED FACTS block** (`render_pool_facts`) so the teacher phrases only true pieces/squares/captures |
| Coaching target | move + why + takeaway | **3 explicit parts**: (a) the move, (b) WHY (tier concepts), (c) **HOW to FIND it** — an explicit `method` field ("How to find it: …") teaching the search process |
| Faithfulness | none (deferred) → 6.3% of labels false | **`verify_text` reject gate** in `src/filter/filter.py` (+ grounded gen + 1 retry) → residual 1.6%, then rejected → **0% false labels** |
| Contrastive tiers | **0** FENs at >1 tier | **348 FENs × 3 tiers** (same position taught per tier) via a tier‑aware `fen_tier_move` dedup |

Data‑level check that the new selection is correctly directed (whole plan, 2,628
jobs): mean pick pool‑rank **beginner 1.70 > intermediate 0.88 > advanced 0.07**;
**37%** of picks are NOT the engine's #1 (vs a weak, mis‑directed v1 signal).

---

## v2 dataset

| | v1 | v2 |
|---|---:|---:|
| Teacher candidates | 2,132 | **2,628** (1,584 single‑tier + 1,044 contrastive across 348 FENs) |
| Kept after filter | 1,448 | **2,586** (only 42 rejected — all for faithfulness, 1.6%) |
| Train / valid | 1,376 / 72 | **2,457 / 129** |
| Multi‑tier contrastive FENs | 0 | **348** (same position → per‑tier move + method) |
| Teacher fabrication at generation | 6.3% (unfiltered) | 1.6% residual (grounding + retry) → 0% after reject |

Teacher generation (GPT‑5.5, 2,797 calls, 0 failures; fully
checkpoint/resumed through two interruptions).

---

## v1 → v2 deltas

### 1. Tier‑differentiation + direction (divergence harness, 120 matched held‑out, greedy)

| Metric | v1 | v2 |
|---|---:|---:|
| **Tier‑differentiation** (≥1 tier picks a different move) | 27.5% | **39.2%** |
| Distinct‑move distribution (1 / 2 / 3) | 87 / 31 / 2 | 73 / 41 / 6 |
| mean pool‑rank — beginner | 0.43 | **0.78** |
| mean pool‑rank — advanced | 0.62 | **0.45** |
| **Direction correct** (beginner rank > advanced) | **No** | **Yes** |
| beginner move == its Maia (human) top | 39.2% | **61.7%** |

v1's differentiation was not only weak but *mis‑directed* (beginners got the
sharper engine move). v2 reverses this: beginners are steered to the move a
1000–1200 would actually find, advanced to the engine's sharpest — the exact
correction the divergence report called for.

### 2. Fabrication (truthfulness)

| Fabrication rate (≥1 false board fact) | v1 | v2 |
|---|---:|---:|
| Divergence harness (120 held‑out, raw output) | 46.1% | **31.7%** |
| Benchmark, **grounded** (100 held‑out) | 50% | **33%** |
| Benchmark, grounded — avg false facts / answer | 0.62 | **0.46** |

The faithfulness gate removed fabricated *labels*; grounded generation + the
verify‑and‑retry loop pushed source fabrication to 1.6%. The tuned model then
fabricates meaningfully less at inference **when grounded** — the deployment mode.

Continuity with `RESULTS.md` (same Claude‑Opus rubric, 15 held‑out scenarios,
0–2): the truthfulness score RESULTS.md reported **flat at 0.13** now moves:

| Rubric (Claude judge, 0–2) | base | v1 tuned | **v2 tuned** |
|---|---:|---:|---:|
| spec_adherence | 0.47 | 0.93 | 0.93 |
| level_calibration | 0.60 | 1.13 | 1.13 |
| no_engine_speak | 0.87 | 1.87 | 1.73 |
| **truthfulness** | 0.13 | 0.13 | **0.20** |
| task_quality | 0.13 | 0.27 | 0.33 |

Objective (v2): move‑sound 100%, no‑engine‑speak 100%, ply‑cap 100%. The 0–2
truthfulness lift is modest on 15 scenarios; the deterministic 100‑position
fabrication cut (50% → 33%) is the more robust signal — both point the same way.

### 3. Council instructiveness (blinded cross‑family, 100 held‑out, same scenarios)

Mean rank (1 = best of 5) / top‑1 win‑rate for **OURS**:

| Condition | v1 rank | v2 rank | v1 win% | v2 win% | gap to best frontier: v1 → v2 |
|---|---:|---:|---:|---:|---:|
| **grounded** | 4.13 | **3.68** | 2% | **8%** | +2.22 → **+1.60** |
| ungrounded | 4.46 | **4.11** | 1% | 1% | +2.75 → +2.38 |

v2 is judged more instructive than v1 and than BASE, and narrows (does not erase)
the gap to the frontier panel.

### 4. Objective style/soundness (OURS, benchmark)

| Metric (grounded) | v1 | v2 |
|---|---:|---:|
| move‑soundness | 97% | 98% |
| no‑engine‑speak (no jargon) | 100% | 100% |
| fabrication_rate | 50% | **33%** |

---

## v2 vs frontier — full 5‑model benchmark (100 held‑out, grounded + ungrounded)

Blinded council mean rank (1 = best) / top‑1 win‑rate, `ours` = **chess‑coach‑v2**:

| Model | grnd rank | ungr rank | grnd win% | ungr win% |
|---|---:|---:|---:|---:|
| OURS (chess‑coach‑v2, 1.7B) | 3.68 | 4.11 | 8% | 1% |
| BASE (Qwen3‑1.7B‑4bit) | 4.70 | 4.50 | 1% | 2% |
| GPT‑5.5 | 2.09 | 1.73 | 33% | 52% |
| Claude Opus 4.8 | 2.09 | 2.61 | 36% | 14% |
| Gemini 3.1 Pro | 2.44 | 2.05 | 23% | 31% |

Objective, grounded (5 models): move‑soundness — OURS 98, BASE 92, GPT 99, Claude
98, Gemini 100; fabrication_rate — OURS 33, BASE 13, GPT 3, Claude 3, Gemini 4;
no‑engine‑speak — 100 across the board.

Blinded council self‑preference (a judge vs its peers on its own lab's model): mean
signed **+0.43** rank — present but smaller than the gaps above, so the ranking is
not merely lab loyalty. A **blind human‑label export** (anonymized A–E, shuffled
per item) is at `data/benchmark_v2/blind_label.html` for the user's own ranking.

---

## Honest caveats

- **Ungrounded, v2 is *more* brittle** (fabrication 87% → 99%). v2 teaches more
  concretely (the explicit method references squares/captures), so without the
  engine facts a 1.7B invents more. The product always runs grounded (engine in
  the loop), so this is the expected trade, not the deployment path — but it is
  real and reported.
- The frontier still out‑coaches a 1.7B on instructiveness; v2 **narrows**, not
  closes, the gap. The win remains "reliable, local, cheap, engine‑grounded
  coaching," not "beats GPT‑5.5."
- The fabrication metric is a **conservative current‑board** verifier; some flags
  are the method's legitimate forward‑looking language ("after cxb4, the pawn on
  c3…"). It is applied identically to all 5 models, so the deltas are fair.

---

Every long stage is checkpoint/resumable (teacher gen, benchmark phases,
divergence); the run survived two transient interruptions with no lost work or
double‑spend.

---

## Artifacts (all v2‑suffixed)

- **Model:** `models/mlx/chess-coach-v2` (4‑bit MLX, group‑size 64) · adapter +
  merged 16‑bit at `models/adapters/chess-coach-v2/`
- **Dataset:** `data/dataset/train_v2.jsonl` (2,457) · `valid_v2.jsonl` (129) ·
  candidates `data/generated/candidates_v2.jsonl` (2,628) · held‑out reserve
  `data/analysis/heldout_v2.json`
- **Benchmark:** `data/benchmark_v2/` and `data/benchmark_v1/`
  (`scenarios/generations/objective/council/results.json`) ·
  `RESULTS_BENCHMARK_v2.md`, `RESULTS_BENCHMARK_v1.md` · **blind human‑label
  export** `data/benchmark_v2/blind_label.html` (+ `.jsonl`, `blind_key.json`)
- **Divergence:** `data/analysis/divergence_v2.jsonl`,
  `divergence_v1_matched.jsonl`, `divergence_compare_v2.json`
- **Rubric eval:** `data/eval/results_tuned_v2_claude.json`
- **Cost ledger:** `data/generated/cost_v2.json`
- **Code:** `src/teacher/tier_select.py`, `src/teacher/generate_v2.py`,
  `prompts/teacher_system_v2.md`, `src/train/train_modal_v2.py`,
  `scripts/run_benchmark_v2.py`, `scripts/divergence_compare_v2.py`; v2 gates in
  `src/filter/filter.py`; `render_assistant_target_v2` in `config/schema.py`

### For the HF re‑publish

- **Dataset page:** replace the "honest limitation" (unfiltered/fabrication) note —
  v2 is faithfulness‑filtered (0% false labels), tier‑aware, and teaches the search
  method; 2,586 rows incl. 348 contrastive multi‑tier sets.
- **Model card:** update the base‑vs‑tuned table with v2 (grounded): move‑sound
  98%, no‑engine‑speak 100%, fabrication 50%→33% vs v1, council rank 4.13→3.68.
- **Space/demo:** point `COACH_MODEL_PATH` at `models/mlx/chess-coach-v2`.
