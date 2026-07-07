# Chess-Coach Benchmark — Grounded vs Ungrounded, Ours vs Frontier

A 2×2×5 grid: **5 models × 2 conditions** (WITHOUT vs WITH Stockfish+Maia grounding), scored on **100 held-out positions** by deterministic objective metrics and a **blinded, cross-family model council**. Generated 2026-07-06 23:15 UTC.

## Setup

| Item | Value |
|---|---|
| Held-out positions | 100 (excluded any board present in `train.jsonl`/`valid.jsonl`, dedup by placement+turn) |
| Competitors | OURS `chess-coach-v1` (1.7B tuned) · BASE `Qwen3-1.7B-4bit` · GPT-5.5 · Claude Opus 4.8 · Gemini 3.1 Pro |
| Conditions | **ungrounded** (tier + position + student move only) vs **grounded** (verified facts + Stockfish sound pool + Maia) |
| Same system prompt + format instruction | Yes — identical for all 5 models, both conditions |
| Objective | move-soundness (Stockfish pool), no-engine-speak, ply-cap, **fabrication rate** (faithfulness verifier) |
| Council | GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro rank the 5 **blinded** outputs by *instructiveness for the tier* |


**Scenario balance** (game phase = material-first: endgame if ≤6 major/minor pieces, else opening if ≤ move 12, else middlegame):

- **tier** — advanced: 27, beginner: 40, intermediate: 33
- **phase** — endgame: 22, middlegame: 51, opening: 27
- **severity** — blunder: 28, inaccuracy: 41, mistake: 31


## Headline

- **Grounding lifts move-soundness across the board.** OURS goes 36% → 100% on picking a Stockfish-sound move; BASE 28% → 92%; the frontier average 50% → 100%. When every model is handed the sound pool, choosing from it is the easy part.
- **Fabrication is the honest metric, and grounding is what moves it.** OURS fabricates a false board fact in 93% of ungrounded outputs vs 38% grounded; the frontier average is 5% → 3%. Verified facts in the prompt are the lever a 1.7B model cannot supply from its own board-tracking.
- **The fine-tune still owns the style gate.** OURS keeps outputs jargon-free (no engine-speak) 100%/100% (ungr/grnd) vs the grounded frontier average 100% — the frontier models, handed evals, are more tempted to leak them.
- **Council (instructiveness) is where the honest gap shows.** OURS mean rank improves 4.44 → 4.19 (1=best of 5) with grounding, while the grounded frontier averages 2.10 (best 1.93). Grounding narrows the coaching gap but does not erase it — a bigger model still explains more instructively.
- **Self-preference check.** Mean signed self-preference across the three judges is +0.44 rank (positive = a judge ranks its own lab's model better than its peers do). Small relative to the gaps above, so the council ranking is not merely lab loyalty.


## Objective metrics (2×2: 5 models × 2 conditions)

### WITHOUT grounding

| Model | move_sound | no_engine_speak | ply_cap_ok | fabrication_rate | avg_violations | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OURS (chess-coach-v1, 1.7B tuned) | 36% | 100% | 100% | 93% | 1.62 | 100 |
| BASE (Qwen3-1.7B-4bit, untuned) | 28% | 100% | 100% | 21% | 0.29 | 100 |
| GPT-5.5 | 56% | 100% | 100% | 2% | 0.02 | 100 |
| Claude Opus 4.8 | 33% | 100% | 100% | 11% | 0.11 | 100 |
| Gemini 3.1 Pro | 60% | 100% | 100% | 3% | 0.03 | 100 |


### WITH grounding

| Model | move_sound | no_engine_speak | ply_cap_ok | fabrication_rate | avg_violations | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OURS (chess-coach-v1, 1.7B tuned) | 100% | 100% | 100% | 38% | 0.45 | 100 |
| BASE (Qwen3-1.7B-4bit, untuned) | 92% | 95% | 100% | 15% | 0.23 | 100 |
| GPT-5.5 | 100% | 100% | 100% | 2% | 0.02 | 100 |
| Claude Opus 4.8 | 100% | 100% | 100% | 6% | 0.06 | 100 |
| Gemini 3.1 Pro | 100% | 100% | 100% | 2% | 0.02 | 100 |


*`move_sound` = recommended move is in the Stockfish sound pool. `fabrication_rate` = share of outputs with ≥1 false board fact (non-LLM faithfulness verifier). `avg_violations` = mean false facts per output.*


## Council — instructiveness ranking (blinded, cross-family)

Each of 3 judges ranked all 5 anonymized outputs per item; lower mean rank = judged more instructive (1 = best of 5).


| Model | mean rank (ungr) | mean rank (grnd) | top-1 win% (ungr) | top-1 win% (grnd) |
| --- | ---: | ---: | ---: | ---: |
| OURS (chess-coach-v1, 1.7B tuned) | 4.44 | 4.19 | 1% | 2% |
| BASE (Qwen3-1.7B-4bit, untuned) | 4.24 | 4.52 | 1% | 1% |
| GPT-5.5 | 1.72 | 2.08 | 50% | 34% |
| Claude Opus 4.8 | 2.71 | 1.93 | 10% | 39% |
| Gemini 3.1 Pro | 1.90 | 2.29 | 39% | 24% |


Items judged: ungrounded 300, grounded 300 (judge-observations).


### Per-dimension rubric (mean 0–2)

**WITHOUT grounding**

| Model | tier_calibration | clarity | correctness |
| --- | ---: | ---: | ---: |
| OURS (chess-coach-v1, 1.7B tuned) | 0.18 | 0.48 | 0.02 |
| BASE (Qwen3-1.7B-4bit, untuned) | 0.27 | 0.63 | 0.01 |
| GPT-5.5 | 1.66 | 1.83 | 1.13 |
| Claude Opus 4.8 | 1.27 | 1.63 | 0.46 |
| Gemini 3.1 Pro | 1.64 | 1.85 | 1.01 |


**WITH grounding**

| Model | tier_calibration | clarity | correctness |
| --- | ---: | ---: | ---: |
| OURS (chess-coach-v1, 1.7B tuned) | 0.92 | 1.06 | 0.28 |
| BASE (Qwen3-1.7B-4bit, untuned) | 0.56 | 0.74 | 0.23 |
| GPT-5.5 | 1.90 | 1.92 | 1.80 |
| Claude Opus 4.8 | 1.88 | 1.90 | 1.57 |
| Gemini 3.1 Pro | 1.84 | 1.90 | 1.54 |


## Bias / self-preference check

Mean rank each judge gives each model, pooled across conditions (a judge favoring its own lab shows a lower number in its own column):


| Model | judge=GPT-5.5 | judge=Claude Opus 4.8 | judge=Gemini 3.1 Pro |
| --- | ---: | ---: | ---: |
| OURS (chess-coach-v1, 1.7B tuned) | 4.52 | 3.94 | 4.47 |
| BASE (Qwen3-1.7B-4bit, untuned) | 4.34 | 4.33 | 4.46 |
| GPT-5.5 | 1.52 | 2.04 | 2.12 |
| Claude Opus 4.8 | 2.48 | 2.23 | 2.25 |
| Gemini 3.1 Pro | 2.15 | 2.46 | 1.69 |


Self-preference (a judge vs its peers, on its own lab's model):


| Judge (lab) | own model | own mean rank | peers' mean rank | self-pref Δ (peers − own) |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.5 | GPT-5.5 | 1.52 | 2.08 | +0.56 |
| Claude Opus 4.8 | Claude Opus 4.8 | 2.23 | 2.37 | +0.14 |
| Gemini 3.1 Pro | Gemini 3.1 Pro | 1.69 | 2.30 | +0.61 |


Mean signed self-preference Δ across judges: **+0.44** rank (positive = judges favor their own lab). Mean magnitude: 0.44.


## Cost

| Model | gen calls | judge calls | in tok | out tok | est. USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| OURS (chess-coach-v1, 1.7B tuned) | 200 | 0 | 0 | 0 | $0.00 |
| BASE (Qwen3-1.7B-4bit, untuned) | 200 | 0 | 0 | 0 | $0.00 |
| GPT-5.5 | 200 | 200 | 418,548 | 328,067 | $3.80 |
| Claude Opus 4.8 | 200 | 200 | 622,845 | 90,375 | $16.12 |
| Gemini 3.1 Pro | 200 | 200 | 432,791 | 349,384 | $4.03 |


**Total estimated cost: $23.96** (local MLX models are free; frontier prices are per-1M-token estimates, see per-model rows). Generations: 1000, council judgments: 600.


## Artifacts

- Scenario set: `data/benchmark/scenarios.jsonl`
- Raw generations: `data/benchmark/generations.jsonl`
- Objective scores: `data/benchmark/objective.jsonl`
- Council rankings: `data/benchmark/council.jsonl`
- Aggregated results: `data/benchmark/results.json`
- Blind human-label export: `data/benchmark/blind_label.jsonl` + `blind_label.html` (viewer) + `blind_key.json` (label→model)
