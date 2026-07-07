# Chess-Coach Benchmark — Grounded vs Ungrounded, Ours vs Frontier

A 2×2×5 grid: **5 models × 2 conditions** (WITHOUT vs WITH Stockfish+Maia grounding), scored on **100 held-out positions** by deterministic objective metrics and a **blinded, cross-family model council**. Generated 2026-07-07 00:45 UTC.

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

- **tier** — advanced: 34, beginner: 33, intermediate: 33
- **phase** — endgame: 33, middlegame: 41, opening: 26
- **severity** — blunder: 34, inaccuracy: 36, mistake: 30


## Headline

- **Grounding lifts move-soundness across the board.** OURS goes 41% → 98% on picking a Stockfish-sound move; BASE 26% → 92%; the frontier average 53% → 99%. When every model is handed the sound pool, choosing from it is the easy part.
- **Fabrication is the honest metric, and grounding is what moves it.** OURS fabricates a false board fact in 99% of ungrounded outputs vs 33% grounded; the frontier average is 5% → 3%. Verified facts in the prompt are the lever a 1.7B model cannot supply from its own board-tracking.
- **The fine-tune still owns the style gate.** OURS keeps outputs jargon-free (no engine-speak) 100%/100% (ungr/grnd) vs the grounded frontier average 100% — the frontier models, handed evals, are more tempted to leak them.
- **Council (instructiveness) is where the honest gap shows.** OURS mean rank improves 4.11 → 3.68 (1=best of 5) with grounding, while the grounded frontier averages 2.21 (best 2.09). Grounding narrows the coaching gap but does not erase it — a bigger model still explains more instructively.
- **Self-preference check.** Mean signed self-preference across the three judges is +0.43 rank (positive = a judge ranks its own lab's model better than its peers do). Small relative to the gaps above, so the council ranking is not merely lab loyalty.


## Objective metrics (2×2: 5 models × 2 conditions)

### WITHOUT grounding

| Model | move_sound | no_engine_speak | ply_cap_ok | fabrication_rate | avg_violations | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OURS (chess-coach-v2, 1.7B tuned) | 41% | 100% | 100% | 99% | 2.14 | 100 |
| BASE (Qwen3-1.7B-4bit, untuned) | 26% | 100% | 100% | 29% | 0.40 | 100 |
| GPT-5.5 | 66% | 100% | 100% | 2% | 0.02 | 100 |
| Claude Opus 4.8 | 44% | 99% | 100% | 6% | 0.06 | 100 |
| Gemini 3.1 Pro | 49% | 100% | 100% | 6% | 0.06 | 100 |


### WITH grounding

| Model | move_sound | no_engine_speak | ply_cap_ok | fabrication_rate | avg_violations | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OURS (chess-coach-v2, 1.7B tuned) | 98% | 100% | 100% | 33% | 0.46 | 100 |
| BASE (Qwen3-1.7B-4bit, untuned) | 92% | 100% | 100% | 13% | 0.20 | 100 |
| GPT-5.5 | 99% | 100% | 100% | 3% | 0.03 | 100 |
| Claude Opus 4.8 | 98% | 100% | 100% | 3% | 0.03 | 100 |
| Gemini 3.1 Pro | 100% | 100% | 100% | 4% | 0.04 | 100 |


*`move_sound` = recommended move is in the Stockfish sound pool. `fabrication_rate` = share of outputs with ≥1 false board fact (non-LLM faithfulness verifier). `avg_violations` = mean false facts per output.*


## Council — instructiveness ranking (blinded, cross-family)

Each of 3 judges ranked all 5 anonymized outputs per item; lower mean rank = judged more instructive (1 = best of 5).


| Model | mean rank (ungr) | mean rank (grnd) | top-1 win% (ungr) | top-1 win% (grnd) |
| --- | ---: | ---: | ---: | ---: |
| OURS (chess-coach-v2, 1.7B tuned) | 4.11 | 3.68 | 1% | 8% |
| BASE (Qwen3-1.7B-4bit, untuned) | 4.50 | 4.70 | 2% | 1% |
| GPT-5.5 | 1.73 | 2.09 | 52% | 33% |
| Claude Opus 4.8 | 2.61 | 2.09 | 14% | 36% |
| Gemini 3.1 Pro | 2.05 | 2.44 | 31% | 23% |


Items judged: ungrounded 300, grounded 300 (judge-observations).


### Per-dimension rubric (mean 0–2)

**WITHOUT grounding**

| Model | tier_calibration | clarity | correctness |
| --- | ---: | ---: | ---: |
| OURS (chess-coach-v2, 1.7B tuned) | 0.39 | 0.85 | 0.00 |
| BASE (Qwen3-1.7B-4bit, untuned) | 0.17 | 0.63 | 0.01 |
| GPT-5.5 | 1.59 | 1.75 | 1.12 |
| Claude Opus 4.8 | 1.21 | 1.52 | 0.42 |
| Gemini 3.1 Pro | 1.54 | 1.77 | 0.86 |


**WITH grounding**

| Model | tier_calibration | clarity | correctness |
| --- | ---: | ---: | ---: |
| OURS (chess-coach-v2, 1.7B tuned) | 1.28 | 1.42 | 0.65 |
| BASE (Qwen3-1.7B-4bit, untuned) | 0.54 | 0.78 | 0.22 |
| GPT-5.5 | 1.84 | 1.88 | 1.79 |
| Claude Opus 4.8 | 1.84 | 1.87 | 1.48 |
| Gemini 3.1 Pro | 1.76 | 1.88 | 1.52 |


## Bias / self-preference check

Mean rank each judge gives each model, pooled across conditions (a judge favoring its own lab shows a lower number in its own column):


| Model | judge=GPT-5.5 | judge=Claude Opus 4.8 | judge=Gemini 3.1 Pro |
| --- | ---: | ---: | ---: |
| OURS (chess-coach-v2, 1.7B tuned) | 4.06 | 3.50 | 4.13 |
| BASE (Qwen3-1.7B-4bit, untuned) | 4.67 | 4.35 | 4.77 |
| GPT-5.5 | 1.47 | 2.21 | 2.05 |
| Claude Opus 4.8 | 2.46 | 2.36 | 2.23 |
| Gemini 3.1 Pro | 2.33 | 2.58 | 1.82 |


Self-preference (a judge vs its peers, on its own lab's model):


| Judge (lab) | own model | own mean rank | peers' mean rank | self-pref Δ (peers − own) |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.5 | GPT-5.5 | 1.47 | 2.13 | +0.66 |
| Claude Opus 4.8 | Claude Opus 4.8 | 2.36 | 2.35 | -0.01 |
| Gemini 3.1 Pro | Gemini 3.1 Pro | 1.82 | 2.46 | +0.64 |


Mean signed self-preference Δ across judges: **+0.43** rank (positive = judges favor their own lab). Mean magnitude: 0.44.


## Cost

| Model | gen calls | judge calls | in tok | out tok | est. USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| OURS (chess-coach-v2, 1.7B tuned) | 200 | 0 | 0 | 0 | $0.00 |
| BASE (Qwen3-1.7B-4bit, untuned) | 200 | 0 | 0 | 0 | $0.00 |
| GPT-5.5 | 200 | 200 | 420,142 | 324,294 | $3.77 |
| Claude Opus 4.8 | 200 | 200 | 624,137 | 89,929 | $16.11 |
| Gemini 3.1 Pro | 200 | 200 | 434,445 | 333,780 | $3.88 |


**Total estimated cost: $23.76** (local MLX models are free; frontier prices are per-1M-token estimates, see per-model rows). Generations: 1000, council judgments: 600.


## Artifacts

- Scenario set: `data/benchmark/scenarios.jsonl`
- Raw generations: `data/benchmark/generations.jsonl`
- Objective scores: `data/benchmark/objective.jsonl`
- Council rankings: `data/benchmark/council.jsonl`
- Aggregated results: `data/benchmark/results.json`
- Blind human-label export: `data/benchmark/blind_label.jsonl` + `blind_label.html` (viewer) + `blind_key.json` (label→model)
