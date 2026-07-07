# DEFINITIVE Chess-Coach Eval — 803 gap positions, all 14 models

The airtight, held-out evaluation of **tier-appropriate move selection** (the moat) and every other axis we optimize for, on the curated **803-position** gap set (`data/eval/gap_positions.jsonl` — 100% discriminating, per-tier Stockfish sound pool + Maia likelihoods + the identified tier-appropriate move, **zero leakage** vs train/valid). Every model coaches the SAME positions at all 3 tiers with byte-identical grounding (`render_pool_facts` + `render_user_prompt` + the tier's Maia block).

## TL;DR

- **Best open model on the balanced score (tier-selection + instructiveness weighted highest): GLM-5** — **NOT Gemma-3-27B** (it is GLM-5).
- **Best open v3 *base* (re-weighted for what's hard to ADD — instructiveness/capacity, faithfulness, local-runnability — since tier-appropriateness is what we fine-tune IN): Qwen3-32B.**
- The balanced winner (GLM-5) and the best-base pick (Qwen3-32B) **differ** — the balanced score rewards raw tier-selection + coaching that a huge model has, but a v3 base must be fine-tunable and locally runnable, which favors Qwen3-32B.
- Tier-appropriate move selection is **weak across the whole field** (it is the trained behavior, not an emergent one) — see the tier table; this is exactly the gap v3 targets.

## Method & cost-smart scope

- **Deterministic metrics** (tier-fit, tier-differentiation, direction, move-safety, no-engine-speak, fabrication) computed on **ALL 803 positions x 3 tiers** for the 2 local models (free) + 9 open models. The **3 frontier references** are measured on a **balanced 150-position stratified subset x 3 tiers** — measuring Claude Opus 4.8 on all 803 would add ~$55 for a *reference* row whose behavior is already established; a stratified subset gives a tight estimate (this mirrors the council-subset rationale).
- **Tier-appropriate move (the moat):** each coach's recommended move is re-extracted with the instrumented, pool-restricted extractor and compared to the canonical tier move from `src/teacher/tier_select.select_tier_move` (beginner=most human-findable sound move, intermediate=eval/Maia blend, advanced=sharpest=engine best). `tier-fit` = pick == that canonical move (mean over the 3 tiers).
- **Instructiveness:** one blinded cross-family council (3 frontier judges: GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro) ranks the unified **14-model** field per item on a **stratified ~120-item** subset (balanced across tier x phase). Council on 803x14 is expensive + statistically unnecessary for a rank estimate.
- **Gates:** move-safety (no blunders) and no-engine-speak are pass/fail floors. **Fabrication is reported but down-weighted** (the project's non-LLM faithfulness verifier neutralizes it at serve time — it is table-stakes, not a differentiator).

## 1. Per-metric leaderboard (all 14 models)

Sorted by the balanced score (below). `tier-fit` is the moat metric; instructiveness is council mean rank (lower = better, of 14).

| # | Model | family | tier-fit↑ | tier-diff↑ | direction↑ | instr rank↓ (top1) | safety↑ | no-jargon↑ | fab↓ | local | n(det) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|
| 1 | GPT-5.5 | frontier | 43% | 31% | 46% | 3.21 (32%) | 99% | 100% | 3% | no | 450 |
| 2 | Gemini 3.1 Pro | frontier | 48% | 28% | 50% | 4.96 (16%) | 99% | 100% | 4% | no | 450 |
| 3 | Claude Opus 4.8 | frontier | 46% | 31% | 50% | 4.28 (24%) | 98% | 100% | 5% | no | 450 |
| 4 | GLM-5 | open | 45% | 37% | 52% | 6.10 (6%) | 100% | 100% | 7% | no | 2409 |
| 5 | Qwen3-32B | open | 37% | 45% | 48% | 8.58 (1%) | 100% | 99% | 6% | yes | 2409 |
| 6 | Kimi-K2.5 | open | 36% | 49% | 48% | 6.73 (5%) | 100% | 99% | 8% | no | 2409 |
| 7 | DeepSeek-R1 | open | 44% | 39% | 50% | 7.47 (1%) | 100% | 100% | 2% | no | 2409 |
| 8 | OURS-v2 (Qwen3-1.7B tuned) | ours | 53% | 44% | 54% | 9.36 (9%) | 99% | 100% | 30% | yes | 2409 |
| 9 | Llama-3.3-70B | open | 40% | 27% | 48% | 8.02 (1%) | 100% | 100% | 0% | tight | 2409 |
| 10 | Gemma-3-27B-it | open | 35% | 23% | 48% | 8.69 (1%) | 100% | 100% | 2% | yes | 2409 |
| 11 | DeepSeek-V3.2 | open | 41% | 41% | 50% | 7.90 (1%) | 100% | 100% | 5% | no | 2409 |
| 12 | Qwen3-Next-80B-A3B | open | 32% | 25% | 51% | 8.04 (0%) | 100% | 100% | 7% | tight | 2409 |
| 13 | Mistral-Large-3 (675B) | open | 37% | 44% | 49% | 8.30 (3%) | 100% | 100% | 7% | no | 2409 |
| 14 | BASE (Qwen3-1.7B untuned) | base | 36% | 50% | 46% | 13.38 (0%) | 96% | 96% | 15% | yes | 2409 |

- **tier-fit** = share of (position,tier) where the coach's pick equals the canonical `select_tier_move` move. **tier-diff** = share of positions where the pick changes across the 3 tiers. **direction** = share where the beginner pick is at least as human-findable (Maia rank) as the advanced pick (correct level gradient).
- **safety** = share of picks that are not blunders (cp-loss < 250). **no-jargon** = no centipawn/engine-speak leaked. **fab** = share of outputs with >=1 false board fact (down-weighted). **n(det)** = deterministic positions x tiers scored (frontier on the 150-subset).

## 2. Tier-appropriate move selection (the moat), per tier

Per-tier `tier-fit` (pick == `select_tier_move` canonical) + engine-mirror rate. A strong leveled coach has HIGH beginner/intermediate fit (finding the *human* move) while advanced fit ~ engine-mirror (the sharp move is correct there).

| Model | fit B | fit I | fit A | mirror B | mirror I | mirror A | diff | mirror@all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.5 | 31% | 45% | 54% | 52% | 51% | 57% | 31% | 43% |
| Gemini 3.1 Pro | 29% | 47% | 69% | 69% | 70% | 74% | 28% | 60% |
| Claude Opus 4.8 | 30% | 46% | 61% | 59% | 60% | 65% | 31% | 47% |
| GLM-5 | 34% | 43% | 57% | 53% | 57% | 59% | 37% | 42% |
| Qwen3-32B | 34% | 38% | 38% | 45% | 41% | 39% | 45% | 28% |
| Kimi-K2.5 | 34% | 36% | 38% | 34% | 36% | 38% | 49% | 22% |
| DeepSeek-R1 | 38% | 42% | 52% | 43% | 47% | 52% | 39% | 34% |
| OURS-v2 (Qwen3-1.7B tuned) | 48% | 50% | 61% | 65% | 59% | 63% | 44% | 42% |
| Llama-3.3-70B | 35% | 39% | 44% | 40% | 43% | 45% | 27% | 35% |
| Gemma-3-27B-it | 28% | 34% | 42% | 43% | 42% | 43% | 23% | 36% |
| DeepSeek-V3.2 | 33% | 38% | 52% | 47% | 50% | 52% | 41% | 37% |
| Qwen3-Next-80B-A3B | 33% | 32% | 30% | 30% | 31% | 30% | 25% | 24% |
| Mistral-Large-3 (675B) | 37% | 37% | 36% | 35% | 37% | 37% | 44% | 24% |
| BASE (Qwen3-1.7B untuned) | 31% | 39% | 40% | 51% | 43% | 41% | 50% | 28% |

## 3. Weighted BALANCED ranking

Transparent weighted score (each component normalized to 0-1, higher = better): **tier-appropriate move selection 40%** + **instructiveness 40%** + fabrication (1-fab) 10% + practical (local+cost) 10%. Safety + no-jargon are pass/fail gates. Score = weighted mean x 100.

| # | Model | family | tier(.40) | instr(.40) | 1-fab(.10) | practical(.10) | **balanced** | gate |
|---|---|---|---:|---:|---:|---:|---:|:--:|
| 1 | GPT-5.5 | frontier | 39.9 | 83.0 | 96.7 | 35.0 | **62.3** | pass |
| 2 | Gemini 3.1 Pro | frontier | 42.3 | 69.6 | 95.8 | 35.0 | **57.8** | pass |
| 3 | Claude Opus 4.8 | frontier | 42.3 | 74.8 | 95.3 | 0.0 | **56.4** | pass |
| 4 | GLM-5 | open | 44.4 | 60.8 | 92.7 | 39.1 | **55.3** | pass |
| 5 | Qwen3-32B | open | 43.3 | 41.7 | 93.9 | 99.7 | **53.4** | pass |
| 6 | Kimi-K2.5 | open | 44.4 | 55.9 | 91.7 | 38.6 | **53.2** | pass |
| 7 | DeepSeek-R1 | open | 44.5 | 50.3 | 98.0 | 37.0 | **51.4** | pass |
| 8 | OURS-v2 (Qwen3-1.7B tuned) | ours | 50.3 | 35.7 | 69.8 | 100.0 | **51.4** | pass |
| 9 | Llama-3.3-70B | open | 37.9 | 46.0 | 99.7 | 75.4 | **51.1** | pass |
| 10 | Gemma-3-27B-it | open | 35.1 | 40.9 | 97.5 | 99.8 | **50.1** | pass |
| 11 | DeepSeek-V3.2 | open | 43.8 | 46.9 | 94.9 | 39.6 | **49.7** | pass |
| 12 | Qwen3-Next-80B-A3B | open | 35.9 | 45.8 | 92.9 | 75.6 | **49.5** | pass |
| 13 | Mistral-Large-3 (675B) | open | 43.3 | 43.8 | 92.7 | 36.4 | **47.8** | pass |
| 14 | BASE (Qwen3-1.7B untuned) | base | 44.2 | 4.8 | 85.5 | 100.0 | **38.1** | **FAIL** |

## 4. Best v3-BASE ranking (re-weighted)

For a fine-tuning *base*, tier-appropriateness is what we ADD, so it is down-weighted; the hard-to-add qualities dominate: **instructiveness/capacity 35%**, **faithfulness 20%**, **local-runnability+cost 35%**, tier 10%. Only locally fine-tunable/runnable models are viable bases.

| # | Model | family | base-fit | local | note |
|---|---|---|---:|:--:|---|
| 1 | Qwen3-32B | open | 72.6 | yes | 4-bit ~18GB — comfortable |
| 2 | Gemma-3-27B-it | open | 72.2 | yes | 4-bit ~15GB — comfortable |
| 3 | OURS-v2 (Qwen3-1.7B tuned) | ours | 66.5 | yes | already local |
| 4 | Llama-3.3-70B | open | 66.2 | tight | 4-bit ~40GB — tight on 64GB |
| 5 | Qwen3-Next-80B-A3B | open | 64.7 | tight | 4-bit ~43GB — tight but fits; 3B active = fast |
| 6 | GPT-5.5 | frontier | 64.6 | no | API-only |
| 7 | Gemini 3.1 Pro | frontier | 60.0 | no | API-only |
| 8 | BASE (Qwen3-1.7B untuned) | base | 58.2 | yes | already local |
| 9 | GLM-5 | open | 57.9 | no | far exceeds 64GB |
| 10 | Kimi-K2.5 | open | 55.9 | no | far exceeds 64GB |
| 11 | DeepSeek-R1 | open | 54.6 | no | far exceeds 64GB |
| 12 | DeepSeek-V3.2 | open | 53.7 | no | far exceeds 64GB |
| 13 | Mistral-Large-3 (675B) | open | 51.0 | no | far exceeds 64GB |
| 14 | Claude Opus 4.8 | frontier | 49.5 | no | API-only |

## 5. Recommendation

- **Best overall (balanced), any provider: GPT-5.5** (frontier) — the frontier still coaches best; it is the distillation-teacher benchmark, not a deployable base.
- **Best OPEN model (balanced): GLM-5.** This is **NOT** Gemma-3-27B — GLM-5 is the strongest open coach (best open instructiveness) with solid tier-selection, but it is far too large to run locally.
- **Best open v3 base: Qwen3-32B** — the best mix of coaching capacity, faithfulness, and 4-bit local fine-tunability/runnability on a 64GB Mac.
- **It is effectively a tie with Gemma-3-27B-it** (base-fit 72.6 vs 72.2 — within noise). Qwen3-32B edges it on tier-selection/capacity; Gemma-3-27B-it is smaller and more faithful (fab 2% vs 6%). Either is a defensible v3 base; prefer Gemma-3-27B-it if faithfulness/size is paramount, Qwen3-32B if raw capacity is.
- **The balanced winner and the base pick differ:** GLM-5 wins the raw open balanced score, but it far exceeds 64GB — not a viable local base. Qwen3-32B is the pragmatic v3 base because tier-appropriateness (where the giant coaches lead) is exactly what we fine-tune IN, while capacity + faithfulness + local-runnability are what a base must bring.

## 6. Cost

| group | calls | in tok | out tok | est. USD |
|---|---:|---:|---:|---:|
| open | 21,681 | 21,910,715 | 4,266,729 | $28.17 |
| frontier_gen | 1,350 | 1,531,055 | 495,278 | $21.71 |
| council | 360 | 1,085,518 | 346,804 | $15.57 |
| local | 4,818 | 0 | 0 | $0.00 |
| **TOTAL** | | | | **$65.45** |

_Open-model + frontier prices are best-effort Bedrock/gateway estimates; local (OURS-v2, BASE) is free. Total definitive-eval spend: **$65.45**._

## Artifacts

- Positions: `data/eval/gap_positions.jsonl` (803, curated, zero-leakage)
- Flattened scenarios: `data/benchmark_gap803/scenarios.jsonl` (803x3)
- Generations: `data/benchmark_gap803/gen/<model>.jsonl` -> `generations.jsonl`
- Objective: `data/benchmark_gap803/objective.jsonl`; safety: `move_safety.json`
- Council: `data/benchmark_gap803/council.jsonl`; leaderboard: `leaderboard.json`
- Drivers: `scripts/gap803_{gen,report,safety,council}.py`, `scripts/gap803_common.py`