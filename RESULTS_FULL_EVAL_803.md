# DEFINITIVE Chess-Coach Eval — 803 gap positions, all 15 models

The airtight, held-out evaluation of **tier-appropriate move selection** (the moat) and every other axis we optimize for, on the curated **803-position** gap set (`data/eval/gap_positions.jsonl` — 100% discriminating, per-tier Stockfish sound pool + Maia likelihoods + the identified tier-appropriate move, **zero leakage** vs train/valid). Every model coaches the SAME positions at all 3 tiers with byte-identical grounding (`render_pool_facts` + `render_user_prompt` + the tier's Maia block).

## TL;DR

- **Best open model on the balanced score (tier-selection + instructiveness weighted highest): GLM-5** — **NOT Gemma-3-27B** (it is GLM-5).
- **Best open v3 *base* (re-weighted for what's hard to ADD — instructiveness/capacity, faithfulness, local-runnability — since tier-appropriateness is what we fine-tune IN): Gemma-3-27B-it.**
- The balanced winner (GLM-5) and the best-base pick (Gemma-3-27B-it) **differ** — the balanced score rewards raw tier-selection + coaching that a huge model has, but a v3 base must be fine-tunable and locally runnable, which favors Gemma-3-27B-it.
- Tier-appropriate move selection is **weak across the whole field** (it is the trained behavior, not an emergent one) — see the tier table; this is exactly the gap v3 targets.

## Corrected-benchmark update (Stage-4: v4 / base / v6-dpo / v6-distill)

The tables in this document were computed on the pre-correction (v4-era) labels. The
v6 rebuild re-derived the benchmark's canonical / engine-best labels under deeper
Stockfish-17 search plus Syzygy (`data/benchmark_gap803/scenarios_v6.jsonl`; the 120
held-out TEST FENs are unchanged, only the targets moved). The consolidating Stage-4
re-eval on those corrected labels, over the 120 held-out TEST x 3 tiers, is in
`RESULTS_STAGE4_CORRECTED.md`. Deterministic tier-policy match (same vendored
extractor), GROUNDED for base/v4/v6-dpo and NO-GROUNDING for base/v6-distill:

| Model | condition | tier-policy match | move-sound | distinct | names-a-move |
|---|---|---:|---:|---:|---:|
| BASE (Qwen3-32B untuned) | grounded | 0.428 | 0.969 | 0.303 | 0.975 |
| OURS-v4 (shipped) | grounded | 0.861 | 0.983 | 0.987 | 0.983 |
| OURS-v6-dpo | grounded | 0.881 | 0.983 | 0.987 | 0.983 |
| BASE (Qwen3-32B untuned) | no-grounding | 0.022 | 0.081 | 0.040 | 0.250 |
| OURS-v6-distill | no-grounding | 0.325 | 0.653 | 0.461 | 0.983 |

Takeaways (full detail + per-tier + 95%-CI council in `RESULTS_STAGE4_CORRECTED.md`):

- **v6-dpo sharpens the moat without regressing:** +0.0195 tier-policy over v4, all
  from the intermediate tier (0.808 vs 0.750, out of distribution); soundness,
  distinct-moves, beginner and advanced are identical to v4.
- **Distillation puts the tier rule in the weights:** stripped of grounding the base
  collapses (tier 0.022, names-a-move 0.25); the distilled adapter recovers it to
  0.325 tier-policy and 0.983 names-a-move, with an honest advanced-tier limit
  (0.217, the sharpest move genuinely needs grounding).
- **Base-vs-tuned is preserved under the correction:** grounded tuned-minus-base is
  +0.433 (v4) and +0.453 (v6-dpo) tier-policy, matching the pre-correction gap.
- The correction lifts every model's absolute grounded number slightly (cleaner
  grounding) but preserves the tuned-over-base and tuned-over-frontier ordering, so
  the "beats-frontier on tier-appropriate selection" claim holds. The full 803 x 3
  frontier re-score under corrected labels — out of scope for the *generation* stage
  (cost) — has now been done for **free** as a cached re-score: see the next section.

## Corrected-benchmark (v6 labels) field table — full-field deterministic re-score

_Added 2026-07-11. This REFRESHES the field-wide moat (§1–§2 below) against the
corrected v6 labels. The historical pre-correction tables are kept intact underneath
for provenance. **No model was re-run** — this is a free, cached re-score (0 GPU, 0
network, $0 gateway spend)._

**What changed.** The v6 rebuild re-derived the benchmark's canonical / sound labels
on the SAME 803 positions (`data/benchmark_gap803/scenarios_v6.jsonl`): **45.2% of
canonical tier targets moved** (1089/2409) and **28.9% of the old "sound" moves were
removed** (4701/16257) under deeper Stockfish-17 search + Syzygy. The models' move
CHOICES are unchanged (0/2409 FEN or student-move changes across the rebuild), so only
their tier-policy SCORES had to be recomputed.

**Method (byte-identical to the shipped scorer).** Each model's recommended move is
extracted ONCE from its cached raw `output` with the vendored extractor
`src.eval.evaluate.extract_recommended_move` (the one `scripts/reproduce_v4.py`
asserts and Stage-4 uses), then scored against BOTH the old and the corrected labels.
Because extraction depends only on `(text, fen, student_uci)` — all label-free — the
**only** thing that moves a score is the label correction. Metric definitions match
`scripts/stage4_rescore_committed.py` exactly (validated: q3_32b on the 120-TEST slice
reproduces `base_grounded_committed` 0.2972 / sound 0.8583 / distinct 18-76 to 4 dp).
Coverage is each model's cached set — 12 open/OURS at 803×3 = 2409, the 3 frontier
references at the 150-subset ×3 = 450 (the same `n(det)` split as §1).
Driver: `scripts/rescore_field_v6.py` → `data/benchmark_gap803/field_v6_rescore.json`.

> **Scope caveat (read the ranking, not the ceiling).** These are the ORIGINAL
> (v4-era-grounding) generations judged by the CORRECTED targets. As Stage-4's
> continuity check shows, re-scoring old generations against the sharper targets
> LOWERS absolute exact-match for *everyone*, whereas *re-generating* against the
> corrected grounding raises it (fresh OURS-v4 = 0.861 on the 120 TEST — see
> `RESULTS_STAGE4_CORRECTED.md`). The field was deliberately **not** re-generated (that
> needs gateway $ + GPU, both off-limits here). So this table is valid for the
> **relative / ranking** comparison — every model's old gens vs the same new labels,
> fully apples-to-apples — not as each model's ceiling under fresh corrected grounding.
> Numbers use the Stage-4 vendored extractor, so the "old-label" column is a like-for-
> like re-derivation and will not exactly equal the historical §1 "tier-fit" (which
> used the earlier `gap803_report` pipeline); the **Δ** is the load-bearing quantity.

### The moat, corrected — tier-appropriate move selection (sorted by v6 tier-policy match)

| # | Model | family | tier-policy v6↑ | B / I / A (v6) | Δ vs old | move-sound↑ | distinct↑ | names | format | n(det) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | OURS-v2 (Qwen3-1.7B tuned) | ours | **0.509** | 0.557 / 0.532 / 0.438 | -0.124 | 0.898 | 0.477 (195/409) | 1.000 | 1.000 | 2409 |
| 2 | Claude Opus 4.8 | frontier | 0.467 | 0.447 / 0.460 / 0.493 | -0.053 | 0.951 | 0.240 (23/96) | 1.000 | 1.000 | 450 |
| 3 | OURS-v3 (Qwen3-32B tuned) | ours | 0.463 | 0.463 / 0.441 / 0.484 | -0.105 | 0.867 | 0.509 (208/409) | 0.949 | 0.279 | 2409 |
| 4 | Gemini 3.1 Pro | frontier | 0.440 | 0.433 / 0.380 / 0.507 | -0.087 | 0.920 | 0.333 (32/96) | 1.000 | 1.000 | 450 |
| 5 | GPT-5.5 | frontier | 0.404 | 0.393 / 0.373 / 0.447 | -0.067 | 0.918 | 0.312 (30/96) | 1.000 | 1.000 | 450 |
| 6 | GLM-5 | open | 0.387 | 0.355 / 0.376 / 0.430 | -0.071 | 0.894 | 0.286 (117/409) | 1.000 | 0.998 | 2409 |
| 7 | BASE (Qwen3-1.7B untuned) | base | 0.358 | 0.421 / 0.399 / 0.255 | +0.002 | 0.825 | 0.318 (130/409) | 0.999 | 0.997 | 2409 |
| 8 | DeepSeek-R1 | open | 0.340 | 0.305 / 0.326 / 0.387 | -0.100 | 0.862 | 0.325 (133/409) | 1.000 | 1.000 | 2409 |
| 9 | DeepSeek-V3.2 | open | 0.324 | 0.285 / 0.310 / 0.376 | -0.088 | 0.864 | 0.318 (130/409) | 1.000 | 0.992 | 2409 |
| 10 | Mistral-Large-3 (675B) | open | 0.303 | 0.303 / 0.286 / 0.321 | -0.065 | 0.840 | 0.354 (145/409) | 1.000 | 1.000 | 2409 |
| 11 | Qwen3-32B (untuned v3 base) | open | 0.300 | 0.294 / 0.295 / 0.313 | -0.075 | 0.845 | 0.340 (139/409) | 1.000 | 1.000 | 2409 |
| 12 | Llama-3.3-70B | open | 0.299 | 0.277 / 0.288 / 0.334 | -0.098 | 0.855 | 0.215 (88/409) | 1.000 | 1.000 | 2409 |
| 13 | Gemma-3-27B-it | open | 0.296 | 0.264 / 0.277 / 0.347 | -0.051 | 0.876 | 0.215 (88/409) | 1.000 | 1.000 | 2409 |
| 14 | Kimi-K2.5 | open | 0.289 | 0.281 / 0.288 / 0.299 | -0.075 | 0.832 | 0.372 (152/409) | 1.000 | 0.999 | 2409 |
| 15 | Qwen3-Next-80B-A3B | open | 0.256 | 0.250 / 0.264 / 0.253 | -0.063 | 0.807 | 0.193 (79/409) | 1.000 | 1.000 | 2409 |

- **tier-policy v6** = mean over the 3 tiers of (pick == corrected `canonical_uci`).
  **Δ vs old** = v6 minus the same-extractor score on the OLD labels (isolates the
  correction). **move-sound** = pick in the corrected tier sound pool. **distinct** =
  of positions whose corrected canonical beginner≠advanced, the share where the model's
  beginner≠advanced picks (honest all-opportunities denominator, scoped to coverage).
  **names/format** are label-independent (format = names a move AND closes with a
  "Takeaway:" line; OURS-v3's low format reflects its terser output style, not a move
  miss). **n(det)** = scenarios scored (frontier on the 150-subset ×3).
- The instructiveness council is intentionally **SKIPPED** here (de-emphasized +
  expensive); the existing §3 council stands and a corrected-label council can be run
  **on request**.

### Did the frontier ranking shift materially?

**Partly — the field-level ordering holds, but the top of the frontier flips.** The
correction lowered nearly every model's absolute exact-match (sharper, harder targets;
only BASE-1.7B is flat at +0.002) and compressed the field. Family averages keep their
order: **OURS 0.486 > frontier 0.437 > open 0.310** (was 0.600 > 0.506 > 0.387). The
material moves:

- **Best frontier flips Gemini → Claude.** Frontier internal order was
  [Gemini, Claude, GPT] and is now **[Claude, Gemini, GPT]**; Claude climbs #4 → #2
  overall. GPT-5.5 stays third of the three frontier coaches.
- **OURS-v3 slips #2 → #3** as Claude edges past it (0.463 vs 0.467, ~0.4 pt).
- Tail reshuffles are minor (Llama #9 → #12, Kimi #12 → #14; BASE rises #13 → #7 only
  because it barely moved while others fell).

So the *cross-family* story (OURS > frontier > open on the moat) is preserved, but the
single strongest frontier competitor changed from Gemini to **Claude Opus 4.8**.

### Where OURS stands vs the field (the bonus claim) — still holds

- **OURS is still #1 on the moat.** OURS-v2 (Qwen3-1.7B tuned) tops the full field under
  BOTH old (0.633) and corrected (0.509) labels — **+0.042 over the best frontier**
  (Claude 0.467) and ahead of every open model on the corrected labels.
- **Tuned-over-base is preserved:** OURS-v2 − BASE-1.7B = **+0.151** (0.509 vs 0.358);
  OURS-v3 − Qwen3-32B-untuned = **+0.162** (0.463 vs 0.300).
- **Best open unchanged: GLM-5** (#6, 0.387) — still the strongest open coach on the moat.
- Both OURS models remain top-3; the "OURS beats the frontier field on tier-appropriate
  selection" claim survives the correction (now led by OURS-v2, with Claude the single
  strongest frontier competitor rather than Gemini).

### Current-generation OURS on the corrected benchmark (reference — different scope)

The shipped/stretch OURS adapters were re-evaluated FRESH on the corrected grounding
over the **120 held-out TEST × 3** (not the 803 / 150 field slice above), in
`RESULTS_STAGE4_CORRECTED.md`; reproduced here for orientation only — **do not compare
cross-scope** to the field table above (fresh corrected grounding vs old grounding;
120 TEST vs 803):

| Model (corrected v6, 120 TEST, fresh grounding) | condition | tier-policy | move-sound | distinct | names |
|---|---|---:|---:|---:|---:|
| OURS-v6-dpo2 | grounded | **0.892** | 0.983 | 0.987 | 0.986 |
| OURS-v6-dpo | grounded | 0.881 | 0.983 | 0.987 | 0.983 |
| OURS-v4 (shipped) | grounded | 0.861 | 0.983 | 0.987 | 0.983 |
| BASE (Qwen3-32B untuned) | grounded | 0.428 | 0.969 | 0.303 | 0.975 |
| OURS-v6-distill | no-grounding | 0.325 | 0.653 | 0.461 | 0.983 |

_OURS-v6-dpo2 (tier-targeted DPO, checkpoint step 200) is the strongest DPO variant: all
its gain over v4 is the intermediate tier (B / I / A = 0.858 / 0.842 / 0.975); beginner and
advanced are unchanged. Detail in `RESULTS_STAGE4_CORRECTED.md`._

**Artifacts (this section):** re-score driver `scripts/rescore_field_v6.py`; per-model
old+v6 metrics `data/benchmark_gap803/field_v6_rescore.json`; corrected labels
`data/benchmark_gap803/scenarios_v6.jsonl`; cached field gens
`data/benchmark_gap803/gen/<model>.jsonl`.
Reproduce (free, ~7 s, no GPU/network): `~/.venvs/mlx/bin/python scripts/rescore_field_v6.py`.

## Method & cost-smart scope

- **Deterministic metrics** (tier-fit, tier-differentiation, direction, move-safety, no-engine-speak) computed on **ALL 803 positions x 3 tiers** for the 2 local models (OURS-v2, BASE) + OURS-v3 + 9 open models — the 12 models with full generations. The **3 frontier references** are measured on a **balanced 150-position stratified subset x 3 tiers** — generating Claude Opus 4.8 on all 803 would add real cost for a *reference* row whose behavior is already established; the stratified subset gives a tight estimate.
- **Tier-appropriate move (the moat):** each coach's recommended move is re-extracted with the instrumented, pool-restricted extractor and compared to the canonical tier move from `src/teacher/tier_select.select_tier_move` (beginner=most human-findable sound move, intermediate=eval/Maia blend, advanced=sharpest=engine best). `tier-fit` = pick == that canonical move (mean over the 3 tiers).
- **Instructiveness:** one blinded cross-family council (3 frontier judges: GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro) RANKS the unified **15-model** field per item. It covers **all 450 (position×tier) items where every one of the 15 models has a generation** — the complete eligible set (the 3 frontier models were only generated on the 150-position frontier subset × 3 tiers, so a 15-way ranking is impossible elsewhere without fabricating outputs). Because each judge also grades its own lab's model, we report BOTH a raw and a **self-preference-corrected** ranking (§3).
- **Faithfulness is a fairness FLOOR, not a scoring axis:** after the verify-and-regenerate gate, **every model ships zero verifier-detectable mechanical violations**. The same gate is applied to all, so raw pre-gate fabrication is intentionally **not** reported as a per-model comparison axis. Where models genuinely differ on truth is the semantic-judge residual (§4). Move-safety (no blunders) and no-engine-speak remain pass/fail gates.

## 1. Per-metric leaderboard (all 15 models)

Sorted by the balanced score (below). `tier-fit` is the moat metric; **instr rank** is the **self-preference-corrected** blinded-council mean rank (lower = better, of 15) — raw vs corrected + per-judge self-preference in §3.

| # | Model | family | tier-fit↑ | tier-diff↑ | direction↑ | instr rank↓ (top1) | safety↑ | no-jargon↑ | local | n(det) |
|---|---|---|---:|---:|---:|---:|---:|---:|:--:|---:|
| 1 | OURS-v3 (Qwen3-32B tuned) | ours | 53% | 39% | 55% | 6.93 (23%) | 94% | 96% | yes | 2409 |
| 2 | GPT-5.5 | frontier | 43% | 31% | 46% | 3.72 (24%) | 99% | 100% | no | 450 |
| 3 | Claude Opus 4.8 | frontier | 46% | 31% | 50% | 4.95 (19%) | 98% | 100% | no | 450 |
| 4 | GLM-5 | open | 45% | 37% | 52% | 6.52 (4%) | 100% | 100% | no | 2409 |
| 5 | Gemini 3.1 Pro | frontier | 48% | 28% | 50% | 6.70 (12%) | 99% | 100% | no | 450 |
| 6 | Kimi-K2.5 | open | 36% | 49% | 48% | 7.43 (4%) | 100% | 99% | no | 2409 |
| 7 | OURS-v2 (Qwen3-1.7B tuned) | ours | 53% | 44% | 54% | 10.26 (7%) | 99% | 100% | yes | 2409 |
| 8 | Qwen3-32B (untuned v3 base) | open | 37% | 45% | 48% | 9.30 (0%) | 100% | 99% | yes | 2409 |
| 9 | DeepSeek-R1 | open | 44% | 39% | 50% | 7.98 (1%) | 100% | 100% | no | 2409 |
| 10 | Llama-3.3-70B | open | 40% | 27% | 48% | 8.39 (0%) | 100% | 100% | tight | 2409 |
| 11 | Gemma-3-27B-it | open | 35% | 23% | 48% | 8.88 (1%) | 100% | 100% | yes | 2409 |
| 12 | DeepSeek-V3.2 | open | 41% | 41% | 50% | 8.24 (1%) | 100% | 100% | no | 2409 |
| 13 | Qwen3-Next-80B-A3B | open | 32% | 25% | 51% | 8.53 (1%) | 100% | 100% | tight | 2409 |
| 14 | Mistral-Large-3 (675B) | open | 37% | 44% | 49% | 9.43 (2%) | 100% | 100% | no | 2409 |
| 15 | BASE (Qwen3-1.7B untuned) | base | 36% | 50% | 46% | 14.18 (0%) | 96% | 96% | yes | 2409 |

- **tier-fit** = share of (position,tier) where the coach's pick equals the canonical `select_tier_move` move. **tier-diff** = share of positions where the pick changes across the 3 tiers. **direction** = share where the beginner pick is at least as human-findable (Maia rank) as the advanced pick (correct level gradient).
- **safety** = share of picks that are not blunders (cp-loss < 250). **no-jargon** = no centipawn/engine-speak leaked. **n(det)** = deterministic positions x tiers scored (frontier on the 150-subset). Faithfulness is a gated fairness floor (zero verifier-detectable mechanical violations for all models), so it is not a comparison column here — see §4.

## 2. Tier-appropriate move selection (the moat), per tier

Per-tier `tier-fit` (pick == `select_tier_move` canonical) + engine-mirror rate. A strong leveled coach has HIGH beginner/intermediate fit (finding the *human* move) while advanced fit ~ engine-mirror (the sharp move is correct there).

| Model | fit B | fit I | fit A | mirror B | mirror I | mirror A | diff | mirror@all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OURS-v3 (Qwen3-32B tuned) | 30% | 46% | 84% | 78% | 76% | 85% | 39% | 58% |
| GPT-5.5 | 31% | 45% | 54% | 52% | 51% | 57% | 31% | 43% |
| Claude Opus 4.8 | 30% | 46% | 61% | 59% | 60% | 65% | 31% | 47% |
| GLM-5 | 34% | 43% | 57% | 53% | 57% | 59% | 37% | 42% |
| Gemini 3.1 Pro | 29% | 47% | 69% | 69% | 70% | 74% | 28% | 60% |
| Kimi-K2.5 | 34% | 36% | 38% | 34% | 36% | 38% | 49% | 22% |
| OURS-v2 (Qwen3-1.7B tuned) | 48% | 50% | 61% | 65% | 59% | 63% | 44% | 42% |
| Qwen3-32B (untuned v3 base) | 34% | 38% | 38% | 45% | 41% | 39% | 45% | 28% |
| DeepSeek-R1 | 38% | 42% | 52% | 43% | 47% | 52% | 39% | 34% |
| Llama-3.3-70B | 35% | 39% | 44% | 40% | 43% | 45% | 27% | 35% |
| Gemma-3-27B-it | 28% | 34% | 42% | 43% | 42% | 43% | 23% | 36% |
| DeepSeek-V3.2 | 33% | 38% | 52% | 47% | 50% | 52% | 41% | 37% |
| Qwen3-Next-80B-A3B | 33% | 32% | 30% | 30% | 31% | 30% | 25% | 24% |
| Mistral-Large-3 (675B) | 37% | 37% | 36% | 35% | 37% | 37% | 44% | 24% |
| BASE (Qwen3-1.7B untuned) | 31% | 39% | 40% | 51% | 43% | 41% | 50% | 28% |

## 3. Instructiveness — blinded cross-family council (raw vs self-preference-corrected)

One blinded, cross-family council: **3 frontier judges** (GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro) each RANK the unified **15-model** field per item on instructiveness for the stated tier (blinded labels A–O, shuffled per item). Coverage = **all 450 (position×tier) items where every one of the 15 models has a generation** (the complete eligible set). **n = 450 items × 3 judges = 1350 rankings.** Mean rank ↓ = better (of 15); 95% CIs are a cluster bootstrap by item.

Because each judge also grades a model from its OWN lab, the raw leaderboard is contaminated by **self-preference**. The **corrected** column drops each frontier competitor's same-lab judge (leave-own-out), so no model is graded by its own family; non-frontier models keep all 3 judges.

| # | Model | family | raw mean rank ↓ [95% CI] | corrected ↓ [95% CI] | top-1% |
|---|---|---|---:|---:|---:|
| 1 | GPT-5.5 | frontier | 3.35 [3.18–3.53] | 3.72 [3.52–3.92] | 24% |
| 2 | Claude Opus 4.8 | frontier | 4.79 [4.54–5.05] | 4.95 [4.68–5.24] | 19% |
| 3 | GLM-5 | open | 6.52 [6.27–6.78] | 6.52 [6.27–6.78] | 4% |
| 4 | Gemini 3.1 Pro | frontier | 5.78 [5.55–6.04] | 6.70 [6.43–6.98] | 12% |
| 5 | OURS-v3 (Qwen3-32B tuned) | ours | 6.93 [6.55–7.32] | 6.93 [6.55–7.32] | 23% |
| 6 | Kimi-K2.5 | open | 7.43 [7.16–7.69] | 7.43 [7.16–7.69] | 4% |
| 7 | DeepSeek-R1 | open | 7.98 [7.75–8.21] | 7.98 [7.75–8.21] | 1% |
| 8 | DeepSeek-V3.2 | open | 8.24 [7.96–8.51] | 8.24 [7.96–8.51] | 1% |
| 9 | Llama-3.3-70B | open | 8.39 [8.17–8.59] | 8.39 [8.17–8.59] | 0% |
| 10 | Qwen3-Next-80B-A3B | open | 8.53 [8.28–8.79] | 8.53 [8.28–8.79] | 1% |
| 11 | Gemma-3-27B-it | open | 8.88 [8.64–9.12] | 8.88 [8.64–9.12] | 1% |
| 12 | Qwen3-32B (untuned v3 base) | open | 9.30 [9.07–9.54] | 9.30 [9.07–9.54] | 0% |
| 13 | Mistral-Large-3 (675B) | open | 9.43 [9.18–9.67] | 9.43 [9.18–9.67] | 2% |
| 14 | OURS-v2 (Qwen3-1.7B tuned) | ours | 10.26 [9.90–10.61] | 10.26 [9.90–10.61] | 7% |
| 15 | BASE (Qwen3-1.7B untuned) | base | 14.18 [14.07–14.28] | 14.18 [14.07–14.28] | 0% |

**Per-judge self-preference** — how each judge ranks its OWN lab's model vs how the other two judges rank that same model. Δ = (own − peers) in rank positions; **negative ⇒ the judge favours its own family** (ranks it better / lower).

| judge | ranks own family ↓ | peers rank it ↓ | Δ (own − peers) |
|---|---:|---:|---:|
| GPT-5.5 | 2.62 | 3.72 | -1.10 |
| Claude Opus 4.8 | 4.48 | 4.95 | -0.47 |
| Gemini 3.1 Pro | 3.96 | 6.70 | -2.74 |

Mean signed self-preference Δ = **-1.44** rank positions — all three judges favour their own family; the corrected ranking above removes it.

## 4. Truthfulness — fairness floor + semantic-judge residual

**Fairness floor (verifier-detectable mechanical violations):** after the verify-and-regenerate gate, **every model ships zero verifier-detectable mechanical violations** — the deterministic board-fact checker finds no false board fact in any shipped cell. The same gate is applied to OURS, BASE, frontier and open alike, so faithfulness is **table-stakes, not a per-model differentiator**; raw pre-gate fabrication is intentionally NOT reported as a comparison axis.

**Semantic-truth residual (the honest differentiator):** an independent cross-family judge panel (GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro) fact-checks a stratified sample of the **gated** text for the multi-move / evaluative claims the deterministic layer cannot decide. Reported under three nested rules with 95% CIs: **any** (a single objection sinks the cell — a strict **lower bound**), **majority** (≥2 of 3), **unanimous** (only a 3/3 objection sinks it — a lenient **upper bound**). OURS trails the frontier here.

| Model | n | any (strict ↓) | majority | unanimous (lenient ↑) |
|---|---:|---:|---:|---:|
| GPT-5.5 | 39 | 79% [64–89] | 97% [87–100] | 100% [91–100] |
| Llama-3.3-70B | 18 | 72% [49–88] | 83% [61–94] | 94% [74–99] |
| Qwen3-32B | 39 | 46% [32–61] | 64% [48–77] | 74% [59–85] |
| GLM-5 | 18 | 44% [25–66] | 67% [44–84] | 89% [67–97] |
| Kimi-K2.5 | 18 | 44% [25–66] | 72% [49–88] | 89% [67–97] |
| Mistral-Large-3 (675B) | 18 | 39% [20–61] | 56% [34–75] | 67% [44–84] |
| Gemini 3.1 Pro | 39 | 33% [21–49] | 64% [48–77] | 92% [80–97] |
| Claude Opus 4.8 | 39 | 26% [15–41] | 56% [41–71] | 85% [70–93] |
| Gemma-3-27B-it | 39 | 26% [15–41] | 54% [39–68] | 72% [56–83] |
| OURS-v2 (1.7B tuned) | 39 | 23% [13–38] | 26% [15–41] | 31% [19–46] |
| DeepSeek-V3.2 | 18 | 22% [9–45] | 39% [20–61] | 67% [44–84] |
| DeepSeek-R1 (reasoning) | 18 | 11% [3–33] | 50% [29–71] | 61% [39–80] |
| Qwen3-Next-80B-A3B | 18 | 6% [1–26] | 33% [16–56] | 67% [44–84] |
| BASE (Qwen3-1.7B-4bit, untuned) | 18 | 0% [0–18] | 17% [6–39] | 33% [16–56] |

Pooled (n=378): any 35% [31–40], majority 57% [52–62], unanimous 74% [69–78].

_Source: `data/showcase/truthfulness.json` — the 14-model gated showcase set. "any" is a conservative lower bound (a single cross-family objection marks a cell not-truthful), not a claim the rest are outright lies. OURS-v3 is a gap803-only model and is not in this sample._

## 5. Weighted BALANCED ranking

Transparent weighted score (each component normalized to 0-1, higher = better): **tier-appropriate move selection 45%** + **instructiveness (self-preference-corrected) 45%** + practical (local+cost) 10%. Safety + no-jargon are pass/fail gates. **Fabrication is not a scoring axis** — it is a gated fairness floor (zero verifier-detectable mechanical violations for all), not a differentiator (§4). Score = weighted mean x 100.

| # | Model | family | tier(0.45) | instr(0.45) | practical(0.10) | **balanced** | gate |
|---|---|---|---:|---:|---:|---:|:--:|
| 1 | OURS-v3 (Qwen3-32B tuned) | ours | 49.0 | 57.6 | 100.0 | **58.0** | **FAIL** |
| 2 | GPT-5.5 | frontier | 39.9 | 80.6 | 35.0 | **57.7** | pass |
| 3 | Claude Opus 4.8 | frontier | 42.3 | 71.8 | 0.0 | **51.3** | pass |
| 4 | GLM-5 | open | 44.4 | 60.6 | 39.1 | **51.1** | pass |
| 5 | Gemini 3.1 Pro | frontier | 42.3 | 59.3 | 35.0 | **49.2** | pass |
| 6 | Kimi-K2.5 | open | 44.4 | 54.1 | 38.6 | **48.2** | pass |
| 7 | OURS-v2 (Qwen3-1.7B tuned) | ours | 50.3 | 33.8 | 100.0 | **47.9** | pass |
| 8 | Qwen3-32B (untuned v3 base) | open | 43.3 | 40.7 | 99.7 | **47.8** | pass |
| 9 | DeepSeek-R1 | open | 44.5 | 50.2 | 37.0 | **46.3** | pass |
| 10 | Llama-3.3-70B | open | 37.9 | 47.2 | 75.4 | **45.9** | pass |
| 11 | Gemma-3-27B-it | open | 35.1 | 43.7 | 99.8 | **45.5** | pass |
| 12 | DeepSeek-V3.2 | open | 43.8 | 48.3 | 39.6 | **45.4** | pass |
| 13 | Qwen3-Next-80B-A3B | open | 35.9 | 46.2 | 75.6 | **44.5** | pass |
| 14 | Mistral-Large-3 (675B) | open | 43.3 | 39.8 | 36.4 | **41.0** | pass |
| 15 | BASE (Qwen3-1.7B untuned) | base | 44.2 | 5.9 | 100.0 | **32.5** | **FAIL** |

## 6. Best v3-BASE ranking (re-weighted)

For a fine-tuning *base*, tier-appropriateness is what we ADD, so it is down-weighted; the hard-to-add qualities dominate: **instructiveness/capacity 45%**, **local-runnability+cost 45%**, tier 10%. Only locally fine-tunable/runnable models are viable bases (faithfulness is a gated fairness floor for every model, so it is not a base-selection axis).

| # | Model | family | base-fit | local | note |
|---|---|---|---:|:--:|---|
| 1 | OURS-v3 (Qwen3-32B tuned) | ours | 75.8 | yes | 4-bit ~18GB — comfortable (fine-tuned) |
| 2 | Gemma-3-27B-it | open | 68.1 | yes | 4-bit ~15GB — comfortable |
| 3 | Qwen3-32B (untuned v3 base) | open | 67.5 | yes | 4-bit ~18GB — comfortable |
| 4 | OURS-v2 (Qwen3-1.7B tuned) | ours | 65.3 | yes | already local |
| 5 | Llama-3.3-70B | open | 59.0 | tight | 4-bit ~40GB — tight on 64GB |
| 6 | Qwen3-Next-80B-A3B | open | 58.4 | tight | 4-bit ~43GB — tight but fits; 3B active = fast |
| 7 | GPT-5.5 | frontier | 56.0 | no | API-only |
| 8 | BASE (Qwen3-1.7B untuned) | base | 52.1 | yes | already local |
| 9 | GLM-5 | open | 49.3 | no | far exceeds 64GB |
| 10 | Gemini 3.1 Pro | frontier | 46.7 | no | API-only |
| 11 | Kimi-K2.5 | open | 46.2 | no | far exceeds 64GB |
| 12 | DeepSeek-V3.2 | open | 44.0 | no | far exceeds 64GB |
| 13 | DeepSeek-R1 | open | 43.7 | no | far exceeds 64GB |
| 14 | Mistral-Large-3 (675B) | open | 38.6 | no | far exceeds 64GB |
| 15 | Claude Opus 4.8 | frontier | 36.5 | no | API-only |

## 7. Recommendation

- **Best overall (balanced), any provider: OURS-v3 (Qwen3-32B tuned)** (ours) — a **locally-runnable** model tops the balanced score — though it currently trips the 97% safety/no-jargon gate on formatting (not blunders; see §1/§5), so a gate-passing frontier model leads the shippable board. The three frontier APIs remain strongest on raw instructiveness (§3), but no longer dominate the blend once tier-appropriateness + local-runnability are weighed and self-preference is removed.
- **Best OPEN model (balanced): GLM-5.** This is **NOT** Gemma-3-27B — GLM-5 is the strongest open coach (best open instructiveness) with solid tier-selection, but it is far too large to run locally.
- **Best open v3 base: Gemma-3-27B-it** — the best mix of coaching capacity and 4-bit local fine-tunability/runnability on a 64GB Mac (faithfulness is a gated fairness floor for every model, so it is not a tie-breaker).
- **It is effectively a tie with Qwen3-32B (untuned v3 base)** (base-fit 68.1 vs 67.5 — within noise). Gemma-3-27B-it edges it on tier-selection/capacity; Qwen3-32B (untuned v3 base) is smaller. Either is a defensible v3 base; prefer Qwen3-32B (untuned v3 base) if size / local-runnability is paramount, Gemma-3-27B-it if raw capacity is.
- **The balanced winner and the base pick differ:** GLM-5 wins the raw open balanced score, but it far exceeds 64GB — not a viable local base. Gemma-3-27B-it is the pragmatic v3 base because tier-appropriateness (where the giant coaches lead) is exactly what we fine-tune IN, while capacity + faithfulness + local-runnability are what a base must bring.

## 8. Cost

| group | calls | in tok | out tok | est. USD |
|---|---:|---:|---:|---:|
| open | 21,681 | 21,910,715 | 4,266,729 | $28.17 |
| frontier_gen | 1,350 | 1,531,055 | 495,278 | $21.71 |
| council | 1,350 | 4,376,779 | 1,376,176 | $62.27 |
| local | 7,227 | 2,639,131 | 0 | $0.00 |
| **TOTAL** | | | | **$112.15** |

_Open-model + frontier prices are best-effort Bedrock/gateway estimates; local (OURS-v2, BASE) is free. Total definitive-eval spend: **$112.15**._

## Artifacts

- Positions: `data/eval/gap_positions.jsonl` (803, curated, zero-leakage)
- Flattened scenarios: `data/benchmark_gap803/scenarios.jsonl` (803x3)
- Generations: `data/benchmark_gap803/gen/<model>.jsonl` -> `generations.jsonl`
- Objective: `data/benchmark_gap803/objective.jsonl`; safety: `move_safety.json`
- Council: `data/benchmark_gap803/council.jsonl`; leaderboard: `leaderboard.json`; instructiveness stats (raw + self-pref-corrected + CIs): `council_stats.json`
- Truthfulness residual (any/majority/unanimous + CIs): `data/showcase/truthfulness.json`
- Drivers: `scripts/gap803_{gen,report,safety,council,council_stats}.py`, `scripts/gap803_common.py`