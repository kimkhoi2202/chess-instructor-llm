# Chess-Coach Move-SELECTION Divergence Report

- **Model:** `models/mlx/chess-coach-v1` (tuned)  ·  **Decoding:** greedy (temp=0, deterministic)
- **Positions:** 120 held-out (none appear in `train.jsonl`/`valid.jsonl`; excluded by board+side-to-move key)
- **Prompt:** identical to the live app — `render_pool_facts` + `render_user_prompt`, system = `coach_system.md` + grounding + format suffix
- **Move extraction:** the live API's `_extract_recommended`, instrumented to separate a genuinely NAMED pick (`cue`/`prose`) from the API's `pool[0]` fallback

> Greedy decoding is used on purpose: any move change across tiers is genuine tier-conditioning (same model, only the tier label / ply-cap / tier-Maia block differ in the prompt), not sampling noise.

## 1. Headline rates

| Metric | Rate |
|---|---|
| **Tier-differentiation** (>=1 tier picks a different move), all positions | ** 25.0%** (30/120) |
| Tier-differentiation, genuine-pick subset (all 3 tiers named a move) |  22.5% (25/111) |
| **Joint "interesting" set** (tiers differ OR model != SF-best) | ** 34.2%** (41/120) |
| Engine-divergence, beginner (move != SF best) |  18.3% |
| Engine-divergence, intermediate |  25.0% |
| Engine-divergence, advanced |  29.2% |

## 2. Tier-differentiation distribution

How many DISTINCT moves the three tiers produce per position (1 = all tiers agree).

| Distinct tier-moves | All positions | Genuine-pick subset |
|---|---|---|
| 1 | 90 ( 75.0%) | 86 ( 77.5%) |
| 2 | 29 ( 24.2%) | 24 ( 21.6%) |
| 3 | 1 (  0.8%) | 1 (  0.9%) |

- All positions: ** 25.0%** differentiate (30/120).
- Genuine subset (n=111): ** 22.5%** differentiate (25/111).

## 3. Engine- and human-divergence, per tier

`!= SF best` = model's move differs from Stockfish's best (pool[0]). `!= Maia top` = differs from that tier's most human-likely move. `== SF best` is the sanity check the task asked for (how often the model just returns the engine's best).

| Tier | == SF best | != SF best | == Maia top | != Maia top |
|---|---|---|---|---|
| beginner |  81.7% |  18.3% |  38.3% |  61.7% |
| intermediate |  75.0% |  25.0% |  42.5% |  57.5% |
| advanced |  70.8% |  29.2% |  39.2% |  60.8% |

### Extraction-mode split per tier (context for `== SF best`)

A `fallback_pool0` row means the model did NOT name a sound move in prose, so the live API *displays* the engine best. Those rows inflate `== SF best` without the model actually choosing it, so they must be read separately.

| Tier | cue (named) | prose (named) | fallback_pool0 | genuine total | == SF among genuine |
|---|---|---|---|---|---|
| beginner | 110 | 1 | 9 | 111 |  80.2% |
| intermediate | 114 | 0 | 6 | 114 |  73.7% |
| advanced | 114 | 2 | 4 | 116 |  69.8% |

## 4. Does it add value beyond copying? (genuine named picks, pooled over tiers)

Across all 341 genuinely named picks (positions x tiers):

| Pick relationship | Count | Share |
|---|---|---|
| mirrors engine only (== SF best, != Maia top) | 158 |  46.3% |
| mirrors human only (== Maia top, != SF best) | 32 |   9.4% |
| == both (engine best is also the human top) | 96 |  28.2% |
| independent (!= SF best AND != Maia top) | 55 |  16.1% |

- Picks that equal the engine best (`sf_only` + `both`):  74.5%.
- Picks that are NOT the engine best (`maia_only` + `neither`):  25.5%.

## 5. Direction of differentiation (positions where tiers genuinely differ)

Among the 25 genuine-pick positions where the tiers disagree, does the beginner tier lean toward the human (Maia) move and the advanced tier toward the engine's best? Mean pool rank: 0 = engine best, higher = further from engine best.

| Tier | picks == its Maia top | picks == SF best | mean pool rank of pick |
|---|---|---|---|
| beginner |  20.0% |  56.0% | 1.24 |
| intermediate |  60.0% |  36.0% | 1.56 |
| advanced |  48.0% |  24.0% | 1.44 |

## 6. Interesting-set composition & where differentiation happens

- Total interesting positions: **41/120** ( 34.2%).
  - tiers differ AND diverge from SF: 30
  - tiers differ only (all tiers == SF best but not all identical is impossible; this counts tiers-differ with every tier == SF best): 0
  - diverge from SF only (tiers agree on a non-SF-best move): 11
  - of interesting positions, also != Maia top on >=1 tier: 39

Differentiation rate by phase / severity:

| Phase | diff / total |   | Severity | diff / total |
|---|---|---|---|---|
| endgame | 12/33 ( 36.4%) |   | blunder | 7/42 ( 16.7%) |
| middlegame | 9/42 ( 21.4%) |   | inaccuracy | 8/35 ( 22.9%) |
| opening | 9/45 ( 20.0%) |   | mistake | 11/37 ( 29.7%) |
|  |  |   | none | 4/6 ( 66.7%) |

## 7. Example differentiating positions

- `mr9QjWOC_23` [opening/inaccuracy] SF best **Qc4+** — B:**Qc4+** I:**Qc4+** A:**Bxd7+**
- `b92oJhdS_51` [middlegame/none] SF best **Nc4** — B:**Nc4** I:**Nc4** A:**Be3**
- `GEHbHVSu_23` [middlegame/blunder] SF best **Bxc6+** — B:**cxb4** I:**cxb4** A:**Bxc6+**
- `VsykaIN0_82` [endgame/none] SF best **Ra3** — B:**Kf6** I:**Kf6** A:**Ra3**
- `X1nwiPgj_124` [endgame/inaccuracy] SF best **f5** — B:**f5** I:**Ra4** A:**Ra4**
- `flY6Tk8T_19` [opening/mistake] SF best **Nxg4** — B:**Nxg4** I:**Bxc6** A:**Bxc6**
- `diZkf8wa_37` [opening/mistake] SF best **Ne2** — B:**Be1** I:**Ne2** A:**Be1**
- `I72qKqM7_69` [endgame/mistake] SF best **Kf4** — B:**Nc6+** I:**Nc6+** A:**Kf4**

_(Full per-tier coaching text for every position is in `divergence.jsonl` under `tiers.<tier>.coaching`.)_

## 8. Root cause — why differentiation is modest and mis-directed

Three joined measurements (all on data, no guessing):

**(a) The model faithfully mirrors its teacher, then regresses toward the engine best.**
On the same 120 held-out positions, at each position's source tier:

| Comparison | Rate |
|---|---|
| model pick == teacher (gpt-5.5) pick | 70.8% (85/120) |
| teacher pick == Stockfish best | 65.8% |
| **model pick == Stockfish best** | **76.7%** |

The 1.7B student reproduces the teacher ~71% of the time but collapses onto `pool[0]`
*more* than the teacher did (76.7% vs 65.8%) — it rounds the teacher's subtler,
non-best picks back toward the obvious engine move. That washes out tier signal.

**(b) The teacher's own move choice is only weakly tier-aware.** Across all 2,132
training candidates (one tier each):

| Teacher tier | == SF best | == Maia top | mean pool rank |
|---|---|---|---|
| beginner | 67.5% | 42.2% | 0.67 |
| intermediate | 71.2% | 41.3% | 0.64 |
| advanced | 79.3% | 37.9% | 0.47 |

There is a mild intended gradient (beginner picks slightly more human / less-best
moves; advanced picks the sharpest), but it is small — and it is a *between-position*
comparison (different positions per tier), so even this weak signal is partly
confounded. The SFT target simply does not strongly encode "the move should change
with tier."

**(c) The training data has ZERO contrastive tier examples.** Of 2,132 unique
training FENs, **0 (0.0%)** appear at more than one tier. The model has never once
seen the *same position* taught with a *different move* for a different tier, so it
has no supervised reason to vary its pick by tier. Whatever differentiation exists
(25%) is an emergent side-effect of the tier label / ply-cap / tier-Maia block
nudging the prompt — and on identical positions it points the *wrong way*: beginners
match the engine's sharp best move **most** (81.7%) and the human/Maia move is flat
across tiers (~38-42%), the opposite of "give beginners the move they'd actually find."

## 9. VERDICT

**Is "same move across tiers" happening, and how often? Is it fine?**
Yes for **75%** of positions (identical move to all three tiers); the model genuinely
differentiates on **25%** (30/120, greedy — not sampling noise). "Same move at every
tier" is therefore the *common* case but **not universal**. Partly fine: many positions
have one clearly-best instructive move for everyone (win the hanging queen), so agreeing
across tiers is correct there. What is **not** fine is that the differentiation that does
happen is (1) modest and (2) mis-directed — the model does not steer beginners toward the
more human-findable move; if anything beginners get the sharp engine move slightly more
than advanced.

**Does the model add move-SELECTION value beyond copying Stockfish/Maia, or is its value
only in the explanation?**
It is **not** a pure engine mirror: **25.5%** of genuinely-named picks are not the engine
best, and **16.1%** are *independent* (neither engine best nor Maia top) — real
"instructive middle" picks. It is **not** a Maia mirror either (~60% differ from the
tier's human-top move). So there *is* some move-selection signal beyond either engine.
**But** it is weak and more engine-anchored than the teacher (77% == SF best), and it is
barely tier-conditioned. **The model's distinctive, reliable value is the EXPLANATION**
(calibrated to tier, jargon-free, grounded in verified facts) — that is where the tuned
delta lives — **far more than in tier-differentiated move selection**, which is marginal
today.

**Differentiation is NOT ~0 (it is 25%)** — so, per the brief, it is quantified above and
the **41 differentiating positions are saved as proof** (see §10). It is, however, weak
and mis-directed, so the corrective direction below still applies.

## 10. What to change (and why "more positions" won't fix it)

Downloading more positions of the same kind cannot help: it re-samples the same weakly
tier-aware teacher rule and the same one-tier-per-position format, so the student keeps
regressing to `pool[0]`. The lever is the **teacher's move-selection rule + the training
format**, then regenerate → filter → retrain:

1. **Make the teacher's move choice explicitly, strongly tier-aware** (in
   `src/teacher/generate.py`), in the *correct* direction:
   - **beginner** → recommend the sound-pool move with the **highest Maia-1100 policy**
     (the move a 1000-1200 would actually find), even when it is not `pool[0]`.
   - **intermediate** → a Maia-1500-weighted sound move (blend normalized eval + policy).
   - **advanced** → `pool[0]` (the sharpest sound move) unless a clearly better teaching idea exists.
2. **Add contrastive tier examples** (currently 0%): for a subset of positions, generate
   the **same FEN at all three tiers** so the model is directly supervised to vary the move
   by tier. This is the single highest-leverage data change.
3. **Counter the small model's regression to the engine best**: e.g., DPO/preference pairs
   that prefer the tier-appropriate move over `pool[0]` for beginners, or emit the pick as a
   structured field so it is weighted in the loss.
4. **Re-run this exact harness** after retraining. Success = tier-differentiation up (target
   >50%) **and correctly directed** (beginner `== Maia` up and `mean pool rank`
   beginner > advanced), while the existing base-vs-tuned explanation-quality judge holds.

## 11. Deliverables & method notes

- Raw per-position records: `data/analysis/divergence.jsonl` (120 rows; each has the sound
  pool, SF best, per-tier Maia tops, per-tier pick + extraction mode + full coaching).
- Filtered "interesting" gallery (tiers differ and/or model != SF best / != Maia top),
  in the exact `library.json` schema, swappable into the Studio:
  **`web/public/library_differentiated.json` — 41 positions** (30 where tiers differ, all 41
  diverge from SF best on at least one tier). `library.json` was left untouched.
- **Held-out check:** all 120 FENs verified absent from `train.jsonl`/`valid.jsonl` (0 leaks
  by board+side-to-move key and by exact FEN).
- **Decoding:** greedy (temp=0) so tier differences are genuine conditioning, not sampling
  noise. Extraction reuses the live API's `_extract_recommended` but is instrumented so the
  `pool[0]` fallback (only 9/6/4 of 120 per tier) is reported separately and never silently
  inflates "== SF best".
