# Frontier Gap Report — is the leveled teaching-move behavior a real gap?

The instructor's framing: **before** we can claim training moved our model into a valuable behavior gap, we must first prove the **frontier** models are *not reliably good* at that narrow behavior. This report measures that on data.

## Setup

- **Positions:** 50 held-out (a balanced phase×severity subset of `data/analysis/divergence.jsonl`; every FEN re-verified absent from `train.jsonl`/`valid.jsonl` by board+side-to-move key).
- **Grounding:** byte-identical to the live app and to the v1 divergence run — `render_pool_facts` (verified piece/threat facts) + `render_user_prompt` (the Stockfish sound pool + the tier's Maia block), system = `coach_system.md` + grounding + format suffix. Only the model changes.
- **Models:** GPT-5.5 (`openai-group/gpt-5.5`), Claude Opus 4.8 (`claude-group/claude-opus-4-8`, temperature omitted), Gemini 3.1 Pro (`gemini-group/gemini-3.1-pro`) via the TrueFoundry gateway, each at ALL THREE tiers. `v1-tuned` = our fine-tuned Qwen3-1.7B coach (picks reused from `divergence.jsonl`).
- **Move extraction:** the live API's `_extract_recommended`, instrumented to separate a genuinely NAMED pick (`cue`/`prose`) from the `pool[0]` fallback (so the fallback never silently inflates `== SF best`).
- **Findability:** each pick's **Maia rank inside the sound pool** at that tier (0 = the most human-likely sound move). The engine-best's own Maia rank is the yardstick.

## The target behavior (the pass definition)

For a stated ELO tier, recommend the move that is **(a) SOUND** (inside Stockfish's tolerance pool — doesn't throw the advantage), **(b) FINDABLE** (high Maia-likelihood at that tier — a move a human at that level would actually consider, not the sharpest engine-only line), and **(c) INSTRUCTIVE** — then coach WHY it is good AND HOW a player at that ELO should think to find it. Serving a beginner the engine-only best move with a GM-level line is a **FAIL**.

## 1. Headline rates (per model)

| Model | Tier-diff | Engine-mirror @ every tier | Beginner pick == engine-best | Beginner pick == most-findable | Fabrication rate |
|---|---|---|---|---|---|
| **gpt-5.5** |  24.0% |  62.0% |  68.0% |  58.0% |   2.7% |
| **claude-opus-4.8** |  24.0% |  66.0% |  70.0% |  52.0% |   5.3% |
| **gemini-3.1-pro** |  20.0% |  78.0% |  84.0% |  50.0% |   2.0% |
| v1-tuned |  22.0% |  70.0% |  84.0% |  54.0% |  51.3% |

- **Tier-diff** = picks ≥1 different move across the three ELO tiers (greedy where the family allows; Claude at default). Higher = more level-aware.
- **Engine-mirror @ every tier** = returns Stockfish's #1 at *all three* tiers. Higher = more of a pure engine mouthpiece, blind to level.
- **Beginner pick == most-findable** = the beginner recommendation is the top Maia-1100 move in the sound pool. This is the behavior we want *high*.
- **Fabrication rate** = share of coaching outputs (position×tier) with ≥1 demonstrably-false board claim (deterministic verifier). Lower is better.

## 2. Tier-differentiation detail

Distinct moves produced across the 3 tiers per position (1 = same move at every tier).

| Model | 1 (same) | 2 | 3 | differentiate (all) | differentiate (genuine subset) |
|---|---|---|---|---|---|
| gpt-5.5 | 38 ( 76.0%) | 11 | 1 | ** 24.0%** |  24.5% (12/49) |
| claude-opus-4.8 | 38 ( 76.0%) | 12 | 0 | ** 24.0%** |  25.0% (12/48) |
| gemini-3.1-pro | 40 ( 80.0%) | 9 | 1 | ** 20.0%** |  21.2% (7/33) |
| v1-tuned | 39 ( 78.0%) | 11 | 0 | ** 22.0%** |  20.8% (10/48) |

## 3. Engine-mirroring per tier (`== Stockfish best`)

| Model | beginner | intermediate | advanced | fallback_pool0 (b/i/a) |
|---|---|---|---|---|
| gpt-5.5 |  68.0% |  72.0% |  78.0% | 1/1/1 |
| claude-opus-4.8 |  70.0% |  80.0% |  78.0% | 1/2/1 |
| gemini-3.1-pro |  84.0% |  86.0% |  90.0% | 12/10/5 |
| v1-tuned |  84.0% |  76.0% |  78.0% | 2/1/1 |

_`fallback_pool0` = the model named no sound move in prose, so extraction falls back to the engine best; those rows are counted but flagged (they inflate `== SF best` without a genuine choice)._

## 4. Findability at the beginner tier (the crux)

A tier-appropriate beginner move is a **findable** sound move (low Maia-1100 rank), not the sharpest engine line. The **findability gap** = `engine_best_maia_rank − pick_maia_rank` (both within the sound pool at Maia-1100): **>0** means the pick is *more* human-findable than the engine best (good); **0** usually means the pick *is* the engine best; **<0** means the model over-leveled the beginner.

| Model | mean findability gap | gap>0 | gap=0 | gap<0 | mean pick Maia-rank | mean Maia-policy gap |
|---|---|---|---|---|---|---|
| gpt-5.5 | 0.26 |  24.0% |  68.0% |   8.0% | 0.94 | 0.021 |
| claude-opus-4.8 | 0.02 |  16.0% |  70.0% |  14.0% | 1.18 | 0.005 |
| gemini-3.1-pro | -0.12 |   6.0% |  84.0% |  10.0% | 1.32 | 0.004 |
| v1-tuned | 0.08 |  10.0% |  84.0% |   6.0% | 1.12 | 0.009 |

### 4b. Opportunity subset — where engine-best ≠ most-findable (26/50 positions)

These are the positions where the engine's sharpest sound move is **not** the move a 1000–1200 player is most likely to find — i.e. the only positions where the leveled teaching-move behavior can actually be *exercised*. On the rest, picking the engine best is already correct for everyone.

| Model | steers beginner to most-findable | still picks engine-best (mirror) | mean findability gap |
|---|---|---|---|
| gpt-5.5 |  30.8% |  50.0% | 0.65 |
| claude-opus-4.8 |  23.1% |  57.7% | 0.23 |
| gemini-3.1-pro |   7.7% |  73.1% | -0.19 |
| v1-tuned |  15.4% |  73.1% | 0.27 |

## 5. Direction check — does findability improve toward beginner?

Mean pick Maia-rank in the sound pool per tier (0 = most human-findable). The intended gradient is **beginner < advanced** (beginners get the more findable move; advanced get the sharper one). Flat or inverted = not level-aware.

| Model | beginner | intermediate | advanced |
|---|---|---|---|
| gpt-5.5 | 0.94 | 0.92 | 0.96 |
| claude-opus-4.8 | 1.18 | 1.12 | 0.98 |
| gemini-3.1-pro | 1.32 | 1.22 | 1.02 |
| v1-tuned | 1.12 | 0.88 | 0.98 |

## 6. Fabrication (coaching truthfulness)

Share of coaching outputs (position×tier) whose prose states a demonstrably-false board fact (a named piece not on the named square, a side lacking a claimed piece), via the deterministic `faithfulness.verify_text`. This is the same truthfulness axis `RESULTS.md` flagged as flat for v1.

| Model | outputs | with ≥1 false claim | total false-claim sentences |
|---|---|---|---|
| gpt-5.5 | 150 |   2.7% | 4 |
| claude-opus-4.8 | 150 |   5.3% | 8 |
| gemini-3.1-pro | 150 |   2.0% | 3 |
| v1-tuned | 150 |  51.3% | 88 |

_Note: frontier coaching is generated in this run; v1 coaching is reused from the v1 divergence run. The verifier is conservative (flags only demonstrably-false claims)._

## VERDICT — is the gap real?

**YES — the gap is REAL.**

Across the three frontier models (same 50 held-out positions, identical grounding as our app), the average rates are:

- **Tier-differentiation:** 22.7% (picks a different move across the three ELO tiers).
- **Beginner findability on the opportunity subset:** 20.5% (when the engine-best is NOT already the most human-findable move, how often the frontier model still steers the beginner to the findable one).
- **Engine-mirroring at EVERY tier:** 68.7% (returns Stockfish's #1 at all three tiers, regardless of level).

The frontier models are strong chess players and their prose is fluent, but on the NARROW target behavior — *tier-appropriate, human-findable move selection* — they mostly recommend the **same move regardless of the stated ELO** (tier-differentiation only 22.7%); on the positions where it matters, they **default to the sharp engine-best instead of the move a beginner would actually find** (findable-pick rate only 20.5% on the opportunity subset); a large share **mirror Stockfish's #1 at every tier** (68.7%).

That is exactly the gap: prompting a frontier model with the same engine+Maia grounding does **not** reliably produce the leveled teaching-move behavior. So there is real room for a trained model to win on this behavior — and the `EVAL_AND_ITERATE` loop is set up to prove (or disprove) that our v2 does.

### The honest counter-finding: faithfulness

The gap above is about *move selection*. On *truthfulness* the result is the **opposite**, and we report it plainly: the frontier models fabricate a board fact in only **3.3%** of coaching outputs, while our **v1 tuned model fabricates in 51.3%** — the flat-truthfulness failure `RESULTS.md` flagged, now quantified against the frontier. So the frontier is *not* uniformly bad at coaching: it is weak at leveling the move but strong at not lying about the board.

That is why the pass bar in `EVAL_AND_ITERATE.md` requires v2 to win on **both** the move-selection gap **and** fabrication (≤ frontier). A v2 that levels the move but keeps v1's fabrication rate would be worse than the frontier where the frontier is already strong — and we would not ship it or call it a win.

### Where v1 stands on the gap today

Our v1 tuned model is currently **in the same weak band as the frontier** on move selection (tier-differentiation  22.0%, opportunity-subset findable-pick 15.4%, engine-mirror-at-every-tier  70.0%). So v1 has **not** yet won the gap either — consistent with `DIVERGENCE_REPORT.md` (differentiation weak and mis-directed). The gap is real and *open*: nobody in this comparison reliably does the behavior, which is exactly the room a targeted v2 data intervention (contrastive multi-tier + a tier-aware teacher rule + a faithfulness gate) is designed to claim.

## Appendix — example positions (beginner tier)

`gap>0` positions are where a frontier model DID steer the beginner to a more findable move; look for how rare they are vs. the mirror cases.

- `diZkf8wa_37` [opening/mistake] SF-best **Ne2** (Maia-rank 1) — gpt:dxe4(mr0) claude:dxe4(mr0) gemini:Ne2(mr1) v1:Be1(mr2)
- `T997RAgX_53` [middlegame/inaccuracy] SF-best **Kf1** (Maia-rank 1) — gpt:Kf1(mr1) claude:Kf1(mr1) gemini:Kf1(mr1) v1:Kf1(mr1)
- `QU2bnwCs_82` [endgame/mistake] SF-best **Nc5** (Maia-rank 1) — gpt:Ke6(mr0) claude:Ke6(mr0) gemini:Ke6(mr0) v1:Ke6(mr0)
- `bWwdtmJb_18` [opening/inaccuracy] SF-best **exd4** (Maia-rank 1) — gpt:exd4(mr1) claude:exd4(mr1) gemini:exd4(mr1) v1:exd4(mr1)
- `rKymuRxn_25` [opening/mistake] SF-best **a4** (Maia-rank 2) — gpt:Re1(mr0) claude:Re1(mr0) gemini:a4(mr2) v1:a4(mr2)
- `Fr1dRrTh_21` [opening/blunder] SF-best **Na4** (Maia-rank 2) — gpt:Be3(mr0) claude:Be3(mr0) gemini:Na4(mr2) v1:Be3(mr0)
- `oPLztjAs_37` [middlegame/mistake] SF-best **Bc4** (Maia-rank 1) — gpt:Bb5(mr0) claude:Bb5(mr0) gemini:Bc4(mr1) v1:Bc4(mr1)
- `X1nwiPgj_124` [endgame/inaccuracy] SF-best **f5** (Maia-rank 1) — gpt:Ra4(mr0) claude:Ra4(mr0) gemini:f5(mr1) v1:f5(mr1)
- `d3EkLSHP_31` [opening/mistake] SF-best **Nc3** (Maia-rank 2) — gpt:O-O(mr7) claude:O-O(mr7) gemini:O-O(mr7) v1:Nc3(mr2)
- `koGdNw9A_31` [middlegame/mistake] SF-best **Bb7** (Maia-rank 1) — gpt:Bb7(mr1) claude:Bb7(mr1) gemini:Bb7(mr1) v1:Bb7(mr1)
- `OxPFVrUV_50` [middlegame/blunder] SF-best **Rf5** (Maia-rank 1) — gpt:Rf5(mr1) claude:Rf5(mr1) gemini:Rf5(mr1) v1:Rd8(mr0)
- `ttG1GNDK_92` [endgame/inaccuracy] SF-best **Rxf5** (Maia-rank 7) — gpt:Rxf5(mr7) claude:Rxf5(mr7) gemini:Rxf5(mr7) v1:Rxf5(mr7)
