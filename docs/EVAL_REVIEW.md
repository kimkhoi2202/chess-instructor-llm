# Eval Review (talking points)

A tight, honest brief for the live eval review. It answers the three questions in order:
what the eval suite measures, how the model performed, and what we would do better next time.
Every number traces to a committed results doc, named inline.

---

## 1. Behavior spec (one sentence)

The single trained behavior is: given a chess position and a stated learner tier
(Beginner, Intermediate, Advanced), select the tier-appropriate, instructive SOUND move
and attach a short principle tag; the prose explanation is an optional display layer, not
the trained target.

- Per tier the rule collapses to: Beginner = the most human-findable sound move,
  Intermediate = a 50/50 engine/human blend, Advanced = the engine-best sound move
  (`RESULTS_PROMPT_CONTROL.md`, `RESULTS_FULL_EVAL_803.md` section 2).
- It passes the litmus that a matched-backend prompt cannot reproduce it. A spec-exact
  system prompt handed to the SAME Qwen3-32B base on the SAME Unsloth/Modal backend, with
  the same grounding, reaches tier-policy match 0.428 vs the v4 tune's 0.767, closing only
  about 19 percent of the base-to-tune gap; extra prompt optimization does not help
  (`RESULTS_PROMPT_CONTROL.md`). Fine-tuning earns its place.

Honest framing carried on every number below: "tier-policy match" is agreement with our
own preregistered `select_tier_move` rule (learnability), not validated pedagogy, and the
same grounding also feeds a ~20-line rule that computes the target move at about 1.0 by
construction, so the model approximates a policy the product already produces
(`RESULTS_HONEST_EVAL_V4.md`, `BRAINLIFT.md`).

## 2. The eval suite: what it measures and how it is built

Deterministic, un-gameable metrics (python-chess over engine-verified labels; no LLM grader):

- tier-policy exact match: pick equals the preregistered `select_tier_move` canonical move,
  averaged over the three tiers.
- distinct-moves-per-level: on positions whose canonical Beginner and Advanced moves differ,
  the share where the model's Beginner and Advanced picks also differ.
- move-soundness: pick lands in the Stockfish sound pool.
- format / names-a-move: reply names a legal move and closes with a "Takeaway:" line.
- gated fabrication: zero verifier-detectable mechanical violations. This is true by
  construction because the shared extractor accepts only a LEGAL move, so an illegal or
  fabricated move is never counted (`RESULTS_STAGE4_CORRECTED.md`).

Scoring uses one vendored extractor (`src/eval/evaluate.py:extract_recommended_move`), the
same one `scripts/reproduce_v4.py` asserts against, so scores are recomputed from raw prose,
not from stored fields.

Blinded cross-family LLM council (secondary, for instructiveness): GPT-5.5 + Claude Opus 4.8
+ Gemini 3.1 Pro via the TrueFoundry gateway, ranking a blinded field per item with bootstrap
95 percent CIs and self-preference correction (`RESULTS_FULL_EVAL_803.md` section 3,
`RESULTS_STAGE4_CORRECTED.md`).

Design and coverage:

- Held-out corrected benchmark: the v6 deep-verified labels, scored on 120 held-out TEST
  positions x 3 tiers = 360 scenarios. The TEST FENs are stable across the v6 rebuild
  (0 FEN changes; only the labels re-derived), so the re-score is comparable to the shipped
  v4 headline (`RESULTS_STAGE4_CORRECTED.md`).
- Base-vs-tuned design in one controlled session: the base is loaded once, adapters are
  swapped, and every condition uses identical greedy decode (`do_sample=False`,
  `repetition_penalty=1.15`, `no_repeat_ngram_size=4`) (`RESULTS_STAGE4_CORRECTED.md`).
- Adversarial / robustness set: 54 cases across five attack categories, run on two tracks
  (live deployed endpoint and raw base-vs-v4) (`RESULTS_ADVERSARIAL.md`).
- Matched prompt-control: same backend, spec-exact prompt, base weights only
  (`RESULTS_PROMPT_CONTROL.md`).
- Reproducible: `scripts/reproduce_v4.py` re-scores the committed generations against the
  committed labels with only python-chess, no GPU and no network; eval code lives in
  `src/eval/` and the drivers `scripts/honest_eval.py`, `scripts/reproduce_v4.py`.

## 3. How the model performed

Headline table (grounded, corrected v6 labels, 120 held-out TEST; `RESULTS_STAGE4_CORRECTED.md`):

| Model (grounded) | tier-policy match | B / I / A | move-sound | distinct |
|---|---:|---|---:|---:|
| BASE (Qwen3-32B untuned) | 0.428 | 0.442 / 0.408 / 0.433 | 0.969 | 0.303 |
| OURS-v4 (shipped) | 0.861 | 0.858 / 0.750 / 0.975 | 0.983 | 0.987 |
| OURS-v6-dpo | 0.881 | 0.858 / 0.808 / 0.975 | 0.983 | 0.987 |
| OURS-v6-dpo2 | 0.892 | 0.858 / 0.842 / 0.975 | 0.986 | 0.987 |

Headlines:

- Base-vs-tuned is the load-bearing result: tuned minus base is +0.433 (v4) and +0.453
  (v6-dpo) on tier-policy, and +0.684 on distinct-moves (0.987 vs 0.303); the gap survives
  the label correction (`RESULTS_STAGE4_CORRECTED.md`).
- The tune ladder improves the moat: v4 0.861 to v6-dpo2 0.892 (+0.031)
  (`RESULTS_STAGE4_CORRECTED.md`).
- Unbiased head-to-head vs the best frontier: 56-24-12 over the 92 diverging positions
  (56-24-40 over all 120), recomputed and asserted by `scripts/reproduce_v4.py`
  (`RESULTS_HONEST_EVAL_V4.md`).
- OURS tops the field on tier-appropriate selection: on the corrected 803-field re-score the
  best OURS model leads at 0.509, +0.042 over the best frontier (Claude 0.467), and family
  averages hold OURS > frontier > open (`RESULTS_FULL_EVAL_803.md`).
- Robustness held: the deployed v4 held 52 of 54 adversarial cases with 0 broke after the one
  malformed-FEN gap was fixed; the fine-tune's clear win is injection resistance (raw base
  broke on 4 of 12 injections, raw and deployed v4 on 0) (`RESULTS_ADVERSARIAL.md`).
- Prompt-control confirms fine-tuning was necessary: matched 32B prompted base 0.428 vs v4
  0.767 (`RESULTS_PROMPT_CONTROL.md`).
- Instructiveness (council, secondary): the three frontier coaches lead on prose; the DPO
  tune does not regress instructiveness vs v4 (overlapping CIs) (`RESULTS_STAGE4_CORRECTED.md`).

Honest caveats (say these out loud):

- v4 traded prose for the moat: its blinded instructiveness grade is 4.67, below the 4B tune
  (5.32) and v3 (6.35), and about 40 percent of raw drafts fail the faithfulness check before
  the gate (`RESULTS_HONEST_EVAL_V4.md`).
- The v6-dpo2 gain is intermediate-only: Intermediate 0.750 to 0.842, while Beginner (0.858)
  and Advanced (0.975) are byte-identical to v4 (`RESULTS_STAGE4_CORRECTED.md`).
- Advanced equals engine-best by rule, so it is already near ceiling and is the distillation's
  weakest tier (0.217 without grounding) (`RESULTS_STAGE4_CORRECTED.md`).
- The small model is the spec-honest form factor: the 1.7B tune leads learnability at 0.578
  (from base 0.358), above every frontier, and beats the 4B tune (0.397), so the behavior
  tracks the data, not capacity; the 32B was the quality push (`RESULTS_HONEST_EVAL_V4.md`).
- v5 was confounded and under-optimized: it regressed to 0.536 under several tangled changes
  at once (about 27 percent less optimization, broken contrastive triads, about 42 percent
  boilerplate-principle pollution, retrained from base not v4, no checkpoint selection), so it
  isolates nothing (`BRAINLIFT.md`).
- Deployment-necessity is false as built: `select_tier_move` computes the canonical move at
  about 1.0 from the same grounding, so the model approximates it rather than being required
  (`RESULTS_HONEST_EVAL_V4.md`).

## 4. What we would do better next time

- Build the held-out eval and a data-quality gate FIRST. The v6 rebuild found the labels were
  off late: about 28.9 percent of old "sound" moves were removed (4701/16257) and about 45.2
  percent of canonical tier targets moved (1089/2409) under deeper Stockfish-17 plus Syzygy,
  and the "advanced equals engine-best" divergence hit about 43 of 803 positions. Catching
  this earlier would have saved a re-eval cycle (`RESULTS_FULL_EVAL_803.md`,
  `RESULTS_HONEST_EVAL_V4.md`).
- Run the matched prompt-control from day 1. The 32B un-promptability claim sat as a
  hypothesis for most of the project and was only settled at the end for about 4.71 dollars;
  it should gate the thesis from the start (`RESULTS_PROMPT_CONTROL.md`).
- Avoid the v5 under-optimization mistake: change one variable at a time, select checkpoints
  on a dev set, and do not guess epochs (`BRAINLIFT.md`).
- Manage council cost earlier: the instructiveness council was 62.27 dollars of the 112.15
  dollar 803 eval, so scope the blinded panel and reuse generations deliberately
  (`RESULTS_FULL_EVAL_803.md`).
- Ship the spec-honest small model as the headline and treat the 32B as the stretch, since
  learnability is led by the small model and is the honest form factor
  (`RESULTS_HONEST_EVAL_V4.md`).
- Tighten the prose/faithfulness tradeoff: write the explanation with engine templates or a
  frontier writer on top of the trained move, and close the format gap (v4 format is the one
  sub-metric that dips, a 256-token-cap artifact, not a move or soundness regression)
  (`RESULTS_STAGE4_CORRECTED.md`, `BRAINLIFT.md`).

---

Reproduce the deterministic headline (no GPU, no network): `python -m scripts.reproduce_v4`.
