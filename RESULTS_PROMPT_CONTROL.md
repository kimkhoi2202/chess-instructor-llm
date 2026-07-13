# MATCHED same-backend, spec-exact PROMPT CONTROL (32B)

**Question.** Can a *well-prompted base* Qwen3-32B reproduce the tier-appropriate
move selection that the v4 QLoRA tune learned — when you remove the two confounds
in the earlier base comparison? The BrainLift currently records 32B
un-promptability as a **falsifiable hypothesis** precisely because the matched
control had never been run: the earlier base comparison used **aws-bedrock**, a
*different backend* than v4's Unsloth/Modal path (a confound the eval audit
flagged). This run turns the hypothesis into a **result**.

> **HEADLINE.** Spec-exact-prompted base (same backend as v4) tier-policy match =
> **0.428** (lightly prompt-optimized variant: **0.431**) vs base-default **0.347**
> and v4-tuned **0.767**. The exact rule + same grounding lifts the base only
> **+0.081** — closing just **~19%** of the base→tune gap — and extra prompt
> optimization does not help.
> **Verdict: the hypothesis HOLDS.** A well-prompted base *cannot* reproduce the
> tier-appropriate move selection at 32B on the matched backend; fine-tuning earns
> its place. (This is now a same-backend RESULT, not a hypothesis.)

---

## What this controls for (both confounds removed at once)

1. **Same backend as v4.** We serve the EXACT base v4 was trained on —
   `unsloth/Qwen3-32B-unsloth-bnb-4bit` — through the SAME Unsloth/Modal/A100 path
   as `src/eval/eval_modal_v4.py` / `eval_modal_v4_val.py`, with the **v4 LoRA
   adapter removed** (base weights only). Same tokenizer, same 4-bit quant, same
   greedy decoding (`do_sample=False`, `repetition_penalty=1.15`,
   `no_repeat_ngram_size=4`), same 512-token cap, same `enable_thinking=False`, and
   the same `_strip_think` + `_clean_lead` post-processing. The Modal image
   definition is byte-identical to the v4 eval so its cached layers (hence the
   resolved dependency versions) are reused — the base runs in the *same
   environment* as v4.

2. **A real, spec-exact prompt.** The system prompt is hand-authored to state the
   EXACT canonical selection algorithm from `src/teacher/tier_select.py` (see the
   verbatim text below). The **user** prompt is the IDENTICAL grounded prompt v4
   saw — `src.eval.benchmark.prompts.build_grounded_user` = verified position
   facts + the Stockfish sound pool with internal evals + the per-tier Maia
   human-likelihoods — so the base gets the *same grounded facts* v4 got, no more
   and no less. The ONLY thing changed vs the shipped coach is the system prompt.

**Data.** The 120 held-out VAL positions × 3 tiers = 360 scenarios
(`data/benchmark_honest/val_ids.txt` over `data/benchmark_gap803/scenarios.jsonl`).
Reusing VAL is legitimate because the prompt is *hand-authored*, not tuned on VAL
(no gradient, no selection, ever touches these positions).

**Scorer (identical to v4).** The recommended move is re-extracted from each raw
output with the same strict any-legal extractor the v4 report uses
(`src.eval.evaluate.extract_recommended_move` →
`src.teacher.coach_gate.pick_recommendation`) and compared to the preregistered
`canonical_uci` from `select_tier_move`; the tier-policy match is the mean of the
three per-tier exact-match rates — byte-for-byte the metric in
`scripts/reproduce_v4.py` / `report_v4.json` (`per_model.*.tier_fit`). So the
number is directly comparable to base `q3_32b` (0.3472) and `ours_v4` (0.7667).

**Ceiling note (honest).** The grounded prompt lists Maia only for the top-6 human
moves, while `canonical_uci` is computed from the full per-pool-move policy — so a
pool move's human-likelihood can be missing from what the model sees. Both v4 and
this prompted base face that identical truncation, so the comparison is fair; v4's
0.767 shows the grounding is rich enough to get most of the way, and the
deterministic `select_tier_move` rule over the *full* policy is ~1.0 by
construction (the named ceiling).

---

## The exact spec-exact system prompt (variant `spec_exact`, verbatim)

```text
You are a chess coach doing move review for a student at a stated rating tier (beginner, intermediate, or advanced).

For each position you are given, in the user message:
- the move the student played;
- a list of engine-SOUND candidate moves, listed best-first, each with an internal evaluation in centipawns (cp) from the side-to-move's point of view (higher cp = better for the mover);
- for THIS tier, how likely a human at that level is to play each move, under "Human-likelihood at this tier (Maia)", given as percentages.

Recommend exactly ONE move, chosen by the following EXACT rule. Do not substitute your own judgement; apply the rule mechanically.

Step 1 - Restrict to the sound list. Only the moves in the "Engine-sound candidate moves" list are eligible. Never recommend a move outside it.

Step 2 - Normalize two signals ACROSS the sound candidates:
- eval_norm: rescale the candidates' centipawn evaluations to the range 0..1, where the highest-cp sound move = 1 and the lowest-cp sound move = 0. (If every candidate has the same cp, set every eval_norm = 1.)
- human_norm: rescale the candidates' human-likelihood percentages to the range 0..1, where the most human-likely sound move = 1 and the least human-likely = 0. If a sound move has no percentage listed, treat its percentage as 0. (If every listed percentage is equal, set every human_norm = 1.)

Step 3 - Blend with the tier weight w on the human term:
- beginner:     w = 1.0
- intermediate: w = 0.5
- advanced:     w = 0.0
score(move) = (1 - w) * eval_norm + w * human_norm

Step 4 - Pick the sound move with the HIGHEST score. Break ties in this EXACT order: (1) higher score; then (2) higher raw centipawn evaluation; then (3) it appears earlier in the best-first sound list.

Equivalently, the rule collapses per tier to:
- ADVANCED (w=0): the engine-best sound move = the FIRST move in the best-first sound list.
- BEGINNER (w=1): the sound move with the HIGHEST human-likelihood percentage (the most findable sound move for a human at this level) - often NOT the engine's top move.
- INTERMEDIATE (w=0.5): the sound move that best balances engine strength and human-likelihood under the 50/50 blended score above.

Then write your reply in exactly this shape: begin with `I'd play <MOVE>.` where <MOVE> is the chosen move in standard algebraic notation; give 2-4 sentences of encouraging coaching tied to the student's actual mistake and a concrete plan; and end with one line `Takeaway: <one transferable sentence>.` Use the centipawn numbers and percentages ONLY to make the selection - never quote them, and never write "engine", "Stockfish", or "computer" in your reply.
```

A second, lightly **prompt-optimized** variant (`spec_exact_opt`) states the same
rule but surfaces the per-tier shortcut first (advanced = first listed; beginner =
argmax human-likelihood among sound moves; intermediate = 50/50 blend). It is the
optional "prompt-optimized" arm; the hand-authored `spec_exact` prompt above is the
key control. Its full text is in `scripts/prompt_control_32b.py`.

---

## Results — tier-policy exact match (vs `canonical_uci`, mean over tiers)

120 held-out VAL positions × 3 tiers = 360 scenarios per prompt; 360/360 scored.

| Model / prompt | backend | overall | beginner | intermediate | advanced | move-sound | distinct↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Base default (`q3_32b`, shipped coach prompt) | aws-bedrock (prior) | **0.347** | 0.300 | 0.383 | 0.358 | 1.000 | 0.29 |
| **Base + spec-exact prompt (`spec_exact`)** | **Unsloth/Modal (same as v4)** | **0.428** | 0.383 | 0.433 | 0.467 | 0.800 | 0.34 |
| Base + spec-exact-optimized (`spec_exact_opt`) | Unsloth/Modal (same as v4) | 0.431 | 0.358 | 0.350 | 0.583 | 0.717 | 0.30 |
| **v4 tuned (`ours_v4`)** | Unsloth/Modal | **0.767** | 0.725 | 0.733 | 0.842 | 0.942 | 0.73 |

Per-tier match counts (of 120 each): `spec_exact` = B 46, I 52, A 56; `spec_exact_opt` = B 43, I 42, A 70.

> The base-default 0.347 row is the published `q3_32b` number (`report_v4.json`),
> generated with the shipped coach system prompt. This run adds the two rows in the
> middle: the SAME base weights on the SAME Unsloth/Modal backend as v4, differing
> from the shipped coach only by the spec-exact system prompt. The earlier
> cross-backend base comparison the eval audit flagged is exactly what these
> same-backend rows replace.

**Gap closed.** `(0.428 − 0.347) / (0.767 − 0.347) = 0.081 / 0.420 ≈ 19%`. The
optimized variant closes ~20% — statistically indistinguishable. Most of the
base→tune gap survives the exact rule.

**Format / soundness diagnostics (why soundness DROPS for the prompted base).**
When asked to *compute* the tier rule, the base fails to name a parseable move at
all on **67/360 (18.6%)** of `spec_exact` scenarios (97/360 = 26.9% for
`spec_exact_opt`), and only **39%** of `spec_exact` replies even begin with the
required `I'd play <MOVE>.` The soundness drop (1.00 → 0.80) is therefore driven by
these *format* failures, not by picking bad moves — of the moves it *does* name,
only 5/360 fall outside the sound pool. The shipped coach reads move-sound 1.00
precisely because the simpler prompt lets the base default to a plausible/engine
move; the harder tier computation degrades even the format floor the tune nails
(v4 well-formed ≈ 0.96, 1.00 after the gate).

---

## Interpretation

**The hypothesis holds — now as a same-backend result.** With both confounds
removed (same Unsloth/Modal backend as v4 *and* a real, spec-exact prompt that
hands the model the exact `select_tier_move` algorithm plus the same grounded
facts v4 saw), the base Qwen3-32B reaches **0.428** tier-policy match — far below
the v4 tune's **0.767**. The exact rule lifts the base only **+0.081** over the
shipped-prompt base (0.347), closing **~19%** of the base→tune gap. A lightly
prompt-optimized variant lands at **0.431** (~20%) — no meaningful improvement — so
this is not a wording problem you can engineer away. The earlier cross-backend
comparison the eval audit flagged is not what drove the deficit: on the matched
backend the base still cannot reproduce the behavior.

**It IS partially promptable — honestly.** The prompt is not inert: stating the
rule moves the base up on all three tiers, and the optimized variant's blunt
"advanced = the first sound move" pushes advanced from 0.358 → 0.583. So a base
32B given the exact rule can execute *parts* of it *some* of the time. But it
plateaus well short of the tune, and "optimizing" the prompt mostly *reshuffles*
where it succeeds (advanced up, the mid-tiers down) rather than raising the
aggregate — the signature of a reliability/capability gap, not a prompt gap.

**Where the base breaks (all genuine, not scorer artifacts):**
- **Commitment/format.** On 19–27% of scenarios the base names no parseable move
  at all (rambles, hedges, or buries the pick), and only ~39% obey the required
  `I'd play <MOVE>.` opening. Reliable in-format commitment to one move is itself
  something the tune supplies.
- **Executing even the trivial rule.** For advanced the rule is just "take the
  first (engine-best) sound move," yet the base tops out at 0.47–0.58 — versus
  0.842 for the tune and ~0.925 for perfect execution of that rule (measured on a
  synthetic oracle). The base does not reliably do the easy thing.
- **Intersecting "human" with "sound."** The core beginner skill is "the most
  human-likely move *that is also sound*." The base repeatedly grabs the
  globally-most-human move instead — e.g. on `vaLVwTHK_77` it played **a6** (the
  42% Maia move, **not in the sound pool**) where the canonical pick is **Ne6+**
  (the highest-Maia *sound* move, 15%). Stating the restriction in the prompt did
  not make the base honor it.

**What fine-tuning buys at 32B.** Reliable, in-format selection of the
tier-appropriate *sound* move, per position — a behavior that does not transfer by
handing the base the exact rule in context on the matched setup. That is the moat
the tune earns.

**Honest caveats.**
- This tests the base reproducing the behavior *as a coach on the matched eval*.
  It does **not** claim the canonical move is uncomputable in general: an external
  scaffold (enable thinking, inject the full per-move policy, or just run
  `select_tier_move` as a tool) would of course get ~1.0 — but that is
  re-implementing the rule around the model, not the base behaving as the coach.
  The trained model internalizes the selection with none of that scaffolding.
- The grounded prompt shows Maia for only the top-6 human moves while
  `canonical_uci` uses the full per-pool-move policy — a real ceiling, but a
  *shared* one: v4 sees the same truncated grounding and still reaches 0.767, so
  the truncation does not explain the base's 0.43.
- VAL reuse is legitimate here because the prompt is hand-authored and never tuned
  on VAL; there is no train/eval leakage to inflate the base.

**Bottom line for the BrainLift.** The "un-promptability at 32B" claim survived its
own falsification attempt on the matched backend. It should move from a FALSIFIABLE
HYPOTHESIS to a RESULT: spec-exact prompting of the same base on the same backend
recovers <20% of the tune's gain (0.347 → 0.428 vs 0.767).

---

## Reproduce

```bash
# build prompts + generate (base only, both variants) on Modal, then download + score:
MODAL_PROFILE=chess-instructor-3 modal run scripts/prompt_control_32b.py --block
# or score an already-generated run:
MODAL_PROFILE=chess-instructor-3 modal run scripts/prompt_control_32b.py --score-only
```

Artifacts: `data/prompt_control_32b/prompts.jsonl` (the 720 prompts),
`data/prompt_control_32b/gen.jsonl` (raw generations), `data/prompt_control_32b/scores.json`
(the numbers above).

- **Backend/decoding:** `unsloth/Qwen3-32B-unsloth-bnb-4bit` (base, no adapter),
  A100-80GB, greedy (`do_sample=False`, `repetition_penalty=1.15`,
  `no_repeat_ngram_size=4`), 512 new tokens, `enable_thinking=False`,
  `_strip_think` + `_clean_lead` — identical to `eval_modal_v4*.py`.
