# Eval rubric — LLM-as-judge (base vs tuned)

Score each model output (base AND tuned) on the SAME held-out positions. Judge
with a DIFFERENT model family than the teacher that generated the data. Report the
delta. This rubric mirrors the teacher spec — one spec, three jobs.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Spec adherence** (picks a sound move from the pool, explained at-tier) | recommends an unsound move OR wrong-tier reasoning | partially | fully embodies the spec |
| **Level calibration** (idea fits the tier; simpler for lower tiers) | tier-inappropriate | wobbles | consistently tier-appropriate |
| **No engine-speak** (behavioral check the spec forbids) | leaks centipawns/eval/engine words/over-deep lines | minor slip | clean, human language only |
| **Truthfulness** (tactics cited exist; matches analysis) | fabricates a threat/line | minor imprecision | fully faithful to the analysis |
| **Task quality** (genuinely useful coaching + takeaway) | wrong/useless | acceptable | genuinely instructive |

## Automated behavioral checks (run before the judge; cheap, deterministic)
1. **Engine-speak regex** — reject/flag any of: digits with "cp"/"+"/"-" evals,
   "eval", "engine", "stockfish", "computer", "#\\d", "centipawn", decimals like
   "1.3".
2. **Soundness** — recommended move must be in the sound pool (not a blunder).
3. **Ply-cap** — no narrated line longer than the tier cap.
4. **Format** — valid JSON with all required fields; non-empty takeaway.

## Required outputs
- Mean score per dimension, **base vs tuned**, on the same held-out scenarios.
- % passing each automated check, base vs tuned.
- A short error-analysis paragraph: where does the tuned model still fail, and is
  it a **data** problem (the lever), not a hyperparameter one?

**Win condition:** tuned beats base on *Spec adherence*, *Level calibration*, and
*No engine-speak*. That is "behavior from data", demonstrated in numbers.
