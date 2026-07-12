# PRELIMINARY — OURS-v4 (32B) vs OURS-4B, deterministic axes

> Partial run: **100/120** VAL positions have all three OURS-v4 tiers (the 32B eval was paused to preserve chess-instructor credit). These are the FREE deterministic axes only — no council spend. OURS-v4 = RAW drafts (ungated); OURS-4B = fully gated shipped pipeline. Instructiveness (blinded frontier council) + the final showcase come with the full resume on the new workspace.

## Deterministic regression: did the 32B (v4) regress vs the 4B?

| axis | OURS-v4 (32B) | OURS-4B | Δ (v4−4b) | better |
|---|---:|---:|---:|:--:|
| tier_fit_mean | 0.670 | 0.377 | 0.293 | higher |
| distinct_rate | 0.787 | 0.272 | 0.515 | higher |
| move_sound | 0.917 | 1.000 | -0.083 | higher |
| no_engine_speak | 0.980 | 1.000 | -0.020 | higher |
| well_formed | 0.957 | 1.000 | -0.043 | higher |
| flat_rate | 0.180 | 0.660 | -0.480 | lower |

- tier-fit by tier — OURS-v4: {'beginner': 0.65, 'intermediate': 0.65, 'advanced': 0.71}; OURS-4B: {'beginner': 0.36, 'intermediate': 0.41, 'advanced': 0.36}.
- untuned Qwen3-4B base (`base_4b`) tier-fit 0.323, distinct 0.235 (reference).

## vs-frontier + distinct-tier signal (deterministic)

Of 100 complete positions, OURS-v4 gives distinct, sound, correctly-graded per-tier moves on **52**; also diverges from the best frontier move on **47**. On that set (objective tier-fit+soundness vs best frontier): **30 wins / 5 losses / 12 ties**.

