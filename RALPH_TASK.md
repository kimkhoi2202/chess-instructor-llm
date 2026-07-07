---
task: Ship the full "Train Your Own Small Learning Model" spec — a reliable, level-calibrated chess-coaching model — end to end, to submission quality, today.
completion_criteria:
  - Dataset published on Hugging Face Hub (the real artifact)
  - Tuned model on Hugging Face Hub + a running local inference demo
  - Eval harness + results table: base vs tuned on the behavior metric
  - BrainLift: behavior thesis + evidence (data->behavior held?)
  - 3-5 min demo video (USER records; agent provides runnable demo + shot script)
  - WIN CONDITION: tuned beats base on Spec-adherence, Level-calibration, No-engine-speak
deadline: today (treat as final Sunday checkpoint)
---

## What we're building
Fine-tune Qwen3-1.7B (QLoRA) into a chess coach that, given a position + the
student's rating tier + engine analysis (Stockfish sound pool + Maia human moves),
recommends the most INSTRUCTIVE move for that tier and explains it in plain human
terms — never leaking engine internals (centipawns, deep lines), never fabricating
tactics — CONSISTENTLY, where a prompted base model drifts.

## Locked design decisions
- Engine-in-the-loop at inference: Stockfish + Maia run first; the model does the
  behavior (select teaching move + level the explanation). It is NOT a standalone
  chess player.
- Teacher = GPT-5.5 (reasoning high), grounded in engine analysis. Discarded after
  data-gen. Base = Qwen3-1.7B. Only the small model is trained.
- Transcripts (Naroditsky/GothamChess) distilled once into principles injected into
  the teacher prompt; dataset is 100% synthetic.
- Tiers: beginner 1000-1200 / intermediate 1300-1600 / advanced 1700-2000.
- Train on Modal (Unsloth QLoRA) or local MLX-LoRA (fallback). Demo runs locally (MLX).

## Constraints
- Fix disappointing models in DATA, not hyperparameters.
- Eval before/independent of training; cross-family judge for the headline.
- Ask the user only on genuine blockers (HF token for publish; recording the video).
