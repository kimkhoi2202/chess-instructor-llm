# Results — Base vs. Fine-tuned Chess Coach

**Headline:** Fine-tuning a small model (Qwen3-1.7B, 4-bit) on a hard-filtered,
level-calibrated coaching dataset **reliably delivers the product behavior**
(sound move, no engine jargon, tier-appropriate depth) — **but does not, on its
own, make the explanations *truthful*.** That gap is the thesis of this project,
and it is now measured, not asserted.

---

## Setup (honest, cross-family)

| Item | Value |
|---|---|
| Base model | `mlx-community/Qwen3-1.7B-4bit` (untuned) |
| Tuned model | `models/mlx/chess-coach-v1` (QLoRA SFT → merged → 4-bit MLX) |
| Judge | **`claude-group/claude-opus-4-8`** via TrueFoundry gateway — **cross-family** (teacher was GPT-5.5, so we do *not* grade with a GPT judge) |
| Scenarios | 15, built from `data/positions/positions.jsonl` — a **separate Lichess crawl** from the 8k-position training bank (held-out; minor overlap possible) |
| Same scenarios for both models | Yes (identical prompts; only the model differs) |
| Objective checks | Non-LLM: move legality/soundness (Stockfish pool), engine-speak regex, ply cap |

Reproduce:
```bash
set -a && source .env && set +a
python -m src.eval.evaluate --model base  --num-scenarios 18 \
  --positions data/positions/positions.jsonl --out data/eval/results_base_claude.json
python -m src.eval.evaluate --model tuned --tuned-path models/mlx/chess-coach-v1 \
  --num-scenarios 18 --positions data/positions/positions.jsonl \
  --compare-to data/eval/results_base_claude.json --out data/eval/results_tuned_claude.json
```

---

## Headline table

### Objective checks (% passing — non-LLM, deterministic)

| Check | Base | Tuned | Δ |
|---|---:|---:|---:|
| produced_nonempty | 100% | 100% | +0 |
| move_parseable | 100% | 100% | +0 |
| **move_sound** | 87% | **100%** | **+13** |
| **no_engine_speak** | 33% | **100%** | **+67** |
| **ply_cap_ok** | 67% | **100%** | **+33** |

### LLM-judge (Claude Opus, mean 0–2)

| Dimension | Base | Tuned | Δ |
|---|---:|---:|---:|
| spec_adherence | 0.47 | **0.93** | **+0.47** |
| level_calibration | 0.60 | **1.13** | **+0.53** |
| no_engine_speak | 0.87 | **1.87** | **+1.00** |
| **truthfulness** | 0.13 | 0.13 | **+0.00** ← flat |
| task_quality | 0.13 | 0.27 | +0.13 |

---

## What this means

**The fine-tune wins decisively on everything it *can* control by shaping the
training distribution:** it picks a sound move every time (100%), never leaks
engine numbers (33% → 100%), obeys the length cap, and adopts tier-appropriate
register. These are *style/format* behaviors — and they are exactly what
resisted prompting on the base model. The fine-tune is a **reliable last-mile
behavior compressor.**

**Truthfulness is the exception, and it is flat.** The judge's rationales are
consistent and damning about *why*:

> "b4 attacks the knight on c3 (not c2), there is no bishop on c1…"
> "the king is on g8 not e8, f1 holds a rook not a pawn…"
> "fabricates a nonexistent hanging bishop on g3 (the bishop is on d3)…"
> "Bxc5 captures a bishop (not a knight)…"

The tuned model **says the right *kind* of thing in the right *voice*, but
invents the board facts** — piece locations, hanging pieces, captures, tactics.
A 1.7B/4-bit model cannot reliably track 32 pieces from a FEN in-context, so it
confabulates the justification for an otherwise-sound move.

**Root cause is in the data, not just the model.** The training labels are
GPT-5.5 teacher outputs that were filtered for *format/soundness* but **not for
faithfulness** (the faithfulness gate was deferred). So the model faithfully
learned the teacher's style *and* the teacher's occasional fabrication. **The
truthfulness ceiling was set by the unfiltered training distribution.**

---

## The fix (v2 — this is the project's real thesis)

Dependability on truth cannot come from more fine-tuning. It comes from
**grounding + a non-LLM verifier**:

1. **Faithfulness filter on the dataset** — before training, reject any teacher
   candidate whose explanation references a piece/square/capture that doesn't
   exist in the position (deterministic check against `python-chess`), plus a
   cross-family LLM faithfulness pass. This removes the fabrication habit from
   the labels.
2. **Inference-time verifier in the product** — every claim the coach makes
   ("the bishop on d3 is hanging", "this attacks the knight on f6") is checked
   against the actual board and the engine before it reaches the student;
   unverifiable claims are stripped or the turn is regenerated.

**Prediction:** with (1)+(2), truthfulness rises sharply while the style deltas
above are preserved — because style was never the problem. That is the
falsifiable claim the eval harness is built to test.

---

## Bottom line for the brief

- The target behavior **genuinely resists prompting** (base is weak across the
  board) → the fine-tune is *justified*, not decorative.
- The fine-tune **reliably delivers the controllable behavior** (sound, calibrated,
  jargon-free) with large, cross-family-judged deltas.
- The **one dimension it can't fix by data-shaping alone (truthfulness) is
  measured, explained, and has a concrete architectural remedy** — which is the
  spiky, defensible claim of the accompanying BrainLift.
