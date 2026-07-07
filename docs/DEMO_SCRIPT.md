# Demo Script — chess-instructor-llm (3–5 min)

A shot-by-shot script for the submission video. Target **~4:30** (hard cap 5:00). Each shot has
**ON SCREEN** (what the viewer sees / what to click) and **SAY** (the voiceover line, trim to taste).

> **✅ Frame every number as `v2` — the current shipped model.** v2 improved faithfulness
> (grounded fabrication 50% → 33%) and fixed tier-direction (27.5% → 39.2%); it still trails the
> frontier on raw instructiveness — it *narrows* the gap (council rank 4.13 → 3.68). Say the win is
> "reliable, local, ~$0, honest-about-its-gap coaching," not "beats GPT-5.5."

---

## Pre-flight (before you hit record)

- [ ] `cd chess-instructor-llm && ./run_platform.sh` → wait for `coach ready (tuned)` and the
      Next.js `Ready` line. Confirm the badge reads **"Tuned coach"** (not "Base model").
- [ ] Open **http://localhost:3000**; let the auto-run `2.Qh5?` reveal finish once so the model is warm.
- [ ] Pick your "tier-difference" position in advance from the **differentiated library**
      (`web/public/library_differentiated.json`, 41 positions where the move genuinely changes by
      tier) so the Beginner→Advanced toggle visibly changes the recommendation on camera.
- [ ] Open three tabs ready to show: `RESULTS_V2.md`, `RESULTS_BENCHMARK_v2.md`, and the HF Space
      ([`spaces/khoilamalphaai/chess-coach-benchmark`](https://huggingface.co/spaces/khoilamalphaai/chess-coach-benchmark)).
- [ ] (Optional) have `data/analysis/GAP_REPORT.md` open for the gap numbers.

---

## 0:00 – 0:30 — The hook: the gap

**ON SCREEN:** The Analysis Room with the `1.e4 e5 2.Qh5?` position already coached. Cursor rests on
the raw engine lines panel (centipawns + PVs) for a beat, then on the plain coaching text.

**SAY:**
> "Ask a 3000-Elo engine for the best move and it'll give you one — plus a wall of centipawns and a
> grandmaster-only line. That's *true*, but it's useless to a 1200. The move a beginner would
> actually *find* and *learn from* is usually a different move, explained a different way. We
> measured whether frontier models get this right, and they mostly don't — same engine grounding,
> and GPT-5.5, Claude, and Gemini recommend the *same* move regardless of rating about 78% of the
> time. That gap is what this project fills — with a fine-tuned 1.7B model that runs on my laptop."

---

## 0:30 – 2:00 — Live platform walkthrough

### 0:30 – 0:50 — The library + coach-a-move

**ON SCREEN:** Open the **position library** (the 45-position study set), click one position. Then on
the board, **drag a piece** to mark "the move I was unsure about" (a rust arrow appears; the FEN
doesn't change). Click **Coach**.

**SAY:**
> "You pick a position — or drop in your own — mark the move you weren't sure about, choose your
> tier, and ask for coaching. Watch what comes back."

### 0:50 – 1:15 — The reveal: one move, grounded + faithful

**ON SCREEN:** The coaching reveal animates in — the brass **recommended-move arrow**, two–four
sentences of plain coaching, and one **Takeaway** line. Point to the fact that there are **no
centipawns and no jargon** in the coaching text.

**SAY:**
> "One move, drawn as the loudest thing on the board, and a plain-language explanation — why it's
> good and how to think about finding it. No centipawns, no 'the engine says,' no ten-move line.
> And every concrete claim here — every piece, square, and capture — was checked against the real
> board before it reached me. If the model invents a fact, the backend catches it and re-generates;
> if it can't get a clean answer, it falls back to an explanation that's true by construction."

### 1:15 – 1:35 — Engine lines (show the grounding)

**ON SCREEN:** Expand the **engine lines / analysis rail** (chess.com-style sound pool + PVs + your
move's severity). Show the recommended move *is* one of the sound moves.

**SAY:**
> "The engine truth is still here for anyone who wants to verify — the sound-move pool, the lines,
> and how bad my move was. The coach is grounded in exactly this. It never picks an unsound move; it
> just chooses the most *instructive* sound one and hides the numbers."

### 1:35 – 2:00 — Tier differences (the core behavior)

**ON SCREEN:** On the pre-chosen differentiated position, toggle **Beginner → Intermediate →
Advanced**. The recommended move and/or the explanation depth visibly change (e.g., beginner gets
the more human-findable move / simpler idea; advanced gets the sharper line).

**SAY:**
> "Here's the whole point. Same position, three ratings. For the beginner it steers toward the move
> a beginner would actually find and keeps the idea simple. Bump it to advanced and the pick sharpens
> and the explanation goes deeper. Same board, different lesson — that's the level-calibration
> behavior we trained in."

---

## 2:00 – 3:00 — The proof: base-vs-tuned + the frontier benchmark

### 2:00 – 2:30 — Base vs. tuned (why the fine-tune is justified)

**ON SCREEN:** Switch to `RESULTS_V2.md`. Highlight the objective + judge deltas.

**SAY:**
> "Does the fine-tune actually do anything, or could you just prompt the base model? We measured it,
> cross-family — GPT-5.5 wrote the training data, so a *Claude* judge grades, no grading your own
> homework. This is **v2**, our current model. The base 1.7B leaks engine-speak two times out of
> three; the tuned model, never — 33% to 100%. Move-soundness 87 to 100. Level-calibration and
> spec-adherence roughly double. And the truthfulness line that was *flat* in v1 finally moves — v2
> nails the style and starts biting into faithfulness too."

### 2:30 – 3:00 — The 5-model benchmark + the honest gap

**ON SCREEN:** Switch to `RESULTS_BENCHMARK_v2.md`. Point to the fabrication row and the council-rank
row (ours vs. frontier), then the cost row.

**SAY:**
> "We also ran a blinded, cross-family council — five models, grounded and ungrounded, ranked by
> instructiveness. Two honest results. One: grounding is what moves truth — our fabrication rate
> drops from 99% to 33% once the position is grounded, and v2 cut that grounded rate from v1's 50
> down to 33. Two: the frontier still *out-teaches* us — it ranks higher on instructiveness even
> grounded, though v2 narrowed the gap from 4.13 to 3.68. Our edge isn't being smarter; the whole
> benchmark cost about $24 to run and our model was **$0** — local, private, offline."

---

## 3:00 – 3:45 — The results dashboard (HF)

**ON SCREEN:** Open the HF Space
([`spaces/khoilamalphaai/chess-coach-benchmark`](https://huggingface.co/spaces/khoilamalphaai/chess-coach-benchmark)).
Scroll the grid; hover the ours-vs-frontier comparison. Briefly show the model and dataset repos
([model](https://huggingface.co/khoilamalphaai/qwen3-1.7b-chess-coach-mlx) ·
[dataset](https://huggingface.co/datasets/khoilamalphaai/chess-coach-benchmark)).

**SAY:**
> "Everything's public. The benchmark dataset — 100 held-out positions, five models, both
> conditions, plus the blinded council rankings — is on the Hub, with a dashboard to explore it, and
> the tuned model is published too. All of this is v2, reproducible with one command."

---

## 3:45 – 4:30 — The honest arc: the gap v2 closed, and the one it didn't

**ON SCREEN:** Back to `RESULTS_V2.md`, cursor on the **v1→v2 delta table** (fabrication 50 → 33,
tier-differentiation 27.5 → 39.2, council 4.13 → 3.68).

**SAY:**
> "The most important part isn't a single win — it's the honest arc. Fine-tuning fixed style
> immediately, but in v1 it did *not* fix truthfulness: a 1.7B can't track 32 pieces from a FEN, and
> we'd filtered the training data for format but not faithfulness. So v2 was a *data* fix — a
> faithfulness filter that produced zero false labels, a strongly tier-aware teacher rule, and
> contrastive same-position-different-tier pairs that were zero percent of v1. It worked: grounded
> fabrication dropped from 50 to 33 percent, and tier-differentiation went from 27 to 39 percent with
> the direction *corrected* — beginners now get the move a beginner would actually find. It still
> doesn't out-teach GPT-5.5, and it isn't meant to. The thesis holds: dependability comes from the
> engine, the grounding, and a non-LLM verifier — the fine-tune is the last-mile compressor for voice
> and form factor. A reliable, local, honest coach."

**ON SCREEN (close):** The Analysis Room, tuned coaching reveal on screen.

**SAY:**
> "A reliable, level-calibrated chess coach — grounded, honest about its gap, and running locally.
> Thanks for watching."

---

## Timing cheat-sheet

| Segment | Window | Beat |
|---|---|---|
| Hook — the gap | 0:00–0:30 | 3000-Elo "best" fails a 1200; frontier ~78% same move across tiers |
| Platform | 0:30–2:00 | library → coach-a-move → reveal → engine lines → **tier toggle** |
| Proof | 2:00–3:00 | base→tuned deltas; benchmark fabrication + council + $0 cost |
| Dashboard | 3:00–3:45 | HF Space + model + dataset, all public |
| The v2 arc | 3:45–4:30 | v2 deltas (fabrication 50→33, tier-dir 27→39) → verifier thesis → standing win |

## Key numbers to say out loud (all `v2` — current)

- Frontier same-move-across-tiers ≈ **78%** (tier-differentiation only **22.7%**); engine-mirror at
  every tier **68.7%**; beginner findable-pick on the opportunity subset **20.5%**. *(the motivating
  frontier gap — a measurement of the big models, unchanged by our retrain)*
- Base → tuned (v2): no-engine-speak **33% → 100%**; move-soundness **87% → 100%**; level-calibration
  **0.60 → 1.13**; truthfulness **no longer flat — 0.13 → 0.20**.
- v1 → v2: grounded fabrication **50% → 33%**; tier-differentiation **27.5% → 39.2%** (direction
  corrected); council rank **4.13 → 3.68**; top-1 instructiveness **2% → 8%**.
- Benchmark (v2): our fabrication **99% → 33%** with grounding vs. frontier ≈ **3%**; council
  instructiveness ours **3.68** vs. frontier ≈ **2.1** (1 = best of 5); total run cost ≈ **$24**,
  our model **$0**.

## Fallback / gotchas

- If **Maia** is unavailable on your machine, the human-likelihood panel shows "unavailable" — the
  coach still runs; just don't dwell on that panel.
- Pick the tier-difference position from `library_differentiated.json` **in advance** — differentiation
  happens on ~39% of positions in v2 (up from ~25% in v1, and now correctly directed), so a random
  position may still give the same move at every tier.
- If a coaching reveal shows a "re-generated / verified" note, that's the faithfulness gate working —
  you can point it out as a feature, not a bug.
- Keep the engine-lines panel collapsed until the "grounding" beat so the reveal stays clean.
