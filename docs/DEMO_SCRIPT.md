# Demo Script - chess-instructor-llm (3-5 min)

A recordable, shot-by-shot script for the submission video. Target runtime **~4:00** (range 3:45-4:20,
hard cap 5:00). Each beat has **SHOT** (what to film / click), **ON-SCREEN TEXT** (lower-third or
slide callouts), and **SAY** (the voiceover, trim to taste).

The one thing this video proves: given a position and a player tier, the model picks the
**tier-appropriate instructive move**, the thing a base model does not do reliably, and the thing a
3000-Elo engine's "best move" is bad at because the best move is a bad teacher for a beginner. This is
framed as **behavior from data**, not "smarter than a frontier model."

Everything said here matches the reproducible facts. Keep it plain, confident, and honest. No hype.

---

## Pre-record checklist (do this before you hit record)

- [ ] **Warm the Modal endpoint.** The live Space (`chess-coach-studio`) is backed by a scale-to-zero
      Modal endpoint (`chess-coach-v4-4bit-maia`) with a **~2.5-3 min cold start**. Open the Space and
      send one coach request 5 minutes before recording so the box is hot. If you do not warm it, the
      first on-camera request will hang for minutes.
- [ ] **Have a fallback ready: the precomputed Showcase.** The Showcase renders real, precomputed
      per-tier coaching with **no backend call** (instant). If the endpoint is cold or flaky at record
      time, demo the tier toggle on the Showcase instead. It is the canonical, deterministic proof.
- [ ] **Pick the fork position in advance and copy its FEN.** Choose a genuine fork position from the
      Showcase (Showcase positions are precomputed to differentiate by tier, so the move is guaranteed
      to change on camera). Paste it here so you can reload it fast:
      `FORK_FEN = ____________________________________`
      Do not assert specific per-tier moves in narration beyond what is visibly on screen.
- [ ] Have these tabs open: the **Space** (warmed), the **Showcase** (fallback), and one slide or file
      with the **base-vs-tuned numbers** (from `RESULTS_HONEST_EVAL_V4.md`) for the proof beat.
- [ ] Sanity check: toggle Beginner -> Advanced once on your chosen position and confirm the
      recommended move actually changes before you record.

---

## Beat 1 - Hook: the gap (~20s)

**SHOT:** A single position on a clean board. Show a 3000-Elo engine readout for a beat (best move plus
centipawns and a deep line), then cut to a confused-beginner framing (or just let the jargon sit there).

**ON-SCREEN TEXT:** `The engine's best move is a bad teacher for a 1200.`

**SAY:**
> "Ask a 3000-Elo engine for the best move and it will give you one, plus a wall of centipawns and a
> grandmaster line. That is true, and it is useless to a beginner. The move a 1200 would actually find
> and learn from is usually a different move. Frontier models do not fix this: given the same position
> at three different ratings, they hand back the same move about three times out of four. That gap is
> what this project fills."

---

## Beat 2 - The behavior and the Behavior Spec (~30s)

**SHOT:** A simple spec slide: INPUT -> OUTPUT, then the three pass/fail checks listed.

**ON-SCREEN TEXT:**
`INPUT: position + tier (Beginner ~1000-1200 / Intermediate ~1300-1600 / Advanced ~1700-2000)`
`OUTPUT: the tier-appropriate move + a short principle (e.g. "Nf3, develop toward the center")`
`GRADED pass/fail: (1) sound  (2) matches the canonical tier move  (3) distinct across levels`
`Metric: tier-policy exact match = agreement with our select_tier_move rule`

**SAY:**
> "Here is the exact behavior, written down before training. Input: a position and the student's rating
> tier. Output: one move, the instructive move for that tier, tagged with a short principle. It is
> graded pass or fail on three things a stranger can check with no opinion in the loop. Is the move
> sound. Does it match the canonical move our rule designates for that tier. And is it distinct across
> levels, so a beginner and an advanced player are not handed the same move. We call the score
> tier-policy exact match: agreement with our own move rule. That is the whole target."

---

## Beat 3 - How the data was made (~40s)

**SHOT:** The data pipeline as a left-to-right flow. Animate each stage in as you say it.

**ON-SCREEN TEXT:**
`~6M raw positions (Lichess bank)  ->  ~6.8k curated training examples`
`Stockfish sound pool  ->  Maia (human-likely move per tier)  ->  deterministic tier rule  ->  GPT-5.5 explanation  ->  hard filter + faithfulness gate`

**SAY:**
> "The data is where the behavior comes from. We start from a raw bank of about six million Lichess
> positions. For each kept position, Stockfish gives the pool of sound moves, moves that are not
> blunders. Maia, a human-move model, ranks which of those a player at each tier would actually
> consider. A short deterministic rule then picks the canonical move for each tier from those two
> signals, and GPT-5.5 writes the plain explanation grounded in that analysis. Then a hard filter
> throws out anything unsound, jargon-heavy, or that fails a faithfulness check. What survives is about
> sixty-eight hundred curated, contrastive, same-position-different-tier examples. That is what we
> fine-tune on."

---

## Beat 4 - Live demo: the tier toggle changes the move (~60-90s)

**SHOT:** The live **chess-coach-studio** Space (warmed). Load your fork position. Set tier to
**Beginner**, coach it, let the recommended-move arrow and the short principle appear. Then toggle to
**Intermediate**, then **Advanced**, re-coaching each time. The point of the shot is the move visibly
**changing** between tiers on the same board. Zoom on the arrow and the principle line each time.

> SHOT NOTE: if the endpoint is cold or slow, cut to the **Showcase** and do the exact same toggle
> there. It is precomputed and instant, and it is the canonical deterministic proof of this behavior.

**ON-SCREEN TEXT:**
`Same position. Three ratings. The move changes.`
`(Beginner: human-findable move  ->  Advanced: sharper move)`

**SAY:**
> "Here is the model doing the thing. Same fork position, and I am the student. As a beginner, it steers
> me to the move a beginner would actually find, and keeps the idea simple. Watch the board as I change
> only my rating. Intermediate. Advanced. The recommended move changes, and the principle changes with
> it. Same board, three different lessons. A base model, and the frontier models, mostly give you one
> move here regardless of level. That per-tier change is the behavior we trained in, and it is the
> whole point."

---

## Beat 5 - The proof: base vs tuned (~45s)

**SHOT:** The numbers slide or `RESULTS_HONEST_EVAL_V4.md`. Lead with the base-vs-tuned rows. Show the
frontier and the ceiling as context, not as the headline.

**ON-SCREEN TEXT:**
`tier-policy exact match (120 held-out positions x 3 tiers, deterministic, no LLM judge)`
`1.7B on-spec:  base 0.358  ->  tuned 0.578   (#2 of 20, above every frontier)`
`32B shipped (v4):  base 0.347  ->  0.767 raw / 0.789 served`
`best frontier (Gemini 3.1 Pro): 0.553     deterministic rule (ceiling): ~1.0`

**SAY:**
> "Does the fine-tune actually do anything, or could you just prompt the base model? We measured it,
> deterministically, with grounding held identical on both sides and no model judge. The genuinely
> small on-spec model carries the result: the 1.7B tune goes from 0.36 to 0.58 on tier-policy match,
> second of a twenty-model field and above every frontier model. The shipped 32B model goes from 0.35
> to 0.77, and 0.79 as the demo actually serves it. And the 1.7B tune beats the 4B tune, so the lift
> comes from the data, not from size. Beating the frontier here is a bonus. The real win is that this
> behavior distills into a small model's weights, and it is reproducible with one command."

---

## Beat 6 - Honest framing and what is next (~20s)

**SHOT:** A single honesty slide, three short lines. Keep it calm, not a disclaimer dump.

**ON-SCREEN TEXT:**
`Honest by design`
`- Reliable GROUNDED execution: it uses Stockfish + Maia at inference. Not "the behavior is magically in the weights."`
`- The metric is agreement with our tier rule, not proven best teaching.`
`- Faithfulness = 0 verifier-detectable violations via a gate. Not "0% fabrication," not certified truth.`
`Next: deeper engine pool + tablebases, Maia-as-constraint, titled-coach validation, a grounding-free local test.`

**SAY:**
> "One honest line, because the framing matters. This is reliable grounded execution: the model still
> uses Stockfish and Maia at inference, so the behavior is not magically in the weights. The score is
> agreement with our own tier rule, not proof of better teaching, and our faithfulness number means
> zero violations a verifier can detect, not zero fabrication. A short deterministic rule already
> computes this move, so the model is the local executor, not the moat. Next up: a deeper engine pool,
> validation with titled coaches, and the fully-local test that would make the model load-bearing."

---

## Beat 7 - Close with the links (~15s)

**SHOT:** A links card. Hold it long enough to read. End on the warmed Space or the Showcase reveal.

**ON-SCREEN TEXT:**
`Model (32B v4):  huggingface.co/khoilamalphaai/chess-coach-32b-v4-qlora`
`Model (1.7B on-spec):  huggingface.co/khoilamalphaai/qwen3-1.7b-chess-coach-mlx`
`Dataset:  huggingface.co/datasets/khoilamalphaai/chess-coach-move-review`
`Demo (Space):  huggingface.co/spaces/khoilamalphaai/chess-coach-studio`
`Code:  github.com/Alpha-AI-Engineering-Khoi/chess-instructor-llm`

**SAY:**
> "Everything is public: the fine-tuned models, the dataset, the live demo, and the code, all
> reproducible. A level-calibrated chess coach whose behavior came from the data. Thanks for watching."

---

## Runtime estimate

| # | Beat | Window | Cumulative |
|---|---|---|---|
| 1 | Hook: the gap | ~20s | 0:20 |
| 2 | The behavior + Behavior Spec | ~30s | 0:50 |
| 3 | How the data was made | ~40s | 1:30 |
| 4 | Live demo: tier toggle changes the move | ~60-90s | 2:30-3:00 |
| 5 | The proof: base vs tuned | ~45s | 3:15-3:45 |
| 6 | Honest framing + what is next | ~20s | 3:35-4:05 |
| 7 | Close with the links | ~15s | 3:50-4:20 |

**Total: ~3:50-4:20** (comfortably inside the 3-5 min window; keep beat 4 near 60s if you are running long).

---

## Key numbers to say out loud (all match the reproducible eval)

- Frontier hands the same move to all three tiers about **77%** of the time (distinct-moves ~0.21-0.28).
- **1.7B on-spec** tier-policy match: **0.358 -> 0.578**, #2 of 20, above every frontier.
- **4B**: base 0.353 / prompt-base 0.378 / tuned 0.397 (tune > prompt > base; and 1.7B tune > 4B tune,
  so it is the data, not capacity).
- **32B shipped v4**: **0.347 -> 0.767** raw, **0.789** as served by the demo.
- Best frontier (Gemini 3.1 Pro): **0.553**. Deterministic `select_tier_move` rule: **~1.0** (the ceiling).
- v4 distinct-moves-per-level **0.730** (73 of 100 beginner!=advanced opportunities).
- Eval: 120 held-out positions x 3 tiers, deterministic, no LLM judge, zero train/test leakage, Maia
  symmetric across all 20 models, greedy decoding, re-scores exactly.

## Do-not-say list (keep it honest)

- Do not say "beats GPT-5.5 at chess" or "smarter than a frontier model." The claim is behavior from
  data, and a bounded lead on our own tier metric.
- Do not say "0% fabrication" or "guaranteed truthful." Say zero verifier-detectable violations.
- Do not claim the behavior "lives in the weights." It needs Maia grounding at inference; without Maia
  the tiers collapse to one move.
- Do not present the 51-5-6 head-to-head as a win rate. It is a selection-conditioned subset; leave it
  out of the video.
- Do not imply pedagogy is validated. We measured agreement with our rule, not student outcomes.
