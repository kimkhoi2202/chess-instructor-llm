# Lichess Puzzles vs. Engine — Objective-Best or Findable-Move?

Date: 2026-07-06 · Probe for the `Lichess/chess-puzzles` ingest decision

**Question.** Do Lichess puzzle *solutions* give the OBJECTIVE BEST move (≈ Stockfish), or the move most SUITABLE / FINDABLE for a player at that ELO? This decides whether the puzzle answer can be used as a recommended-move label in our tier-appropriate coaching pipeline.

## Method

- **Sample:** 150 puzzles from `Lichess/chess-puzzles`, balanced across rating buckets ~800, ~1200, ~1600, ~2000, ~2400 (±100 each). Puzzle `Rating` is *difficulty*, not player Elo.
- **Puzzle position:** `board = chess.Board(FEN); board.push(Moves[0])` (the opponent setup); the graded **solution is `Moves[1]`** on that position.
- **Stockfish** (`stockfish_engine`): MultiPV-8, 500ms, sound pool = within 150cp of best and not a blunder (>= 250cp loss). `is_sf_best` = solution == engine #1.
- **Maia** (`maia_engine.human_moves`): full per-move policy at **beginner `maia-1100`** and at the **net matched to the puzzle rating** (<1300→1100, 1300-1699→1500, ≥1700→1900). `policy` = P(a human at that tier plays the move). 'findable' = solution is Maia's #1 **or** policy ≥ 0.25; 'hard-to-find' = policy < 0.05.

## 1. Stockfish alignment — is the solution the objective best?

| Rating bucket | n | mean rating | solution == SF best | in SF sound pool | median cp-loss (solution) | mean pool size | mate puzzles |
|---|---|---|---|---|---|---|---|
| ~800 | 30 | 810 | 96.7% | 100.0% | 0 | 1.0 | 73% |
| ~1200 | 30 | 1177 | 100.0% | 100.0% | 0 | 1.0 | 43% |
| ~1600 | 30 | 1616 | 100.0% | 100.0% | 0 | 1.1 | 23% |
| ~2000 | 30 | 1986 | 100.0% | 100.0% | 0 | 1.0 | 3% |
| ~2400 | 30 | 2391 | 100.0% | 100.0% | 0 | 1.0 | 0% |
| **ALL** | 150 | 1596 | 99.3% | 100.0% | 0 | 1.0 | 29% |

**Read-out.** The solution is Stockfish's #1 move in **99.3%** of puzzles and is in our sound pool in **100%** — median cp-loss **0**. The mean sound-pool size is **1.03** and **98%** of positions have exactly ONE sound move: puzzles are, by construction, single-best-move ('only-move') positions.
The only non-match (1/150, 1 a dual mate) is a *tie for best*: puzzle `00DPQ` (r765) plays **Qxh2#** while Stockfish prefers the equally-winning **Rxh2#** (cp-loss 0, still in pool).

## 2. Findability — how human-likely is the solution?

### 2a. At a BEGINNER (`maia-1100`)

| Rating bucket | n | mean policy | median policy | solution == Maia top | findable (top or ≥0.25) | hard-to-find (<0.05) |
|---|---|---|---|---|---|---|
| ~800 | 30 | 0.489 | 0.456 | 80.0% | 86.7% | 0.0% |
| ~1200 | 30 | 0.393 | 0.352 | 76.7% | 83.3% | 3.3% |
| ~1600 | 30 | 0.282 | 0.234 | 50.0% | 56.7% | 10.0% |
| ~2000 | 30 | 0.260 | 0.195 | 46.7% | 46.7% | 20.0% |
| ~2400 | 30 | 0.178 | 0.136 | 20.0% | 26.7% | 30.0% |
| **ALL** | 150 | 0.320 | 0.281 | 54.7% | 60.0% | 12.7% |

### 2b. At the tier MATCHED to the puzzle rating

| Rating bucket | n | matched net | mean policy | median policy | solution == Maia top | findable (top or ≥0.25) | hard-to-find (<0.05) |
|---|---|---|---|---|---|---|---|
| ~800 | 30 | 1100 | 0.489 | 0.456 | 80.0% | 86.7% | 0.0% |
| ~1200 | 30 | 1100 | 0.393 | 0.352 | 76.7% | 83.3% | 3.3% |
| ~1600 | 30 | 1500 | 0.337 | 0.264 | 56.7% | 60.0% | 10.0% |
| ~2000 | 30 | 1900 | 0.284 | 0.192 | 43.3% | 46.7% | 30.0% |
| ~2400 | 30 | 1900 | 0.252 | 0.186 | 40.0% | 46.7% | 10.0% |
| **ALL** | 150 | mixed | 0.351 | 0.303 | 59.3% | 64.7% | 10.7% |

## 3. Beginner-findability distribution (solution policy @ `maia-1100`)

| Rating bucket | >=0.50 | 0.25-0.50 | 0.10-0.25 | 0.05-0.10 | <0.05 |
|---|---|---|---|---|---|
| ~800 | 14 (47%) | 11 (37%) | 5 (17%) | 0 (0%) | 0 (0%) |
| ~1200 | 9 (30%) | 14 (47%) | 6 (20%) | 0 (0%) | 1 (3%) |
| ~1600 | 5 (17%) | 10 (33%) | 9 (30%) | 3 (10%) | 3 (10%) |
| ~2000 | 5 (17%) | 7 (23%) | 8 (27%) | 4 (13%) | 6 (20%) |
| ~2400 | 1 (3%) | 5 (17%) | 15 (50%) | 0 (0%) | 9 (30%) |
| **ALL** | 34 (23%) | 47 (31%) | 43 (29%) | 7 (5%) | 19 (13%) |

## 4. Concrete examples — objective-best but hard to find

Hard-to-find best moves (lowest beginner Maia policy) — the objective best a beginner would essentially never play:

| PuzzleId | rating | solution | Maia-1100 policy (rank) | themes |
|---|---|---|---|---|
| `008lc` | 1973 | Qxg7+ | 0.003 (rank 31) | attraction, crushing, exposedKing |
| `004kB` | 1209 | Qxf2+ | 0.003 (rank 33) | kingsideAttack, long, mate |
| `00Ksk` | 2355 | Qd3 | 0.011 (rank 16) | crushing, endgame, long |
| `008Sk` | 2007 | Rxf2 | 0.011 (rank 2) | crushing, endgame, long |

Naturally-findable solutions (an easy ~800 puzzle) — beginner Maia already plays it:

| PuzzleId | rating | solution | Maia-1100 policy | is Maia top | themes |
|---|---|---|---|---|---|
| `009L0` | 706 | Rxd1# | 0.866 | True | backRankMate, hangingPiece, mate |
| `00Bot` | 733 | Rxa6+ | 0.798 | True | crushing, endgame, rookEndgame |

## 5. Motif coverage in the sample (`Themes`)

Top themes across 150 puzzles: `middlegame` 73, `endgame` 71, `short` 69, `crushing` 65, `mate` 43, `long` 43, `advantage` 41, `mateIn1` 24, `oneMove` 24, `master` 20, `mateIn2` 15, `rookEndgame` 14, `veryLong` 14, `fork` 13, `defensiveMove` 11, `kingsideAttack` 10, `quietMove` 10, `advancedPawn` 9.

## 6. Verdict

**Q1 — Are puzzle solutions ≈ the objective best (Stockfish)?  YES.** The solution equals Stockfish's #1 move **99.3%** of the time and is inside our sound pool **100.0%** of the time (median solution cp-loss 0). A puzzle answer is an **engine-best / near-best** label, essentially by construction (Lichess mines puzzles as positions with one decisive best line).

**Q2 — Is it the tier-appropriate FINDABLE move, or a hard-to-find best?  It is the OBJECTIVE BEST, and for beginners it is often HARD TO FIND.** At `maia-1100`, only **60.0%** of solutions are naturally findable (Maia top or policy ≥ 0.25) and **12.7%** are low-policy (< 0.05) hard-to-find moves. Findability falls as puzzle rating rises (see §2) — the `Rating` really does encode 'how hard to find', which is a Maia-low, engine-high signal, i.e. the *opposite* of a 'what a 1000 would play' label.

**Q3 — How should puzzles be used in our pipeline?**

- ✅ **Use puzzles for MOTIF COVERAGE + as positions.** Filter by `Themes` to fill our measured motif holes (fork/pin/skewer/discovered/deflection/back-rank/endgame) and bucket by `Rating` as a difficulty tier. Push `Moves[0]` to get the position to coach.
- ✅ **Always re-ground through our own Stockfish + Maia + teacher.** Let the teacher pick the **tier-appropriate** move from our sound pool; synthesize the student's mistake (e.g. a high-Maia non-solution move) rather than assuming one.
- ❌ **Do NOT use `Moves[1]` as the recommended-move label for beginner/intermediate tiers.** It is the engine-best, frequently a low-Maia move a player at that tier would not find — using it as the label would re-teach 'always play the engine move' and undo tier differentiation.
- ➕ **Advanced tier is the exception.** When the sharp tactic *is* the teaching point (higher-rated puzzles, `advanced`), the puzzle solution and our sound-pool best usually coincide, so the solution can legitimately be the recommended move there.

**Bottom line:** puzzles are excellent *fuel* (positions + motifs + a strong hint for the teacher's context), not a drop-in coaching label. This matches the pipeline's existing invariant (`docs/EXTERNAL_DATASETS.md`, `DIVERGENCE_REPORT.md`): external solutions are context, never labels; the recommended move is chosen by our tier-aware teacher over our SF+Maia grounding.

## 7. Deliverables

- Raw per-puzzle records: `data/analysis/puzzles_vs_engine.jsonl`
- Cached balanced sample: `data/analysis/puzzles_sample.jsonl`
- This report: `data/analysis/PUZZLES_REPORT.md`
