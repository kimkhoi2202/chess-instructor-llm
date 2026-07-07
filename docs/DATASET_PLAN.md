# Dataset Expansion & Quality Plan — chess-instructor-llm

> Mantra: **"training the model is easy; the dataset is the most important deliverable."**
> This is a research/recommendation doc. Nothing here has been executed — every action is a
> proposed next step with a concrete command. It does not touch the running platform/backend.

Date: 2026-07-06 · Author: dataset-research pass

---

## 0. TL;DR

- **Volume is nearly a solved problem; quality is not.** The format/soundness gates already
  pass ~99.9% of teacher output (1 reject in 1,449). The one metric that is **flat is
  truthfulness (0.13 → 0.13)**, and its root cause is a *missing faithfulness filter*, not a
  small dataset. **The highest-leverage work is the v2 faithfulness filter, not "more rows."**
- **But there is free, in-hand growth**: ~683 already-generated candidates are not yet in the
  dataset. Re-filtering + re-splitting lifts the training set from **1,376 → ~2,030 rows
  (+47%) with zero new generation or API spend.**
- **Real coverage gaps exist** and are measured: concrete **tactics** (skewer/discovered/
  double-attack/deflection/decoy all <1%) and **endgame technique** (K+P opposition ≈ 0.1%)
  are thin, and **0%** of positions are coached at more than one tier (so the core
  "same position, different tier → different lesson" behavior isn't in the data as pairs).
- **Transcripts are few by throttle, not by failure**: only 3 playlists are hardcoded and the
  harvest ran a "polite" 3-videos-each pass. Captions succeeded on 100% of attempted videos.

---

## 1. Current state (exact counts)

| File | Rows | Notes |
|---|---:|---|
| `data/positions/positions_v1.jsonl` | **8,000** | tiers 2,667 / 2,667 / 2,666 (perfectly balanced) |
| `data/generated/candidates_v1.jsonl` | **2,132** | beginner 804 / intermediate 705 / advanced 623 |
| `data/generated/candidates_v1_snap.jsonl` | 1,449 | 14:45 snapshot the current dataset was built from |
| `data/dataset/train.jsonl` | **1,376** | beginner 520 / intermediate 455 / advanced 401 |
| `data/dataset/valid.jsonl` | **72** | beginner 24 / intermediate 26 / advanced 22 |
| `data/generated/candidates.jsonl` | 3 | tiny smoke file — ignore |

**Filter yield is essentially 100%.** `train + valid = 1,448` came from the 1,449-row snapshot
→ exactly **1 reject** (an `engine_speak` leak). The hard gates (soundness, no-engine-speak,
ply-cap, validity, dedup) barely bite anymore: **the teacher output is already clean on
everything the current filter checks.** There is no faithfulness gate, which is the whole point
of `RESULTS.md`.

**Generation headroom (from `_genv1.log`):** 2,132 positions coached + **3,897 skipped as
"severity none"** (the played move was fine → nothing to coach) = 6,029 / 8,000 positions
processed. So **only ~35% of real human positions are coachable**, and the existing 8k bank
tops out at roughly **~2,800 candidates**. Growing much past that needs *new* positions or
*multi-tier reuse* (see §4b).

---

## 2. Why transcripts are "only a few" (concrete reason)

Nine cleaned transcripts exist, from **exactly 3 hardcoded playlists** (3 videos each):

| Playlist (hardcoded in `src/ingest/transcripts.py`) | Videos | Level |
|---|---:|---|
| GothamChess – Win At Chess | 3 | beginner |
| Naroditsky – Beginner to Master Speedrun | 3 | beginner→intermediate |
| Naroditsky – Master Class Speedrun | 3 | advanced |

Root cause is **throttle + hardcoding, not a technical limit**:

1. **Only 3 playlists** are in the `PLAYLISTS` constant.
2. The harvest was run as a **"polite first pass" with ~3 videos/playlist** (the CLI default is
   actually 15). Captions were available for **100% of attempted videos** — 9 raw video-ids →
   9 clean transcripts, **zero caption failures**.
3. The distiller (`distill_principles.py`) further caps sampling at `--per-tier 3` /
   `--max-transcripts 9` (`principles.md` header says "Distilled from 8 transcript(s)").
4. Minor: `manifest.json` is **stale** — it lists only 3 of the 9 transcripts. It doesn't
   matter for correctness because the distiller globs `clean/*/*.txt` directly, but it should
   be regenerated.

**Do we need more?** Transcripts only shape the **teacher's coaching STYLE** (distilled ONCE
into `prompts/principles.md` + `prompts/fewshots.json`); they are never SFT rows. So we don't
need *many* — we need **more diverse coaches across all three levels**. A modest expansion
(more playlists by level, limit raised to ~12, re-distill) is a cheap quality win, not a
volume play.

---

## 3. Q1 — "Should we get more dataset?"

**Yes, but targeted — and volume is the *least* important lever.** Evidence:

- The brief calls for "hundreds to a few thousand" for **one** behavior on a **1.7B** model.
  We're at 1,448 usable rows with ~683 more already generated and unused. **~3,000–5,000
  high-quality, well-covered rows is more than enough**; past ~5k is diminishing returns for a
  single narrow behavior.
- Tiers and severities are **already well balanced** (see below), so "more of the same" adds
  little. The value is in **coverage** and **faithfulness**, not raw count.

### Coverage audit (measured over the 2,132 candidates)

| Dimension | Distribution | Verdict |
|---|---|---|
| Tier | beginner 38% / intermediate 33% / advanced 29% | ✅ balanced |
| Severity | inaccuracy 39% / mistake 38% / blunder 23% | ✅ balanced |
| Phase | middlegame 71% / endgame 20% / **opening 10%** | ⚠️ opening thin (by sampler design: skips first 16 plies) |
| Teaching-move divergence | **32%** recommend a NON-#1 sound move | ✅ core behavior present |
| Strategic concepts | king-safety 701, initiative 367, tempo 363, activity 358, prophylaxis 351 | ✅ rich |
| **Concrete tactics** | fork 4.4%, pin 4.1%, back-rank 1.9%, **skewer 0.2%, discovered 0.6%, double-attack 0.3%, deflection 0.6%, decoy 0.05%** | ❌ **thin** |
| **Endgame technique** | passed-pawn 5.4%, outpost 1.9%, **opposition 0.1%, weak/isolated-pawn 0.2%** | ❌ **thin** |
| Multi-tier reuse | **0** FENs coached at >1 tier (2,132 unique FENs = 2,132 rows) | ❌ **contrastive calibration pairs absent** |

### Q1 recommendation (in priority order)

1. **Harvest what's already generated (free):** re-filter `candidates_v1.jsonl` (2,132) →
   split. **1,376 → ~2,030 train rows, no spend.**
2. **Finish generation** on the remaining ~2k unprocessed positions (resume) → ~2,800 candidates.
3. **Rebalance for coverage, not count:** inject **tactical-motif** positions (fork/pin/
   skewer/discovered/back-rank) and **endgame-technique** positions (K+P opposition, rook
   endings, passed pawns). These are the measured holes.
4. **Add multi-tier contrastive pairs** (same FEN at all 3 tiers) — directly trains the
   calibration behavior and is a cheap ~2–3× multiplier on chosen positions.
5. **Then stop chasing volume** and spend the effort on §5 (faithfulness) — the actual ceiling.

**Target:** ~3,000–5,000 rows, balanced on tier × phase × severity × **motif**, **faithfulness-
filtered**. A smaller faithfulness-filtered set will beat a larger unfiltered one.

---

## 4. Q2 — Transcripts + free/open sources (by level, with harvest methods)

Two distinct data types come out of these sources:
- **STYLE** → feeds the teacher's `principles.md` / `fewshots.json` (paraphrased, internal, distilled once).
- **ANNOTATED POSITIONS / EXTRA POSITIONS** → new FENs (± human commentary) to widen coverage
  and, later, to ground the faithfulness verifier.

### A) YouTube transcripts — STYLE only (distilled, never verbatim)

The existing `transcripts.py` already does the hard part (yt-dlp playlist enumeration, caption
download, VTT→plaintext, tier parsing). Just broaden `PLAYLISTS` and raise the limit.

| Level | Add these channels / playlists | Why |
|---|---|---|
| Beginner | GothamChess guides; **John Bartholomew – Chess Fundamentals / Climbing the Rating Ladder**; Chess Vibes; Naroditsky Beginner-speedrun (already have) | slow, first-principles, plain language |
| Intermediate | **Hanging Pawns** (structured strategy/openings, unusually clear diction → clean captions); Naroditsky mid-speedrun; ChessNetwork (Jerry) | plan-based, calm explanation |
| Advanced | **Saint Louis Chess Club** GM lectures (Finegold/Seirawan/Yasser); Naroditsky Master Class (have); Power Play (Daniel King) | deeper prophylaxis/structure vocabulary |

- **Harvest:** add `Playlist(...)` entries → `python src/ingest/transcripts.py --per-playlist-limit 12`
  then re-run the distiller. Code already prefers human `en` captions over ASR `en-orig`.
- **Licensing/ToS:** personal/research use; the pipeline distills to **paraphrase** and never
  ships verbatim quotes (design invariant in `README.md`). Do not redistribute transcripts.
- **Yield:** 6–10 playlists × 10–25 videos ≈ 60–150 transcripts. Note the distiller only
  *samples* ~9–12, so **diversity across levels/coaches matters more than volume.**

### B) Lichess Studies + Broadcasts — ANNOTATED positions (new, high-value)

Real coach/human annotations tied to specific positions (FEN + prose comment). Endpoints
(verified 2026):

- Whole study: `GET https://lichess.org/api/study/{studyId}.pgn?comments=true&variations=true`
- One chapter: `GET /api/study/{studyId}/{chapterId}.pgn`
- All of a user's studies: `GET /api/study/by/{username}/export.pgn`
- Broadcast (all rounds): `GET /api/broadcast/{tournamentId}.pgn` · one round: `.../round/{roundId}.pgn`
- Official broadcast index: `GET /api/broadcast` (find tournament/round ids)
- Discover annotated studies: `lichess.org/study/search?q=annotated`

- **What it gives:** extra positions AND human explanations (for grounding / contrast).
- **Licensing:** Lichess is free/open; respect rate limits (reuse the `LICHESS_TOKEN` +
  polite-throttle pattern already in `lichess_sampler.py`). Attribute studies if surfaced.
- **Yield:** thousands of positions, but **quality varies** — many "studies" are bare engine
  lines. Filter for chapters with real prose comments (`{ ... }` blocks with words, not evals).
- **Harvest:** `requests` → `.pgn`, parse with `python-chess` (already a dep), walk the game and
  emit `(fen, comment)` at each commented ply.

### C) Public-domain / free annotated PGN databases — EXTRA positions (± annotations)

| Source | Gives | License | Use |
|---|---|---|---|
| **PGN Mentor** (`pgnmentor.com/files.html`, updated Jan 2026) | huge PGN by player/opening/event (mostly unannotated) | free | **position source** to diversify beyond Lichess blitz (classical, tactical, endgame) |
| Annotated collections (Path-to-Chess-Mastery; chessgames.com collections) | 950+ annotated games, 30k+ comment blocks | mixed — check each | human-commentary positions; verbatim only if PD/CC |
| **TCEC archive** (GitHub) / **Stockfish `fishtest_pgns`** (HF dataset) | engine-annotated games (eval/depth comments) | TCEC = CC BY-SA 3.0 | **positions only** — strip engine comments (they're engine-speak); also useful as ground-truth evals for the faithfulness verifier |

- **Note on copyright:** chess *moves/facts* aren't copyrightable, but annotation **prose** can
  be → use modern prose only from PD/CC sources; use everything else as **positions only**.
- **Harvest:** download PGN → `python-chess` → feed FENs into the existing
  Stockfish→Maia→teacher pipeline (they become normal candidates).

### D) Public-domain chess books — STYLE + annotated illustrative games

| Book | Where | Gives |
|---|---|---|
| Capablanca, *Chess Fundamentals* | Project Gutenberg #33870 (PD) | principles + 14 annotated illustrative games |
| Lasker, *Manual of Chess* | Internet Archive (PD mark) | principles + models |
| Tarrasch, *The Game of Chess*; Lasker, *Common Sense in Chess* | Gutenberg (PD) | principles |

- **Use:** feed the prose into `distill_principles.py` alongside transcripts (timeless,
  jargon-light pedagogy); extract embedded games as extra positions. Style is archaic → prefer
  **distilling paraphrased principles** over quoting.
- **Licensing:** public domain — safe to use verbatim, but paraphrase for voice consistency.

### E) Chessable — ⚠️ do NOT scrape

Free tier is now thin (Short & Sweet moved to PRO; only "community courses" stay free), and the
**Terms of Service explicitly prohibit scrapers/spiders/crawlers.** Treat Chessable as **manual
human inspiration only** — not a harvest target.

---

## 5. Prioritized plan

All commands run from the repo root with the pinned interpreter
(`/Users/khoilam/.venvs/mlx/bin/python`) and, where the teacher/Lichess is involved,
`set -a && source .env && set +a` first. **These are recommendations — run when ready.**

### (a) Quick wins — broaden the transcript/principle corpus

1. Add playlists by level to `PLAYLISTS` in `src/ingest/transcripts.py` (see §4A table).
2. Re-harvest with a higher limit + regenerate the stale manifest:
   ```bash
   python src/ingest/transcripts.py --per-playlist-limit 12
   ```
3. Re-distill principles + few-shots across more transcripts/levels:
   ```bash
   set -a && source .env && set +a
   python src/teacher/distill_principles.py --max-transcripts 12 --per-tier 4
   ```
4. (Optional) Add 1–2 public-domain books' text as an extra distiller input for breadth.

*Effect:* richer, multi-coach, multi-level `principles.md` → better teacher STYLE. Low cost,
one-time.

### (b) Grow + rebalance the SFT candidate set

1. **Free, in-hand (+47%, no spend):** re-filter the full candidate file, then re-split.
   ```bash
   python src/filter/filter.py --candidates data/generated/candidates_v1.jsonl \
       --train-out data/dataset/train.jsonl --rejects-out data/generated/rejects_v1.jsonl
   python src/train/split_data.py                      # 1,376 -> ~2,030 train rows
   ```
2. **Finish generation** (resumes automatically by skipping done ids):
   ```bash
   set -a && source .env && set +a
   python -m src.teacher.generate --positions data/positions/positions_v1.jsonl \
       --out data/generated/candidates_v1.jsonl --concurrency 6
   ```
3. **Targeted coverage crawls** (fill the measured tactics + endgame holes):
   - Tactics: pull motif-labeled positions from the **Lichess puzzle DB / API** (themes
     `fork,pin,skewer,discoveredAttack,backRankMate,deflection`) or extract forcing moments
     from **PGN Mentor** games; run them through the existing pipeline.
   - Endgames: sample low-piece FENs (K+P opposition, rook endings) from PGN Mentor / Lichess.
   - Practically: add a small "coverage" sampler mode or a motif filter over new PGNs; keep the
     same `positions_*.jsonl` schema so `generate.py` consumes them unchanged.
4. **Multi-tier contrastive pairs** (trains calibration directly; cheap ~3×): transform chosen
   positions into 3 rows sharing the FEN but with distinct `id` + `tier`, then regenerate. This
   is the single cheapest way to exercise "same position → simpler for Beginner than Advanced."
5. **Target:** ~3,000–5,000 rows, balanced on tier × phase × severity × **motif**.

### (c) Highest-leverage upgrade — faithfulness-filtered v2 (do this before chasing volume)

This is the project's real thesis (`RESULTS.md`: truthfulness flat at **0.13**, because labels
were filtered for format/soundness but **not faithfulness**).

1. **Deterministic faithfulness gate** (new gate in `src/filter/filter.py`): parse the coaching
   text with `python-chess` and reject any candidate that references a piece/square/capture/
   tactic that doesn't exist in the FEN (e.g. "the bishop on d3 is hanging" when d3 is empty).
   This removes the fabrication habit from the labels.
2. **Cross-family LLM faithfulness pass** (Claude, since the teacher is GPT-5.5 — no grading own
   homework): flag remaining candidates whose justification isn't supported by the board +
   engine analysis; drop or regenerate them.
3. Re-split → train v2 → re-run the base-vs-tuned eval and check the truthfulness delta.

*Expected:* truthfulness rises sharply while the style deltas (sound move, no engine-speak,
calibration) are preserved — the falsifiable claim the eval harness exists to test.

---

## 6. Do-not-touch / caveats

- Do **not** scrape Chessable (ToS).
- Keep transcripts **paraphrased/internal**; the SFT dataset stays 100% synthetic.
- Engine-annotated PGNs (TCEC/fishtest) are **positions only** — their eval comments are exactly
  the engine-speak the coach must never emit.
- Another agent owns the platform/backend; the commands above are dataset-pipeline only.
