# External Ready-Made Datasets — chess-instructor-llm

> Companion to `docs/DATASET_PLAN.md`. That doc covered **raw sources you harvest
> yourself** (YouTube transcripts, Lichess Studies/Broadcasts API, PGN Mentor, PD
> books). **This doc is only about already-packaged, downloadable datasets** on
> Hugging Face Hub and other platforms — what to reuse instead of re-harvesting.
>
> Research/recommendation only. **Nothing here has been downloaded or wired into the
> pipeline.** Every "how to use" is a proposed next step. Do not bulk-download the
> multi-TB sets; stream/filter a slice.

Date: 2026-07-06 · Author: external-dataset discovery pass

---

## 0. TL;DR

- **~14 datasets are genuinely useful to us**, out of dozens of chess datasets on the
  Hub (most are robot-arm/vision/engine-selfplay noise). They split into four jobs:
  **(A) tier-specific human positions**, **(B) themed puzzles for motif coverage**,
  **(C) engine evals for grounding shortcuts**, **(D) NL commentary for teacher style.**
- **The single most valuable dataset for us is [`Lichess/chess-puzzles`](https://huggingface.co/datasets/Lichess/chess-puzzles)**
  (CC0, 6.0M rows). It carries the two labels we lack together: a **`Rating`** (→ tier
  bucket) and **`Themes`** (→ the exact motifs measured <1% in our set: fork, pin,
  skewer, discoveredAttack, deflection, backRankMate, and endgame themes). It directly
  closes **both** our motif-coverage gap **and** the tier-specific-position gap.
- **Nobody has already built our deliverable.** There is *no* existing rating-calibrated,
  faithful, tier-appropriate *coaching* dataset. The closest move-explanation set,
  [`aicrowd/ChessExplained`](https://huggingface.co/datasets/aicrowd/ChessExplained),
  is full of vague, verdict-laden, sometimes fabricated explanations — a textbook example
  of the failure mode our faithfulness filter exists to kill. **This validates the thesis:
  the dataset is still the deliverable.** External data is *fuel* (positions + motifs +
  style), not a finished label set.
- **Everything external must be re-grounded through our own Stockfish (MultiPV-8 sound
  pool + our tolerances) + Maia (tier policy).** External FENs/solutions/evals are inputs
  to `generate.py`, never drop-in labels. Commentary is **STYLE only** (distilled,
  paraphrased, internal — never SFT rows), exactly like transcripts.

---

## 1. How external data maps onto our pipeline

Recall the pipeline and the target schema everything must become:

```
positions_*.jsonl  →  Stockfish (sound pool + mistake)  →  Maia (human likelihood)
   →  GPT-5.5 teacher  →  hard filter  →  train.jsonl
```

`positions_*.jsonl` row (the one integration contract — from `src/ingest/lichess_sampler.py`):

```json
{"id","fen","tier","played_move_uci","played_move_san","side_to_move",
 "mover_rating","game_id","ply","time_control"}
```

So a dataset is "useful" to the degree it cheaply yields **(fen, tier, played_move)** or
enriches **teacher style**. Four plug points:

| Plug point | What we need | Best external fuel |
|---|---|---|
| **Positions bank by tier** | real human FEN + the move a player *at that Elo* actually played, bucketed 1000–1200 / 1300–1600 / 1700–2000 | `Lichess/standard-chess-games` (bulk, Elo-filterable); raw Lichess games DB |
| **Motif coverage** | positions carrying fork/pin/skewer/discovered/deflection/back-rank + endgame technique, at a tier-appropriate difficulty | `Lichess/chess-puzzles` (+ `-with-games`) via `Themes` × `Rating` |
| **Grounding shortcut** | precomputed Stockfish eval to pre-screen "coachable" positions / fallback eval | `Lichess/chess-position-evaluations` |
| **Teacher STYLE** | human "how to explain a move" prose to distill into `principles.md` | ChessGPT (`Waterhorse/chess_data`), Jhamtani commentary, `Icannos/chess_studies` |
| **Eval (bonus)** | held-out motif/tactics understanding to measure base-vs-tuned | `wieeii/ChessQA-Benchmark` |

---

## 2. Ranked table

Tiers = usefulness to **our** narrow behavior (leveled move-review coaching), not dataset quality in general.

### Tier A — use these next

| # | Dataset · link | Contents / schema (verified) | Size | License | How it plugs in | Caveats |
|---|---|---|---|---|---|---|
| 1 | **[Lichess/chess-puzzles](https://huggingface.co/datasets/Lichess/chess-puzzles)** | `PuzzleId, GameId, FEN, Moves`(UCI solution), `Rating, RatingDeviation, Popularity, NbPlays, Themes[]`, `OpeningTags[]`. Themes seen: `fork, discoveredAttack, sacrifice, hangingPiece, backRankMate, rookEndgame, mateIn2, …` | 6.0M rows · 865 MB parquet | **CC0** | **Motif coverage + tier bucket in one file.** Filter `Themes` to our missing motifs; bucket by `Rating` into our 3 tiers; `push(FEN, Moves[0])` → the to-move position; re-ground with our SF+Maia. | Puzzle FEN is *before* the opponent's setup move — apply move 1 first. `Rating` is puzzle *difficulty* (a good tier proxy), not a player Elo. No "student blunder" — synthesize one (top Maia move ≠ solution). |
| 2 | **[Lichess/chess-puzzles-with-games](https://huggingface.co/datasets/Lichess/chess-puzzles-with-games)** | Everything in #1 **plus** full game context: `movetext`, `WhiteElo/BlackElo`, `White/BlackAcpl`, per-move `analysis[]`(json evals), `Opening`, `OpeningPly`, `speed`, `clock` (45 cols) | 3.0M rows · 18.4 GB | **CC0** | Richer sibling of #1: gives **real player Elo + move history + per-move eval** for each puzzle → fills our schema's `ply`, `mover_rating`, and `move_history_san` and enables true tier attribution + eval pre-screen. | Puzzle snapshot is Sep-2022. 18 GB — pull a themed/rated slice, don't grab whole. Marked WIP. |
| 3 | **[Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games)** | `Event, White, Black, Result, WhiteElo, BlackElo, WhiteRatingDiff, BlackRatingDiff, ECO, Opening, Termination, TimeControl, movetext`, dates/titles | 7.1B games · ~4.9 TB parquet | **CC0** | **Bulk, offline version of our `lichess_sampler.py`.** Predicate-pushdown filter rows by `WhiteElo/BlackElo` in-band → parse `movetext` with python-chess → reuse `extract_positions()` → our schema. Scales tier-specific positions far past the API crawl. | Enormous — stream via HF/DuckDB/polars, pull only in-band rows for the months you want. Same content as the raw monthly PGN.zst (see §5). |

### Tier B — grounding shortcuts + teacher style

| # | Dataset · link | Contents / schema | Size | License | How it plugs in | Caveats |
|---|---|---|---|---|---|---|
| 4 | **[Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations)** | `fen, line`(PV, UCI), `depth, knodes, cp, mate` | 945M rows · 41.4 GB | **CC0** | **Grounding shortcut.** Join by FEN to pre-screen "coachable" positions (clear best-vs-mistake gap) before spending SF time; or a fallback eval. | Not a MultiPV-8 *sound pool* and no Maia → **still re-ground** for the pool + human-likelihood. FEN is 4-field (no move clocks) → join on FEN prefix. |
| 5 | **[Waterhorse/chess_data](https://huggingface.co/datasets/Waterhorse/chess_data)** (ChessGPT, arXiv 2306.09200) | ChessCLIP annotated PGNs + ChessGPT base (game/language/mixed) + Chat (conversational NL) | ~15.6K downloads; multi-file | **Apache-2.0** (mixed provenance) | **Teacher STYLE**: NL move/plan commentary → distill into `principles.md`/`fewshots.json` (paraphrase). | Dataset-viewer is broken (cast error) — read repo files directly. Book/forum/blog subsets withheld for legal reasons. STYLE only, never verbatim. |
| 6 | **[harsh19/ChessCommentaryGeneration](https://github.com/harsh19/ChessCommentaryGeneration)** (Jhamtani et al., ACL 2018) | **298K move→commentary pairs across 11K games**, categorized (Description / Move-quality / Comparative / Planning-rationale / Contextual). From gameknot.com | ~298K pairs | code MIT; **data = gameknot provenance** | Academic gold for **teacher STYLE** and a reference for "how humans phrase a move rationale per category" (useful when tuning `principles.md`). | Old (py2.7, crawler-built). Provenance = forum users → **STYLE/paraphrase only**, do not redistribute or SFT verbatim. |
| 7 | **[Icannos/chess_studies](https://huggingface.co/datasets/Icannos/chess_studies)** | `text` = annotated PGN studies (top Lichess studies + angelfire smartbridge), 2 configs `lichess`/`others` | 6.1K rows · 6 MB | **CC0** | Cleanest-license **STYLE** source: parse `{comments}` from annotated PGN → extra pedagogy prose for distillation. | Small; quality varies (some bare lines). Parse for prose, drop engine-only comments. |

### Tier C — eval, format references, and use-with-caution

| # | Dataset · link | Contents / schema | Size | License | How it plugs in | Caveats |
|---|---|---|---|---|---|---|
| 8 | **[wieeii/ChessQA-Benchmark](https://huggingface.co/datasets/wieeii/ChessQA-Benchmark)** (Toronto CSSLab = Maia's lab; arXiv 2510.23948) | Expert QA, configs `motifs / short_tactics / structural / semantic / position_judgement`; `input, question, correct_answer, …` | 3.5K rows | **MIT** | **Eval complement**: measure whether tuning improved **motif/tactic understanding** (base-vs-tuned) on an independent, expert set — same lab that made the Maia we already use. | Eval only, not training. Multiple-choice/QA format ≠ our coaching format; use as a probe. |
| 9 | **[aicrowd/ChessExplained](https://huggingface.co/datasets/aicrowd/ChessExplained)** | `fen, move`(UCI), `explanation`(NL), `messages`(Qwen chat w/ `<think>`), `text` | 2.5M rows · 1 GB | **MIT** | Only as a **negative / contrastive** example or a chat-format scaffold. | ⚠️ Explanations are generic, template-y, contain **eval verdicts** ("Black is winning") and **fabricated motifs** — the exact anti-behavior our filter targets. **Do NOT use as labels.** |
| 10 | **[Thytu/ChessInstruct](https://huggingface.co/datasets/Thytu/ChessInstruct)** | `task, input, expected_output, KIND` (find-best-move / who-won / list-moves), derived from `laion/strategic_game_chess` | 100K rows | **CC-BY-4.0** | Instruction-format reference only. | Engine-selfplay derived, **not tier-calibrated, not human coaching**. Low value for our behavior. |
| 11 | **[Lichess/tournament-chess-games](https://huggingface.co/datasets/Lichess/tournament-chess-games)** | Broadcast/master games, game+movetext schema | 931K games | **CC-BY-SA-4.0** | Positions-only diversification (classical/master style) beyond blitz. | Players are **above** our 1000–2000 bands; SA license (attribution + share-alike). Positions only. |
| 12 | **[mateuszgrzyb/lichess-stockfish-normalized](https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized)** | `fen, depth, cp, mate` (deduped, max-depth per FEN) | 316M rows · 6.6 GB | **CC-BY-4.0** | Compact eval lookup (smaller than #4). | **Drops the `line`/PV** → less useful than #4 for us (we want the line). Value-net oriented. |
| 13 | **Kaggle chess sets** (category) — e.g. [350k-chess-positions-analyzed](https://www.kaggle.com/datasets/ffatty/350k-chess-positions-analyzed), [chess-fens-evaluations](https://www.kaggle.com/datasets/dev102/chess-fens-evaluations-dataset), `arevel/chess-games` (Lichess 2016, ~6.25M w/ eval), `datasnaek/chess` (20k w/ ratings) | FEN+eval CSVs; some games w/ Elo | 10K–6M | per-dataset (varies) | Alternative FEN+eval or small rated-games sources. | Needs Kaggle account/API; licenses vary; **mostly redundant with the CC0 Lichess sets**. Lower priority. |
| 14 | **NL/board-understanding & reasoning** — [ssingh22/ChessPositionUnderstanding](https://huggingface.co/datasets/ssingh22/ChessPositionUnderstanding) (llama3.1), [oscar128372/chess_spatial_reasoning_400kv1](https://huggingface.co/datasets/oscar128372/chess_spatial_reasoning_400kv1), [lucasdino/chess-reasoning-data](https://huggingface.co/datasets/lucasdino/chess-reasoning-data) (CC0) | position captions / piece-on-square / legal-move / motif reasoning | 0.4M–many | mixed | Possible fuel for a **deterministic faithfulness verifier** (does text match the board?) — cf. DATASET_PLAN §5. | Machine-generated; check license (llama3.1 is restrictive). Not coaching labels. |

**Not useful (excluded):** robot-arm "chess move" LeRobot sets (`Chojins/*`, `dopaul/*`), board-image/OCR/YOLO sets (`surawut/*`, `bingbangboom/*`, `smallchess/kaggle-chess-positions`), engine self-play (`laion/strategic_game_chess`, `*/chess-selfplay`), mechanistic-interp hidden-states (`austindavis/*`), and bulk unlabeled PGN-for-LLM-pretraining (`adamkarvonen/chess_games`, `BlueSunflower/ChessGames`, `angeluriot/chess_games`, `patrickfrank1/chess-pgn-games`) — none give tier + motif + coachable-move together.

---

## 3. Top 3 to actually use next

### 🥇 1. `Lichess/chess-puzzles` — closes the motif-coverage gap (our #1 measured hole)
Our audit: skewer 0.2%, discovered 0.6%, double-attack 0.3%, deflection 0.6%, decoy 0.05%,
opposition 0.1%. This set fixes all of them because every puzzle is **themed and rated**.

**Integration step (proposed):**
1. Stream the parquet; keep rows whose `Themes` intersect our target motifs
   (`fork, pin, skewer, discoveredAttack, doubleCheck, deflection, decoy, attraction,
   backRankMate, hangingPiece` + endgame: `rookEndgame, pawnEndgame, endgame`).
2. Bucket by `Rating` → `beginner` ≤1200, `intermediate` 1300–1600, `advanced` 1700–2000
   (drop the inter-band gaps, mirroring our existing tier logic).
3. For each puzzle: `board = chess.Board(FEN); board.push_uci(Moves[0])` → this is the
   position to coach (student to move). Emit our `positions_*.jsonl` row with
   `tier` from step 2; set a plausible `played_move` = top Maia move ≠ solution.
4. Run the **unchanged** `generate.py` (SF sound pool + Maia + teacher). The solution is a
   *hint* to the teacher's context, **not** the label.
5. Balance ~a few hundred per (tier × motif) to hit the coverage target in DATASET_PLAN §3.

### 🥈 2. `Lichess/standard-chess-games` — scalable tier-specific positions
Removes the API-crawl bottleneck for "real human decision positions at 1000–2000."

**Integration step (proposed):**
1. With DuckDB/polars over the HF parquet (or monthly PGN.zst from §5), select rows where
   `WhiteElo` or `BlackElo` ∈ a target band; take a bounded sample per band/month.
2. Parse `movetext` with python-chess; reuse `sample_plies()` + `extract_positions()` from
   `lichess_sampler.py` (attribute tier by the *mover's* Elo — the logic already exists).
3. Write our `positions_*.jsonl` and feed `generate.py`. This is a **drop-in offline path**
   next to the existing crawl — same schema, no teacher/filter changes.

### 🥉 3. `Waterhorse/chess_data` (ChessGPT) + Jhamtani commentary — teacher STYLE
Directly enriches "how to explain / how to find it" beyond the 9 YouTube transcripts.

**Integration step (proposed):**
1. Pull the ChessGPT annotated-PGN / language subset (repo files; viewer is broken) and, for
   breadth, the Jhamtani categorized commentary.
2. Extract `(fen/move, comment)` prose; **paraphrase-distill** into `prompts/principles.md`
   and `prompts/fewshots.json` via `distill_principles.py`, alongside transcripts.
3. Keep it **internal STYLE only** — never an SFT row (preserves the "100% synthetic dataset"
   invariant). Bonus: this commentary is also a corpus of *faithful human phrasings* you can
   use to sanity-check the v2 faithfulness filter.

**Single best for closing motif-coverage + tier-specific-position gaps → `Lichess/chess-puzzles`.**
It is the only ready-made set that carries **both** a difficulty `Rating` (tier) **and**
`Themes` (motif) on every row, is **CC0**, and drops straight into our existing
positions→SF→Maia→teacher pipeline with zero changes to `generate.py`.

---

## 4. Cross-cutting caveats (read before downloading)

- **Re-grounding is mandatory.** Our behavior depends on *our* Stockfish MultiPV-8 sound
  pool (150cp tolerance, 250cp blunder cutoff) + *our* Maia tier policy. External evals
  (`cp/line`) and puzzle solutions are **context/pre-filters**, never labels.
- **Puzzle rating ≠ player rating.** It's puzzle *difficulty*. It's a good, arguably cleaner,
  proxy for "motif at a tier," but document it as difficulty-tier, not Elo attribution.
- **Commentary provenance is mixed.** ChessGPT and Jhamtani draw on forum/book text →
  **paraphrase-and-distill only**; `Icannos/chess_studies` (CC0) is the cleanest for any
  quoting. The SFT set stays synthetic.
- **`aicrowd/ChessExplained` is a trap as labels.** Use it only to *demonstrate* the
  fabrication/verdict failure mode (or as a `<think>`/chat format scaffold).
- **Size discipline.** `standard-chess-games` (~4.9 TB) and `chess-position-evaluations`
  (41 GB) must be filtered server-side/streamed. Never `load_dataset(..., split="train")`
  the whole thing.
- **License summary:** all Lichess-authored sets are **CC0** (free for anything);
  `tournament-chess-games` is **CC-BY-SA-4.0** (attribution + share-alike);
  `mateuszgrzyb`/`Thytu`/`laion` are **CC-BY-4.0**; ChessGPT is **Apache-2.0** (mixed source);
  Kaggle varies per dataset.

---

## 5. Appendix — the canonical raw source (behind every HF mirror)

The official **[Lichess Open Database](https://database.lichess.org/)** (CC0, updated
2026-07-05) is what all the Lichess HF datasets mirror. Useful when you want the freshest or
the richest form:

- **Games:** `standard/lichess_db_standard_rated_YYYY-MM.pgn.zst` — monthly, ~28–31 GB each,
  full Elo + `[%eval]`/clock tags. (Same data as `Lichess/standard-chess-games`.)
- **Puzzles:** `lichess_db_puzzle.csv.zst` — 6.06M, columns
  `PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags`.
  (Same as `Lichess/chess-puzzles`, one file.)
- **Evals:** `lichess_db_eval.jsonl.zst` — ~394M positions; **multi-PV** (`pvs[]` with
  `cp/mate/line`), i.e. *richer than* the HF-denormalized `chess-position-evaluations`.
- **Puzzle themes list:** [`Lichess/puzzle-themes`](https://huggingface.co/datasets/Lichess/puzzle-themes)
  — the controlled vocabulary to filter `Themes` against.

Access = plain HTTPS/torrent download + `zstd` + `python-chess` (all already deps). Same
politeness/attribution norms as in `lichess_sampler.py`.
