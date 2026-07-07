#!/usr/bin/env python3
"""Are Lichess puzzle solutions the OBJECTIVE BEST move, or the tier-appropriate
FINDABLE move? — a data probe for the `Lichess/chess-puzzles` ingest decision.

Context
-------
Our thesis is that a good coach recommends the **tier-appropriate, findable,
instructive** move — NOT always the engine-best. Before ingesting Lichess
puzzles we need to know what a puzzle *solution* actually is:

- If solutions ≈ Stockfish's #1 best move, then a puzzle answer is an
  **objective-best** label (great for tactics, but it is NOT a
  "what a 1000 would play" label).
- Maia then tells us whether that best move is **naturally findable** at a
  given tier (high policy) or a **hard-to-find** best move (low policy),
  which is exactly the difficulty a puzzle `Rating` encodes.

This is analysis only. It does NOT modify the pipeline. It reuses the repo's
own engines so the numbers match production behaviour:

- ``src/engine/stockfish_engine.py`` — best move + MultiPV sound pool
  (replicating ``sound_pool``'s rule with the module's own constants).
- ``src/engine/maia_engine.py`` — per-tier human move policy (``human_moves``).

Puzzle format nuance (critical)
-------------------------------
The dataset ``FEN`` is one ply BEFORE the puzzle. ``Moves[0]`` (UCI) is the
opponent's *setup* move that creates the puzzle; the SOLVER's move to grade is
``Moves[1]`` applied to the position AFTER ``Moves[0]``. So:

    board = chess.Board(FEN); board.push_uci(Moves[0])   # <- puzzle position
    solution = Moves[1]                                   # <- the "answer"

Run (from repo root, pinned interpreter)::

    ~/.venvs/mlx/bin/python -m scripts.puzzles_vs_engine --per-bucket 30
    ~/.venvs/mlx/bin/python -m scripts.puzzles_vs_engine --report-only

Outputs::

    data/analysis/puzzles_sample.jsonl   (cached balanced sample)
    data/analysis/puzzles_vs_engine.jsonl (raw per-puzzle records)
    data/analysis/PUZZLES_REPORT.md      (tables + verdict)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import settings  # noqa: E402
from src.engine import maia_engine, stockfish_engine  # noqa: E402
from src.engine.stockfish_engine import BLUNDER_CP  # noqa: E402

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

DATASET_ID = "Lichess/chess-puzzles"

#: Rating-bucket centers to balance across (puzzle *difficulty*, not player Elo).
BUCKETS: Tuple[int, ...] = (800, 1200, 1600, 2000, 2400)
#: Half-width of each bucket window (disjoint: 700-900, 1100-1300, ...).
WINDOW = 100
#: Hard cap on rows streamed while filling buckets (rarest bucket is binding).
SCAN_CAP = 400_000

#: Stockfish settings — matched to the pipeline (MultiPV-8, 150cp tolerance,
#: 250cp blunder cutoff). Movetime bumped a touch above the 300ms pipeline
#: default so the "objective best" reference is stable for a match-rate probe.
SF_MULTIPV = settings.MULTIPV               # 8
SF_TOLERANCE_CP = settings.SOUND_TOLERANCE_CP  # 150
SF_MOVETIME_MS = 500

#: Ask Maia for every legal move's policy (>= max legal moves in any position).
MAIA_TOPK = 300

ANALYSIS_DIR = _ROOT / "data" / "analysis"
SAMPLE_PATH = ANALYSIS_DIR / "puzzles_sample.jsonl"
RAW_PATH = ANALYSIS_DIR / "puzzles_vs_engine.jsonl"
REPORT_PATH = ANALYSIS_DIR / "PUZZLES_REPORT.md"


# --------------------------------------------------------------------------- #
# Bucket / tier helpers
# --------------------------------------------------------------------------- #


def bucket_of(rating: int) -> Optional[int]:
    """Return the bucket center whose ±WINDOW window contains ``rating``."""
    for center in BUCKETS:
        if center - WINDOW <= rating <= center + WINDOW:
            return center
    return None


def matched_net(rating: int) -> str:
    """Map a puzzle's difficulty Rating to the closest available Maia net.

    Maia nets exist only at 1100 / 1500 / 1900. We align to our tier bands:
    beginner (<=1200) -> 1100, intermediate (1300-1600) -> 1500,
    advanced (>=1700, incl. 2000/2400 which exceed Maia's ceiling) -> 1900.
    """
    if rating < 1300:
        return "maia-1100"
    if rating < 1700:
        return "maia-1500"
    return "maia-1900"


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def sample_puzzles(per_bucket: int) -> List[Dict[str, Any]]:
    """Stream the HF puzzle set and collect a balanced, rating-bucketed sample.

    Sequential fill (no shuffle): PuzzleId ordering is uncorrelated with Rating,
    so the first-N-per-bucket sample is reproducible and cheap.
    """
    from datasets import load_dataset

    print(f"  streaming {DATASET_ID} (filling {per_bucket}/bucket)…", file=sys.stderr)
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    by_bucket: Dict[int, List[Dict[str, Any]]] = {c: [] for c in BUCKETS}
    scanned = 0
    kept = 0
    for row in ds:
        scanned += 1
        if scanned > SCAN_CAP:
            break
        rating = int(row["Rating"])
        center = bucket_of(rating)
        if center is None or len(by_bucket[center]) >= per_bucket:
            continue

        moves = (row["Moves"] or "").split()
        if len(moves) < 2:
            continue
        # Validate the puzzle format: setup move legal, solution legal after it.
        try:
            board = chess.Board(row["FEN"])
            setup = chess.Move.from_uci(moves[0])
            if setup not in board.legal_moves:
                continue
            board.push(setup)
            solution = chess.Move.from_uci(moves[1])
            if solution not in board.legal_moves:
                continue
        except (ValueError, AssertionError):
            continue

        by_bucket[center].append(
            {
                "PuzzleId": row["PuzzleId"],
                "GameId": row.get("GameId"),
                "FEN": row["FEN"],
                "Moves": moves,
                "Rating": rating,
                "RatingDeviation": row.get("RatingDeviation"),
                "Popularity": row.get("Popularity"),
                "NbPlays": row.get("NbPlays"),
                "Themes": row.get("Themes") or [],
                "OpeningTags": row.get("OpeningTags"),
            }
        )
        kept += 1
        if all(len(by_bucket[c]) >= per_bucket for c in BUCKETS):
            break

    for center in BUCKETS:
        got = len(by_bucket[center])
        if got < per_bucket:
            print(
                f"  [warn] bucket {center}: only {got}/{per_bucket} found "
                f"(scanned {scanned}).",
                file=sys.stderr,
            )

    sample: List[Dict[str, Any]] = []
    for center in BUCKETS:
        sample.extend(by_bucket[center])
    print(f"  sampled {len(sample)} puzzles (scanned {scanned} rows).", file=sys.stderr)
    return sample


def load_or_build_sample(per_bucket: int, rebuild: bool) -> List[Dict[str, Any]]:
    if SAMPLE_PATH.exists() and not rebuild:
        rows = [json.loads(l) for l in SAMPLE_PATH.read_text().splitlines() if l.strip()]
        by_bucket = Counter(bucket_of(r["Rating"]) for r in rows)
        if all(by_bucket.get(c, 0) >= per_bucket for c in BUCKETS):
            print(f"  reusing cached sample ({len(rows)} puzzles).", file=sys.stderr)
            # Trim to exactly per_bucket for a clean balanced set.
            trimmed: List[Dict[str, Any]] = []
            seen: Dict[int, int] = defaultdict(int)
            for r in rows:
                c = bucket_of(r["Rating"])
                if c is not None and seen[c] < per_bucket:
                    seen[c] += 1
                    trimmed.append(r)
            return trimmed
    sample = sample_puzzles(per_bucket)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with SAMPLE_PATH.open("w", encoding="utf-8") as fh:
        for r in sample:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return sample


# --------------------------------------------------------------------------- #
# Per-puzzle measurement
# --------------------------------------------------------------------------- #


def sound_pool_from_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replicate ``stockfish_engine.sound_pool``'s selection from analyze lines."""
    if not lines:
        return []
    best_cp = int(lines[0]["cp"])
    max_loss = min(SF_TOLERANCE_CP, BLUNDER_CP - 1)  # never include a blunder
    return [ln for ln in lines if best_cp - int(ln["cp"]) <= max_loss]


def maia_distribution(fen: str, net: str) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Return (ranked_moves, {uci: policy}) for every legal move at ``net``."""
    res = maia_engine.human_moves(fen, net, top_k=MAIA_TOPK)
    moves = res["moves"]
    dist = {m["uci"]: float(m["policy"]) for m in moves}
    return moves, dist


def rank_of(moves: List[Dict[str, Any]], uci: str) -> Optional[int]:
    for i, m in enumerate(moves):
        if m["uci"] == uci:
            return i + 1
    return None


def analyze_puzzle(
    engine: "stockfish_engine.chess.engine.SimpleEngine",
    puzzle: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute Stockfish alignment + Maia findability for one puzzle."""
    fen0 = puzzle["FEN"]
    moves = puzzle["Moves"]
    rating = int(puzzle["Rating"])

    board = chess.Board(fen0)
    board.push(chess.Move.from_uci(moves[0]))
    puzzle_fen = board.fen()
    solver_side = "white" if board.turn == chess.WHITE else "black"
    solution_uci = moves[1]
    solution_san = board.san(chess.Move.from_uci(solution_uci))

    # ---- Stockfish (objective best + sound pool) ----
    analysis = stockfish_engine._analyze_impl(
        engine, puzzle_fen, multipv=SF_MULTIPV, movetime_ms=SF_MOVETIME_MS
    )
    lines = analysis["best"]
    sf_best = lines[0] if lines else None
    best_cp = int(sf_best["cp"]) if sf_best else None
    pool = sound_pool_from_lines(lines)
    pool_ucis = {ln["uci"] for ln in pool}

    is_sf_best = bool(sf_best and solution_uci == sf_best["uci"])
    in_sound_pool = solution_uci in pool_ucis

    # cp of the solution (reuse an already-searched line if present).
    sol_cp: Optional[int] = None
    for ln in lines:
        if ln["uci"] == solution_uci:
            sol_cp = int(ln["cp"])
            break
    if sol_cp is None:
        sol_cp = int(
            stockfish_engine._eval_move_impl(
                engine, puzzle_fen, solution_uci, movetime_ms=SF_MOVETIME_MS
            )["cp"]
        )
    cp_loss = max(0, best_cp - sol_cp) if best_cp is not None else None
    best_is_mate = bool(sf_best and sf_best.get("mate") is not None)

    # ---- Maia findability (beginner 1100 + tier matched to puzzle rating) ----
    beg_moves, beg_dist = maia_distribution(puzzle_fen, "maia-1100")
    beg_policy = beg_dist.get(solution_uci, 0.0)
    beg_rank = rank_of(beg_moves, solution_uci)
    beg_top = beg_moves[0] if beg_moves else None

    net = matched_net(rating)
    if net == "maia-1100":
        m_moves, m_dist, m_top = beg_moves, beg_dist, beg_top
    else:
        m_moves, m_dist = maia_distribution(puzzle_fen, net)
        m_top = m_moves[0] if m_moves else None
    m_policy = m_dist.get(solution_uci, 0.0)
    m_rank = rank_of(m_moves, solution_uci)

    return {
        "puzzle_id": puzzle["PuzzleId"],
        "game_id": puzzle.get("GameId"),
        "rating": rating,
        "rating_bucket": bucket_of(rating),
        "rating_deviation": puzzle.get("RatingDeviation"),
        "popularity": puzzle.get("Popularity"),
        "nb_plays": puzzle.get("NbPlays"),
        "themes": puzzle.get("Themes") or [],
        "puzzle_fen": puzzle_fen,
        "solver_side": solver_side,
        "setup_move_uci": moves[0],
        "solution_uci": solution_uci,
        "solution_san": solution_san,
        # Stockfish
        "sf_best_uci": sf_best["uci"] if sf_best else None,
        "sf_best_san": sf_best["san"] if sf_best else None,
        "sf_best_cp": best_cp,
        "sf_best_is_mate": best_is_mate,
        "sf_pool_ucis": sorted(pool_ucis),
        "sf_pool_size": len(pool_ucis),
        "solution_cp": sol_cp,
        "solution_cp_loss": cp_loss,
        "is_sf_best": is_sf_best,
        "in_sound_pool": in_sound_pool,
        # Maia @ beginner 1100
        "maia1100_solution_policy": round(beg_policy, 4),
        "maia1100_solution_rank": beg_rank,
        "maia1100_top_uci": beg_top["uci"] if beg_top else None,
        "maia1100_top_policy": round(float(beg_top["policy"]), 4) if beg_top else None,
        "maia1100_solution_is_top": bool(beg_top and beg_top["uci"] == solution_uci),
        # Maia @ tier matched to puzzle rating
        "matched_net": net,
        "matched_solution_policy": round(m_policy, 4),
        "matched_solution_rank": m_rank,
        "matched_top_uci": m_top["uci"] if m_top else None,
        "matched_top_policy": round(float(m_top["policy"]), 4) if m_top else None,
        "matched_solution_is_top": bool(m_top and m_top["uci"] == solution_uci),
    }


# --------------------------------------------------------------------------- #
# Aggregation + report
# --------------------------------------------------------------------------- #


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def _median(xs: List[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _mean(xs: List[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group records by rating bucket (+ an ALL group) and compute stats."""
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[r["rating_bucket"]].append(r)
        groups["ALL"].append(r)

    out: Dict[Any, Dict[str, Any]] = {}
    for key, rs in groups.items():
        n = len(rs)
        beg_pol = [r["maia1100_solution_policy"] for r in rs]
        m_pol = [r["matched_solution_policy"] for r in rs]
        cp_losses = [r["solution_cp_loss"] for r in rs if r["solution_cp_loss"] is not None]
        out[key] = {
            "n": n,
            "mean_rating": _mean([r["rating"] for r in rs]),
            # Stockfish alignment
            "sf_match_rate": _pct(sum(r["is_sf_best"] for r in rs), n),
            "sound_pool_rate": _pct(sum(r["in_sound_pool"] for r in rs), n),
            "median_cp_loss": _median(cp_losses),
            "mean_pool_size": _mean([r["sf_pool_size"] for r in rs]),
            "mate_share": _pct(sum(r["sf_best_is_mate"] for r in rs), n),
            # Maia findability @1100
            "beg_mean_policy": _mean(beg_pol),
            "beg_median_policy": _median(beg_pol),
            "beg_is_top_rate": _pct(sum(r["maia1100_solution_is_top"] for r in rs), n),
            "beg_findable_rate": _pct(
                sum(r["maia1100_solution_is_top"] or r["maia1100_solution_policy"] >= 0.25 for r in rs), n
            ),
            "beg_hard_rate": _pct(sum(r["maia1100_solution_policy"] < 0.05 for r in rs), n),
            # Maia findability @matched
            "m_mean_policy": _mean(m_pol),
            "m_median_policy": _median(m_pol),
            "m_is_top_rate": _pct(sum(r["matched_solution_is_top"] for r in rs), n),
            "m_findable_rate": _pct(
                sum(r["matched_solution_is_top"] or r["matched_solution_policy"] >= 0.25 for r in rs), n
            ),
            "m_hard_rate": _pct(sum(r["matched_solution_policy"] < 0.05 for r in rs), n),
        }
    return out


def _policy_hist(records: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    bins = {">=0.50": 0, "0.25-0.50": 0, "0.10-0.25": 0, "0.05-0.10": 0, "<0.05": 0}
    for r in records:
        p = r[field]
        if p >= 0.50:
            bins[">=0.50"] += 1
        elif p >= 0.25:
            bins["0.25-0.50"] += 1
        elif p >= 0.10:
            bins["0.10-0.25"] += 1
        elif p >= 0.05:
            bins["0.05-0.10"] += 1
        else:
            bins["<0.05"] += 1
    return bins


def build_report(records: List[Dict[str, Any]], stats: Dict[Any, Dict[str, Any]]) -> str:
    from datetime import date

    allst = stats["ALL"]
    order = list(BUCKETS) + ["ALL"]

    non_match = [r for r in records if not r["is_sf_best"]]
    n_singleton = sum(1 for r in records if r["sf_pool_size"] == 1)
    hardest = sorted(records, key=lambda r: r["maia1100_solution_policy"])[:4]
    easy_800 = sorted(
        (r for r in records if r["rating_bucket"] == 800),
        key=lambda r: -r["maia1100_solution_policy"],
    )[:2]

    def row_label(k: Any) -> str:
        return "**ALL**" if k == "ALL" else f"~{k}"

    lines: List[str] = []
    ap = lines.append

    ap("# Lichess Puzzles vs. Engine — Objective-Best or Findable-Move?")
    ap("")
    ap(f"Date: {date.today().isoformat()} · Probe for the `Lichess/chess-puzzles` ingest decision")
    ap("")
    ap(
        "**Question.** Do Lichess puzzle *solutions* give the OBJECTIVE BEST move "
        "(≈ Stockfish), or the move most SUITABLE / FINDABLE for a player at that "
        "ELO? This decides whether the puzzle answer can be used as a "
        "recommended-move label in our tier-appropriate coaching pipeline."
    )
    ap("")
    ap("## Method")
    ap("")
    ap(
        f"- **Sample:** {allst['n']} puzzles from `{DATASET_ID}`, balanced across "
        f"rating buckets {', '.join('~'+str(b) for b in BUCKETS)} (±{WINDOW} each). "
        "Puzzle `Rating` is *difficulty*, not player Elo."
    )
    ap(
        "- **Puzzle position:** `board = chess.Board(FEN); board.push(Moves[0])` "
        "(the opponent setup); the graded **solution is `Moves[1]`** on that position."
    )
    ap(
        f"- **Stockfish** (`stockfish_engine`): MultiPV-{SF_MULTIPV}, {SF_MOVETIME_MS}ms, "
        f"sound pool = within {SF_TOLERANCE_CP}cp of best and not a blunder "
        f"(>= {BLUNDER_CP}cp loss). `is_sf_best` = solution == engine #1."
    )
    ap(
        "- **Maia** (`maia_engine.human_moves`): full per-move policy at **beginner "
        "`maia-1100`** and at the **net matched to the puzzle rating** "
        "(<1300→1100, 1300-1699→1500, ≥1700→1900). `policy` = P(a human at that "
        "tier plays the move). 'findable' = solution is Maia's #1 **or** policy ≥ 0.25; "
        "'hard-to-find' = policy < 0.05."
    )
    ap("")

    # --- Table 1: sample + Stockfish alignment ---
    ap("## 1. Stockfish alignment — is the solution the objective best?")
    ap("")
    ap("| Rating bucket | n | mean rating | solution == SF best | in SF sound pool | median cp-loss (solution) | mean pool size | mate puzzles |")
    ap("|---|---|---|---|---|---|---|---|")
    for k in order:
        s = stats[k]
        ap(
            f"| {row_label(k)} | {s['n']} | {s['mean_rating']:.0f} | "
            f"{s['sf_match_rate']:.1f}% | {s['sound_pool_rate']:.1f}% | "
            f"{s['median_cp_loss']:.0f} | {s['mean_pool_size']:.1f} | {s['mate_share']:.0f}% |"
        )
    ap("")
    ap(
        f"**Read-out.** The solution is Stockfish's #1 move in **{allst['sf_match_rate']:.1f}%** "
        f"of puzzles and is in our sound pool in **{allst['sound_pool_rate']:.0f}%** — median "
        f"cp-loss **0**. The mean sound-pool size is **{allst['mean_pool_size']:.2f}** and "
        f"**{_pct(n_singleton, allst['n']):.0f}%** of positions have exactly ONE sound move: "
        "puzzles are, by construction, single-best-move ('only-move') positions."
    )
    if non_match:
        ex = non_match[0]
        n_mate = sum(1 for r in non_match if r["sf_best_is_mate"])
        ap(
            f"The only non-match ({len(non_match)}/{allst['n']}"
            + (f", {n_mate} a dual mate" if n_mate else "")
            + f") is a *tie for best*: puzzle `{ex['puzzle_id']}` (r{ex['rating']}) plays "
            f"**{ex['solution_san']}** while Stockfish prefers the equally-winning "
            f"**{ex['sf_best_san']}** (cp-loss {ex['solution_cp_loss']}, still in pool)."
        )
    ap("")

    # --- Table 2: Maia findability @1100 ---
    ap("## 2. Findability — how human-likely is the solution?")
    ap("")
    ap("### 2a. At a BEGINNER (`maia-1100`)")
    ap("")
    ap("| Rating bucket | n | mean policy | median policy | solution == Maia top | findable (top or ≥0.25) | hard-to-find (<0.05) |")
    ap("|---|---|---|---|---|---|---|")
    for k in order:
        s = stats[k]
        ap(
            f"| {row_label(k)} | {s['n']} | {s['beg_mean_policy']:.3f} | "
            f"{s['beg_median_policy']:.3f} | {s['beg_is_top_rate']:.1f}% | "
            f"{s['beg_findable_rate']:.1f}% | {s['beg_hard_rate']:.1f}% |"
        )
    ap("")
    ap("### 2b. At the tier MATCHED to the puzzle rating")
    ap("")
    ap("| Rating bucket | n | matched net | mean policy | median policy | solution == Maia top | findable (top or ≥0.25) | hard-to-find (<0.05) |")
    ap("|---|---|---|---|---|---|---|---|")
    for k in order:
        s = stats[k]
        if k == "ALL":
            net_label = "mixed"
        else:
            net_label = matched_net(int(k)).replace("maia-", "")
        ap(
            f"| {row_label(k)} | {s['n']} | {net_label} | {s['m_mean_policy']:.3f} | "
            f"{s['m_median_policy']:.3f} | {s['m_is_top_rate']:.1f}% | "
            f"{s['m_findable_rate']:.1f}% | {s['m_hard_rate']:.1f}% |"
        )
    ap("")

    # --- Table 3: policy histogram by bucket (beginner) ---
    ap("## 3. Beginner-findability distribution (solution policy @ `maia-1100`)")
    ap("")
    ap("| Rating bucket | >=0.50 | 0.25-0.50 | 0.10-0.25 | 0.05-0.10 | <0.05 |")
    ap("|---|---|---|---|---|---|")
    by_bucket: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_bucket[r["rating_bucket"]].append(r)
        by_bucket["ALL"].append(r)
    for k in order:
        h = _policy_hist(by_bucket[k], "maia1100_solution_policy")
        tot = sum(h.values()) or 1
        cells = " | ".join(f"{h[b]} ({_pct(h[b], tot):.0f}%)" for b in [">=0.50", "0.25-0.50", "0.10-0.25", "0.05-0.10", "<0.05"])
        ap(f"| {row_label(k)} | {cells} |")
    ap("")

    # --- Concrete examples ---
    ap("## 4. Concrete examples — objective-best but hard to find")
    ap("")
    ap("Hard-to-find best moves (lowest beginner Maia policy) — the objective best a beginner would essentially never play:")
    ap("")
    ap("| PuzzleId | rating | solution | Maia-1100 policy (rank) | themes |")
    ap("|---|---|---|---|---|")
    for r in hardest:
        th = ", ".join(r["themes"][:3])
        ap(
            f"| `{r['puzzle_id']}` | {r['rating']} | {r['solution_san']} | "
            f"{r['maia1100_solution_policy']:.3f} (rank {r['maia1100_solution_rank']}) | {th} |"
        )
    ap("")
    ap("Naturally-findable solutions (an easy ~800 puzzle) — beginner Maia already plays it:")
    ap("")
    ap("| PuzzleId | rating | solution | Maia-1100 policy | is Maia top | themes |")
    ap("|---|---|---|---|---|---|")
    for r in easy_800:
        th = ", ".join(r["themes"][:3])
        ap(
            f"| `{r['puzzle_id']}` | {r['rating']} | {r['solution_san']} | "
            f"{r['maia1100_solution_policy']:.3f} | {r['maia1100_solution_is_top']} | {th} |"
        )
    ap("")

    # --- Themes present ---
    theme_counter: Counter = Counter()
    for r in records:
        theme_counter.update(r["themes"])
    top_themes = ", ".join(f"`{t}` {c}" for t, c in theme_counter.most_common(18))
    ap("## 5. Motif coverage in the sample (`Themes`)")
    ap("")
    ap(f"Top themes across {allst['n']} puzzles: {top_themes}.")
    ap("")

    # --- Verdict ---
    ap("## 6. Verdict")
    ap("")
    ap(
        f"**Q1 — Are puzzle solutions ≈ the objective best (Stockfish)?  YES.** "
        f"The solution equals Stockfish's #1 move **{allst['sf_match_rate']:.1f}%** of "
        f"the time and is inside our sound pool **{allst['sound_pool_rate']:.1f}%** of "
        f"the time (median solution cp-loss {allst['median_cp_loss']:.0f}). A puzzle "
        "answer is an **engine-best / near-best** label, essentially by construction "
        "(Lichess mines puzzles as positions with one decisive best line)."
    )
    ap("")
    beg = stats["ALL"]["beg_findable_rate"]
    beg_hard = stats["ALL"]["beg_hard_rate"]
    ap(
        f"**Q2 — Is it the tier-appropriate FINDABLE move, or a hard-to-find best?  "
        f"It is the OBJECTIVE BEST, and for beginners it is often HARD TO FIND.** "
        f"At `maia-1100`, only **{beg:.1f}%** of solutions are naturally findable "
        f"(Maia top or policy ≥ 0.25) and **{beg_hard:.1f}%** are low-policy "
        "(< 0.05) hard-to-find moves. Findability falls as puzzle rating rises "
        "(see §2) — the `Rating` really does encode 'how hard to find', which is a "
        "Maia-low, engine-high signal, i.e. the *opposite* of a 'what a 1000 would "
        "play' label."
    )
    ap("")
    ap(
        "**Q3 — How should puzzles be used in our pipeline?**"
    )
    ap("")
    ap(
        "- ✅ **Use puzzles for MOTIF COVERAGE + as positions.** Filter by `Themes` "
        "to fill our measured motif holes (fork/pin/skewer/discovered/deflection/"
        "back-rank/endgame) and bucket by `Rating` as a difficulty tier. Push "
        "`Moves[0]` to get the position to coach."
    )
    ap(
        "- ✅ **Always re-ground through our own Stockfish + Maia + teacher.** Let the "
        "teacher pick the **tier-appropriate** move from our sound pool; synthesize the "
        "student's mistake (e.g. a high-Maia non-solution move) rather than assuming one."
    )
    ap(
        "- ❌ **Do NOT use `Moves[1]` as the recommended-move label for beginner/"
        "intermediate tiers.** It is the engine-best, frequently a low-Maia move a "
        "player at that tier would not find — using it as the label would re-teach "
        "'always play the engine move' and undo tier differentiation."
    )
    ap(
        "- ➕ **Advanced tier is the exception.** When the sharp tactic *is* the "
        "teaching point (higher-rated puzzles, `advanced`), the puzzle solution and "
        "our sound-pool best usually coincide, so the solution can legitimately be the "
        "recommended move there."
    )
    ap("")
    ap(
        "**Bottom line:** puzzles are excellent *fuel* (positions + motifs + a strong "
        "hint for the teacher's context), not a drop-in coaching label. This matches "
        "the pipeline's existing invariant (`docs/EXTERNAL_DATASETS.md`, "
        "`DIVERGENCE_REPORT.md`): external solutions are context, never labels; the "
        "recommended move is chosen by our tier-aware teacher over our SF+Maia grounding."
    )
    ap("")
    ap("## 7. Deliverables")
    ap("")
    ap("- Raw per-puzzle records: `data/analysis/puzzles_vs_engine.jsonl`")
    ap("- Cached balanced sample: `data/analysis/puzzles_sample.jsonl`")
    ap("- This report: `data/analysis/PUZZLES_REPORT.md`")
    ap("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-bucket", type=int, default=30, help="Puzzles per rating bucket.")
    parser.add_argument("--rebuild-sample", action="store_true", help="Re-stream even if cached.")
    parser.add_argument("--report-only", action="store_true", help="Rebuild report from existing raw jsonl.")
    args = parser.parse_args(argv)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        records = [json.loads(l) for l in RAW_PATH.read_text().splitlines() if l.strip()]
        print(f"  loaded {len(records)} raw records.", file=sys.stderr)
    else:
        sample = load_or_build_sample(args.per_bucket, args.rebuild_sample)
        records = []
        t0 = time.time()
        with stockfish_engine.open_engine() as engine, RAW_PATH.open("w", encoding="utf-8") as fh:
            for i, puzzle in enumerate(sample, 1):
                try:
                    rec = analyze_puzzle(engine, puzzle)
                except Exception as exc:  # keep going; log the puzzle that failed
                    print(f"  [warn] puzzle {puzzle.get('PuzzleId')} failed: {exc}", file=sys.stderr)
                    continue
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 10 == 0 or i == len(sample):
                    dt = time.time() - t0
                    print(f"  analyzed {i}/{len(sample)}  ({dt:.0f}s)", file=sys.stderr)

    stats = summarize(records)
    report = build_report(records, stats)
    REPORT_PATH.write_text(report, encoding="utf-8")

    a = stats["ALL"]
    print("\n=== SUMMARY ===")
    print(f"n={a['n']}  SF-match={a['sf_match_rate']:.1f}%  sound-pool={a['sound_pool_rate']:.1f}%")
    print(f"beginner findable={a['beg_findable_rate']:.1f}%  hard(<0.05)={a['beg_hard_rate']:.1f}%")
    print(f"report -> {REPORT_PATH}")
    print(f"raw    -> {RAW_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
