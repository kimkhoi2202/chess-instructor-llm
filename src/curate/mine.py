#!/usr/bin/env python3
"""MINE discriminating multi-tier positions from the Lichess puzzle bank (CPU).

The moat of this project is the *discriminating* position: one where the
tier-appropriate move genuinely differs across levels — the beginner's most
human-findable sound move is NOT the advanced player's sharpest move. This module
finds those, at quality, on CPU only (Stockfish + Maia), never touching a GPU.

Pipeline (two resumable stages)
-------------------------------
``sample``  Stream the 6M-row Lichess puzzle CSV (pulled from the private HF
            mirror ``khoilamalphaai/chess-data-bank`` — a single ~302 MB read;
            falls back to the Modal ``chess-data`` volume) and take a STRATIFIED
            reservoir sample across ``(rating bucket x motif)`` so the mined set
            spans all tiers and covers many motifs, not just mates-in-2. Each
            sampled row is turned into the *solver-to-move* position (the Lichess
            convention: the first move in ``Moves`` is the opponent's setup move;
            after it, the student is to move). Deduped here against the existing
            train/eval FENs so nothing that could leak is ever mined.

``mine``    For each sampled position, on a small CPU process pool: compute the
            Stockfish sound pool, and — only when the pool has >= 2 sound moves —
            the Maia human policy per tier and the deterministic canonical
            per-tier move (``tier_select.select_tier_move``). KEEP the position
            iff ``beginner_pick != advanced_pick`` (the discriminating test).
            Advanced == ``pool[0]`` by construction (``w=0``), so the beginner
            Maia pass alone decides the moat; the other two tiers' Maia passes run
            only for the (few thousand) survivors. Emits ``mined.jsonl`` with the
            grounded engine/Maia facts + the per-tier canonical move + a
            per-tier "student move" (the most human move in the pool at that
            tier) for the coaching frame.

CLI
---
    python -m src.curate.mine sample --target 6000
    python -m src.curate.mine mine   --workers 4
    python -m src.curate.mine run    --target 6000 --workers 4   # both
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chess  # noqa: E402

from config import settings  # noqa: E402

log = logging.getLogger("curate.mine")

# --------------------------------------------------------------------------- #
# Paths / layout
# --------------------------------------------------------------------------- #
BANK_DIR = settings.DATA / "bank"
CURATE_DIR = settings.DATA / "curate"
PUZZLE_ZST = BANK_DIR / "lichess_db_puzzle.csv.zst"
SAMPLE_OUT = BANK_DIR / "puzzle_sample.jsonl"
MINED_OUT = CURATE_DIR / "mined.jsonl"
SAMPLE_META = BANK_DIR / "puzzle_sample_meta.json"
MINE_META = CURATE_DIR / "mine_meta.json"

HF_MIRROR_REPO = "khoilamalphaai/chess-data-bank"
HF_PUZZLE_PATH = "lichess/puzzles/lichess_db_puzzle.csv.zst"
MODAL_PROFILE = "chess-instructor-2"
MODAL_VOLUME = "chess-data"
MODAL_PUZZLE_PATH = "/lichess/puzzles/lichess_db_puzzle.csv.zst"

TIER_ORDER = ("beginner", "intermediate", "advanced")
SEED = 20260708

# Rating buckets for stratification (span the three project tiers + neighbours so
# the sample is rating-diverse; the discriminating test itself is rating-agnostic).
RATING_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (600, 1000), (1000, 1300), (1300, 1600),
    (1600, 1900), (1900, 2200), (2200, 2600),
)

# Motif priority: classify each puzzle to ONE primary motif for balanced coverage.
# Ordered most-specific/instructive first; generic phase tags are the last resort.
MOTIF_PRIORITY: Tuple[str, ...] = (
    "fork", "pin", "skewer", "discoveredAttack", "doubleCheck", "deflection",
    "attraction", "clearance", "interference", "xRayAttack", "zugzwang",
    "intermezzo", "sacrifice", "hangingPiece", "trappedPiece", "advancedPawn",
    "promotion", "enPassant", "capturingDefender", "attackingF2F7",
    "kingsideAttack", "queensideAttack", "defensiveMove", "quietMove",
    "rookEndgame", "pawnEndgame", "queenEndgame", "bishopEndgame",
    "knightEndgame", "endgame", "advantage", "middlegame", "opening",
)
# Forcing/near-singleton motifs get a smaller per-stratum quota (low yield of
# multi-move positions), but are not excluded — a few discriminating mates exist.
FORCING_MOTIFS = {"mate", "mateIn1", "mateIn2", "mateIn3", "mateIn4",
                  "mateIn5", "crushing", "short", "oneMove"}


# --------------------------------------------------------------------------- #
# Existing-FEN dedup set (no train/eval leakage)
# --------------------------------------------------------------------------- #
def _epd_key(fen: str) -> Optional[str]:
    """Canonical position key (placement + stm + castling + ep; no move counts)."""
    try:
        return chess.Board(fen).epd()
    except (ValueError, IndexError):
        return None


def _iter_fens_in_obj(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        v = obj.get("fen")
        if isinstance(v, str):
            out.append(v)
        ti = obj.get("teacher_input")
        if isinstance(ti, dict) and isinstance(ti.get("fen"), str):
            out.append(ti["fen"])
    return out


def load_seen_epds() -> set:
    """Every position FEN already used in training or evaluation, as EPD keys.

    Pulled from the raw candidate/position/scenario sources (which carry the FEN
    verbatim), not the rendered chat rows (which only embed an ASCII board).
    """
    seen: set = set()
    globs = [
        "generated/candidates_v1.jsonl", "generated/candidates_v2.jsonl",
        "generated/candidates_v3.jsonl", "generated/candidates.jsonl",
        "generated/plan_v3.jsonl", "generated/plan_v2.jsonl",
        "positions/positions.jsonl", "positions/positions_v1.jsonl",
        "positions/v3_analysis.jsonl", "positions/v3_candidates.jsonl",
    ]
    files: List[Path] = [settings.DATA / g for g in globs]
    files += list(settings.DATA.glob("benchmark*/scenarios.jsonl"))
    files += list(settings.DATA.glob("showcase/*/scenarios.jsonl"))
    n_files = 0
    for fp in files:
        if not fp.exists():
            continue
        n_files += 1
        try:
            with fp.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for fen in _iter_fens_in_obj(obj):
                        k = _epd_key(fen)
                        if k:
                            seen.add(k)
        except OSError:
            continue
    log.info("dedup: %d existing positions from %d source files", len(seen), n_files)
    return seen


# --------------------------------------------------------------------------- #
# Stage 1: pull + stratified sample of the puzzle bank
# --------------------------------------------------------------------------- #
def ensure_puzzle_zst() -> Path:
    """Ensure the puzzle ``.csv.zst`` is local (HF mirror first, Modal fallback)."""
    if PUZZLE_ZST.exists() and PUZZLE_ZST.stat().st_size > 100_000_000:
        log.info("puzzle zst already local: %s (%.0f MB)", PUZZLE_ZST,
                 PUZZLE_ZST.stat().st_size / 1e6)
        return PUZZLE_ZST
    BANK_DIR.mkdir(parents=True, exist_ok=True)

    # 1) HF mirror (private; token from .env) — a single ~302 MB read, no Modal.
    try:
        from dotenv import load_dotenv
        from huggingface_hub import hf_hub_download

        load_dotenv(settings.ROOT / ".env")
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        log.info("pulling puzzle zst from HF mirror %s ...", HF_MIRROR_REPO)
        path = hf_hub_download(
            repo_id=HF_MIRROR_REPO, filename=HF_PUZZLE_PATH, repo_type="dataset",
            token=tok, local_dir=str(BANK_DIR), local_dir_use_symlinks=False,
        )
        src = Path(path)
        if src.resolve() != PUZZLE_ZST.resolve():
            PUZZLE_ZST.parent.mkdir(parents=True, exist_ok=True)
            if not PUZZLE_ZST.exists():
                try:
                    os.replace(src, PUZZLE_ZST)
                except OSError:
                    import shutil
                    shutil.copyfile(src, PUZZLE_ZST)
        if PUZZLE_ZST.exists():
            log.info("HF pull ok: %.0f MB", PUZZLE_ZST.stat().st_size / 1e6)
            return PUZZLE_ZST
    except Exception as exc:  # noqa: BLE001
        log.warning("HF mirror pull failed (%s); trying Modal volume", exc)

    # 2) Modal volume fallback (chess-instructor-2 chess-data).
    import subprocess
    env = dict(os.environ)
    env.pop("MODAL_TOKEN_ID", None)
    env.pop("MODAL_TOKEN_SECRET", None)
    env["MODAL_PROFILE"] = MODAL_PROFILE
    cmd = ["/Users/khoilam/.venvs/mlx/bin/modal", "volume", "get", "--force",
           MODAL_VOLUME, MODAL_PUZZLE_PATH, str(PUZZLE_ZST)]
    log.info("pulling puzzle zst from Modal volume %s ...", MODAL_VOLUME)
    subprocess.run(cmd, env=env, check=True)
    if not PUZZLE_ZST.exists():
        raise SystemExit("BLOCKED: could not obtain puzzle csv.zst from HF or Modal")
    return PUZZLE_ZST


def _primary_motif(themes: List[str]) -> str:
    tset = set(themes)
    for m in MOTIF_PRIORITY:
        if m in tset:
            return m
    for m in themes:
        if m not in ("veryLong", "long", "short", "master", "masterVsMaster",
                     "superGM", "equality"):
            return m
    return "other"


def _bucket(rating: int) -> Optional[str]:
    for lo, hi in RATING_BUCKETS:
        if lo <= rating < hi:
            return f"{lo}-{hi}"
    return None


def solver_position(fen: str, moves: str) -> Optional[Tuple[str, str, str]]:
    """(solver_fen, setup_uci, setup_san) after the opponent's first move.

    Lichess convention: ``FEN`` is the pre-puzzle position and ``Moves[0]`` is the
    opponent's move INTO the puzzle; after it the student is to move.
    """
    parts = moves.split()
    if not parts:
        return None
    try:
        board = chess.Board(fen)
    except ValueError:
        return None
    try:
        mv = chess.Move.from_uci(parts[0])
    except ValueError:
        return None
    if mv not in board.legal_moves:
        return None
    san = board.san(mv)
    board.push(mv)
    if board.is_game_over():
        return None
    return board.fen(), parts[0], san


def cmd_sample(args: argparse.Namespace) -> int:
    """Stream the puzzle CSV and write a stratified, deduped candidate sample."""
    import zstandard as zstd

    zst = ensure_puzzle_zst()
    seen = load_seen_epds() if not args.no_dedup else set()

    per_stratum = args.per_stratum
    rng = random.Random(SEED)
    # Reservoir per (bucket, motif). Forcing motifs get a fraction of the quota.
    reservoirs: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    counts: Dict[Tuple[str, str], int] = defaultdict(int)  # seen count per stratum

    t0 = time.time()
    n_rows = n_usable = n_dupe = 0
    dctx = zstd.ZstdDecompressor()
    with open(zst, "rb") as fh, dctx.stream_reader(fh) as reader:
        text = _text_stream(reader)
        rd = csv.DictReader(text)
        for row in rd:
            n_rows += 1
            if n_rows % 500_000 == 0:
                kept = sum(len(v) for v in reservoirs.values())
                log.info("  scanned %d rows, reservoir=%d (%.0fs)",
                         n_rows, kept, time.time() - t0)
            try:
                rating = int(row["Rating"])
            except (KeyError, ValueError):
                continue
            bucket = _bucket(rating)
            if bucket is None:
                continue
            themes = (row.get("Themes") or "").split()
            motif = _primary_motif(themes)
            cap = per_stratum
            if motif in FORCING_MOTIFS or (set(themes) & FORCING_MOTIFS):
                cap = max(1, per_stratum // 4)
            key = (bucket, motif)
            counts[key] += 1
            res = reservoirs[key]
            # Reservoir sampling: fill to cap, then replace with prob cap/count.
            if len(res) < cap:
                cand = _mk_candidate(row, rating, themes, motif, bucket)
                if cand is None:
                    continue
                res.append(cand)
            else:
                j = rng.randint(0, counts[key] - 1)
                if j < cap:
                    cand = _mk_candidate(row, rating, themes, motif, bucket)
                    if cand is not None:
                        res[j] = cand

    # Flatten, dedup vs existing + within-sample, cap to target.
    sample: List[dict] = []
    sample_epds: set = set()
    for res in reservoirs.values():
        for cand in res:
            n_usable += 1
            ek = cand.pop("_epd")
            if ek in seen or ek in sample_epds:
                n_dupe += 1
                continue
            sample_epds.add(ek)
            sample.append(cand)
    rng.shuffle(sample)
    if args.target and len(sample) > args.target:
        sample = _proportional_cap(sample, args.target, rng)

    SAMPLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE_OUT.open("w", encoding="utf-8") as out:
        for cand in sample:
            out.write(json.dumps(cand, ensure_ascii=False) + "\n")

    bucket_ct = Counter(c["bucket"] for c in sample)
    motif_ct = Counter(c["motif"] for c in sample)
    meta = {
        "rows_scanned": n_rows,
        "reservoir_usable": n_usable,
        "deduped_out": n_dupe,
        "sample_size": len(sample),
        "per_stratum": per_stratum,
        "target": args.target,
        "n_strata": len(reservoirs),
        "by_bucket": dict(sorted(bucket_ct.items())),
        "by_motif": dict(motif_ct.most_common()),
        "seconds": round(time.time() - t0, 1),
        "seed": SEED,
        "source": str(zst),
    }
    SAMPLE_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {len(sample)} candidates -> {SAMPLE_OUT}")
    return 0


def _text_stream(reader):
    """Wrap a binary zstd reader as a decoded text line iterator for csv."""
    import io
    return io.TextIOWrapper(io.BufferedReader(reader, buffer_size=1 << 20),
                            encoding="utf-8", newline="")


def _mk_candidate(row: dict, rating: int, themes: List[str], motif: str,
                  bucket: str) -> Optional[dict]:
    """Build a mining candidate from the puzzle's SETUP position (the raw FEN).

    We mine the *setup* position (side to move = the player about to continue the
    game), not the tactical solver position: puzzle solutions are forcing
    (singleton pool -> not discriminating; measured ~0.7% yield), while the setup
    position is a normal, non-forcing position where the tier-appropriate move
    genuinely varies (measured ~44% discriminating). The puzzle's rating/motif
    still label it, and the solution is used only to validate the row + keep the
    game continuation as context.
    """
    fen = row.get("FEN", "")
    sp = solver_position(fen, row.get("Moves", ""))  # validates puzzle + game move
    if sp is None:
        return None
    solver_fen, game_uci, game_san = sp
    try:
        board = chess.Board(fen)
    except ValueError:
        return None
    if board.is_game_over():
        return None
    ek = _epd_key(fen)
    if ek is None:
        return None
    return {
        "id": row.get("PuzzleId", ""),
        "puzzle_id": row.get("PuzzleId", ""),
        "fen": fen,                    # SETUP position (the position we coach)
        "solver_fen": solver_fen,      # tactical position (context only)
        "rating": rating,
        "themes": themes,
        "motif": motif,
        "bucket": bucket,
        "opening_tags": row.get("OpeningTags", ""),
        "popularity": _safe_int(row.get("Popularity")),
        "nb_plays": _safe_int(row.get("NbPlays")),
        "game_move_uci": game_uci,     # what was played next in the game (context)
        "game_move_san": game_san,
        "_epd": ek,
    }


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def _proportional_cap(sample: List[dict], target: int, rng: random.Random) -> List[dict]:
    """Cap to ``target`` while preserving motif balance (round-robin by motif)."""
    by_motif: Dict[str, List[dict]] = defaultdict(list)
    for c in sample:
        by_motif[c["motif"]].append(c)
    for lst in by_motif.values():
        rng.shuffle(lst)
    out: List[dict] = []
    motifs = list(by_motif)
    rng.shuffle(motifs)
    idx = 0
    while len(out) < target and any(by_motif.values()):
        m = motifs[idx % len(motifs)]
        if by_motif[m]:
            out.append(by_motif[m].pop())
        idx += 1
    return out


# --------------------------------------------------------------------------- #
# Stage 2: engine + Maia mining (CPU process pool)
# --------------------------------------------------------------------------- #
# Per-worker persistent engines (set in the initializer, reused across tasks).
_SF = None  # chess.engine.SimpleEngine


def _worker_init(stockfish_path: str) -> None:
    """Open one persistent, single-threaded Stockfish per worker process."""
    global _SF
    import chess.engine
    _SF = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        _SF.configure({"Threads": 1, "Hash": 96})
    except Exception:  # noqa: BLE001 - non-fatal; defaults are fine
        pass


def _sound_pool_persistent(fen: str, movetime_ms: int, multipv: int,
                           tolerance_cp: int) -> List[Dict[str, Any]]:
    """Stockfish sound pool using the worker's persistent engine (best-first)."""
    from src.engine import stockfish_engine as se
    analysis = se._analyze_impl(_SF, fen, multipv, movetime_ms)  # type: ignore[arg-type]
    lines = analysis["best"]
    if not lines:
        return []
    best_cp = int(lines[0]["cp"])
    max_loss = min(tolerance_cp, settings.BLUNDER_CP - 1)
    pool: List[Dict[str, Any]] = []
    for line in lines:
        cp = int(line["cp"])
        if best_cp - cp <= max_loss:
            pool.append({"uci": line["uci"], "san": line["san"], "cp": cp,
                         "pv": line["pv"]})
    return pool


def _student_move_for_tier(pool: List[Dict[str, Any]], policy: Dict[str, float],
                           best_cp: int) -> Dict[str, Any]:
    """The most human move IN the pool at this tier (the coaching frame move)."""
    ranked = sorted(pool, key=lambda m: (-policy.get(m["uci"], 0.0), -m["cp"]))
    top = ranked[0]
    return {"san": top["san"], "uci": top["uci"],
            "cp_loss": max(0, best_cp - int(top["cp"])),
            "severity": _severity(max(0, best_cp - int(top["cp"])))}


def _severity(cp_loss: int) -> str:
    if cp_loss < settings.INACCURACY_CP:
        return "none"
    if cp_loss < settings.MISTAKE_CP:
        return "inaccuracy"
    if cp_loss < settings.BLUNDER_CP:
        return "mistake"
    return "blunder"


def _mine_one(cand: dict, movetime_ms: int, multipv: int, tolerance_cp: int,
              maia_top_k: int) -> Optional[dict]:
    """Return an enriched mined row iff the position is discriminating, else None."""
    from src.teacher.tier_select import maia_policy_map, select_tier_move

    fen = cand["fen"]
    try:
        pool = _sound_pool_persistent(fen, movetime_ms, multipv, tolerance_cp)
    except Exception:  # noqa: BLE001 - one bad position must not kill the worker
        return None
    if len(pool) < 2:
        return None

    advanced_uci = pool[0]["uci"]  # w=0 => sharpest sound move (engine best)
    # Beginner Maia decides the moat first (cheapest path to a reject).
    pol_b = maia_policy_map(fen, "beginner")
    if not pol_b:
        return None
    beg = select_tier_move("beginner", pool, pol_b)
    if beg.uci == advanced_uci:
        return None  # not discriminating: beginner already lands on the sharp move

    # Survivor: compute the other two tiers' Maia + canonical picks.
    pol_i = maia_policy_map(fen, "intermediate")
    pol_a = maia_policy_map(fen, "advanced")
    inter = select_tier_move("intermediate", pool, pol_i)
    adv = select_tier_move("advanced", pool, pol_a)
    best_cp = int(pool[0]["cp"])

    def top_maia(pol: Dict[str, float]) -> List[Dict[str, Any]]:
        board = chess.Board(fen)
        items = sorted(pol.items(), key=lambda kv: -kv[1])[:maia_top_k]
        out = []
        for uci, p in items:
            try:
                mv = chess.Move.from_uci(uci)
                if mv in board.legal_moves:
                    out.append({"uci": uci, "san": board.san(mv),
                                "policy": round(float(p), 4)})
            except ValueError:
                continue
        return out

    picks = {"beginner": beg, "intermediate": inter, "advanced": adv}
    maia_by_tier = {"beginner": top_maia(pol_b), "intermediate": top_maia(pol_i),
                    "advanced": top_maia(pol_a)}
    student_by_tier = {
        "beginner": _student_move_for_tier(pool, pol_b, best_cp),
        "intermediate": _student_move_for_tier(pool, pol_i, best_cp),
        "advanced": _student_move_for_tier(pool, pol_a, best_cp),
    }
    picks_out = {t: {"uci": p.uci, "san": p.san, "pool_rank": p.pool_rank,
                     "is_engine_best": p.is_engine_best, "policy": p.policy,
                     "score": p.score} for t, p in picks.items()}

    # Board-class of the triad (how the three tiers diverge) — reporting signal.
    b, i, a = beg.uci, inter.uci, adv.uci
    if b == i == a:
        cls = "all_same"
    elif b == a and b != i:
        cls = "collapse_BA"
    elif b == i and i != a:
        cls = "BI"
    elif i == a and i != b:
        cls = "IA"
    else:
        cls = "full"

    return {
        "id": cand["id"],
        "puzzle_id": cand["puzzle_id"],
        "fen": fen,
        "rating": cand["rating"],
        "themes": cand["themes"],
        "motif": cand["motif"],
        "bucket": cand["bucket"],
        "opening_tags": cand.get("opening_tags", ""),
        "game_move_san": cand.get("game_move_san"),
        "sound_pool": pool,
        "best_cp": best_cp,
        "tier_picks": picks_out,
        "maia_by_tier": maia_by_tier,
        "student_by_tier": student_by_tier,
        "board_class": cls,
        "n_pool": len(pool),
    }


def _existing_mined_ids(path: Path) -> set:
    ids: set = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(str(json.loads(line).get("id")))
            except json.JSONDecodeError:
                continue
    return ids


def cmd_mine(args: argparse.Namespace) -> int:
    """Engine+Maia the sampled candidates; keep discriminating positions."""
    if not SAMPLE_OUT.exists():
        raise SystemExit(f"BLOCKED: no sample at {SAMPLE_OUT}; run `sample` first")
    cands: List[dict] = []
    with SAMPLE_OUT.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cands.append(json.loads(line))
    if args.limit:
        cands = cands[: args.limit]

    CURATE_DIR.mkdir(parents=True, exist_ok=True)
    done = _existing_mined_ids(MINED_OUT) if not args.fresh else set()
    if done:
        before = len(cands)
        cands = [c for c in cands if str(c["id"]) not in done]
        log.info("resume: %d already mined, %d remaining", before - len(cands), len(cands))

    mode = "w" if args.fresh else "a"
    already = len(done)
    t0 = time.time()
    n_kept = n_seen = 0
    kept_cls: Counter = Counter()
    kept_motif: Counter = Counter()
    kept_bucket: Counter = Counter()

    with MINED_OUT.open(mode, encoding="utf-8") as out:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_worker_init,
            initargs=(settings.STOCKFISH_BIN,),
        ) as pool:
            futs = {
                pool.submit(_mine_one, c, args.movetime, args.multipv,
                            args.tolerance, args.maia_top_k): c
                for c in cands
            }
            stop = False
            for fut in as_completed(futs):
                n_seen += 1
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log.debug("mine worker error: %s", exc)
                    row = None
                if row is not None:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    n_kept += 1
                    kept_cls[row["board_class"]] += 1
                    kept_motif[row["motif"]] += 1
                    kept_bucket[row["bucket"]] += 1
                if n_seen % 200 == 0:
                    rate = 100.0 * n_kept / max(1, n_seen)
                    log.info("  mined %d/%d, kept %d (%.1f%% discriminating) %.0fs",
                             n_seen, len(cands), n_kept, rate, time.time() - t0)
                # Early stop once we have enough discriminating positions.
                if args.max_keep and (already + n_kept) >= args.max_keep and not stop:
                    stop = True
                    log.info("reached --max-keep=%d (kept %d this run + %d prior); "
                             "cancelling remaining", args.max_keep, n_kept, already)
                    for f in futs:
                        f.cancel()
                    break

    total_kept = len(_existing_mined_ids(MINED_OUT))
    meta = {
        "analyzed": n_seen,
        "kept_this_run": n_kept,
        "kept_total": total_kept,
        "discriminating_rate": round(100.0 * n_kept / max(1, n_seen), 2),
        "by_board_class": dict(kept_cls.most_common()),
        "by_motif": dict(kept_motif.most_common()),
        "by_bucket": dict(sorted(kept_bucket.items())),
        "movetime_ms": args.movetime, "multipv": args.multipv,
        "tolerance_cp": args.tolerance, "workers": args.workers,
        "seconds": round(time.time() - t0, 1),
    }
    MINE_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {n_kept} discriminating positions this run -> {MINED_OUT} "
          f"({total_kept} total)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    rc = cmd_sample(args)
    if rc != 0:
        return rc
    return cmd_mine(args)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_sample(sp):
        sp.add_argument("--target", type=int, default=6000,
                        help="Max candidates to keep in the sample (motif-balanced).")
        sp.add_argument("--per-stratum", type=int, default=80,
                        help="Reservoir cap per (rating-bucket x motif) stratum.")
        sp.add_argument("--no-dedup", action="store_true",
                        help="Skip dedup against existing train/eval FENs.")

    def add_mine(sp):
        sp.add_argument("--workers", type=int, default=4)
        sp.add_argument("--movetime", type=int, default=settings.DEFAULT_MOVETIME_MS)
        sp.add_argument("--multipv", type=int, default=settings.MULTIPV)
        sp.add_argument("--tolerance", type=int, default=settings.SOUND_TOLERANCE_CP)
        sp.add_argument("--maia-top-k", dest="maia_top_k", type=int, default=6)
        sp.add_argument("--limit", type=int, default=None)
        sp.add_argument("--max-keep", type=int, default=None,
                        help="Stop after this many discriminating positions.")
        sp.add_argument("--fresh", action="store_true")

    ps = sub.add_parser("sample", help="Stratified deduped puzzle sample.")
    add_sample(ps)
    ps.set_defaults(func=cmd_sample)

    pm = sub.add_parser("mine", help="Engine+Maia mine discriminating positions.")
    add_mine(pm)
    pm.set_defaults(func=cmd_mine)

    pr = sub.add_parser("run", help="sample + mine.")
    add_sample(pr)
    add_mine(pr)
    pr.set_defaults(func=cmd_run)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
