"""Held-out scenario set: sample, dedup, and compute ground truth ONCE.

This builds the shared input every model is scored on. Three guarantees:

1. **Held out.** A position is eligible only if its board (piece placement +
   side to move) does NOT appear in ``data/dataset/train.jsonl`` or
   ``valid.jsonl``. Those files are chat rows whose user message embeds the
   ASCII board, so we reconstruct the placement-FEN from that grid and dedup on
   it — reading the *actual* training data, not re-deriving a split.

2. **Balanced.** Positions are drawn to spread across tier x game-phase x
   mistake-severity cells (a per-cell cap on a deterministic shuffle), then
   topped up to the target count from whatever coachable positions were found.

3. **Ground truth computed once.** For each accepted position we run Stockfish
   (mistake severity + sound-move pool) and Maia (human-likelihoods) and persist
   everything, so all five models — and the objective scorer — see identical
   inputs and identical soundness truth.

Resumable: each scenario is appended immediately; re-running skips positions
already on disk and keeps filling until the target is met.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import chess

from config import settings
from src.engine import maia_engine, stockfish_engine

from . import config as bcfg
from .io_utils import append_jsonl, read_jsonl

log = logging.getLogger("benchmark.scenarios")

TIER_ORDER: Tuple[str, ...] = ("beginner", "intermediate", "advanced")
PHASE_ORDER: Tuple[str, ...] = ("opening", "middlegame", "endgame")
SEVERITY_ORDER: Tuple[str, ...] = ("inaccuracy", "mistake", "blunder")

_MAJOR_MINOR = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)


# --------------------------------------------------------------------------- #
# Board-key dedup (held-out check against the training data)
# --------------------------------------------------------------------------- #


def board_key_from_fen(fen: str) -> Optional[str]:
    """Placement + side-to-move key for a FEN, e.g. ``"rnbq... w"`` (or None)."""
    try:
        b = chess.Board(fen)
    except ValueError:
        return None
    return f"{b.board_fen()} {'w' if b.turn else 'b'}"


def _rank_from_ascii(tokens: List[str]) -> Optional[str]:
    """Turn one ASCII board row (8 tokens, '.'=empty) into a FEN rank string."""
    if len(tokens) != 8:
        return None
    out: List[str] = []
    empty = 0
    for t in tokens:
        if t == ".":
            empty += 1
            continue
        if len(t) != 1 or t not in "prnbqkPRNBQK":
            return None
        if empty:
            out.append(str(empty))
            empty = 0
        out.append(t)
    if empty:
        out.append(str(empty))
    return "".join(out)


def board_key_from_user_content(content: str) -> Optional[str]:
    """Reconstruct the placement+turn key from a rendered coach user message."""
    lines = content.split("\n")
    grid_start = None
    for i, line in enumerate(lines):
        if line.strip() == "Board:":
            grid_start = i + 1
            break
    if grid_start is None or grid_start + 8 > len(lines):
        return None
    ranks: List[str] = []
    for row in lines[grid_start : grid_start + 8]:
        rank = _rank_from_ascii(row.split())
        if rank is None:
            return None
        ranks.append(rank)
    placement = "/".join(ranks)
    turn = "w" if "White to move" in content else ("b" if "Black to move" in content else None)
    if turn is None:
        return None
    return f"{placement} {turn}"


def dataset_board_keys(paths: List[Path]) -> Set[str]:
    """Every placement+turn key present in the given chat-format JSONL files."""
    keys: Set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            for msg in row.get("messages", []):
                if msg.get("role") == "user":
                    key = board_key_from_user_content(str(msg.get("content", "")))
                    if key:
                        keys.add(key)
                    break
    return keys


# --------------------------------------------------------------------------- #
# Game-phase classification
# --------------------------------------------------------------------------- #


def classify_phase(board: chess.Board) -> str:
    """Coarse game phase from material + move number (documented in the report)."""
    pieces = board.piece_map().values()
    major_minor = sum(1 for p in pieces if p.piece_type in _MAJOR_MINOR)
    if major_minor <= 6:
        return "endgame"
    if board.fullmove_number <= 12:
        return "opening"
    return "middlegame"


# --------------------------------------------------------------------------- #
# Ground truth for one position
# --------------------------------------------------------------------------- #


def _san_from_uci(fen: str, uci: str) -> str:
    try:
        return chess.Board(fen).san(chess.Move.from_uci(uci))
    except (ValueError, AssertionError):
        return uci


def compute_ground_truth(
    pos: Dict[str, Any],
    *,
    movetime_ms: int,
    tolerance_cp: int,
    multipv: int,
    maia_top_k: int,
) -> Optional[Dict[str, Any]]:
    """Return a full scenario dict for ``pos``, or None if it is not coachable.

    Not coachable == the played move is not a real mistake (severity ``none``)
    or the position has an empty sound pool. Maia is best-effort (empty list if
    unavailable) since it is a helpful signal, not a hard requirement.
    """
    fen = pos["fen"]
    tier = pos["tier"]
    played = pos["played_move_uci"]

    mistake = stockfish_engine.classify_mistake(fen, played, movetime_ms=movetime_ms)
    if mistake["severity"] == "none":
        return None

    pool_raw = stockfish_engine.sound_pool(
        fen, tolerance_cp=tolerance_cp, multipv=multipv, movetime_ms=movetime_ms
    )
    sound_pool = [
        {"uci": m["uci"], "san": m["san"], "cp": int(m["cp"]), "pv": list(m.get("pv") or [])}
        for m in pool_raw
        if m.get("san") and m.get("uci")
    ]
    if not sound_pool:
        return None

    try:
        maia = [
            {"uci": m["uci"], "san": m["san"], "policy": float(m["policy"])}
            for m in maia_engine.human_moves(fen, tier, top_k=maia_top_k)["moves"]
        ]
    except Exception as exc:  # noqa: BLE001 - Maia is a helpful signal, not required
        log.warning("%s: Maia unavailable (%s); continuing without it", pos.get("id"), exc)
        maia = []

    board = chess.Board(fen)
    return {
        "id": str(pos.get("id")),
        "fen": fen,
        "tier": tier,
        "phase": classify_phase(board),
        "severity": str(mistake["severity"]),
        "student_move": {
            "san": pos.get("played_move_san") or _san_from_uci(fen, played),
            "uci": played,
            "cp_loss": int(mistake["cp_loss"]),
            "severity": str(mistake["severity"]),
        },
        "sound_pool": sound_pool,
        "sound_uci": [m["uci"] for m in sound_pool],
        "maia": maia,
        "best_san": sound_pool[0]["san"],
        "best_cp": sound_pool[0]["cp"],
        "meta": {
            "movetime_ms": movetime_ms,
            "tolerance_cp": tolerance_cp,
            "multipv": multipv,
        },
    }


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #


def _load_positions(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def build_scenarios(
    *,
    positions_path: Path,
    n_target: int,
    per_cell_cap: int,
    movetime_ms: int,
    tolerance_cp: int,
    multipv: int,
    maia_top_k: int,
    max_eval: int,
) -> List[Dict[str, Any]]:
    """Sample + persist ~``n_target`` balanced, held-out, coachable scenarios."""
    positions = _load_positions(positions_path)
    if not positions:
        raise SystemExit(f"BLOCKED: no positions loaded from {positions_path}")

    excluded = dataset_board_keys([settings.DATASET / "train.jsonl", settings.DATASET / "valid.jsonl"])
    log.info("held-out filter: %d board-keys in train/valid to exclude", len(excluded))

    held_out = [p for p in positions if board_key_from_fen(p.get("fen", "")) not in excluded]
    log.info("positions: %d total, %d held-out eligible", len(positions), len(held_out))

    # Deterministic shuffle so the sample is reproducible.
    rng = random.Random(bcfg.SEED)
    rng.shuffle(held_out)

    # Resume: seed state from whatever is already on disk.
    existing = read_jsonl(bcfg.SCENARIOS_PATH)
    accepted: List[Dict[str, Any]] = list(existing)
    accepted_ids: Set[str] = {s["id"] for s in existing}
    cell_count: Dict[Tuple[str, str, str], int] = {}
    for s in existing:
        cell = (s["tier"], s["phase"], s["severity"])
        cell_count[cell] = cell_count.get(cell, 0) + 1
    if accepted:
        log.info("resume: %d scenarios already on disk", len(accepted))

    coachable_cache: List[Dict[str, Any]] = []
    evaluated = 0

    # Pass 1: balanced accept, capped per cell. Stop once the target is met OR we
    # have gathered enough coachable positions to top up to the target (this
    # bounds Stockfish work even when common cells cap out early).
    for pos in held_out:
        if len(accepted) >= n_target:
            break
        if len(coachable_cache) >= n_target + 40:
            break
        if evaluated >= max_eval:
            break
        pid = str(pos.get("id"))
        if pid in accepted_ids:
            continue
        evaluated += 1
        try:
            scn = compute_ground_truth(
                pos,
                movetime_ms=movetime_ms,
                tolerance_cp=tolerance_cp,
                multipv=multipv,
                maia_top_k=maia_top_k,
            )
        except Exception as exc:  # noqa: BLE001 - one bad position must not abort
            log.warning("skip %s: ground-truth failed (%s)", pid, exc)
            continue
        if scn is None:
            continue
        coachable_cache.append(scn)
        cell = (scn["tier"], scn["phase"], scn["severity"])
        if cell_count.get(cell, 0) < per_cell_cap and len(accepted) < n_target:
            cell_count[cell] = cell_count.get(cell, 0) + 1
            accepted.append(scn)
            accepted_ids.add(pid)
            append_jsonl(bcfg.SCENARIOS_PATH, scn)
            log.info(
                "  + [%3d/%d] %s  %s/%s/%s (pool=%d)",
                len(accepted), n_target, pid, scn["tier"], scn["phase"],
                scn["severity"], len(scn["sound_pool"]),
            )

    # Pass 2: top up from the coachable cache (cap-agnostic) if still short.
    if len(accepted) < n_target:
        for scn in sorted(coachable_cache, key=lambda s: s["id"]):
            if len(accepted) >= n_target:
                break
            if scn["id"] in accepted_ids:
                continue
            accepted.append(scn)
            accepted_ids.add(scn["id"])
            append_jsonl(bcfg.SCENARIOS_PATH, scn)
            log.info("  + [%3d/%d] %s (top-up)", len(accepted), n_target, scn["id"])

    log.info("scenarios built: %d (evaluated %d positions)", len(accepted), evaluated)
    return accepted


def load_scenarios() -> List[Dict[str, Any]]:
    """Load the persisted scenario set (ordered as written)."""
    return read_jsonl(bcfg.SCENARIOS_PATH)


def distribution(scenarios: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Counts by tier / phase / severity for the report + a sanity check."""
    out: Dict[str, Dict[str, int]] = {"tier": {}, "phase": {}, "severity": {}}
    for s in scenarios:
        for axis in ("tier", "phase", "severity"):
            out[axis][s[axis]] = out[axis].get(s[axis], 0) + 1
    return out
