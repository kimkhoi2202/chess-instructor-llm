"""Stockfish UCI wrapper for the chess-coaching dataset pipeline.

This module is the engine *guardrail* used later to (a) judge how bad a human
move was and (b) build the "sound move pool" that a teacher LLM picks a
teaching move from. It intentionally does NOT decide the lesson — it only
supplies engine truth (soundness + mistake magnitude).

Design notes
------------
- All centipawn (``cp``) values are reported from the **side-to-move POV** of
  the *input* FEN (positive = good for the player to move), via
  ``score.pov(board.turn).score(mate_score=MATE_SCORE)``.
- Mate is reported separately as a signed integer (``+N`` = side to move mates
  in N, ``-N`` = side to move gets mated in N) or ``None``.
- Principal variations are converted to SAN on a board copy and capped to
  ``PV_MAX_PLIES`` plies.
- Each public function opens the engine once via a context manager and quits
  it, so there are no leaked engine processes.
- Illegal / unparseable moves and FENs raise a clear ``ValueError``.

No secrets and no network access are used here.

CLI
---
    python -m src.engine.stockfish_engine analyze     --fen "<FEN>" [--multipv 5]  [--movetime 300]
    python -m src.engine.stockfish_engine eval-move   --fen "<FEN>" --move Qh5     [--movetime 300]
    python -m src.engine.stockfish_engine classify    --fen "<FEN>" --move Qh5     [--movetime 300]
    python -m src.engine.stockfish_engine sound-pool  --fen "<FEN>" [--tolerance 150] [--multipv 8] [--movetime 300]
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence

import chess
import chess.engine

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Default Stockfish binary (overridable via the ``STOCKFISH_PATH`` env var or
#: the ``--engine`` CLI flag).
DEFAULT_STOCKFISH_PATH: str = os.environ.get(
    "STOCKFISH_PATH", "/opt/homebrew/bin/stockfish"
)

#: Magnitude used to project mate scores onto the centipawn scale.
MATE_SCORE: int = 100_000

#: Maximum number of half-moves (plies) kept in a returned principal variation.
PV_MAX_PLIES: int = 6

#: Centipawn loss above which a move is a "blunder" (used by ``sound_pool`` and
#: ``classify_mistake``).
BLUNDER_CP: int = 250

#: Ordered (upper-bound-exclusive, label) thresholds for mistake severity.
_SEVERITY_THRESHOLDS: Sequence[tuple[int, str]] = (
    (50, "none"),
    (100, "inaccuracy"),
    (250, "mistake"),
)


# --------------------------------------------------------------------------- #
# Engine lifecycle
# --------------------------------------------------------------------------- #


@contextmanager
def open_engine(path: str = DEFAULT_STOCKFISH_PATH) -> Iterator[chess.engine.SimpleEngine]:
    """Open a Stockfish engine as a context manager and guarantee ``quit()``.

    Parameters
    ----------
    path:
        Path to the Stockfish UCI binary.

    Yields
    ------
    chess.engine.SimpleEngine
        A ready-to-use engine instance.
    """
    try:
        engine = chess.engine.SimpleEngine.popen_uci(path)
    except FileNotFoundError as exc:  # pragma: no cover - environment specific
        raise ValueError(f"Stockfish binary not found at {path!r}") from exc
    try:
        yield engine
    finally:
        engine.quit()


# --------------------------------------------------------------------------- #
# Small internal helpers
# --------------------------------------------------------------------------- #


def _board_from_fen(fen: str) -> chess.Board:
    """Parse ``fen`` into a validated :class:`chess.Board` or raise ``ValueError``."""
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"Invalid FEN {fen!r}: {exc}") from exc
    if not board.is_valid():
        raise ValueError(f"Invalid (illegal) FEN position: {fen!r}")
    return board


def _parse_move(board: chess.Board, move: str) -> chess.Move:
    """Parse a UCI or SAN move string that is legal in ``board``.

    Raises
    ------
    ValueError
        If the move cannot be parsed or is illegal in the given position.
    """
    text = move.strip()
    if not text:
        raise ValueError("Empty move string")

    # Try UCI first (e.g. "g1f3", "e7e8q").
    try:
        candidate = chess.Move.from_uci(text)
    except ValueError:
        candidate = None
    if candidate is not None and candidate in board.legal_moves:
        return candidate

    # Fall back to SAN (e.g. "Nf3", "Qh5", "O-O", "exd5").
    try:
        return board.parse_san(text)
    except ValueError:
        pass

    raise ValueError(f"Illegal or unparseable move {move!r} for FEN {board.fen()!r}")


def _pv_to_san(board: chess.Board, pv: Sequence[chess.Move], cap: int = PV_MAX_PLIES) -> List[str]:
    """Convert a principal variation to SAN on a board copy, capped to ``cap`` plies."""
    preview = board.copy(stack=False)
    sans: List[str] = []
    for mv in list(pv)[:cap]:
        try:
            sans.append(preview.san(mv))
        except (ValueError, AssertionError):
            break
        preview.push(mv)
    return sans


def _decode(board: chess.Board, info: chess.engine.InfoDict) -> Dict[str, Any]:
    """Turn one engine ``info`` dict into cp/mate/uci/san/pv fields (side-to-move POV)."""
    pov = info["score"].pov(board.turn)
    cp = pov.score(mate_score=MATE_SCORE)
    mate = pov.mate()
    pv_moves: List[chess.Move] = list(info.get("pv") or [])

    uci: Optional[str] = pv_moves[0].uci() if pv_moves else None
    san: Optional[str] = None
    if pv_moves:
        try:
            san = board.san(pv_moves[0])
        except (ValueError, AssertionError):
            san = None

    return {
        "uci": uci,
        "san": san,
        "cp": int(cp),
        "mate": mate,
        "pv": _pv_to_san(board, pv_moves),
    }


def _analyse(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    *,
    multipv: int,
    movetime_ms: int,
    root_moves: Optional[Sequence[chess.Move]] = None,
) -> List[chess.engine.InfoDict]:
    """Run ``engine.analyse`` and always return a list of info dicts."""
    limit = chess.engine.Limit(time=movetime_ms / 1000.0)
    infos = engine.analyse(board, limit, multipv=multipv, root_moves=root_moves)
    if isinstance(infos, dict):  # multipv==1 may return a single dict on some versions
        return [infos]
    return list(infos)


def _severity(cp_loss: int) -> str:
    """Map a (clamped) centipawn loss to a severity label."""
    for upper, label in _SEVERITY_THRESHOLDS:
        if cp_loss < upper:
            return label
    return "blunder"


# --------------------------------------------------------------------------- #
# Engine-bound implementations (reused so each public call opens one engine)
# --------------------------------------------------------------------------- #


def _analyze_impl(
    engine: chess.engine.SimpleEngine, fen: str, multipv: int, movetime_ms: int
) -> Dict[str, Any]:
    board = _board_from_fen(fen)
    best: List[Dict[str, Any]] = []
    if not board.is_game_over():
        infos = _analyse(engine, board, multipv=multipv, movetime_ms=movetime_ms)
        for rank, info in enumerate(infos, start=1):
            best.append({"rank": rank, **_decode(board, info)})
    return {
        "fen": board.fen(),
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "best": best,
    }


def _eval_move_impl(
    engine: chess.engine.SimpleEngine, fen: str, move: str, movetime_ms: int
) -> Dict[str, Any]:
    board = _board_from_fen(fen)
    mv = _parse_move(board, move)
    san = board.san(mv)
    infos = _analyse(
        engine, board, multipv=1, movetime_ms=movetime_ms, root_moves=[mv]
    )
    pov = infos[0]["score"].pov(board.turn)
    return {
        "uci": mv.uci(),
        "san": san,
        "cp": int(pov.score(mate_score=MATE_SCORE)),
        "mate": pov.mate(),
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def analyze(fen: str, multipv: int = 5, movetime_ms: int = 300) -> Dict[str, Any]:
    """Analyze ``fen`` and return the top ``multipv`` engine lines.

    Parameters
    ----------
    fen:
        Position in Forsyth-Edwards Notation.
    multipv:
        Number of distinct best lines to return.
    movetime_ms:
        Per-position search time budget in milliseconds.

    Returns
    -------
    dict
        ``{"fen", "side_to_move", "best": [{"rank", "uci", "san", "cp",
        "mate", "pv"}, ...]}`` where ``cp`` is from the side-to-move POV and
        ``pv`` is a SAN list capped to :data:`PV_MAX_PLIES` plies.
    """
    with open_engine() as engine:
        return _analyze_impl(engine, fen, multipv, movetime_ms)


def eval_move(fen: str, move_uci_or_san: str, movetime_ms: int = 300) -> Dict[str, Any]:
    """Evaluate one specific move in ``fen`` (UCI or SAN accepted).

    Returns
    -------
    dict
        ``{"uci", "san", "cp", "mate"}`` with ``cp`` from the side-to-move POV.

    Raises
    ------
    ValueError
        If the move is illegal or cannot be parsed.
    """
    with open_engine() as engine:
        return _eval_move_impl(engine, fen, move_uci_or_san, movetime_ms)


def classify_mistake(fen: str, played_move: str, movetime_ms: int = 300) -> Dict[str, Any]:
    """Score how much a played move loses vs. the engine's best move.

    Severity thresholds (on the clamped ``cp_loss``): ``<50`` none, ``<100``
    inaccuracy, ``<250`` mistake, ``>=250`` blunder.

    Returns
    -------
    dict
        ``{"best_cp", "played_cp", "cp_loss", "severity"}`` (all side-to-move POV).
    """
    with open_engine() as engine:
        analysis = _analyze_impl(engine, fen, multipv=1, movetime_ms=movetime_ms)
        if not analysis["best"]:
            raise ValueError(f"No legal moves to analyze for FEN {fen!r}")
        best_cp = int(analysis["best"][0]["cp"])
        played_cp = int(_eval_move_impl(engine, fen, played_move, movetime_ms)["cp"])

    cp_loss = max(0, best_cp - played_cp)
    return {
        "best_cp": best_cp,
        "played_cp": played_cp,
        "cp_loss": cp_loss,
        "severity": _severity(cp_loss),
    }


def sound_pool(
    fen: str,
    tolerance_cp: int = 150,
    multipv: int = 8,
    movetime_ms: int = 300,
) -> List[Dict[str, Any]]:
    """Return the pool of *sound* moves for ``fen``.

    A move is sound when its eval is within ``tolerance_cp`` of the best eval
    **and** it is not a blunder (cp loss ``< BLUNDER_CP``). This is the
    guardrailed candidate set a teacher LLM later picks a teaching move from.

    Returns
    -------
    list of dict
        Each entry is ``{"uci", "san", "cp", "pv"}`` (cp from side-to-move POV,
        pv a SAN list capped to :data:`PV_MAX_PLIES` plies), ordered best-first.
    """
    with open_engine() as engine:
        analysis = _analyze_impl(engine, fen, multipv=multipv, movetime_ms=movetime_ms)

    lines = analysis["best"]
    if not lines:
        return []

    best_cp = int(lines[0]["cp"])
    max_loss = min(tolerance_cp, BLUNDER_CP - 1)  # never include a blunder
    pool: List[Dict[str, Any]] = []
    for line in lines:
        cp = int(line["cp"])
        if best_cp - cp <= max_loss:
            pool.append(
                {"uci": line["uci"], "san": line["san"], "cp": cp, "pv": line["pv"]}
            )
    return pool


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stockfish analysis wrapper for the chess-coaching pipeline.",
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_STOCKFISH_PATH,
        help="Path to the Stockfish binary (default: %(default)s).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="Top-N engine lines for a FEN.")
    p_an.add_argument("--fen", required=True)
    p_an.add_argument("--multipv", type=int, default=5)
    p_an.add_argument("--movetime", type=int, default=300, help="ms per position")

    p_ev = sub.add_parser("eval-move", help="Evaluate one specific move.")
    p_ev.add_argument("--fen", required=True)
    p_ev.add_argument("--move", required=True, help="UCI or SAN")
    p_ev.add_argument("--movetime", type=int, default=300, help="ms per position")

    p_cl = sub.add_parser("classify", help="Classify a played move's mistake severity.")
    p_cl.add_argument("--fen", required=True)
    p_cl.add_argument("--move", required=True, help="UCI or SAN")
    p_cl.add_argument("--movetime", type=int, default=300, help="ms per position")

    p_sp = sub.add_parser("sound-pool", help="Pool of sound (non-blunder) moves.")
    p_sp.add_argument("--fen", required=True)
    p_sp.add_argument("--tolerance", type=int, default=150, help="cp tolerance")
    p_sp.add_argument("--multipv", type=int, default=8)
    p_sp.add_argument("--movetime", type=int, default=300, help="ms per position")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Prints JSON results to stdout."""
    args = _build_parser().parse_args(argv)

    # Allow --engine to override the module default for this invocation.
    global DEFAULT_STOCKFISH_PATH
    DEFAULT_STOCKFISH_PATH = args.engine

    if args.command == "analyze":
        result: Any = analyze(args.fen, multipv=args.multipv, movetime_ms=args.movetime)
    elif args.command == "eval-move":
        result = eval_move(args.fen, args.move, movetime_ms=args.movetime)
    elif args.command == "classify":
        result = classify_mistake(args.fen, args.move, movetime_ms=args.movetime)
    elif args.command == "sound-pool":
        result = sound_pool(
            args.fen,
            tolerance_cp=args.tolerance,
            multipv=args.multipv,
            movetime_ms=args.movetime,
        )
    else:  # pragma: no cover - argparse enforces valid commands
        raise SystemExit(2)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
