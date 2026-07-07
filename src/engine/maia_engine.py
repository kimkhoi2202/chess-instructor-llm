"""Maia (human-at-rating) move prediction via lc0.

Where Stockfish (``stockfish_engine.py``) is the *guardrail* that supplies engine
truth (soundness + mistake magnitude), **Maia** answers a different question:

    "Which of these moves would a human at this rating band *actually* consider?"

That lets the downstream teacher LLM pick a teaching move from Stockfish's sound
pool that a real student at the target tier could plausibly find -- never a
superhuman-only move.

How it works
------------
Maia nets are lc0-compatible weight files (a residual policy/value net trained on
human Lichess games at a given rating). We run lc0 with a Maia net at
``go nodes 1`` -- a single neural-network forward pass with **no** tree search --
so the reported per-move **policy prior** ``P`` is exactly Maia's estimate of how
likely a human at that rating is to play each move.

The per-move policy distribution is read from lc0's ``VerboseMoveStats``
``info string`` lines, e.g.::

    info string e2e4  (322 ) N:  0 (+ 0) (P: 24.13%) (WL: ...) ... (V: ...)

We intentionally do **not** override ``PolicyTemperature`` -- each Maia net embeds
its own calibrated policy-softmax temperature, and lc0 adopts it on load, so the
raw ``P`` values already reflect the human move distribution the net was built to
produce.

Tier -> net mapping (matches ``config/settings.py``)
----------------------------------------------------
- ``beginner``     (1000-1200) -> ``maia-1100``
- ``intermediate`` (1300-1600) -> ``maia-1500``
- ``advanced``     (1700-2000) -> ``maia-1900``

One lc0 process is cached per net (started lazily, reused across calls, torn down
at interpreter exit). No secrets and no network access are used here.

CLI
---
    python -m src.engine.maia_engine human-moves --fen "<FEN>" --tier beginner [--top-k 5]
    python -m src.engine.maia_engine compare     --fen "<FEN>" [--tiers beginner advanced] [--top-k 5]
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import queue
import re
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict

import chess

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Coaching tier -> Maia network name. Kept in sync with ``config/settings.py``.
TIER_TO_NET: Dict[str, str] = {
    "beginner": "maia-1100",
    "intermediate": "maia-1500",
    "advanced": "maia-1900",
}

#: Project root (``.../chess-instructor-llm``); this file is ``src/engine/…``.
_ROOT: Path = Path(__file__).resolve().parents[2]

#: Directory holding the ``maia-XXXX.pb.gz`` weight files (``models/`` is
#: gitignored). Overridable via the ``MAIA_DIR`` env var.
MAIA_DIR: Path = Path(os.environ.get("MAIA_DIR", str(_ROOT / "models" / "maia")))

#: lc0 binary. Overridable via ``LC0_BIN`` / ``LC0_PATH``.
LC0_BIN: str = os.environ.get("LC0_BIN") or os.environ.get("LC0_PATH") or "lc0"

#: Seconds to wait for the engine to load its net during the UCI handshake.
_HANDSHAKE_TIMEOUT_S: float = 60.0

#: Seconds to wait for a single ``go nodes 1`` forward pass to return ``bestmove``.
_SEARCH_TIMEOUT_S: float = 30.0

#: Parses a ``VerboseMoveStats`` line, capturing the UCI move and its policy %.
#: Example: ``info string e2e4  (322 ) N: 0 (+ 0) (P: 24.13%) (WL: ...) ...``
_VERBOSE_RE = re.compile(
    r"^info string\s+([a-h][1-8][a-h][1-8][qrbnQRBN]?)\b.*?\(P:\s*([0-9.]+)%\)"
)

#: Sentinel pushed onto the read queue when lc0's stdout closes.
_EOF = object()


# --------------------------------------------------------------------------- #
# Typed results
# --------------------------------------------------------------------------- #


class MovePolicy(TypedDict):
    """One candidate move and how human-likely it is at the target tier."""

    uci: str
    san: str
    policy: float  # Maia policy prior in [0, 1] (higher = more human-likely)


class HumanMovesResult(TypedDict):
    """Return payload of :func:`human_moves`."""

    fen: str
    tier: str
    maia_net: str
    moves: List[MovePolicy]


# --------------------------------------------------------------------------- #
# Small helpers
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


def net_for_tier(tier: str) -> str:
    """Resolve a tier label (or a raw net name) to a Maia network name.

    Parameters
    ----------
    tier:
        A tier label (``"beginner"``/``"intermediate"``/``"advanced"``,
        case-insensitive) or a concrete net name such as ``"maia-1500"``.

    Returns
    -------
    str
        The Maia network name, e.g. ``"maia-1100"``.

    Raises
    ------
    ValueError
        If ``tier`` is neither a known tier nor a known net name.
    """
    key = tier.strip().lower()
    if key in TIER_TO_NET:
        return TIER_TO_NET[key]
    if key in set(TIER_TO_NET.values()):
        return key
    valid = ", ".join(sorted(TIER_TO_NET)) + ", " + ", ".join(TIER_TO_NET.values())
    raise ValueError(f"Unknown tier {tier!r}. Expected one of: {valid}.")


def weights_path(net_name: str) -> Path:
    """Return the ``.pb.gz`` weights path for ``net_name`` or raise ``ValueError``."""
    path = MAIA_DIR / f"{net_name}.pb.gz"
    if not path.is_file():
        raise ValueError(
            f"Maia weights not found: {path}. Download the Maia nets into "
            f"{MAIA_DIR} (maia-1100.pb.gz, maia-1500.pb.gz, maia-1900.pb.gz)."
        )
    return path


# --------------------------------------------------------------------------- #
# lc0 process wrapper (one per net, reused across calls)
# --------------------------------------------------------------------------- #


class MaiaEngine:
    """A persistent lc0 process pinned to a single Maia network.

    The process is launched once, kept alive, and reused for every query. A
    background thread continuously drains lc0's stdout into a queue so we never
    deadlock on a full pipe, and a lock serializes queries (one search at a
    time) so a single instance is safe to share across threads.
    """

    def __init__(self, net_name: str, lc0_bin: str = LC0_BIN) -> None:
        """Start lc0 with ``net_name`` and complete the UCI handshake.

        Parameters
        ----------
        net_name:
            Maia network name, e.g. ``"maia-1100"``.
        lc0_bin:
            Path to (or name of) the lc0 binary.

        Raises
        ------
        ValueError
            If the weights file is missing.
        RuntimeError
            If lc0 cannot be launched or fails the handshake.
        """
        self.net_name = net_name
        self._weights = weights_path(net_name)
        self._lock = threading.Lock()
        self._queue: "queue.Queue[object]" = queue.Queue()
        try:
            self._proc = subprocess.Popen(
                [lc0_bin, f"--weights={self._weights}"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"lc0 binary not found ({lc0_bin!r}). Install with `brew install lc0` "
                f"or set LC0_BIN."
            ) from exc

        self._reader = threading.Thread(target=self._pump_stdout, daemon=True)
        self._reader.start()
        self._handshake()

    # -- internals --------------------------------------------------------- #

    def _pump_stdout(self) -> None:
        """Forward every stdout line onto the queue; push ``_EOF`` at close."""
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._queue.put(line.rstrip("\n"))
        self._queue.put(_EOF)

    def _send(self, command: str) -> None:
        """Write a single UCI ``command`` to lc0's stdin."""
        if self._proc.poll() is not None or self._proc.stdin is None:
            raise RuntimeError(f"lc0 ({self.net_name}) is not running.")
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()

    def _read_until(self, sentinel: str, timeout: float) -> List[str]:
        """Collect queued lines until one equals ``sentinel`` (stripped)."""
        collected: List[str] = []
        while True:
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty as exc:
                raise RuntimeError(
                    f"lc0 ({self.net_name}) timed out waiting for {sentinel!r}."
                ) from exc
            if item is _EOF:
                raise RuntimeError(f"lc0 ({self.net_name}) exited unexpectedly.")
            line = str(item)
            collected.append(line)
            if line.strip() == sentinel:
                return collected

    def _drain_pending(self) -> None:
        """Discard any lines already sitting in the queue (defensive)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _handshake(self) -> None:
        """Run ``uci`` + enable verbose stats + wait for the net to load."""
        self._send("uci")
        self._read_until("uciok", _HANDSHAKE_TIMEOUT_S)
        self._send("setoption name VerboseMoveStats value true")
        self._send("isready")
        self._read_until("readyok", _HANDSHAKE_TIMEOUT_S)

    # -- query ------------------------------------------------------------- #

    def raw_policy(self, fen: str) -> Tuple[Dict[str, float], Optional[str]]:
        """Return ``({uci: policy_percent}, bestmove_uci)`` for ``fen``.

        Runs a single forward pass (``go nodes 1``) and parses the verbose
        per-move policy. Percentages are as reported by lc0 (they sum to ~100
        across all legal moves).

        Raises
        ------
        ValueError
            If ``fen`` is invalid.
        RuntimeError
            If lc0 dies or times out.
        """
        _board_from_fen(fen)  # validate before touching the engine
        with self._lock:
            self._drain_pending()
            self._send(f"position fen {fen}")
            self._send("go nodes 1")
            lines = self._read_until_prefix("bestmove", _SEARCH_TIMEOUT_S)

        policy: Dict[str, float] = {}
        bestmove: Optional[str] = None
        for line in lines:
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "(none)":
                    bestmove = parts[1]
                continue
            match = _VERBOSE_RE.match(line)
            if match:
                policy[match.group(1).lower()] = float(match.group(2))
        return policy, bestmove

    def _read_until_prefix(self, prefix: str, timeout: float) -> List[str]:
        """Collect queued lines until one starts with ``prefix``."""
        collected: List[str] = []
        while True:
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty as exc:
                raise RuntimeError(
                    f"lc0 ({self.net_name}) timed out waiting for {prefix!r}."
                ) from exc
            if item is _EOF:
                raise RuntimeError(f"lc0 ({self.net_name}) exited unexpectedly.")
            line = str(item)
            collected.append(line)
            if line.startswith(prefix):
                return collected

    def close(self) -> None:
        """Quit lc0 and reap the process (idempotent)."""
        if self._proc.poll() is None:
            try:
                self._send("quit")
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        try:
            self._proc.wait(timeout=5)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Per-net engine cache
# --------------------------------------------------------------------------- #

_ENGINES: Dict[str, MaiaEngine] = {}
_ENGINES_LOCK = threading.Lock()


def get_engine(net_name: str) -> MaiaEngine:
    """Return a cached :class:`MaiaEngine` for ``net_name`` (creating it once)."""
    with _ENGINES_LOCK:
        engine = _ENGINES.get(net_name)
        if engine is None:
            engine = MaiaEngine(net_name)
            _ENGINES[net_name] = engine
        return engine


def close_all() -> None:
    """Terminate every cached lc0 process. Registered to run at exit."""
    with _ENGINES_LOCK:
        for engine in _ENGINES.values():
            engine.close()
        _ENGINES.clear()


atexit.register(close_all)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def human_moves(fen: str, tier: str, top_k: int = 5) -> HumanMovesResult:
    """Predict the moves a human at ``tier`` would likely play in ``fen``.

    Uses the tier's Maia network (via lc0 at ``nodes=1``) to score every legal
    move by policy, then returns the ``top_k`` most human-likely moves.

    Parameters
    ----------
    fen:
        Position in Forsyth-Edwards Notation.
    tier:
        Coaching tier (``"beginner"``/``"intermediate"``/``"advanced"``,
        case-insensitive) or a raw net name (``"maia-1500"``).
    top_k:
        Maximum number of moves to return (ranked most-human-first).

    Returns
    -------
    HumanMovesResult
        ``{"fen", "tier", "maia_net", "moves": [{"uci", "san", "policy"}, ...]}``
        where ``policy`` is Maia's move probability in ``[0, 1]`` and ``moves``
        is sorted from most to least human-likely. ``moves`` is empty for a
        terminal position (no legal moves).

    Raises
    ------
    ValueError
        If ``fen`` is invalid, ``tier`` is unknown, ``top_k`` < 1, or the net's
        weights are missing.
    RuntimeError
        If lc0 cannot be launched or fails to respond.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")

    board = _board_from_fen(fen)
    net_name = net_for_tier(tier)
    canonical_fen = board.fen()

    # Terminal position -> nothing for a human to choose.
    if not any(board.legal_moves):
        return HumanMovesResult(
            fen=canonical_fen, tier=tier.strip().lower(), maia_net=net_name, moves=[]
        )

    engine = get_engine(net_name)
    policy_pct, bestmove = engine.raw_policy(canonical_fen)

    # Map lc0's UCI tokens back to legal moves (defensively skip any stragglers).
    legal_by_uci = {mv.uci(): mv for mv in board.legal_moves}

    scored: List[MovePolicy] = []
    if policy_pct:
        for uci, pct in policy_pct.items():
            move = legal_by_uci.get(uci)
            if move is None:
                continue
            scored.append(
                MovePolicy(uci=uci, san=board.san(move), policy=pct / 100.0)
            )
    elif bestmove and bestmove in legal_by_uci:
        # Fallback: verbose policy unavailable -> at least the single most-human
        # move (bestmove at nodes=1). Policy set to 1.0 as a degenerate marker.
        move = legal_by_uci[bestmove]
        scored.append(MovePolicy(uci=bestmove, san=board.san(move), policy=1.0))

    # Most human-likely first; stable tie-break by UCI for determinism.
    scored.sort(key=lambda m: (-m["policy"], m["uci"]))

    return HumanMovesResult(
        fen=canonical_fen,
        tier=tier.strip().lower(),
        maia_net=net_name,
        moves=scored[:top_k],
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Maia (human-at-rating) move prediction via lc0.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_hm = sub.add_parser("human-moves", help="Top human-likely moves for one tier.")
    p_hm.add_argument("--fen", required=True)
    p_hm.add_argument("--tier", required=True, help="beginner | intermediate | advanced")
    p_hm.add_argument("--top-k", type=int, default=5)

    p_cmp = sub.add_parser("compare", help="Compare human-likely moves across tiers.")
    p_cmp.add_argument("--fen", required=True)
    p_cmp.add_argument(
        "--tiers",
        nargs="+",
        default=["beginner", "advanced"],
        help="Tiers to compare (default: beginner advanced).",
    )
    p_cmp.add_argument("--top-k", type=int, default=5)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Prints JSON results to stdout."""
    args = _build_parser().parse_args(argv)

    if args.command == "human-moves":
        result = human_moves(args.fen, args.tier, top_k=args.top_k)
        print(json.dumps(result, indent=2))
    elif args.command == "compare":
        results = [
            human_moves(args.fen, tier, top_k=args.top_k) for tier in args.tiers
        ]
        print(json.dumps(results, indent=2))
    else:  # pragma: no cover - argparse enforces valid commands
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
