"""Deterministic, verifiable facts about a chess position and its moves.

**Truth by construction.** Everything here is computed from the board with
``python-chess`` — never guessed by a language model. Two jobs:

1. :func:`render_fact_sheet` — a compact, VERIFIED brief handed to the coach so it
   phrases real facts (what a move captures, attacks, defends, threatens; what is
   hanging) instead of inventing them.
2. The companion :mod:`src.engine.faithfulness` module consumes
   :func:`piece_at_claim_ok` / :func:`color_has_piece` / :func:`is_hanging` to
   check a generated explanation against reality.

This is the non-LLM half of the "dependability = grounding + verifier" thesis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import chess

PIECE_NAME: Dict[int, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

PIECE_VALUE: Dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def color_name(color: chess.Color) -> str:
    return "White" if color == chess.WHITE else "Black"


def piece_map(board: chess.Board, color: chess.Color) -> List[Tuple[int, chess.Piece]]:
    """(square, piece) list for ``color``, king first then by descending value."""
    out = [(sq, p) for sq, p in board.piece_map().items() if p.color == color]
    out.sort(key=lambda sp: (-PIECE_VALUE[sp[1].piece_type], sp[0]))
    return out


def piece_map_str(board: chess.Board) -> str:
    def side(color: chess.Color) -> str:
        toks = [
            f"{p.symbol().upper()}{chess.square_name(sq)}" for sq, p in piece_map(board, color)
        ]
        return f"{color_name(color)}: {', '.join(toks)}"

    return side(chess.WHITE) + " | " + side(chess.BLACK)


def is_hanging(board: chess.Board, square: chess.Square) -> bool:
    """SEE-lite: is the piece on ``square`` attacked in a way that likely wins material?

    Conservative and cheap: a piece is 'hanging' if it is attacked by the enemy
    AND (undefended, or attacked by something cheaper, or attackers outnumber
    defenders). Kings are never 'hanging'.
    """
    piece = board.piece_at(square)
    if piece is None or piece.piece_type == chess.KING:
        return False
    attackers = board.attackers(not piece.color, square)
    if not attackers:
        return False
    defenders = board.attackers(piece.color, square)
    if not defenders:
        return True
    min_attacker = min(PIECE_VALUE[board.piece_at(a).piece_type] for a in attackers)
    if min_attacker < PIECE_VALUE[piece.piece_type]:
        return True
    return len(attackers) > len(defenders)


def hanging_pieces(board: chess.Board, color: chess.Color) -> List[Tuple[chess.Square, chess.Piece]]:
    return [(sq, p) for sq, p in piece_map(board, color) if is_hanging(board, sq)]


def _valuable_targets(board: chess.Board, from_square: chess.Square, mover: chess.Color) -> List[Tuple[int, chess.Piece]]:
    """Enemy pieces attacked by the piece now standing on ``from_square``."""
    out: List[Tuple[int, chess.Piece]] = []
    for tgt in board.attacks(from_square):
        p = board.piece_at(tgt)
        if p is not None and p.color != mover:
            out.append((tgt, p))
    out.sort(key=lambda sp: -PIECE_VALUE[sp[1].piece_type])
    return out


def move_facts(board: chess.Board, move: chess.Move) -> Dict[str, Any]:
    """Structured, verified facts about ``move`` played in ``board``."""
    mover = board.turn
    piece = board.piece_at(move.from_square)
    facts: Dict[str, Any] = {
        "san": board.san(move),
        "uci": move.uci(),
        "piece": PIECE_NAME[piece.piece_type] if piece else "piece",
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
        "is_capture": board.is_capture(move),
        "captured": None,
        "is_check": board.gives_check(move),
        "castle": None,
        "promotion": PIECE_NAME[move.promotion] if move.promotion else None,
        "develops": False,
        "attacks": [],   # [(square_name, piece_name)] newly attacked enemy pieces
        "defends": [],   # [(square_name, piece_name)] friendly pieces now guarded
    }

    if board.is_capture(move):
        if board.is_en_passant(move):
            facts["captured"] = "pawn"
        else:
            cap = board.piece_at(move.to_square)
            facts["captured"] = PIECE_NAME[cap.piece_type] if cap else "pawn"

    if board.is_castling(move):
        facts["castle"] = "kingside" if chess.square_file(move.to_square) > 4 else "queenside"

    home_rank = 0 if mover == chess.WHITE else 7
    if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(move.from_square) == home_rank:
        facts["develops"] = True

    # What the moved piece attacks / defends from its destination.
    tmp = board.copy(stack=False)
    tmp.push(move)
    for tgt in tmp.attacks(move.to_square):
        p = tmp.piece_at(tgt)
        if p is None:
            continue
        if p.color != mover:
            facts["attacks"].append((chess.square_name(tgt), PIECE_NAME[p.piece_type]))
        elif p.piece_type != chess.KING:
            facts["defends"].append((chess.square_name(tgt), PIECE_NAME[p.piece_type]))
    facts["attacks"].sort(key=lambda t: t[0])
    facts["defends"].sort(key=lambda t: t[0])
    return facts


def _describe_move(facts: Dict[str, Any]) -> str:
    bits: List[str] = []
    if facts["castle"]:
        bits.append(f"castles {facts['castle']}")
    else:
        verb = "captures" if facts["is_capture"] else "moves to"
        if facts["is_capture"] and facts["captured"]:
            bits.append(f"{facts['piece']} {verb} the {facts['captured']} on {facts['to']}")
        else:
            bits.append(f"{facts['piece']} to {facts['to']}")
    if facts["develops"]:
        bits.append("develops a minor piece")
    if facts["is_check"]:
        bits.append("gives check")
    if facts["promotion"]:
        bits.append(f"promotes to a {facts['promotion']}")
    if facts["attacks"]:
        tgts = ", ".join(f"the {n} on {s}" for s, n in facts["attacks"][:3])
        bits.append(f"attacks {tgts}")
    if facts["defends"]:
        tgts = ", ".join(f"the {n} on {s}" for s, n in facts["defends"][:2])
        bits.append(f"defends {tgts}")
    return "; ".join(bits) + "."


def render_fact_sheet(
    fen: str,
    selected_uci: str,
    sound_pool: List[Dict[str, Any]],
) -> str:
    """A compact VERIFIED brief for the coach — real facts, no invention.

    ``sound_pool`` entries are dicts with at least ``uci``/``san`` (as returned by
    the engine). Facts are recomputed here from the board, so they are always true.
    """
    board = chess.Board(fen)
    lines: List[str] = [f"VERIFIED FACTS (use only these; do not invent others):"]
    lines.append(f"- Side to move: {color_name(board.turn)}.")
    lines.append(f"- Pieces on the board — {piece_map_str(board)}")

    loose = hanging_pieces(board, board.turn) + hanging_pieces(board, not board.turn)
    if loose:
        toks = ", ".join(
            f"{color_name(p.color)} {PIECE_NAME[p.piece_type]} on {chess.square_name(sq)}"
            for sq, p in loose
        )
        lines.append(f"- Loose / attacked pieces: {toks}.")
    else:
        lines.append("- Loose / attacked pieces: none obvious.")

    try:
        sel = chess.Move.from_uci(selected_uci)
        if sel in board.legal_moves:
            lines.append(f"- Recommended move {board.san(sel)}: {_describe_move(move_facts(board, sel))}")
    except ValueError:
        pass

    others: List[str] = []
    for m in sound_pool:
        u = m.get("uci")
        if not u or u == selected_uci:
            continue
        try:
            mv = chess.Move.from_uci(u)
            if mv in board.legal_moves:
                others.append(board.san(mv))
        except ValueError:
            continue
    if others:
        lines.append(f"- Other sound moves available: {', '.join(others[:6])}.")
    return "\n".join(lines)


def render_pool_facts(fen: str, sound_pool: List[Dict[str, Any]], top_n: int = 6) -> str:
    """Verified, pre-computed facts for the coach prompt — grounding, not guessing.

    Gives the model the piece list (in words), which pieces are loose, and exactly
    what each candidate move does (captures/checks/attacks/defends). The model then
    only has to *phrase* true facts instead of reading them off an ASCII grid.
    """
    board = chess.Board(fen)
    lines: List[str] = [
        "VERIFIED FACTS — use ONLY these. Never mention a piece, square, capture, "
        "or threat that is not listed here.",
        f"- Side to move: {color_name(board.turn)}.",
        f"- Pieces on the board — {piece_map_str(board)}",
    ]

    loose = hanging_pieces(board, board.turn) + hanging_pieces(board, not board.turn)
    if loose:
        toks = ", ".join(
            f"{color_name(p.color)} {PIECE_NAME[p.piece_type]} on {chess.square_name(sq)}"
            for sq, p in loose
        )
        lines.append(f"- Undefended / attacked pieces: {toks}.")
    else:
        lines.append("- Undefended / attacked pieces: none.")

    lines.append("- What each candidate move concretely does:")
    for m in sound_pool[:top_n]:
        u = m.get("uci")
        if not u:
            continue
        try:
            mv = chess.Move.from_uci(u)
            if mv not in board.legal_moves:
                continue
        except ValueError:
            continue
        lines.append(f"    * {board.san(mv)} — {_describe_move(move_facts(board, mv))}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Verification helpers (used by src.engine.faithfulness)
# --------------------------------------------------------------------------- #


def piece_at_claim_ok(board: chess.Board, square_name: str, piece_word: str,
                      color_word: Optional[str] = None) -> bool:
    """True iff a 'PIECE on SQUARE' claim matches the real board.

    ``color_word`` may be 'white'/'black' (or None if unspecified).
    """
    try:
        sq = chess.parse_square(square_name)
    except ValueError:
        return True  # not a real square reference; don't flag
    piece = board.piece_at(sq)
    if piece is None:
        return False
    want = _PIECE_WORD_TO_TYPE.get(piece_word.lower())
    if want is not None and piece.piece_type != want:
        return False
    if color_word in ("white", "black"):
        want_color = chess.WHITE if color_word == "white" else chess.BLACK
        if piece.color != want_color:
            return False
    return True


def color_has_piece(board: chess.Board, color: chess.Color, piece_word: str) -> bool:
    want = _PIECE_WORD_TO_TYPE.get(piece_word.lower())
    if want is None:
        return True
    return any(p.piece_type == want and p.color == color for p in board.piece_map().values())


_PIECE_WORD_TO_TYPE: Dict[str, int] = {
    "pawn": chess.PAWN,
    "knight": chess.KNIGHT,
    "bishop": chess.BISHOP,
    "rook": chess.ROOK,
    "queen": chess.QUEEN,
    "king": chess.KING,
}
