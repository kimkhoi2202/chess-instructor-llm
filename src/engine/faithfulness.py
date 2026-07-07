"""Faithfulness verifier — the non-LLM truth gate.

Given a coach's free-text explanation and the position it is about, this checks
every *board claim* against reality and removes any sentence that states
something false about the pieces. It is deliberately **conservative**: it only
drops a sentence when a claim is *demonstrably* false (a named piece is not on a
named square, or a side is said to have a piece it does not have). Vague or
purely strategic sentences pass untouched.

This is the guarantee behind "the fine-tune nails style; truth needs a verifier":
no fabricated board fact reaches the learner, regardless of what the model wrote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import chess

from src.engine.position_facts import (
    color_has_piece,
    piece_at_claim_ok,
)

_PIECES = r"(pawn|knight|bishop|rook|queen|king)"
_SQ = r"([a-h][1-8])"

# "white/black/your/opponent's <piece> on <square>"  (piece located on a square)
_ON_SQUARE = re.compile(
    rf"\b(white|black|your|our|the opponent'?s?|opponent'?s?|their|his|her|its)?\s*"
    rf"{_PIECES}\s+(?:on|is on|sits on|sitting on|stands on|standing on)\s+(?:the\s+)?{_SQ}\b",
    re.IGNORECASE,
)

# "the f6 knight" / "e4-pawn" / "d3 bishop"  (square then piece)
_SQ_PIECE = re.compile(rf"\b(?:the\s+)?{_SQ}[-\s]{_PIECES}\b", re.IGNORECASE)

# "white's queen" / "your rook" / "opponent's bishop"  (existence, no square)
_POSSESSIVE = re.compile(
    rf"\b(white'?s?|black'?s?|your|our|the opponent'?s?|opponent'?s?|their)\s+{_PIECES}\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _color_from_word(word: Optional[str], side_to_move: chess.Color) -> Optional[chess.Color]:
    if not word:
        return None
    w = word.lower().rstrip("'s").strip()
    if w == "white":
        return chess.WHITE
    if w == "black":
        return chess.BLACK
    if w in ("your", "our"):
        return side_to_move
    if w in ("their", "opponent", "the opponent", "opponents"):
        return not side_to_move
    return None  # his/her/its — ambiguous, don't judge


@dataclass
class Violation:
    sentence: str
    reason: str


@dataclass
class VerifyResult:
    clean: str
    violations: List[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def _sentence_is_false(sentence: str, board: chess.Board) -> Optional[str]:
    """Return a reason string if the sentence makes a false board claim, else None."""
    stm = board.turn

    # 1) "<piece> on <square>"
    for m in _ON_SQUARE.finditer(sentence):
        color_word, piece_word, sq = m.group(1), m.group(2), m.group(3)
        cw = None
        c = _color_from_word(color_word, stm)
        if c is not None:
            cw = "white" if c == chess.WHITE else "black"
        if not piece_at_claim_ok(board, sq, piece_word, cw):
            return f"no {piece_word} on {sq}"

    # 2) "<square> <piece>"
    for m in _SQ_PIECE.finditer(sentence):
        sq, piece_word = m.group(1), m.group(2)
        if not piece_at_claim_ok(board, sq, piece_word, None):
            return f"no {piece_word} on {sq}"

    # 3) "<color>'s <piece>" existence (only flag if that side has ZERO of it)
    for m in _POSSESSIVE.finditer(sentence):
        color_word, piece_word = m.group(1), m.group(2)
        c = _color_from_word(color_word, stm)
        if c is None:
            continue
        # Skip if this same phrase was an "on square" claim (already handled).
        if not color_has_piece(board, c, piece_word):
            side = "White" if c == chess.WHITE else "Black"
            return f"{side} has no {piece_word}"
    return None


def verify_text(text: str, fen: str) -> VerifyResult:
    """Drop sentences that state a demonstrably false board fact."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return VerifyResult(clean=text)

    sentences = [s for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]
    kept: List[str] = []
    violations: List[Violation] = []
    for s in sentences:
        reason = _sentence_is_false(s, board)
        if reason is None:
            kept.append(s.strip())
        else:
            violations.append(Violation(sentence=s.strip(), reason=reason))

    clean = " ".join(kept).strip()
    return VerifyResult(clean=clean, violations=violations)
