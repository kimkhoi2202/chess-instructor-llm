"""FEN completeness validation for the coach endpoints (:func:`_validate_position`).

Both ``/api/coach`` and ``/api/coach_all`` funnel their FEN through
``_validate_position`` before any board / coaching work. This locks in the
robustness fix the adversarial eval surfaced: a truncated (e.g. board-only) FEN
must be REJECTED with a clean 4xx rather than silently coached. python-chess
fills defaults for the missing side-to-move / castling / en passant / clock
fields, so a board-only string parses to a *legal* position and — without this
guard — would be coached with an invented side-to-move. These tests need no MLX
model or Stockfish binary: they exercise pure validation with python-chess.
"""

from __future__ import annotations

import chess
import pytest
from fastapi import HTTPException

from src.api.server import _validate_position

# The exact board-only string the adversarial `malformed_truncated` case sends
# (1.e4 e5, no side-to-move / rights / clocks). python-chess would fill defaults.
TRUNCATED_BOARD_ONLY = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR"
#: The same position as a complete, standard 6-field FEN (a legitimate caller).
FULL_FEN = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"


def test_board_only_fen_is_rejected_4xx():
    with pytest.raises(HTTPException) as exc:
        _validate_position(TRUNCATED_BOARD_ONLY)
    assert 400 <= exc.value.status_code < 500
    assert "6-field" in exc.value.detail


@pytest.mark.parametrize(
    "fen",
    [
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w",  # + side-to-move only
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq",  # missing ep+clocks
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0",  # missing fullmove
    ],
)
def test_partial_field_fens_are_rejected_4xx(fen):
    # Anything short of the full six standard fields is a truncated FEN.
    with pytest.raises(HTTPException) as exc:
        _validate_position(fen)
    assert 400 <= exc.value.status_code < 500
    assert "6-field" in exc.value.detail


def test_full_fen_is_accepted_unchanged():
    fen, board = _validate_position(FULL_FEN)
    assert fen == FULL_FEN
    assert isinstance(board, chess.Board)
    assert board.turn == chess.WHITE  # side-to-move honored from the FEN, not defaulted
    assert board.fen() == FULL_FEN


def test_full_fen_with_surrounding_whitespace_is_accepted():
    # Leading/trailing whitespace is stripped; the six fields still validate.
    fen, board = _validate_position(f"  {FULL_FEN}  ")
    assert fen == FULL_FEN
    assert board.turn == chess.WHITE


def test_empty_fen_still_rejected():
    with pytest.raises(HTTPException) as exc:
        _validate_position("   ")
    assert exc.value.status_code == 400
    assert "required" in exc.value.detail.lower()


def test_other_malformed_full_fens_still_rejected():
    # A full 6-field but illegal placement (two white kings) keeps its existing
    # 4xx behavior — the new length guard does not shadow the legality checks.
    with pytest.raises(HTTPException) as exc:
        _validate_position("4k3/8/8/8/8/8/8/RK2K3 w - - 0 1")
    assert 400 <= exc.value.status_code < 500
