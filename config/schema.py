"""Shared data contracts + rendering for the whole pipeline.

Every stage (generate / filter / eval) imports these so the record shapes and the
exact text the model sees are identical everywhere. Engine analysis (with evals)
goes INTO the prompt on purpose — the trained behavior is to NOT leak those numbers
back out. That is the demonstrable base-vs-tuned delta.
"""
from __future__ import annotations
import re
from typing import TypedDict, Optional
import chess
from config import settings


class Position(TypedDict):
    id: str
    fen: str
    tier: str
    played_move_uci: str
    played_move_san: str
    side_to_move: str
    mover_rating: int
    game_id: str
    ply: int
    time_control: str


class SoundMove(TypedDict):
    san: str
    uci: str
    cp: int
    pv: list[str]


class MaiaMove(TypedDict):
    san: str
    uci: str
    policy: float


class StudentMove(TypedDict):
    san: str
    uci: str
    cp_loss: int
    severity: str


class TeacherInput(TypedDict):
    tier: str
    fen: str
    move_history_san: Optional[str]
    student_move: StudentMove
    sound_pool: list[SoundMove]
    maia_human_moves: list[MaiaMove]


class TeacherOutput(TypedDict):
    tier: str
    recommended_move_san: str
    recommended_move_uci: str
    coaching: str
    takeaway: str
    concepts_used: list[str]


def ascii_board(fen: str) -> str:
    return str(chess.Board(fen))


def render_user_prompt(ti: TeacherInput) -> str:
    """The exact user-message text shown to BOTH teacher and student models."""
    b = chess.Board(ti["fen"])
    t = settings.TIERS[ti["tier"]]
    L: list[str] = []
    L.append(f"Student rating tier: {ti['tier']} ({t['low']}-{t['high']}).")
    if ti.get("move_history_san"):
        L.append(f"Moves so far: {ti['move_history_san']}")
    L.append("Board:\n" + ascii_board(ti["fen"]))
    L.append(f"{'White' if b.turn else 'Black'} to move.")
    sm = ti["student_move"]
    L.append(
        f"The student played {sm['san']} (severity: {sm['severity']}; "
        f"it loses about {sm['cp_loss']} centipawns)."
    )
    L.append("Engine-sound candidate moves [internal reference — never quote these numbers]:")
    for m in ti["sound_pool"]:
        idea = " ".join(m["pv"][:3])
        L.append(f"  - {m['san']} (eval {m['cp']}cp) idea: {idea}")
    L.append("Human-likelihood at this tier (Maia):")
    for m in ti["maia_human_moves"][:6]:
        L.append(f"  - {m['san']}: {round(m['policy'] * 100)}%")
    L.append(
        f"\nRecommend exactly ONE move from the sound list — the most instructive for a "
        f"{ti['tier']} player — and coach them. Ply cap: {t['ply_cap']}."
    )
    return "\n".join(L)


def render_assistant_target(to: TeacherOutput) -> str:
    """The natural-language coaching the student model learns to produce (v1)."""
    return f"I'd play {to['recommended_move_san']}. {to['coaching']} Takeaway: {to['takeaway']}"


def _strip_leading_move_restatement(coaching: str, san: str) -> str:
    """Drop a redundant leading move-command from coaching.

    The target always opens with ``"I'd play <MOVE>."``; if the teacher's coaching
    also starts with a command form of the same move (``"Play Nf3."`` / ``"I'd play
    Nf3,"``) we strip that so the row does not read ``"I'd play Nf3. Play Nf3. ..."``.
    Only a command lead-in ("play"/"consider"/"go with") is stripped, so coaching
    that uses the move as a sentence subject ("Nf3 develops ...") is left intact.
    """
    pat = re.compile(
        rf"^\s*(?:i['\u2019]?d\s+play|let['\u2019]?s\s+play|play|consider|go\s+with)\s+"
        rf"{re.escape(san)}\b[\s.:,;!\-]*",
        re.IGNORECASE,
    )
    return pat.sub("", coaching, count=1).strip()


def render_assistant_target_v2(to: TeacherOutput) -> str:
    """v2 target: teaches THREE things — the move, WHY, and HOW to FIND it.

    v1 taught the move + a takeaway. v2 adds an explicit **method** clause (the
    reasoning routine a player at this tier runs to find the move themselves),
    which is the whole point of v2. The ``method`` key is a v2 addition to the
    teacher output; if it is missing this degrades to the v1 shape.
    """
    san = to["recommended_move_san"]
    method = str(to.get("method") or "").strip()  # type: ignore[attr-defined]
    coaching = _strip_leading_move_restatement(str(to.get("coaching") or "").strip(), san)
    takeaway = str(to.get("takeaway") or "").strip()
    body = f"I'd play {san}. {coaching}".strip()
    if method:
        body = f"{body} How to find it: {method}"
    return f"{body} Takeaway: {takeaway}".strip()


def build_chat_example(system_prompt: str, ti: TeacherInput, to: TeacherOutput) -> dict:
    """One SFT training row in chat format (Unsloth/TRL compatible)."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": render_user_prompt(ti)},
            {"role": "assistant", "content": render_assistant_target(to)},
        ]
    }


def build_chat_example_v2(system_prompt: str, ti: TeacherInput, to: TeacherOutput) -> dict:
    """v2 SFT row — same prompt shape as v1, but the target teaches the METHOD.

    Keeping the system + user prompt identical to v1 makes the v1->v2 comparison a
    clean DATA intervention (faithful labels, tier-aware moves, explicit method,
    contrastive tiers) rather than a prompt-format change.
    """
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": render_user_prompt(ti)},
            {"role": "assistant", "content": render_assistant_target_v2(to)},
        ]
    }
