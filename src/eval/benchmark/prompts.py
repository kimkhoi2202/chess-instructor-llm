"""Prompt assembly for the two conditions (identical system + format for all).

Fairness rules baked in here:

* **System prompt** — every model, both conditions, gets ``prompts/coach_system.md``
  verbatim.
* **Format instruction** — every model, both conditions, gets the SAME closing
  instruction on how to shape the answer, so the deterministic move-extractor is
  fair to all of them.

The only thing that differs between conditions is the *content* of the user
message:

* ``ungrounded`` — tier + board + the student's move. Nothing else. The coach
  must supply the chess itself.
* ``grounded`` — the same verified block every model sees: ``render_pool_facts``
  (piece list / loose pieces / what each candidate does) followed by
  ``render_user_prompt`` (sound pool with internal evals + Maia human-likelihoods
  + the "pick one from the sound list" instruction).
"""

from __future__ import annotations

from typing import Any, Dict

import chess

from config import schema, settings
from src.engine.position_facts import render_pool_facts

#: Shown to EVERY model in BOTH conditions. Keeps outputs comparable + parseable
#: without prescribing the chess (which is the thing under test).
FORMAT_INSTRUCTION: str = (
    "Format your reply as plain prose in exactly this shape:\n"
    'Start with "I\'d play <MOVE>." where <MOVE> is one move in standard '
    "algebraic notation (e.g. Nf3, exd5, O-O). Then give 2-4 sentences of "
    "coaching in plain, encouraging language tied to the student's mistake and a "
    'concrete plan. End with one line "Takeaway: <one transferable sentence>." '
    "Do not output JSON, bullet lists, headers, or long move-number sequences."
)


def load_system_prompt() -> str:
    """The single coaching system prompt used for every model + condition."""
    return (settings.PROMPTS / "coach_system.md").read_text(encoding="utf-8").strip()


def scenario_to_teacher_input(scn: Dict[str, Any]) -> schema.TeacherInput:
    """Rebuild the shared :class:`schema.TeacherInput` contract from a scenario."""
    sm = scn["student_move"]
    return {
        "tier": scn["tier"],
        "fen": scn["fen"],
        "move_history_san": None,
        "student_move": {
            "san": sm["san"],
            "uci": sm["uci"],
            "cp_loss": int(sm["cp_loss"]),
            "severity": sm["severity"],
        },
        "sound_pool": [
            {"uci": m["uci"], "san": m["san"], "cp": int(m["cp"]), "pv": list(m.get("pv") or [])}
            for m in scn["sound_pool"]
        ],
        "maia_human_moves": [
            {"uci": m["uci"], "san": m["san"], "policy": float(m["policy"])}
            for m in scn.get("maia", [])
        ],
    }


def build_ungrounded_user(scn: Dict[str, Any]) -> str:
    """User message for the WITHOUT-grounding condition (no engine, no Maia)."""
    fen = scn["fen"]
    tier = scn["tier"]
    board = chess.Board(fen)
    t = settings.TIERS[tier]
    lines = [
        f"Student rating tier: {tier} ({t['low']}-{t['high']}).",
        "Board:\n" + schema.ascii_board(fen),
        f"{'White' if board.turn else 'Black'} to move.",
        f"The student played {scn['student_move']['san']}.",
        "",
        (
            f"Recommend exactly ONE move that is genuinely sound AND the most "
            f"instructive for a {tier} player, and coach them on why it beats what "
            f"they played. Keep any concrete line within {t['ply_cap']} plies."
        ),
        "",
        FORMAT_INSTRUCTION,
    ]
    return "\n".join(lines)


def build_grounded_user(scn: Dict[str, Any]) -> str:
    """User message for the WITH-grounding condition (verified facts + pool + Maia)."""
    ti = scenario_to_teacher_input(scn)
    facts = render_pool_facts(scn["fen"], ti["sound_pool"])
    body = schema.render_user_prompt(ti)
    return f"{facts}\n\n{body}\n\n{FORMAT_INSTRUCTION}"


def build_user_prompt(scn: Dict[str, Any], condition: str) -> str:
    """Dispatch to the right user-message builder for ``condition``."""
    if condition == "ungrounded":
        return build_ungrounded_user(scn)
    if condition == "grounded":
        return build_grounded_user(scn)
    raise ValueError(f"unknown condition {condition!r}")
