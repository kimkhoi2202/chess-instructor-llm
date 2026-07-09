#!/usr/bin/env python3
"""Quality gates + best-of-N scoring for curated coaching labels.

The engine + these deterministic gates are the arbiter of a label's *correctness*
(the task's contract: faithfulness = 0 fabrication, tier-appropriateness, a named
principle in the takeaway from the controlled vocabulary, no engine-speak,
well-formed, and no *wrong* heuristics). Best-of-N ranking of the SURVIVORS is a
light instructiveness score (family-agnostic, cheap); an optional single blinded
cross-family judge call breaks ties. No multi-round debate.

Everything reuses the shipped pipeline so the curated rows match the existing
format exactly: ``clean_lead`` / ``beginner_scrub`` / ``apply_takeaway_gate`` /
``PRINCIPLE_CLAUSE`` from :mod:`src.teacher.build_4b_dataset`, the engine-speak /
ply-cap / legality gates from :mod:`src.filter.filter`, and the STRONG widened
faithfulness checker :func:`src.engine.faithfulness_ext.verify_text_ext`.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import settings  # noqa: E402
from src.engine.faithfulness_ext import verify_text_ext  # noqa: E402
from src.filter.filter import detect_engine_speak, longest_san_run, move_is_legal  # noqa: E402
from src.teacher.build_4b_dataset import (  # noqa: E402
    PRINCIPLE_CLAUSE,
    apply_takeaway_gate,
    beginner_scrub,
    clean_lead,
    _principle_families,
)

# --------------------------------------------------------------------------- #
# Controlled principle vocabulary (same detector the audit + v5 build use)
# --------------------------------------------------------------------------- #
_FAMS = None


def principle_families() -> Dict[str, "re.Pattern[str]"]:
    global _FAMS
    if _FAMS is None:
        _FAMS = _principle_families()
    return _FAMS


def has_named_principle(text: str) -> bool:
    """True iff ``text`` names a transferable principle from the controlled vocab."""
    return any(p.search(text) for p in principle_families().values())


# --------------------------------------------------------------------------- #
# REJECTED / inverted heuristics (principle_library_v5.md §D). Conservative: only
# the unambiguous inverted slogans fire, so the correct forms pass untouched.
# --------------------------------------------------------------------------- #
_BAD_HEURISTICS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("trade_when_behind", re.compile(
        r"\b(?:trade|exchang\w+|simplif\w+|swap\w*)\b[^.]{0,60}?\b(?:when|if|because|since|while)\b"
        r"[^.]{0,30}?\b(?:behind|losing|worse|down (?:in )?material|at a disadvantage)\b", re.I)),
    ("trade_when_behind_rev", re.compile(
        r"\b(?:when|if|because|since|while)\b[^.]{0,30}?\b(?:behind|losing|worse|"
        r"down (?:in )?material|at a disadvantage)\b[^.]{0,50}?\b(?:trade|exchang\w+|simplif\w+|swap\w*)\b", re.I)),
    ("passed_must_push", re.compile(
        r"\bpassed pawns?\b[^.]{0,20}\bmust\b[^.]{0,15}\b(?:be pushed|push|advance)\b", re.I)),
    ("always_push_passed", re.compile(
        r"\balways\b[^.]{0,15}\bpush\w*\b[^.]{0,15}\bpassed\b", re.I)),
    ("always_castle", re.compile(
        r"\b(?:always castle|must always castle|castl\w+ is always (?:the )?(?:right|best|priority)|"
        r"you (?:should )?always castle)\b", re.I)),
    ("bishop_pair_always", re.compile(
        r"\b(?:bishop pair|two bishops)\b[^.]{0,15}\b(?:is|are)\b[^.]{0,10}\balways\b[^.]{0,15}"
        r"\b(?:better|stronger|good)\b", re.I)),
    ("space_always_good", re.compile(
        r"\bspace\b[^.]{0,20}\b(?:is )?always\b[^.]{0,10}\b(?:good|better|an advantage)\b", re.I)),
    ("always_grab_pawn", re.compile(
        r"\balways\b[^.]{0,10}\b(?:grab|take|snatch|win)\b[^.]{0,15}\b(?:free )?pawn\b", re.I)),
    ("queen_out_early_good", re.compile(
        r"\bbring\b[^.]{0,15}\bqueen\b[^.]{0,15}\bout\b[^.]{0,15}\bearly\b[^.]{0,20}"
        r"\b(?:to be active|for activity|is good|to attack)\b", re.I)),
]


def bad_heuristic(text: str) -> Optional[str]:
    """Name of the first rejected heuristic asserted in ``text`` (or ``None``)."""
    for name, pat in _BAD_HEURISTICS:
        if pat.search(text):
            return name
    return None


# --------------------------------------------------------------------------- #
# Render (native vs augmented takeaway) — mirrors build_4b_dataset.render_v5_target
# --------------------------------------------------------------------------- #
def render_target(to: Dict[str, Any], tier: str, *, augment: bool
                  ) -> Tuple[str, bool]:
    """Render the training target. If ``augment`` and the takeaway names no
    principle, splice in a correct, on-topic principle clause (last-resort only).
    Returns ``(target, augmented?)``.
    """
    san = str(to.get("recommended_move_san") or "").strip()
    coaching = clean_lead(str(to.get("coaching") or "").strip())
    method = str(to.get("method") or "").strip()
    takeaway = str(to.get("takeaway") or "").strip()
    concepts = [str(c) for c in (to.get("concepts_used") or [])]

    augmented = False
    if augment:
        takeaway, augmented = apply_takeaway_gate(
            takeaway, coaching, concepts, tier, principle_families())

    body = f"I'd play {san}. {coaching}".strip()
    if method:
        body = f"{body} How to find it: {method}"
    target = f"{body} Takeaway: {takeaway}".strip()
    if tier == "beginner":
        target = beginner_scrub(target)
    target = re.sub(r"  +", " ", target)
    return target, augmented


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
@dataclass
class GateResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    target: Optional[str] = None          # rendered training target (if ok)
    native_principle: bool = False        # teacher named a principle unaided
    augmented: bool = False               # a principle clause was spliced in
    score: float = 0.0                    # instructiveness (best-of-N ranker)
    words: int = 0


def gate_candidate(
    to: Dict[str, Any], fen: str, tier: str, canonical_uci: str,
    sound_ucis: List[str], *, allow_augment: bool = False,
) -> GateResult:
    """Run every correctness gate on one teacher label; render if it survives.

    ``canonical_uci`` is the deterministic per-tier move the label MUST teach
    (tier-appropriateness). ``allow_augment`` is the last-resort principle splice,
    only used when NO candidate for the example named a principle natively.
    """
    reasons: List[str] = []
    rec_uci = str(to.get("recommended_move_uci") or "").strip().lower()
    rec_san = str(to.get("recommended_move_san") or "").strip()
    coaching = str(to.get("coaching") or "").strip()
    method = str(to.get("method") or "").strip()
    takeaway = str(to.get("takeaway") or "").strip()

    # Structural presence
    if not coaching:
        reasons.append("empty_coaching")
    if not method:
        reasons.append("missing_method")
    if not takeaway:
        reasons.append("empty_takeaway")

    # Tier-appropriateness: must teach EXACTLY the canonical per-tier move.
    if not rec_uci:
        reasons.append("missing_move")
    elif rec_uci != canonical_uci.lower():
        reasons.append("tier_move_mismatch")

    # Soundness (belt-and-suspenders; canonical move is always in the pool)
    if rec_uci and rec_uci not in {u.lower() for u in sound_ucis}:
        reasons.append("soundness")

    # Legality
    if fen and (rec_uci or rec_san):
        legal, _ = move_is_legal(fen, rec_uci, rec_san)
        if not legal:
            reasons.append("illegal_move")

    if reasons:
        return GateResult(ok=False, reasons=reasons)

    # Native principle in takeaway?
    native = has_named_principle(takeaway)

    # Render (augment only as a sanctioned last resort)
    target, augmented = render_target(
        to, tier, augment=(allow_augment and not native))
    words = len(target.split())

    # No engine-speak
    if detect_engine_speak(target):
        reasons.append("engine_speak")
    # Ply cap
    ply_cap = settings.TIERS[tier]["ply_cap"]
    if longest_san_run(target) > ply_cap:
        reasons.append("ply_cap")
    # Well-formed shape
    if not target.startswith(f"I'd play {rec_san}."):
        reasons.append("format_lead")
    if "Takeaway:" not in target:
        reasons.append("format_takeaway")
    # Principle in takeaway (native, or augmented when allowed)
    final_takeaway = target.split("Takeaway:", 1)[1] if "Takeaway:" in target else ""
    if not has_named_principle(final_takeaway):
        reasons.append("no_principle_in_takeaway")
    # Correctness: no rejected/inverted heuristics
    bad = bad_heuristic(target)
    if bad:
        reasons.append(f"bad_heuristic:{bad}")
    # Faithfulness — STRONG widened checker, zero fabrication tolerated
    if verify_text_ext(target, fen, rec_uci).violations:
        reasons.append("faithfulness_ext")

    if reasons:
        return GateResult(ok=False, reasons=reasons, native_principle=native)

    score = instructiveness_score(target, coaching, method, takeaway,
                                  to.get("concepts_used") or [], tier, rec_san)
    return GateResult(ok=True, target=target, native_principle=native,
                      augmented=augmented, score=score, words=words)


# --------------------------------------------------------------------------- #
# Instructiveness score (family-agnostic best-of-N primary ranker)
# --------------------------------------------------------------------------- #
_SQ_RE = re.compile(r"\b[a-h][1-8]\b")
_METHOD_ROUTINE = re.compile(
    r"\b(ask|look for|look at|check|scan|count|first|before you|every|which|"
    r"spot|find|consider|compare|when you)\b", re.I)
_CONTRAST = re.compile(
    r"\b(at your level|a stronger player|stronger player|simpler|for now|"
    r"just as sound|easier to handle|easier to play|keep it simple)\b", re.I)
_CHECKLIST_CLONE = re.compile(r"run this checklist", re.I)

_LEN_BAND = {"beginner": (55, 145), "intermediate": (85, 185), "advanced": (100, 225)}


def instructiveness_score(target: str, coaching: str, method: str, takeaway: str,
                          concepts: List[str], tier: str, rec_san: str) -> float:
    """A light, deterministic instructiveness rubric (higher = better)."""
    score = 0.0
    # Board-specific reasoning (E3): distinct square references in the coaching.
    sqs = {m.group(0) for m in _SQ_RE.finditer(coaching)}
    score += min(len(sqs), 4) * 1.0
    # Named principle in the takeaway (E2) — the weak axis; weight it.
    if has_named_principle(takeaway):
        score += 3.0
    # A genuine reusable method (E4).
    if _METHOD_ROUTINE.search(method):
        score += 2.0
    # Tier-contrast phrasing (the moat, made explicit).
    if _CONTRAST.search(target):
        score += 1.5
    # Concept richness without dumping.
    nconc = len([c for c in concepts if str(c).strip()])
    if 1 <= nconc <= 4:
        score += 1.0
    # Length band per tier.
    words = len(target.split())
    lo, hi = _LEN_BAND[tier]
    if lo <= words <= hi:
        score += 1.5
    else:
        dev = (lo - words) if words < lo else (words - hi)
        score -= min(dev / 40.0, 3.0)
    # Penalize formulaic / repetitive shapes.
    if rec_san and target.count(rec_san) >= 4:
        score -= 2.0
    if _CHECKLIST_CLONE.search(target):
        score -= 1.0
    if words > 270:
        score -= 2.0
    return round(score, 3)


# --------------------------------------------------------------------------- #
# Self-test (no engine / network) — asserts the correctness gates behave.
# --------------------------------------------------------------------------- #
def _self_test() -> bool:
    ok = True
    checks = [
        ("bad: trade when behind",
         bad_heuristic("When you are behind, trade pieces to reach an endgame.") is not None),
        ("good: keep pieces when behind",
         bad_heuristic("When you are behind, keep pieces on to retain counterplay.") is None),
        ("bad: passed pawns must be pushed",
         bad_heuristic("Passed pawns must be pushed at once.") is not None),
        ("good: support the passed pawn",
         bad_heuristic("Support a passed pawn before you push it.") is None),
        ("bad: always castle",
         bad_heuristic("You should always castle as early as possible.") is not None),
        ("good: king safety first",
         bad_heuristic("King safety first: get your king out of the center.") is None),
        ("bad: space always good",
         bad_heuristic("Space is always good, so grab it.") is not None),
        ("principle detected",
         has_named_principle("Develop your pieces before you attack.")),
        ("no principle in bare narration",
         not has_named_principle("This wins the knight on c6.")),
    ]
    for label, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        ok = ok and cond
    return ok


if __name__ == "__main__":
    print("=== curate.gates self-test ===")
    raise SystemExit(0 if _self_test() else 1)
