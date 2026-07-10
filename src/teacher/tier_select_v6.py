"""v6 tier-aware teaching-move selection — Maia as a CONSTRAINT, not the objective.

This replaces the v2 ``tier_select.select_tier_move`` min-max blend
(``score = (1-w)*eval_norm + w*policy_norm``) whose artifacts the v5 audit
flagged (``data/analysis/V5_AUDIT_AND_PLAN.md`` §3): the 50/50 intermediate blend
could maximize on a *third* "compromise" move, producing the pathological
``B=A != I`` collapse, and beginner chased pure Maia popularity even onto shaky
moves.

The v6 rule (operates on a **deep, WDL/tablebase-verified** sound pool — see
``src/engine/deep_label.py``):

- **advanced = the VERIFIED engine-best move** (``pool[0]`` == persisted
  ``engine_best``). Never diverges from ``engine_best`` (fixes the ~43/803
  advanced!=engine-best bug, audit fix #2).
- **beginner** — require a *reasonable human-likelihood* (Maia policy >= a tier
  gate); among the moves that clear the gate, rank by **robustness/clarity**
  (engine-derived: depth-agreement + WDL-band safety + eval proximity), with
  human-likelihood as the tie-break so the pick stays *findable*. If no move
  clears the gate, fall back to the engine-best (a sound move is always better
  than an unsound "findable" one). (audit fix #3)
- **intermediate** — same gate+rank at the 1500 Maia net (a stronger human finds
  sharper moves, so this naturally sits between). **If beginner == advanced, the
  intermediate pick is forced to that same move** — a position whose best move is
  also human-findable is taught consistently, killing the ``B=A != I`` collapse
  (audit fix #4).

Everything here is a **pure function** of the deep label + Maia policies (no
engine process), so it is deterministic and unit-testable. The companion
``review_student_move`` implements coherent move-review (audit fix #5): a student
move is only *corrected* when it is genuinely worse by a margin; otherwise it is
*endorsed*.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

TIER_ORDER = ("beginner", "intermediate", "advanced")

# --- Human-likelihood GATE per tier (the constraint). A pool move must be at
# least this human-likely at the tier's Maia net to be an eligible pick. Higher
# for beginner (must be genuinely findable by a 1100); advanced ignores the gate
# (advanced always = engine best). Calibrated so ~most positions have >=1 move
# clearing the beginner gate; when none do, we fall back to the sound engine best.
MAIA_GATE: Dict[str, float] = {
    "beginner": 0.10,
    "intermediate": 0.05,
    "advanced": 0.0,
}

# WDL expected-score band edges (side-to-move POV). exp = (win + 0.5*draw)/1000.
WIN_BAND = 0.75
LOSS_BAND = 0.25
# A pool move may sit at most this far below the best move's expected score.
WDL_DROP_MAX = 0.10

# Move-review margins (cp loss vs the deep engine best), audit fix #5.
ENDORSE_CP = 30     # <= this: the student's move is as good as best -> endorse
CORRECT_CP = 90     # >  this (or a WDL-band cross): genuinely worse -> correct


@dataclass(frozen=True)
class TierPickV6:
    uci: str
    san: str
    policy: float          # Maia policy for this tier's net in [0, 1]
    robust: int            # coarse robustness/clarity score (higher = safer)
    pool_rank: int         # rank in the deep pool (0 = engine best)
    is_engine_best: bool
    findable: bool         # cleared this tier's Maia gate
    fallback: bool         # no move cleared the gate -> fell back to engine best

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _exp_score(wdl: Optional[List[int]]) -> Optional[float]:
    """Expected score in [0,1] from a [win, draw, loss] permille triple."""
    if not wdl or len(wdl) != 3:
        return None
    tot = sum(wdl)
    if tot <= 0:
        return None
    return (wdl[0] + 0.5 * wdl[1]) / tot


def _band(exp: Optional[float]) -> Optional[str]:
    if exp is None:
        return None
    if exp >= WIN_BAND:
        return "win"
    if exp <= LOSS_BAND:
        return "loss"
    return "draw"


def _robust_score(move: Mapping[str, Any], best: Mapping[str, Any]) -> int:
    """Coarse, engine-derived robustness/clarity (NOT an eval/Maia blend).

    Rewards a move that is sound at both search depths and shares the engine
    best's WDL band, penalising eval distance from best in coarse ~40cp steps.
    Coarse on purpose: most deep-pool moves tie here, so the tie-break
    (human-likelihood) keeps the beginner pick *findable*.
    """
    score = 0
    if move.get("depth_agree", True):
        score += 2
    mb = _band(_exp_score(move.get("wdl")))
    bb = _band(_exp_score(best.get("wdl")))
    if mb is not None and bb is not None and mb == bb:
        score += 1
    if move.get("tb") in (None, "ok", "win", "draw"):
        # tablebase-sound (or not a tablebase position); a tb_worse move never
        # reaches the pool, so this is defensive.
        score += 0
    gap = int(best.get("cp", 0)) - int(move.get("cp", 0))
    score -= max(0, gap) // 40
    return score


def _pick_for_tier(
    tier: str,
    pool: List[Mapping[str, Any]],
    policy_by_uci: Mapping[str, float],
    best: Mapping[str, Any],
) -> TierPickV6:
    """Gate on human-likelihood, then rank survivors by robustness/clarity."""
    gate = MAIA_GATE.get(tier, 0.0)
    best_uci = str(best["uci"])

    def _mk(move: Mapping[str, Any], fallback: bool) -> TierPickV6:
        u = str(move["uci"])
        return TierPickV6(
            uci=u,
            san=str(move.get("san", u)),
            policy=round(float(policy_by_uci.get(u, 0.0)), 4),
            robust=_robust_score(move, best),
            pool_rank=[str(m["uci"]) for m in pool].index(u),
            is_engine_best=(u == best_uci),
            findable=float(policy_by_uci.get(u, 0.0)) >= gate,
            fallback=fallback,
        )

    if tier == "advanced":
        # Advanced ALWAYS = the verified engine best (pool[0]).
        return _mk(pool[0], fallback=False)

    eligible = [m for m in pool if float(policy_by_uci.get(str(m["uci"]), 0.0)) >= gate]
    if not eligible:
        # Nothing human-findable clears the gate -> a sound engine best beats an
        # unsound "findable" move. Endorse the engine best (marked fallback).
        return _mk(pool[0], fallback=True)

    # Rank by (robustness desc, human-likelihood desc, eval desc, pool order).
    def _key(m: Mapping[str, Any]) -> Tuple:
        u = str(m["uci"])
        return (
            _robust_score(m, best),
            float(policy_by_uci.get(u, 0.0)),
            int(m.get("cp", 0)),
            -[str(x["uci"]) for x in pool].index(u),
        )

    best_move = max(eligible, key=_key)
    return _mk(best_move, fallback=False)


def select_tiers_v6(
    pool: List[Mapping[str, Any]],
    maia_by_tier: Mapping[str, Mapping[str, float]],
    engine_best: Mapping[str, Any],
) -> Dict[str, Any]:
    """Pick the teaching move for all three tiers atomically (complete triad).

    Parameters
    ----------
    pool:
        The **deep, WDL/tablebase-verified** sound pool, best-first, each entry
        ``{"uci","san","cp","wdl":[w,d,l],"depth_agree":bool,"tb":...}``.
    maia_by_tier:
        ``{tier: {uci: policy}}`` for the pool moves (per tier's Maia net).
    engine_best:
        The verified deep engine-best move (``pool[0]`` shape).

    Returns
    -------
    dict with ``picks`` (tier -> TierPickV6 dict), ``distinct_moves``,
    ``discriminating`` (beginner != advanced), ``high_conf_discriminating``,
    ``pattern`` and a short ``rationale``.
    """
    if not pool:
        raise ValueError("select_tiers_v6: empty sound pool")

    picks: Dict[str, TierPickV6] = {}
    for tier in TIER_ORDER:
        picks[tier] = _pick_for_tier(tier, pool, maia_by_tier.get(tier, {}), engine_best)

    # Audit fix #4: kill the B=A != I collapse. A position whose engine best is
    # also the human pick for beginner should be taught consistently.
    if picks["beginner"].uci == picks["advanced"].uci and picks["intermediate"].uci != picks["beginner"].uci:
        picks["intermediate"] = picks["beginner"]

    ucis = {t: picks[t].uci for t in TIER_ORDER}
    distinct = len(set(ucis.values()))
    discriminating = ucis["beginner"] != ucis["advanced"]

    # Pattern label (post-fix, so B=A!=I can no longer occur).
    b, i, a = ucis["beginner"], ucis["intermediate"], ucis["advanced"]
    if b == i == a:
        pattern = "B=I=A"
    elif b == i != a:
        pattern = "B=I!=A"
    elif i == a != b:
        pattern = "I=A!=B"
    elif b == a != i:
        pattern = "B=A!=I"  # should not happen after the fix (defensive)
    else:
        pattern = "B!=I!=A"

    # High-confidence discriminating: beginner differs from advanced AND the
    # beginner pick is genuinely findable while the engine best is NOT easily
    # findable at 1100 (a real teaching fork, not a near-tie).
    beg_pol = picks["beginner"].policy
    eb_beg_pol = float(maia_by_tier.get("beginner", {}).get(str(engine_best["uci"]), 0.0))
    high_conf = bool(
        discriminating
        and picks["beginner"].findable
        and beg_pol >= 0.10
        and (beg_pol - eb_beg_pol) >= 0.05
    )

    rationale = ""
    if discriminating:
        rationale = (
            f"beginner plays the findable {picks['beginner'].san} "
            f"(Maia {round(beg_pol*100)}%), advanced the sharp engine-best "
            f"{picks['advanced'].san}"
        )

    return {
        "picks": {t: picks[t].to_dict() for t in TIER_ORDER},
        "distinct_moves": distinct,
        "discriminating": discriminating,
        "high_conf_discriminating": high_conf,
        "pattern": pattern,
        "rationale": rationale,
    }


# --------------------------------------------------------------------------- #
# Coherent move-review (audit fix #5)
# --------------------------------------------------------------------------- #

def review_student_move(
    student: Optional[Mapping[str, Any]],
    pool: List[Mapping[str, Any]],
    engine_best: Mapping[str, Any],
    tier_pick_uci: str,
) -> Dict[str, Any]:
    """Decide whether to endorse / soften / correct the student's played move.

    Returns ``{"action", "student_uci", "student_in_pool", "cp_loss",
    "crosses_band", "teach_uci"}`` where ``action`` is one of
    ``endorse`` (student move is as good as best), ``soft`` (reasonable but a
    better idea exists), or ``correct`` (genuinely worse — steer to the tier
    move). Never synthesises the student move as the canonical pick.
    """
    if not student or not student.get("uci"):
        return {"action": "none", "student_uci": None, "student_in_pool": False,
                "cp_loss": None, "crosses_band": False, "teach_uci": tier_pick_uci}

    s_uci = str(student["uci"])
    pool_ucis = {str(m["uci"]) for m in pool}
    s_in_pool = s_uci in pool_ucis
    best_cp = int(engine_best.get("cp", 0))
    s_cp = student.get("cp")
    cp_loss = max(0, best_cp - int(s_cp)) if s_cp is not None else student.get("cp_loss")
    cp_loss = int(cp_loss) if cp_loss is not None else None

    best_band = _band(_exp_score(engine_best.get("wdl")))
    s_band = _band(_exp_score(student.get("wdl")))
    crosses = bool(best_band and s_band and best_band != s_band
                   and {"win": 2, "draw": 1, "loss": 0}[s_band] < {"win": 2, "draw": 1, "loss": 0}[best_band])

    if crosses or (cp_loss is not None and cp_loss > CORRECT_CP):
        action = "correct"
    elif cp_loss is not None and cp_loss <= ENDORSE_CP and s_in_pool:
        action = "endorse"
    elif s_in_pool:
        action = "soft"
    else:
        action = "correct"

    return {
        "action": action,
        "student_uci": s_uci,
        "student_in_pool": s_in_pool,
        "cp_loss": cp_loss,
        "crosses_band": crosses,
        "teach_uci": tier_pick_uci,
    }


# --------------------------------------------------------------------------- #
# Self-test (synthetic; no engine dependency)
# --------------------------------------------------------------------------- #

def _self_test() -> bool:
    # Deep pool, best-first by cp; all WDL 'win' band, all depth-agree.
    pool = [
        {"uci": "g1f3", "san": "Nf3", "cp": 60, "wdl": [700, 250, 50], "depth_agree": True, "tb": None},
        {"uci": "b1c3", "san": "Nc3", "cp": 45, "wdl": [680, 260, 60], "depth_agree": True, "tb": None},
        {"uci": "f1c4", "san": "Bc4", "cp": 40, "wdl": [670, 260, 70], "depth_agree": True, "tb": None},
    ]
    best = pool[0]
    # Beginner finds Bc4 most; advanced/engine-best is Nf3.
    maia = {
        "beginner":     {"g1f3": 0.05, "b1c3": 0.15, "f1c4": 0.40},
        "intermediate": {"g1f3": 0.20, "b1c3": 0.25, "f1c4": 0.30},
        "advanced":     {"g1f3": 0.45, "b1c3": 0.20, "f1c4": 0.15},
    }
    r = select_tiers_v6(pool, maia, best)
    p = r["picks"]
    checks = [
        ("advanced == engine best", p["advanced"]["uci"] == "g1f3" and p["advanced"]["is_engine_best"]),
        ("beginner is findable (gate)", p["beginner"]["findable"]),
        ("beginner != advanced (discriminating)", r["discriminating"]),
        ("beginner picks a gate-clearing move", p["beginner"]["uci"] in ("b1c3", "f1c4")),
        ("no B=A!=I", r["pattern"] != "B=A!=I"),
    ]

    # Convergence case: engine best is ALSO the most human at every tier.
    pool2 = [
        {"uci": "d1h5", "san": "Qh5", "cp": 900, "wdl": [980, 15, 5], "depth_agree": True, "tb": None},
        {"uci": "f1c4", "san": "Bc4", "cp": 300, "wdl": [800, 150, 50], "depth_agree": True, "tb": None},
    ]
    maia2 = {t: {"d1h5": 0.80, "f1c4": 0.10} for t in TIER_ORDER}
    r2 = select_tiers_v6(pool2, maia2, pool2[0])
    checks.append(("benign convergence B=I=A", r2["pattern"] == "B=I=A"))
    checks.append(("convergence not discriminating", not r2["discriminating"]))

    # Force a B=A!=I scenario and confirm the fix collapses it to B=I=A.
    # beginner & advanced both land on Qh5; make intermediate's gate pick differ.
    pool3 = [
        {"uci": "d1h5", "san": "Qh5", "cp": 500, "wdl": [900, 80, 20], "depth_agree": True, "tb": None},
        {"uci": "f1c4", "san": "Bc4", "cp": 480, "wdl": [890, 90, 20], "depth_agree": True, "tb": None},
    ]
    maia3 = {
        "beginner":     {"d1h5": 0.60, "f1c4": 0.05},   # beginner -> Qh5
        "intermediate": {"d1h5": 0.05, "f1c4": 0.40},   # would pick Bc4 ...
        "advanced":     {"d1h5": 0.50, "f1c4": 0.10},   # advanced -> Qh5 (engine best)
    }
    r3 = select_tiers_v6(pool3, maia3, pool3[0])
    checks.append(("B=A!=I collapsed to B=I=A", r3["pattern"] == "B=I=A"))

    # Move-review: endorse a best move, correct a blunder.
    rev_endorse = review_student_move(
        {"uci": "g1f3", "cp": 60, "wdl": [700, 250, 50]}, pool, best, "b1c3")
    rev_correct = review_student_move(
        {"uci": "a2a4", "cp": -200, "wdl": [200, 300, 500]}, pool, best, "g1f3")
    checks.append(("endorse student's best move", rev_endorse["action"] == "endorse"))
    checks.append(("correct a genuinely worse move", rev_correct["action"] == "correct"))

    ok = True
    for label, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        ok = ok and cond
    print(f"  case1 pattern={r['pattern']} picks="
          f"B:{p['beginner']['san']} I:{p['intermediate']['san']} A:{p['advanced']['san']}")
    return ok


if __name__ == "__main__":
    import sys

    print("=== tier_select_v6 self-test ===")
    sys.exit(0 if _self_test() else 1)
