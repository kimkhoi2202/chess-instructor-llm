"""Deterministic, tier-aware teaching-move selection (v2).

The v1 teacher let the LLM pick the "teaching move" from the sound pool. The
divergence analysis (``data/analysis/DIVERGENCE_REPORT.md``) measured that this
was only weakly tier-aware and, on identical positions, *mis-directed* —
beginners were steered toward the sharp engine move slightly MORE than advanced
players, the opposite of "give the beginner the move they'd actually find."

This module replaces that with an explicit, correctly-directed rule. Given the
Stockfish **sound pool** (already guardrailed: every move is non-blunder) and the
tier's **Maia** human-move policy, it picks ONE move per tier:

- **beginner**      -> the most human-FINDABLE sound move (highest Maia@1100 in
                       the pool), even when it is not the engine's #1.
- **advanced**      -> the sharpest sound move (the engine best, ``pool[0]``).
- **intermediate**  -> a blend of normalized engine eval and Maia policy.

Formally, with ``w`` = the weight on human-likelihood (Maia) vs engine eval:

    score(move) = (1 - w) * eval_norm(move) + w * policy_norm(move)
    w = {beginner: 1.0, intermediate: 0.5, advanced: 0.0}

so advanced collapses to ``pool[0]`` (pure eval) and beginner to the highest-Maia
sound move (pure human), with intermediate genuinely in between. The pick is
**always inside the sound pool**, so it is never "a move to unlearn."

This is a pure function of ``(tier, sound_pool, maia_policy_by_uci)`` — no engine
process is touched here — so it is trivially unit-testable. ``maia_policy_map``
is the one thin convenience that *does* call Maia, kept separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

# Weight on the human-likelihood (Maia) term vs the engine-eval term, per tier.
# 1.0 => pick purely by "would a human at this tier find it" (most findable).
# 0.0 => pick purely by engine strength (the sharpest sound move = pool[0]).
TIER_HUMAN_WEIGHT: Dict[str, float] = {
    "beginner": 1.0,
    "intermediate": 0.5,
    "advanced": 0.0,
}


@dataclass(frozen=True)
class TierPick:
    """The selected teaching move for a tier plus the signals behind it."""

    uci: str
    san: str
    pool_rank: int          # 0 = engine best (pool[0]); higher = further from best
    eval_norm: float        # normalized engine eval in [0, 1] (1 = best in pool)
    policy: float           # raw Maia policy for the pick in [0, 1]
    policy_norm: float      # normalized Maia policy within the pool in [0, 1]
    score: float            # the blended selection score
    weight: float           # the human-weight w used for this tier
    is_engine_best: bool    # convenience: pool_rank == 0
    n_pool: int             # size of the sound pool considered


def _norm(values: List[float]) -> List[float]:
    """Min-max normalize to [0, 1]; a flat list maps to all 1.0 (no signal)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [1.0 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


def select_tier_move(
    tier: str,
    sound_pool: List[Mapping[str, Any]],
    maia_policy_by_uci: Mapping[str, float],
) -> TierPick:
    """Pick the teaching move for ``tier`` from the (best-first) sound pool.

    Parameters
    ----------
    tier:
        ``"beginner"`` / ``"intermediate"`` / ``"advanced"`` (case-insensitive).
        Unknown tiers fall back to the engine best (``w = 0``).
    sound_pool:
        Non-empty list of ``{"uci", "san", "cp", ...}`` ordered best-first
        (Stockfish ``sound_pool``). ``cp`` is side-to-move POV.
    maia_policy_by_uci:
        Map ``uci -> Maia policy`` in ``[0, 1]`` for this tier's Maia net. Moves
        absent from the map are treated as policy ``0.0``.

    Returns
    -------
    TierPick
        The chosen move and the signals behind it.

    Raises
    ------
    ValueError
        If ``sound_pool`` is empty.
    """
    if not sound_pool:
        raise ValueError("select_tier_move: sound_pool is empty")

    w = TIER_HUMAN_WEIGHT.get(str(tier).strip().lower(), 0.0)

    cps = [float(m["cp"]) for m in sound_pool]
    pols = [float(maia_policy_by_uci.get(str(m["uci"]), 0.0)) for m in sound_pool]
    eval_norms = _norm(cps)
    pol_norms = _norm(pols)

    best_idx = 0
    best_key: Optional[tuple] = None
    for i, m in enumerate(sound_pool):
        score = (1.0 - w) * eval_norms[i] + w * pol_norms[i]
        # Tie-break: higher score, then higher raw eval (sounder), then earlier
        # in the pool (closer to engine best) for full determinism.
        key = (score, cps[i], -i)
        if best_key is None or key > best_key:
            best_key = key
            best_idx = i

    m = sound_pool[best_idx]
    return TierPick(
        uci=str(m["uci"]),
        san=str(m.get("san", m["uci"])),
        pool_rank=best_idx,
        eval_norm=round(eval_norms[best_idx], 4),
        policy=round(pols[best_idx], 4),
        policy_norm=round(pol_norms[best_idx], 4),
        score=round(best_key[0], 4),
        weight=w,
        is_engine_best=best_idx == 0,
        n_pool=len(sound_pool),
    )


def maia_policy_map(fen: str, tier: str, *, top_k: int = 64) -> Dict[str, float]:
    """Return ``{uci: Maia policy}`` for (almost) every legal move at ``tier``.

    Thin convenience over :func:`src.engine.maia_engine.human_moves` with a large
    ``top_k`` so every sound-pool move has a policy to look up. Import is local so
    :func:`select_tier_move` stays engine-free (and unit-testable). Returns an
    empty dict if Maia is unavailable — callers should treat that as "no human
    signal" (selection then falls back to eval, i.e. the engine best).
    """
    from src.engine import maia_engine

    try:
        res = maia_engine.human_moves(fen, tier, top_k=top_k)["moves"]
    except Exception:  # noqa: BLE001 - Maia is a helpful signal, not required
        return {}
    return {str(m["uci"]): float(m["policy"]) for m in res}


# --------------------------------------------------------------------------- #
# Self-test (synthetic; no engine dependency)
# --------------------------------------------------------------------------- #

def _self_test() -> bool:
    """Assert the tier gradient behaves as specified on a synthetic pool."""
    # Pool best-first by cp. The engine best (Kg7) is NOT the most human move.
    pool = [
        {"uci": "g8g7", "san": "Kg7", "cp": 40},   # engine best, low human policy
        {"uci": "e7e5", "san": "e5", "cp": 20},    # middling
        {"uci": "b8c6", "san": "Nc6", "cp": 10},   # most human, weakest (but sound)
    ]
    maia = {"g8g7": 0.05, "e7e5": 0.25, "b8c6": 0.55}

    b = select_tier_move("beginner", pool, maia)
    i = select_tier_move("intermediate", pool, maia)
    a = select_tier_move("advanced", pool, maia)

    ok = True
    checks = [
        ("advanced picks engine best", a.uci == "g8g7" and a.pool_rank == 0),
        ("beginner picks highest-Maia sound move", b.uci == "b8c6"),
        ("beginner pool_rank >= advanced pool_rank", b.pool_rank >= a.pool_rank),
        ("intermediate between (rank)", i.pool_rank <= b.pool_rank),
        ("single-move pool is unambiguous",
         select_tier_move("beginner", [pool[0]], maia).uci == "g8g7"),
        ("no-maia falls back to engine best",
         select_tier_move("beginner", pool, {}).uci == "g8g7"),
    ]
    for label, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        ok = ok and cond
    print(f"  beginner={b.san}(rank{b.pool_rank}) "
          f"intermediate={i.san}(rank{i.pool_rank}) advanced={a.san}(rank{a.pool_rank})")
    return ok


if __name__ == "__main__":
    import sys

    print("=== tier_select self-test ===")
    sys.exit(0 if _self_test() else 1)
