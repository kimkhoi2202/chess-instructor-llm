#!/usr/bin/env python3
"""Build web/public/library_differentiated.json from divergence.jsonl.

Filters to the "interesting" positions (the 3 tiers pick different moves, and/or
the tuned model's pick diverges from Stockfish best / that tier's Maia top) and
emits them in the EXACT schema of web/public/library.json so the set can be
swapped into the Studio later. Does NOT overwrite library.json and does NOT edit
any frontend source.

For each interesting position we pick a *display tier* that best shows the
model's move-selection value: a genuinely-named pick that diverges from the
engine best (preferring one that matches the human/Maia top). The label encodes
all three tier picks so the differentiation is visible in the gallery; the full
per-tier coaching lives in divergence.jsonl.

Run::  ~/.venvs/mlx/bin/python -m scripts.build_differentiated_library
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import chess

_ROOT = Path(__file__).resolve().parents[1]
TIERS = ("beginner", "intermediate", "advanced")
IN_PATH = _ROOT / "data" / "analysis" / "divergence.jsonl"
OUT_PATH = _ROOT / "web" / "public" / "library_differentiated.json"
MODEL_NAME = "models/mlx/chess-coach-v1"


def _load(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _display_tier(row: Dict[str, Any]) -> str:
    """Choose the tier whose pick best demonstrates move-selection value."""
    best_uci = row["stockfish_best"]["uci"]
    # 1) genuine, diverges from SF, and == human/Maia top (clearest value)
    for t in TIERS:
        td = row["tiers"][t]
        if td["genuine"] and td["rec_uci"] != best_uci and td["eq_maia_top"]:
            return t
    # 2) genuine and diverges from SF
    for t in TIERS:
        td = row["tiers"][t]
        if td["genuine"] and td["rec_uci"] != best_uci:
            return t
    # 3) any genuine pick, else advanced
    for t in ("advanced", "intermediate", "beginner"):
        if row["tiers"][t]["genuine"]:
            return t
    return "advanced"


def _coach_obj(row: Dict[str, Any], tier: str) -> Dict[str, Any]:
    board = chess.Board(row["fen"])
    td = row["tiers"][tier]
    sm = row["student_move"]
    student_block: Optional[Dict[str, Any]] = None
    if sm.get("uci"):
        student_block = {
            "san": sm["san"],
            "uci": sm["uci"],
            "cp_loss": sm["cp_loss"],
            "severity": sm["severity"],
        }
    return {
        "recommended_move_san": td["rec_san"],
        "recommended_move_uci": td["rec_uci"],
        "coaching": td["coaching"],
        "takeaway": td["takeaway"],
        "concepts_used": [],
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "engine": {
            "best_san": row["stockfish_best"]["san"],
            "best_cp": row["stockfish_best"]["cp"],
            "sound_pool": [
                {"san": m["san"], "uci": m["uci"], "cp": m["cp"], "pv": m["pv"]}
                for m in row["sound_pool"]
            ],
            "student_move": student_block,
        },
        "maia": [
            {"san": m["san"], "uci": m["uci"], "policy": m["policy"]}
            for m in row["maia_by_tier"][tier]["moves"]
        ],
        "meta": {"model": MODEL_NAME, "tuned": True, "notes": []},
    }


def _label(row: Dict[str, Any], tier: str) -> str:
    base = f"{tier.capitalize()} \u00b7 {row['phase']}"
    if row["student_move"].get("san") and row["student_move"]["san"] != "(none provided)":
        base += f" \u00b7 played {row['student_move']['san']}"
    b = row["beginner_move"]["san"]
    im = row["intermediate_move"]["san"]
    a = row["advanced_move"]["san"]
    if row["n_distinct_tier_moves"] > 1:
        base += f" \u00b7 tiers B:{b}/I:{im}/A:{a}"
    else:
        base += f" \u00b7 picks {row['tiers'][tier]['rec_san']} over engine {row['stockfish_best']['san']}"
    return base


def main() -> int:
    if not IN_PATH.exists():
        print(f"missing {IN_PATH}", file=sys.stderr)
        return 1
    rows = _load(IN_PATH)
    interesting = [r for r in rows if r["interesting"]]

    library: List[Dict[str, Any]] = []
    for r in interesting:
        tier = _display_tier(r)
        library.append(
            {
                "id": r["id"],
                "label": _label(r, tier),
                "fen": r["fen"],
                "tier": tier,
                "phase": r["phase"],
                "severity": r["student_move"]["severity"],
                "student_move": (
                    r["student_move"]["san"]
                    if r["student_move"].get("san") != "(none provided)"
                    else None
                ),
                "coach": _coach_obj(r, tier),
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(library, indent=2, ensure_ascii=False), encoding="utf-8")

    n_diff = sum(1 for r in interesting if r["n_distinct_tier_moves"] > 1)
    n_nesf = sum(1 for r in interesting if r["diverges_from_sf_any_tier"])
    print(f"wrote {len(library)} interesting entries -> {OUT_PATH}")
    print(f"  of which tiers-differ: {n_diff} | diverge-from-SF (any tier): {n_nesf}")
    print(f"  (total analyzed positions: {len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
