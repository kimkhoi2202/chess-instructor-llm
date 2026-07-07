#!/usr/bin/env python3
"""Compare v1 vs v2 on the SAME held-out set: tier-differentiation + fabrication.

Reads two divergence artifacts (produced by ``scripts.divergence_analysis`` on a
matched held-out sample) and computes, for each model, the two metrics v2 targets:

* **TIER-DIFFERENTIATION** — % of positions where the three tiers do NOT all pick
  the same move, plus the *direction* (mean pool-rank per tier: beginner should be
  HIGHER than advanced = beginners steered to the more human-findable move), and
  the per-tier ``== SF best`` / ``== Maia top`` rates.
* **FABRICATION RATE** — % of (position x tier) coaching outputs that state >=1
  demonstrably-false board fact (the non-LLM ``verify_text`` truth gate), overall
  and per tier — the honest hallucination metric from RESULTS.md.

Writes a side-by-side JSON + prints a markdown block for RESULTS_V2.md.

Run::  ~/.venvs/mlx/bin/python -m scripts.divergence_compare_v2 \
          --v1 data/analysis/divergence_v1_matched.jsonl \
          --v2 data/analysis/divergence_v2.jsonl \
          --out data/analysis/divergence_compare_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.engine.faithfulness import verify_text  # noqa: E402

TIERS = ("beginner", "intermediate", "advanced")


def _load(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _pool_rank(row: Dict[str, Any], uci: Optional[str]) -> Optional[int]:
    if uci is None:
        return None
    for i, m in enumerate(row["sound_pool"]):
        if m["uci"] == uci:
            return i
    return None


def compute(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    n_diff = sum(1 for r in rows if r["n_distinct_tier_moves"] > 1)
    distinct = Counter(r["n_distinct_tier_moves"] for r in rows)

    per_tier: Dict[str, Any] = {}
    for t in TIERS:
        eq_sf = sum(1 for r in rows if r["tiers"][t]["eq_pool0"])
        eq_maia = sum(1 for r in rows if r["tiers"][t]["eq_maia_top"])
        ranks = [_pool_rank(r, r["tiers"][t]["rec_uci"]) for r in rows]
        ranks = [x for x in ranks if x is not None]
        per_tier[t] = {
            "eq_sf_pct": round(100 * eq_sf / n, 1) if n else None,
            "eq_maia_pct": round(100 * eq_maia / n, 1) if n else None,
            "mean_pool_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
        }

    # Fabrication over every (position x tier) raw coaching output.
    fab_hits = 0
    fab_total = 0
    fab_by_tier: Dict[str, List[int]] = {t: [0, 0] for t in TIERS}
    for r in rows:
        for t in TIERS:
            text = r["tiers"][t].get("raw") or r["tiers"][t].get("coaching") or ""
            fab_total += 1
            fab_by_tier[t][1] += 1
            if not verify_text(text, r["fen"]).ok:
                fab_hits += 1
                fab_by_tier[t][0] += 1

    # Direction: is beginner's mean pool-rank > advanced's? (the target)
    b_rank = per_tier["beginner"]["mean_pool_rank"]
    a_rank = per_tier["advanced"]["mean_pool_rank"]
    directed = (b_rank is not None and a_rank is not None and b_rank > a_rank)

    return {
        "n": n,
        "differentiation_pct": round(100 * n_diff / n, 1) if n else None,
        "distinct_distribution": {str(k): distinct.get(k, 0) for k in (1, 2, 3)},
        "per_tier": per_tier,
        "direction_correct_beginner_gt_advanced": directed,
        "fabrication_rate_pct": round(100 * fab_hits / fab_total, 1) if fab_total else None,
        "fabrication_by_tier_pct": {
            t: round(100 * fab_by_tier[t][0] / fab_by_tier[t][1], 1) if fab_by_tier[t][1] else None
            for t in TIERS
        },
    }


def _md(v1: Dict[str, Any], v2: Dict[str, Any]) -> str:
    def row(label: str, a: Any, b: Any) -> str:
        return f"| {label} | {a} | {b} |"

    L = ["| Metric | v1 | v2 |", "|---|---:|---:|"]
    L.append(row("Held-out positions", v1["n"], v2["n"]))
    L.append(row("**Tier-differentiation** (≥1 tier differs)",
                 f"{v1['differentiation_pct']}%", f"{v2['differentiation_pct']}%"))
    L.append(row("Distinct-move distribution (1/2/3)",
                 "/".join(str(v1["distinct_distribution"][k]) for k in ("1", "2", "3")),
                 "/".join(str(v2["distinct_distribution"][k]) for k in ("1", "2", "3"))))
    for t in TIERS:
        L.append(row(f"mean pool-rank — {t}",
                     v1["per_tier"][t]["mean_pool_rank"], v2["per_tier"][t]["mean_pool_rank"]))
    L.append(row("Direction correct (beginner rank > advanced)",
                 v1["direction_correct_beginner_gt_advanced"],
                 v2["direction_correct_beginner_gt_advanced"]))
    for t in TIERS:
        L.append(row(f"beginner→ {t} == Maia top %",
                     v1["per_tier"][t]["eq_maia_pct"], v2["per_tier"][t]["eq_maia_pct"]))
    L.append(row("**Fabrication rate** (≥1 false board fact)",
                 f"{v1['fabrication_rate_pct']}%", f"{v2['fabrication_rate_pct']}%"))
    for t in TIERS:
        L.append(row(f"fabrication — {t}",
                     f"{v1['fabrication_by_tier_pct'][t]}%", f"{v2['fabrication_by_tier_pct'][t]}%"))
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v1", default="data/analysis/divergence_v1_matched.jsonl")
    p.add_argument("--v2", default="data/analysis/divergence_v2.jsonl")
    p.add_argument("--out", default="data/analysis/divergence_compare_v2.json")
    args = p.parse_args(argv)

    def _abs(x: str) -> Path:
        pp = Path(x)
        return pp if pp.is_absolute() else _ROOT / pp

    v1_rows = _load(_abs(args.v1))
    v2_rows = _load(_abs(args.v2))
    v1 = compute(v1_rows)
    v2 = compute(v2_rows)
    out = {"v1": v1, "v2": v2}
    _abs(args.out).parent.mkdir(parents=True, exist_ok=True)
    _abs(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=== v1 vs v2 — differentiation + fabrication (matched held-out) ===\n")
    print(_md(v1, v2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
