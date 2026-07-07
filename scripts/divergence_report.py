#!/usr/bin/env python3
"""Compute divergence metrics from divergence.jsonl and write DIVERGENCE_REPORT.md.

Reports (exact rates):
  * TIER-DIFFERENTIATION: % of positions where the 3 tiers do NOT all agree,
    plus the all-same / 2-distinct / 3-distinct distribution. Reported both over
    ALL positions (live-app extraction incl. pool[0] fallback) and over the
    GENUINE subset (all three tiers named a sound move in prose).
  * ENGINE-DIVERGENCE per tier: % where the model's move != Stockfish best
    (pool[0]) and != that tier's Maia top.
  * JOINT "interesting" set: tiers differ OR model diverges from SF-best.
  * SANITY: how often the model's move is simply pool[0], per tier, and the
    extraction-mode split (cue / prose / fallback_pool0) that context requires.
  * VALUE 2x2: among genuinely named picks, mirrors-engine-only vs
    mirrors-human-only vs both vs neither.

Run::  ~/.venvs/mlx/bin/python -m scripts.divergence_report
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import chess

_ROOT = Path(__file__).resolve().parents[1]
TIERS = ("beginner", "intermediate", "advanced")
IN_PATH = _ROOT / "data" / "analysis" / "divergence.jsonl"
OUT_PATH = _ROOT / "data" / "analysis" / "DIVERGENCE_REPORT.md"


def _load(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _pct(n: int, d: int) -> str:
    return f"{100.0*n/d:5.1f}%" if d else "  n/a"


def _tier_move(row: Dict[str, Any], tier: str) -> Optional[str]:
    return row["tiers"][tier]["rec_uci"]


def _pool_rank(row: Dict[str, Any], uci: Optional[str]) -> Optional[int]:
    if uci is None:
        return None
    for i, m in enumerate(row["sound_pool"]):
        if m["uci"] == uci:
            return i
    return None


def main() -> int:
    if not IN_PATH.exists():
        print(f"missing {IN_PATH}", file=sys.stderr)
        return 1
    rows = _load(IN_PATH)
    N = len(rows)
    if N == 0:
        print("no rows", file=sys.stderr)
        return 1

    # ---- Tier differentiation (all positions) --------------------------- #
    distinct_counts = Counter(r["n_distinct_tier_moves"] for r in rows)
    n_diff = sum(1 for r in rows if r["n_distinct_tier_moves"] > 1)

    # Genuine subset: all three tiers actually named a sound move.
    genuine_rows = [r for r in rows if r["genuine_all_tiers"]]
    Ng = len(genuine_rows)
    distinct_counts_g = Counter(r["n_distinct_tier_moves"] for r in genuine_rows)
    n_diff_g = sum(1 for r in genuine_rows if r["n_distinct_tier_moves"] > 1)

    # ---- Per-tier engine / maia divergence + sanity -------------------- #
    per_tier: Dict[str, Dict[str, Any]] = {}
    for tier in TIERS:
        eq_sf = sum(1 for r in rows if r["tiers"][tier]["eq_pool0"])
        eq_maia = sum(1 for r in rows if r["tiers"][tier]["eq_maia_top"])
        modes = Counter(r["tiers"][tier]["mode"] for r in rows)
        genuine = [r for r in rows if r["tiers"][tier]["genuine"]]
        eq_sf_gen = sum(1 for r in genuine if r["tiers"][tier]["eq_pool0"])
        eq_maia_gen = sum(1 for r in genuine if r["tiers"][tier]["eq_maia_top"])
        ranks = [_pool_rank(r, _tier_move(r, tier)) for r in genuine]
        ranks = [x for x in ranks if x is not None]
        per_tier[tier] = {
            "eq_sf": eq_sf,
            "ne_sf": N - eq_sf,
            "eq_maia": eq_maia,
            "ne_maia": N - eq_maia,
            "modes": modes,
            "n_genuine": len(genuine),
            "eq_sf_gen": eq_sf_gen,
            "ne_sf_gen": len(genuine) - eq_sf_gen,
            "eq_maia_gen": eq_maia_gen,
            "mean_pick_rank": (sum(ranks) / len(ranks)) if ranks else None,
            "pick_is_maia_top": eq_maia_gen,
        }

    # ---- Joint interesting set ----------------------------------------- #
    interesting = [r for r in rows if r["interesting"]]
    n_interesting = len(interesting)
    n_diff_only = sum(1 for r in rows if r["n_distinct_tier_moves"] > 1 and not r["diverges_from_sf_any_tier"])
    n_nesf_only = sum(1 for r in rows if r["diverges_from_sf_any_tier"] and r["n_distinct_tier_moves"] == 1)
    n_both = sum(1 for r in rows if r["n_distinct_tier_moves"] > 1 and r["diverges_from_sf_any_tier"])
    # also diverges from maia (any tier) intersect
    n_interesting_ne_maia = sum(
        1 for r in interesting
        if any(not r["tiers"][t]["eq_maia_top"] for t in TIERS)
    )

    # ---- Value 2x2 over genuine picks (pooled across tiers) ------------ #
    v = Counter()  # keys: sf_only / maia_only / both / neither
    total_genuine_picks = 0
    for r in rows:
        for t in TIERS:
            td = r["tiers"][t]
            if not td["genuine"]:
                continue
            total_genuine_picks += 1
            sf = td["eq_pool0"]
            ma = td["eq_maia_top"]
            if sf and ma:
                v["both"] += 1
            elif sf and not ma:
                v["sf_only"] += 1
            elif ma and not sf:
                v["maia_only"] += 1
            else:
                v["neither"] += 1

    # ---- Direction analysis on DIFF genuine rows ----------------------- #
    diff_genuine = [r for r in genuine_rows if r["n_distinct_tier_moves"] > 1]
    dir_rows = len(diff_genuine)
    lean = {t: {"maia": 0, "sf": 0} for t in TIERS}
    rank_by_tier = {t: [] for t in TIERS}
    for r in diff_genuine:
        for t in TIERS:
            td = r["tiers"][t]
            if td["eq_maia_top"]:
                lean[t]["maia"] += 1
            if td["eq_pool0"]:
                lean[t]["sf"] += 1
            rk = _pool_rank(r, _tier_move(r, t))
            if rk is not None:
                rank_by_tier[t].append(rk)

    # ---- Phase / severity breakdown of differentiation ----------------- #
    by_phase = defaultdict(lambda: [0, 0])   # phase -> [diff, total]
    by_sev = defaultdict(lambda: [0, 0])
    for r in rows:
        d = 1 if r["n_distinct_tier_moves"] > 1 else 0
        by_phase[r["phase"]][0] += d
        by_phase[r["phase"]][1] += 1
        sev = r["student_move"]["severity"]
        by_sev[sev][0] += d
        by_sev[sev][1] += 1

    # =================================================================== #
    # Render markdown
    # =================================================================== #
    L: List[str] = []
    A = L.append
    A("# Chess-Coach Move-SELECTION Divergence Report")
    A("")
    A(f"- **Model:** `models/mlx/chess-coach-v1` (tuned)  ·  **Decoding:** greedy (temp=0, deterministic)")
    A(f"- **Positions:** {N} held-out (none appear in `train.jsonl`/`valid.jsonl`; "
      f"excluded by board+side-to-move key)")
    A(f"- **Prompt:** identical to the live app — `render_pool_facts` + "
      f"`render_user_prompt`, system = `coach_system.md` + grounding + format suffix")
    A(f"- **Move extraction:** the live API's `_extract_recommended`, instrumented to "
      f"separate a genuinely NAMED pick (`cue`/`prose`) from the API's `pool[0]` fallback")
    A("")
    A("> Greedy decoding is used on purpose: any move change across tiers is genuine "
      "tier-conditioning (same model, only the tier label / ply-cap / tier-Maia block "
      "differ in the prompt), not sampling noise.")
    A("")

    A("## 1. Headline rates")
    A("")
    A("| Metric | Rate |")
    A("|---|---|")
    A(f"| **Tier-differentiation** (>=1 tier picks a different move), all positions | "
      f"**{_pct(n_diff, N)}** ({n_diff}/{N}) |")
    A(f"| Tier-differentiation, genuine-pick subset (all 3 tiers named a move) | "
      f"{_pct(n_diff_g, Ng)} ({n_diff_g}/{Ng}) |")
    A(f"| **Joint \"interesting\" set** (tiers differ OR model != SF-best) | "
      f"**{_pct(n_interesting, N)}** ({n_interesting}/{N}) |")
    A(f"| Engine-divergence, beginner (move != SF best) | "
      f"{_pct(per_tier['beginner']['ne_sf'], N)} |")
    A(f"| Engine-divergence, intermediate | {_pct(per_tier['intermediate']['ne_sf'], N)} |")
    A(f"| Engine-divergence, advanced | {_pct(per_tier['advanced']['ne_sf'], N)} |")
    A("")

    A("## 2. Tier-differentiation distribution")
    A("")
    A("How many DISTINCT moves the three tiers produce per position "
      "(1 = all tiers agree).")
    A("")
    A("| Distinct tier-moves | All positions | Genuine-pick subset |")
    A("|---|---|---|")
    for k in (1, 2, 3):
        A(f"| {k} | {distinct_counts.get(k,0)} ({_pct(distinct_counts.get(k,0), N)}) | "
          f"{distinct_counts_g.get(k,0)} ({_pct(distinct_counts_g.get(k,0), Ng)}) |")
    A("")
    A(f"- All positions: **{_pct(n_diff, N)}** differentiate ({n_diff}/{N}).")
    A(f"- Genuine subset (n={Ng}): **{_pct(n_diff_g, Ng)}** differentiate ({n_diff_g}/{Ng}).")
    A("")

    A("## 3. Engine- and human-divergence, per tier")
    A("")
    A("`!= SF best` = model's move differs from Stockfish's best (pool[0]). "
      "`!= Maia top` = differs from that tier's most human-likely move. "
      "`== SF best` is the sanity check the task asked for (how often the model just "
      "returns the engine's best).")
    A("")
    A("| Tier | == SF best | != SF best | == Maia top | != Maia top |")
    A("|---|---|---|---|---|")
    for t in TIERS:
        pt = per_tier[t]
        A(f"| {t} | {_pct(pt['eq_sf'], N)} | {_pct(pt['ne_sf'], N)} | "
          f"{_pct(pt['eq_maia'], N)} | {_pct(pt['ne_maia'], N)} |")
    A("")
    A("### Extraction-mode split per tier (context for `== SF best`)")
    A("")
    A("A `fallback_pool0` row means the model did NOT name a sound move in prose, so the "
      "live API *displays* the engine best. Those rows inflate `== SF best` without the "
      "model actually choosing it, so they must be read separately.")
    A("")
    A("| Tier | cue (named) | prose (named) | fallback_pool0 | genuine total | "
      "== SF among genuine |")
    A("|---|---|---|---|---|---|")
    for t in TIERS:
        pt = per_tier[t]
        m = pt["modes"]
        A(f"| {t} | {m.get('cue',0)} | {m.get('prose',0)} | {m.get('fallback_pool0',0)} | "
          f"{pt['n_genuine']} | {_pct(pt['eq_sf_gen'], pt['n_genuine'])} |")
    A("")

    A("## 4. Does it add value beyond copying? (genuine named picks, pooled over tiers)")
    A("")
    A(f"Across all {total_genuine_picks} genuinely named picks (positions x tiers):")
    A("")
    A("| Pick relationship | Count | Share |")
    A("|---|---|---|")
    A(f"| mirrors engine only (== SF best, != Maia top) | {v['sf_only']} | {_pct(v['sf_only'], total_genuine_picks)} |")
    A(f"| mirrors human only (== Maia top, != SF best) | {v['maia_only']} | {_pct(v['maia_only'], total_genuine_picks)} |")
    A(f"| == both (engine best is also the human top) | {v['both']} | {_pct(v['both'], total_genuine_picks)} |")
    A(f"| independent (!= SF best AND != Maia top) | {v['neither']} | {_pct(v['neither'], total_genuine_picks)} |")
    A("")
    A(f"- Picks that equal the engine best (`sf_only` + `both`): "
      f"{_pct(v['sf_only']+v['both'], total_genuine_picks)}.")
    A(f"- Picks that are NOT the engine best (`maia_only` + `neither`): "
      f"{_pct(v['maia_only']+v['neither'], total_genuine_picks)}.")
    A("")

    A("## 5. Direction of differentiation (positions where tiers genuinely differ)")
    A("")
    A(f"Among the {dir_rows} genuine-pick positions where the tiers disagree, does the "
      "beginner tier lean toward the human (Maia) move and the advanced tier toward the "
      "engine's best? Mean pool rank: 0 = engine best, higher = further from engine best.")
    A("")
    A("| Tier | picks == its Maia top | picks == SF best | mean pool rank of pick |")
    A("|---|---|---|---|")
    for t in TIERS:
        mr = rank_by_tier[t]
        mean_rank = f"{sum(mr)/len(mr):.2f}" if mr else "n/a"
        A(f"| {t} | {_pct(lean[t]['maia'], dir_rows)} | {_pct(lean[t]['sf'], dir_rows)} | {mean_rank} |")
    A("")

    A("## 6. Interesting-set composition & where differentiation happens")
    A("")
    A(f"- Total interesting positions: **{n_interesting}/{N}** ({_pct(n_interesting, N)}).")
    A(f"  - tiers differ AND diverge from SF: {n_both}")
    A(f"  - tiers differ only (all tiers == SF best but not all identical is impossible; "
      f"this counts tiers-differ with every tier == SF best): {n_diff_only}")
    A(f"  - diverge from SF only (tiers agree on a non-SF-best move): {n_nesf_only}")
    A(f"  - of interesting positions, also != Maia top on >=1 tier: {n_interesting_ne_maia}")
    A("")
    A("Differentiation rate by phase / severity:")
    A("")
    A("| Phase | diff / total |   | Severity | diff / total |")
    A("|---|---|---|---|---|")
    phases = sorted(by_phase)
    sevs = sorted(by_sev)
    for i in range(max(len(phases), len(sevs))):
        lp = f"{phases[i]} | {by_phase[phases[i]][0]}/{by_phase[phases[i]][1]} ({_pct(by_phase[phases[i]][0], by_phase[phases[i]][1])})" if i < len(phases) else " | "
        ls = f"{sevs[i]} | {by_sev[sevs[i]][0]}/{by_sev[sevs[i]][1]} ({_pct(by_sev[sevs[i]][0], by_sev[sevs[i]][1])})" if i < len(sevs) else " | "
        A(f"| {lp} |   | {ls} |")
    A("")

    # ---- Example interesting positions --------------------------------- #
    A("## 7. Example differentiating positions")
    A("")
    ex = [r for r in genuine_rows if r["n_distinct_tier_moves"] > 1][:8]
    if not ex:
        A("_None: no position produced a genuinely different named move across tiers._")
    for r in ex:
        b = r["beginner_move"]["san"]; im = r["intermediate_move"]["san"]; a = r["advanced_move"]["san"]
        A(f"- `{r['id']}` [{r['phase']}/{r['student_move']['severity']}] SF best "
          f"**{r['stockfish_best']['san']}** — B:**{b}** I:**{im}** A:**{a}**")
    A("")

    report = "\n".join(L)
    OUT_PATH.write_text(report, encoding="utf-8")

    # ---- console summary ----------------------------------------------- #
    print("=" * 68)
    print(f"N={N} | genuine-all={Ng}")
    print(f"TIER-DIFFERENTIATION (all):     {_pct(n_diff, N)}  ({n_diff}/{N})")
    print(f"TIER-DIFFERENTIATION (genuine): {_pct(n_diff_g, Ng)}  ({n_diff_g}/{Ng})")
    print(f"  distinct dist (all): 1={distinct_counts.get(1,0)} 2={distinct_counts.get(2,0)} 3={distinct_counts.get(3,0)}")
    print(f"INTERESTING SET: {_pct(n_interesting, N)} ({n_interesting}/{N})")
    for t in TIERS:
        pt = per_tier[t]
        print(f"  {t:12s} !=SF {_pct(pt['ne_sf'],N)} | ==SF {_pct(pt['eq_sf'],N)} | "
              f"!=Maia {_pct(pt['ne_maia'],N)} | fallback {pt['modes'].get('fallback_pool0',0)}")
    print(f"VALUE 2x2 (genuine picks={total_genuine_picks}): "
          f"sf_only {_pct(v['sf_only'],total_genuine_picks)} | "
          f"maia_only {_pct(v['maia_only'],total_genuine_picks)} | "
          f"both {_pct(v['both'],total_genuine_picks)} | "
          f"neither {_pct(v['neither'],total_genuine_picks)}")
    print(f"wrote {OUT_PATH}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
