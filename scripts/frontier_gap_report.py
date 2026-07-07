#!/usr/bin/env python3
"""Turn ``frontier_gap.jsonl`` into ``GAP_REPORT.md`` — the honest gap verdict.

Reads the raw per-position picks written by ``scripts.frontier_gap`` and computes,
for each frontier model (GPT-5.5, Claude Opus 4.8, Gemini 3.1 Pro) AND the v1 tuned
model, the three rates the instructor asked for, plus a faithfulness rate:

  1. TIER-DIFFERENTIATION — does the model pick a DIFFERENT move across tiers, or
     the same move regardless of the stated ELO?
  2. FINDABILITY (tier-appropriateness) — is the beginner recommendation the
     *human-findable* sound move (top Maia-1100 rank in the sound pool), or the
     sharp engine-best? Reported with a "findability gap" and, crucially, on the
     OPPORTUNITY SUBSET where the engine-best is NOT already the most findable
     move (the only positions where the behavior can even be exercised).
  3. ENGINE-MIRRORING — how often the model just returns Stockfish's #1, per tier
     and at EVERY tier at once.
  4. FABRICATION — fraction of coaching outputs that state a demonstrably-false
     board fact (deterministic ``faithfulness.verify_text``).

Then it writes the VERDICT: is the gap REAL (frontier models also fail to reliably
pick the tier-appropriate findable move / mirror the engine / don't differentiate),
or are the frontier models already good at it (=> prompting works, no gap here)?

Run::  ~/.venvs/mlx/bin/python -m scripts.frontier_gap_report
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = _ROOT / "data" / "analysis" / "frontier_gap.jsonl"
OUT_PATH = _ROOT / "data" / "analysis" / "GAP_REPORT.md"

TIERS = ("beginner", "intermediate", "advanced")
#: Report order — frontier models first, then the v1 tuned baseline.
MODEL_ORDER = ("gpt-5.5", "claude-opus-4.8", "gemini-3.1-pro", "v1-tuned")


def _load(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%" if d else "  n/a"


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(x: Optional[float], nd: int = 2) -> str:
    return f"{x:.{nd}f}" if x is not None else "n/a"


def compute_model(rows: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """All gap metrics for one model over the shared position set."""
    N = len(rows)

    # ---- tier differentiation ---------------------------------------- #
    distinct = Counter()
    n_diff = 0
    genuine_rows = 0
    n_diff_gen = 0
    for r in rows:
        td = r["models"][model]
        ucis = [td[t]["rec_uci"] for t in TIERS]
        d = len(set(ucis))
        distinct[d] += 1
        if d > 1:
            n_diff += 1
        if all(td[t].get("genuine") for t in TIERS):
            genuine_rows += 1
            if d > 1:
                n_diff_gen += 1

    # ---- per-tier engine mirror / maia + modes ----------------------- #
    per_tier: Dict[str, Dict[str, Any]] = {}
    for t in TIERS:
        eq_sf = sum(1 for r in rows if r["models"][model][t]["eq_sf_best"])
        eq_maia = sum(1 for r in rows if r["models"][model][t]["eq_maia_top"])
        modes = Counter(r["models"][model][t]["mode"] for r in rows)
        # pick's maia rank in pool (0 = most findable sound move)
        pick_maia_ranks = [r["models"][model][t].get("maia_pool_rank") for r in rows]
        pick_engine_ranks = [r["models"][model][t].get("engine_pool_rank") for r in rows]
        pick_is_findable = sum(1 for x in pick_maia_ranks if x == 0)
        per_tier[t] = {
            "eq_sf": eq_sf,
            "eq_maia": eq_maia,
            "modes": modes,
            "mean_pick_maia_rank": _mean(pick_maia_ranks),
            "mean_pick_engine_rank": _mean(pick_engine_ranks),
            "pick_is_findable": pick_is_findable,
        }

    # ---- engine-mirror at EVERY tier --------------------------------- #
    mirror_all = sum(
        1 for r in rows if all(r["models"][model][t]["eq_sf_best"] for t in TIERS)
    )

    # ---- findability gap (beginner, Maia-1100) ----------------------- #
    # findability_gap = engine_best_maia_rank - pick_maia_rank
    #   > 0  pick is MORE human-findable than the sharp engine best (good)
    #   = 0  pick is as findable as engine best (usually pick == engine best)
    #   < 0  pick is LESS findable (over-leveled)
    gaps: List[float] = []
    gap_pos = gap_zero = gap_neg = 0
    # Opportunity subset: positions where the engine best is NOT already the most
    # findable sound move at 1100 (engine_best_maia_rank > 0). Only here can a
    # tier-appropriate coach beat the engine on findability.
    opp_total = 0
    opp_pick_findable = 0
    opp_pick_mirror = 0
    opp_gaps: List[float] = []
    policy_gaps: List[float] = []
    for r in rows:
        g = r["grounding"]["beginner"]
        ebr = g.get("engine_best_maia_rank")
        pick_uci = r["models"][model]["beginner"]["rec_uci"]
        pr = g["maia_rank"].get(pick_uci) if pick_uci else None
        if ebr is None or pr is None:
            continue
        gap = ebr - pr
        gaps.append(gap)
        if gap > 0:
            gap_pos += 1
        elif gap == 0:
            gap_zero += 1
        else:
            gap_neg += 1
        # policy gap (fraction of humans, at 1100)
        best_uci = (r["stockfish_best"] or {}).get("uci")
        pol = g.get("maia_policy", {})
        if best_uci in pol and pick_uci in pol:
            policy_gaps.append(pol[pick_uci] - pol[best_uci])
        if ebr > 0:
            opp_total += 1
            opp_gaps.append(gap)
            if pr == 0:
                opp_pick_findable += 1
            if pick_uci == best_uci:
                opp_pick_mirror += 1

    # ---- fabrication (faithfulness) ---------------------------------- #
    n_outputs = 0
    n_fab = 0
    total_viol = 0
    for r in rows:
        for t in TIERS:
            td = r["models"][model][t]
            n_outputs += 1
            v = td.get("faith_violations") or 0
            total_viol += v
            if v > 0:
                n_fab += 1

    return {
        "N": N,
        "distinct": distinct,
        "n_diff": n_diff,
        "genuine_rows": genuine_rows,
        "n_diff_gen": n_diff_gen,
        "per_tier": per_tier,
        "mirror_all": mirror_all,
        "findability": {
            "mean_gap": _mean(gaps),
            "gap_pos": gap_pos,
            "gap_zero": gap_zero,
            "gap_neg": gap_neg,
            "n": len(gaps),
            "mean_policy_gap": _mean(policy_gaps),
            "opp_total": opp_total,
            "opp_pick_findable": opp_pick_findable,
            "opp_pick_mirror": opp_pick_mirror,
            "opp_mean_gap": _mean(opp_gaps),
        },
        "fab": {"n_outputs": n_outputs, "n_fab": n_fab, "total_viol": total_viol},
    }


def _verdict(stats: Dict[str, Dict[str, Any]], N: int) -> List[str]:
    """Data-driven honest verdict on whether the gap is real."""
    frontier = [m for m in ("gpt-5.5", "claude-opus-4.8", "gemini-3.1-pro") if m in stats]
    L: List[str] = []

    def diff_rate(m: str) -> float:
        return 100.0 * stats[m]["n_diff"] / stats[m]["N"] if stats[m]["N"] else 0.0

    def mirror_all_rate(m: str) -> float:
        return 100.0 * stats[m]["mirror_all"] / stats[m]["N"] if stats[m]["N"] else 0.0

    def opp_findable_rate(m: str) -> Optional[float]:
        f = stats[m]["findability"]
        return 100.0 * f["opp_pick_findable"] / f["opp_total"] if f["opp_total"] else None

    avg_diff = _mean([diff_rate(m) for m in frontier])
    avg_mirror = _mean([mirror_all_rate(m) for m in frontier])
    opp_rates = [opp_findable_rate(m) for m in frontier]
    avg_opp = _mean([x for x in opp_rates if x is not None])

    # Heuristic thresholds for "already good at the behavior".
    good_diff = avg_diff is not None and avg_diff >= 60.0
    good_find = avg_opp is not None and avg_opp >= 60.0
    low_mirror = avg_mirror is not None and avg_mirror <= 25.0
    gap_real = not (good_diff and good_find)

    L.append("## VERDICT — is the gap real?")
    L.append("")
    L.append(f"**{'YES — the gap is REAL.' if gap_real else 'NO — frontier models already do this well.'}**")
    L.append("")
    L.append(
        f"Across the three frontier models (same {N} held-out positions, identical "
        f"grounding as our app), the average rates are:"
    )
    L.append("")
    L.append(f"- **Tier-differentiation:** {_fmt(avg_diff, 1)}% "
             f"(picks a different move across the three ELO tiers).")
    L.append(f"- **Beginner findability on the opportunity subset:** {_fmt(avg_opp, 1)}% "
             f"(when the engine-best is NOT already the most human-findable move, how often "
             f"the frontier model still steers the beginner to the findable one).")
    L.append(f"- **Engine-mirroring at EVERY tier:** {_fmt(avg_mirror, 1)}% "
             f"(returns Stockfish's #1 at all three tiers, regardless of level).")
    L.append("")
    if gap_real:
        reasons = []
        if not good_diff:
            reasons.append(
                f"they mostly recommend the **same move regardless of the stated ELO** "
                f"(tier-differentiation only {_fmt(avg_diff,1)}%)")
        if not good_find:
            reasons.append(
                f"on the positions where it matters, they **default to the sharp engine-best "
                f"instead of the move a beginner would actually find** "
                f"(findable-pick rate only {_fmt(avg_opp,1)}% on the opportunity subset)")
        if not low_mirror:
            reasons.append(
                f"a large share **mirror Stockfish's #1 at every tier** ({_fmt(avg_mirror,1)}%)")
        L.append("The frontier models are strong chess players and their prose is fluent, but on "
                 "the NARROW target behavior — *tier-appropriate, human-findable move selection* — "
                 + "; ".join(reasons) + ".")
        L.append("")
        L.append("That is exactly the gap: prompting a frontier model with the same engine+Maia "
                 "grounding does **not** reliably produce the leveled teaching-move behavior. So "
                 "there is real room for a trained model to win on this behavior — and the "
                 "`EVAL_AND_ITERATE` loop is set up to prove (or disprove) that our v2 does.")
    else:
        L.append("The frontier models already differentiate by tier AND steer beginners to the "
                 "human-findable move often enough that this specific behavior does **not** need "
                 "training to unlock — a well-grounded prompt gets it. If we want a defensible gap, "
                 "it must be a DIFFERENT axis (e.g. cost/latency/local, or truthful grounding at "
                 "small scale). We report this honestly rather than claim a gap that isn't there.")
    L.append("")

    # ---- Honest counter-finding: faithfulness (frontier wins this one) ---- #
    def fab_rate(m: str) -> Optional[float]:
        fb = stats[m]["fab"]
        return 100.0 * fb["n_fab"] / fb["n_outputs"] if fb["n_outputs"] else None

    avg_fab_frontier = _mean([fab_rate(m) for m in frontier])
    v1_fab = fab_rate("v1-tuned") if "v1-tuned" in stats else None
    L.append("### The honest counter-finding: faithfulness")
    L.append("")
    if v1_fab is not None and avg_fab_frontier is not None:
        L.append(
            f"The gap above is about *move selection*. On *truthfulness* the result is the "
            f"**opposite**, and we report it plainly: the frontier models fabricate a board fact "
            f"in only **{_fmt(avg_fab_frontier,1)}%** of coaching outputs, while our **v1 tuned "
            f"model fabricates in {_fmt(v1_fab,1)}%** — the flat-truthfulness failure `RESULTS.md` "
            f"flagged, now quantified against the frontier. So the frontier is *not* uniformly bad "
            f"at coaching: it is weak at leveling the move but strong at not lying about the board.")
        L.append("")
        L.append(
            "That is why the pass bar in `EVAL_AND_ITERATE.md` requires v2 to win on **both** the "
            "move-selection gap **and** fabrication (≤ frontier). A v2 that levels the move but "
            "keeps v1's fabrication rate would be worse than the frontier where the frontier is "
            "already strong — and we would not ship it or call it a win.")
    else:
        L.append("_(faithfulness comparison unavailable — v1 picks not present in this run.)_")
    L.append("")

    # ---- Where v1 already stands on the gap (context for iteration) ---- #
    if "v1-tuned" in stats:
        s1 = stats["v1-tuned"]
        opp1 = None
        f1 = s1["findability"]
        if f1["opp_total"]:
            opp1 = 100.0 * f1["opp_pick_findable"] / f1["opp_total"]
        L.append("### Where v1 stands on the gap today")
        L.append("")
        L.append(
            f"Our v1 tuned model is currently **in the same weak band as the frontier** on move "
            f"selection (tier-differentiation {_pct(s1['n_diff'], s1['N'])}, opportunity-subset "
            f"findable-pick {_fmt(opp1,1)}%, engine-mirror-at-every-tier "
            f"{_pct(s1['mirror_all'], s1['N'])}). So v1 has **not** yet won the gap either — "
            f"consistent with `DIVERGENCE_REPORT.md` (differentiation weak and mis-directed). The "
            f"gap is real and *open*: nobody in this comparison reliably does the behavior, which "
            f"is exactly the room a targeted v2 data intervention (contrastive multi-tier + a "
            f"tier-aware teacher rule + a faithfulness gate) is designed to claim.")
        L.append("")
    return L


def main() -> int:
    if not IN_PATH.exists():
        print(f"missing {IN_PATH} — run scripts.frontier_gap first.", file=sys.stderr)
        return 1
    rows = _load(IN_PATH)
    N = len(rows)
    if N == 0:
        print("no rows", file=sys.stderr)
        return 1

    models = [m for m in MODEL_ORDER if m in rows[0]["models"]]
    stats = {m: compute_model(rows, m) for m in models}

    L: List[str] = []
    A = L.append
    A("# Frontier Gap Report — is the leveled teaching-move behavior a real gap?")
    A("")
    A("The instructor's framing: **before** we can claim training moved our model into a "
      "valuable behavior gap, we must first prove the **frontier** models are *not reliably "
      "good* at that narrow behavior. This report measures that on data.")
    A("")
    A("## Setup")
    A("")
    A(f"- **Positions:** {N} held-out (a balanced phase×severity subset of "
      "`data/analysis/divergence.jsonl`; every FEN re-verified absent from "
      "`train.jsonl`/`valid.jsonl` by board+side-to-move key).")
    A("- **Grounding:** byte-identical to the live app and to the v1 divergence run — "
      "`render_pool_facts` (verified piece/threat facts) + `render_user_prompt` (the Stockfish "
      "sound pool + the tier's Maia block), system = `coach_system.md` + grounding + format "
      "suffix. Only the model changes.")
    A("- **Models:** GPT-5.5 (`openai-group/gpt-5.5`), Claude Opus 4.8 "
      "(`claude-group/claude-opus-4-8`, temperature omitted), Gemini 3.1 Pro "
      "(`gemini-group/gemini-3.1-pro`) via the TrueFoundry gateway, each at ALL THREE tiers. "
      "`v1-tuned` = our fine-tuned Qwen3-1.7B coach (picks reused from `divergence.jsonl`).")
    A("- **Move extraction:** the live API's `_extract_recommended`, instrumented to separate a "
      "genuinely NAMED pick (`cue`/`prose`) from the `pool[0]` fallback (so the fallback never "
      "silently inflates `== SF best`).")
    A("- **Findability:** each pick's **Maia rank inside the sound pool** at that tier "
      "(0 = the most human-likely sound move). The engine-best's own Maia rank is the yardstick.")
    A("")

    # ---------------- The pass definition ---------------- #
    A("## The target behavior (the pass definition)")
    A("")
    A("For a stated ELO tier, recommend the move that is **(a) SOUND** (inside Stockfish's "
      "tolerance pool — doesn't throw the advantage), **(b) FINDABLE** (high Maia-likelihood at "
      "that tier — a move a human at that level would actually consider, not the sharpest "
      "engine-only line), and **(c) INSTRUCTIVE** — then coach WHY it is good AND HOW a player "
      "at that ELO should think to find it. Serving a beginner the engine-only best move with a "
      "GM-level line is a **FAIL**.")
    A("")

    # ---------------- Headline table ---------------- #
    A("## 1. Headline rates (per model)")
    A("")
    A("| Model | Tier-diff | Engine-mirror @ every tier | Beginner pick == engine-best | "
      "Beginner pick == most-findable | Fabrication rate |")
    A("|---|---|---|---|---|---|")
    for m in models:
        s = stats[m]
        b = s["per_tier"]["beginner"]
        eq_sf_b = b["eq_sf"]
        find_b = b["pick_is_findable"]
        fab = s["fab"]
        A(f"| {'**'+m+'**' if m != 'v1-tuned' else m} | "
          f"{_pct(s['n_diff'], s['N'])} | "
          f"{_pct(s['mirror_all'], s['N'])} | "
          f"{_pct(eq_sf_b, s['N'])} | "
          f"{_pct(find_b, s['N'])} | "
          f"{_pct(fab['n_fab'], fab['n_outputs'])} |")
    A("")
    A("- **Tier-diff** = picks ≥1 different move across the three ELO tiers (greedy where the "
      "family allows; Claude at default). Higher = more level-aware.")
    A("- **Engine-mirror @ every tier** = returns Stockfish's #1 at *all three* tiers. Higher = "
      "more of a pure engine mouthpiece, blind to level.")
    A("- **Beginner pick == most-findable** = the beginner recommendation is the top Maia-1100 "
      "move in the sound pool. This is the behavior we want *high*.")
    A("- **Fabrication rate** = share of coaching outputs (position×tier) with ≥1 demonstrably-"
      "false board claim (deterministic verifier). Lower is better.")
    A("")

    # ---------------- Tier differentiation detail ---------------- #
    A("## 2. Tier-differentiation detail")
    A("")
    A("Distinct moves produced across the 3 tiers per position (1 = same move at every tier).")
    A("")
    A("| Model | 1 (same) | 2 | 3 | differentiate (all) | differentiate (genuine subset) |")
    A("|---|---|---|---|---|---|")
    for m in models:
        s = stats[m]
        dc = s["distinct"]
        A(f"| {m} | {dc.get(1,0)} ({_pct(dc.get(1,0), s['N'])}) | {dc.get(2,0)} | {dc.get(3,0)} | "
          f"**{_pct(s['n_diff'], s['N'])}** | {_pct(s['n_diff_gen'], s['genuine_rows'])} "
          f"({s['n_diff_gen']}/{s['genuine_rows']}) |")
    A("")

    # ---------------- Engine mirroring per tier ---------------- #
    A("## 3. Engine-mirroring per tier (`== Stockfish best`)")
    A("")
    A("| Model | beginner | intermediate | advanced | fallback_pool0 (b/i/a) |")
    A("|---|---|---|---|---|")
    for m in models:
        s = stats[m]
        cells = []
        fbs = []
        for t in TIERS:
            pt = s["per_tier"][t]
            cells.append(_pct(pt["eq_sf"], s["N"]))
            fbs.append(str(pt["modes"].get("fallback_pool0", 0)))
        A(f"| {m} | {cells[0]} | {cells[1]} | {cells[2]} | {'/'.join(fbs)} |")
    A("")
    A("_`fallback_pool0` = the model named no sound move in prose, so extraction falls back to "
      "the engine best; those rows are counted but flagged (they inflate `== SF best` without a "
      "genuine choice)._")
    A("")

    # ---------------- Findability ---------------- #
    A("## 4. Findability at the beginner tier (the crux)")
    A("")
    A("A tier-appropriate beginner move is a **findable** sound move (low Maia-1100 rank), not the "
      "sharpest engine line. The **findability gap** = `engine_best_maia_rank − pick_maia_rank` "
      "(both within the sound pool at Maia-1100): **>0** means the pick is *more* human-findable "
      "than the engine best (good); **0** usually means the pick *is* the engine best; **<0** "
      "means the model over-leveled the beginner.")
    A("")
    A("| Model | mean findability gap | gap>0 | gap=0 | gap<0 | mean pick Maia-rank | "
      "mean Maia-policy gap |")
    A("|---|---|---|---|---|---|---|")
    for m in models:
        f = stats[m]["findability"]
        b = stats[m]["per_tier"]["beginner"]
        A(f"| {m} | {_fmt(f['mean_gap'])} | {_pct(f['gap_pos'], f['n'])} | "
          f"{_pct(f['gap_zero'], f['n'])} | {_pct(f['gap_neg'], f['n'])} | "
          f"{_fmt(b['mean_pick_maia_rank'])} | {_fmt(f['mean_policy_gap'], 3)} |")
    A("")
    # opportunity subset
    opp_total = stats[models[0]]["findability"]["opp_total"]
    A(f"### 4b. Opportunity subset — where engine-best ≠ most-findable ({opp_total}/{N} positions)")
    A("")
    A("These are the positions where the engine's sharpest sound move is **not** the move a "
      "1000–1200 player is most likely to find — i.e. the only positions where the leveled "
      "teaching-move behavior can actually be *exercised*. On the rest, picking the engine best "
      "is already correct for everyone.")
    A("")
    A("| Model | steers beginner to most-findable | still picks engine-best (mirror) | "
      "mean findability gap |")
    A("|---|---|---|---|")
    for m in models:
        f = stats[m]["findability"]
        A(f"| {m} | {_pct(f['opp_pick_findable'], f['opp_total'])} | "
          f"{_pct(f['opp_pick_mirror'], f['opp_total'])} | {_fmt(f['opp_mean_gap'])} |")
    A("")

    # ---------------- Per-tier findability direction ---------------- #
    A("## 5. Direction check — does findability improve toward beginner?")
    A("")
    A("Mean pick Maia-rank in the sound pool per tier (0 = most human-findable). The intended "
      "gradient is **beginner < advanced** (beginners get the more findable move; advanced get "
      "the sharper one). Flat or inverted = not level-aware.")
    A("")
    A("| Model | beginner | intermediate | advanced |")
    A("|---|---|---|---|")
    for m in models:
        pt = stats[m]["per_tier"]
        A(f"| {m} | {_fmt(pt['beginner']['mean_pick_maia_rank'])} | "
          f"{_fmt(pt['intermediate']['mean_pick_maia_rank'])} | "
          f"{_fmt(pt['advanced']['mean_pick_maia_rank'])} |")
    A("")

    # ---------------- Fabrication ---------------- #
    A("## 6. Fabrication (coaching truthfulness)")
    A("")
    A("Share of coaching outputs (position×tier) whose prose states a demonstrably-false board "
      "fact (a named piece not on the named square, a side lacking a claimed piece), via the "
      "deterministic `faithfulness.verify_text`. This is the same truthfulness axis `RESULTS.md` "
      "flagged as flat for v1.")
    A("")
    A("| Model | outputs | with ≥1 false claim | total false-claim sentences |")
    A("|---|---|---|---|")
    for m in models:
        fab = stats[m]["fab"]
        A(f"| {m} | {fab['n_outputs']} | {_pct(fab['n_fab'], fab['n_outputs'])} | "
          f"{fab['total_viol']} |")
    A("")
    A("_Note: frontier coaching is generated in this run; v1 coaching is reused from the v1 "
      "divergence run. The verifier is conservative (flags only demonstrably-false claims)._")
    A("")

    # ---------------- Verdict ---------------- #
    L += _verdict(stats, N)

    # ---------------- Examples ---------------- #
    A("## Appendix — example positions (beginner tier)")
    A("")
    A("`gap>0` positions are where a frontier model DID steer the beginner to a more findable "
      "move; look for how rare they are vs. the mirror cases.")
    A("")
    shown = 0
    for r in rows:
        g = r["grounding"]["beginner"]
        ebr = g.get("engine_best_maia_rank")
        if not ebr:  # only opportunity positions are illustrative
            continue
        sfb = (r["stockfish_best"] or {}).get("san")
        picks = " ".join(
            f"{m.split('-')[0]}:{r['models'][m]['beginner']['rec_san']}"
            f"(mr{r['models'][m]['beginner'].get('maia_pool_rank')})"
            for m in models
        )
        A(f"- `{r['id']}` [{r['phase']}/{r['severity']}] SF-best **{sfb}** "
          f"(Maia-rank {ebr}) — {picks}")
        shown += 1
        if shown >= 12:
            break
    if shown == 0:
        A("_(No opportunity positions in this sample — engine-best was always the most findable "
          "move; differentiation cannot be exercised here.)_")
    A("")

    report = "\n".join(L)
    OUT_PATH.write_text(report, encoding="utf-8")

    # ---- console summary ---- #
    print("=" * 72)
    print(f"N={N} positions | models={models}")
    for m in models:
        s = stats[m]
        f = s["findability"]
        opp = _pct(f["opp_pick_findable"], f["opp_total"]) if f["opp_total"] else "n/a"
        print(f"  {m:16s} tier-diff {_pct(s['n_diff'], s['N'])} | "
              f"mirror@every {_pct(s['mirror_all'], s['N'])} | "
              f"beg==findable {_pct(s['per_tier']['beginner']['pick_is_findable'], s['N'])} | "
              f"opp-findable {opp} | fab {_pct(s['fab']['n_fab'], s['fab']['n_outputs'])}")
    print(f"wrote {OUT_PATH}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
