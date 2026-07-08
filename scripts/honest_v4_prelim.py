#!/usr/bin/env python3
"""PRELIMINARY 32B(v4)-vs-4B comparison on the v4 gens completed SO FAR.

The full v4 eval was paused (chess-instructor billing headroom) with only part of
the 120-position VAL slice generated for OURS-v4. This computes the FREE,
deterministic axes (no council spend) on exactly the positions where OURS-v4 has
all three tiers, comparing to the fully-gated OURS-4B (iter1) and giving a
deterministic vs-frontier divergence/tier-fit signal. Instructiveness (blinded
council) is deferred to the full resume.

    ~/.venvs/mlx/bin/python -m scripts.honest_v4_prelim
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chess  # noqa: E402
from src.eval.benchmark.objective import score_one  # noqa: E402
from src.eval.evaluate import find_engine_speak  # noqa: E402

HB = _ROOT / "data" / "benchmark_honest"
GEN = HB / "gen"
VAL_IDS = HB / "val_ids.txt"
GAP_SCN = _ROOT / "data" / "benchmark_gap803" / "scenarios.jsonl"
V4_MERGED = _ROOT / "data" / "benchmark_v4" / "gen" / "ours_v4_val_merged.jsonl"
OUT_MD = _ROOT / "RESULTS_HONEST_EVAL_V4_PRELIM.md"
TIERS = ("beginner", "intermediate", "advanced")
FRONTIER = ("gpt", "claude", "gemini")


def _read(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _outputs(path: Path) -> Dict[str, str]:
    return {r["scenario_id"]: r.get("output", "") for r in _read(path)}


def main() -> int:
    keep = set(VAL_IDS.read_text().split())
    scns = [s for s in _read(GAP_SCN) if s.get("pos_id") in keep]
    by_id = {s["id"]: s for s in scns}

    v4 = _outputs(V4_MERGED)
    # complete-3-tier v4 positions
    seen: Dict[str, set] = defaultdict(set)
    for sid in v4:
        s = by_id.get(sid)
        if s:
            seen[s["pos_id"]].add(s["tier"])
    C = sorted(p for p, ts in seen.items() if len(ts) == 3)

    models = {
        "ours_v4": v4,
        "ours_4b": _outputs(GEN / "ours_4b.jsonl"),
        "base_4b": _outputs(GEN / "base_4b.jsonl"),
        "gpt": _outputs(GEN / "gpt.jsonl"),
        "claude": _outputs(GEN / "claude.jsonl"),
        "gemini": _outputs(GEN / "gemini.jsonl"),
    }

    # objective + rec per (model, pos, tier), restricted to C
    obj: Dict[str, Dict[str, Dict[str, Any]]] = {m: {} for m in models}
    for m, outs in models.items():
        for pid in C:
            for t in TIERS:
                sid = f"{pid}#{t}"
                scn = by_id.get(sid)
                if scn and sid in outs:
                    obj[m][sid] = score_one(scn, outs[sid])

    def rec(m, pid, t):
        return (obj[m].get(f"{pid}#{t}") or {}).get("rec_uci")

    def canon(pid, t):
        return (by_id.get(f"{pid}#{t}") or {}).get("canonical_uci")

    def det(m) -> Dict[str, Any]:
        tf = {t: [0, 0] for t in TIERS}
        snd = es = wf = 0
        n = 0
        for pid in C:
            for t in TIERS:
                o = obj[m].get(f"{pid}#{t}")
                if not o:
                    continue
                n += 1
                tf[t][1] += 1
                if o["rec_uci"] and o["rec_uci"] == canon(pid, t):
                    tf[t][0] += 1
                if o["move_sound"]:
                    snd += 1
                if o["no_engine_speak"]:
                    es += 1
                if o["rec_uci"]:
                    wf += 1
        vals = [tf[t][0] / tf[t][1] for t in TIERS if tf[t][1]]
        # distinct + flat on differentiating positions (canonical b!=a)
        dn = dd = flat = 0
        for pid in C:
            cb, ca = canon(pid, "beginner"), canon(pid, "advanced")
            mb, ma, mi = rec(m, pid, "beginner"), rec(m, pid, "advanced"), rec(m, pid, "intermediate")
            if mb and ma and mi and mb == ma == mi:
                flat += 1
            if cb and ca and cb != ca and mb and ma:
                dn += 1
                if mb != ma:
                    dd += 1
        return {
            "tier_fit_by_tier": {t: round(tf[t][0] / tf[t][1], 4) if tf[t][1] else None for t in TIERS},
            "tier_fit_mean": round(mean(vals), 4) if vals else None,
            "move_sound": round(snd / n, 4) if n else None,
            "no_engine_speak": round(es / n, 4) if n else None,
            "well_formed": round(wf / n, 4) if n else None,
            "distinct_rate": round(dd / dn, 4) if dn else None, "differentiating_n": dn,
            "flat_rate": round(flat / len(C), 4) if C else None,
        }

    dv4, d4b, dbase = det("ours_v4"), det("ours_4b"), det("base_4b")

    # ---- vs-frontier deterministic signal on C -------------------------- #
    def pos_obj_quality(m, pid) -> Tuple[int, int]:
        tfc = sum(1 for t in TIERS if rec(m, pid, t) and rec(m, pid, t) == canon(pid, t))
        sc = sum(1 for t in TIERS if (obj[m].get(f"{pid}#{t}") or {}).get("move_sound"))
        return tfc, sc

    wins = losses = ties = 0
    ndistinct = ndiv = 0
    examples = []
    for pid in C:
        mb, ma, mi = rec("ours_v4", pid, "beginner"), rec("ours_v4", pid, "advanced"), rec("ours_v4", pid, "intermediate")
        if not (mb and ma and mi):
            continue
        sound_all = all((obj["ours_v4"].get(f"{pid}#{t}") or {}).get("move_sound") for t in TIERS)
        distinct = len({mb, mi, ma}) >= 2 and mb != ma
        polB = (by_id.get(f"{pid}#beginner") or {}).get("pool_policy", {}) or {}
        grad_ok = polB.get(mb, 0.0) >= polB.get(ma, 0.0)
        if not (distinct and sound_all and grad_ok):
            continue
        ndistinct += 1
        # best frontier by objective (tier-fit then sound)
        bf, bq = None, (-1, -1)
        for fk in FRONTIER:
            q = pos_obj_quality(fk, pid)
            if q > bq:
                bq, bf = q, fk
        if not bf:
            continue
        diverge = [t for t in TIERS if rec("ours_v4", pid, t) and rec(bf, pid, t)
                   and rec("ours_v4", pid, t) != rec(bf, pid, t)]
        if not diverge:
            continue
        ndiv += 1
        oq = pos_obj_quality("ours_v4", pid)
        if oq > bq:
            wins += 1
            v = "win"
        elif bq > oq:
            losses += 1
            v = "loss"
        else:
            ties += 1
            v = "tie"
        if len(examples) < 12:
            examples.append({"pos_id": pid, "best_frontier": bf, "verdict": v,
                             "ours_tierfit_sound": oq, "frontier_tierfit_sound": list(bq),
                             "diverge_tiers": diverge})

    def sub(a, b):
        return None if a is None or b is None else round(a - b, 4)

    reg = {
        "tier_fit_mean": (dv4["tier_fit_mean"], d4b["tier_fit_mean"], sub(dv4["tier_fit_mean"], d4b["tier_fit_mean"]), "higher"),
        "distinct_rate": (dv4["distinct_rate"], d4b["distinct_rate"], sub(dv4["distinct_rate"], d4b["distinct_rate"]), "higher"),
        "move_sound": (dv4["move_sound"], d4b["move_sound"], sub(dv4["move_sound"], d4b["move_sound"]), "higher"),
        "no_engine_speak": (dv4["no_engine_speak"], d4b["no_engine_speak"], sub(dv4["no_engine_speak"], d4b["no_engine_speak"]), "higher"),
        "well_formed": (dv4["well_formed"], d4b["well_formed"], sub(dv4["well_formed"], d4b["well_formed"]), "higher"),
        "flat_rate": (dv4["flat_rate"], d4b["flat_rate"], sub(dv4["flat_rate"], d4b["flat_rate"]), "lower"),
    }

    out = {
        "n_complete_v4_positions": len(C), "n_val_positions_target": len(keep),
        "note": "PRELIMINARY — deterministic axes only (no council spend); OURS-v4 measured on "
                "raw drafts, OURS-4B fully gated. Instructiveness council deferred to full resume.",
        "det": {"ours_v4": dv4, "ours_4b": d4b, "base_4b": dbase},
        "regression_v4_vs_4b_deterministic": reg,
        "vs_frontier_signal": {"n_distinct": ndistinct, "n_distinct_and_diverge": ndiv,
                               "wins": wins, "losses": losses, "ties": ties,
                               "criterion": "objective tier-fit+soundness vs best frontier; distinct=sound gradient b!=a"},
        "vs_frontier_examples": examples,
    }
    print(json.dumps(out, indent=2))

    # markdown
    def f(x):
        return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
    L = ["# PRELIMINARY — OURS-v4 (32B) vs OURS-4B, deterministic axes\n",
         f"> Partial run: **{len(C)}/{len(keep)}** VAL positions have all three OURS-v4 tiers "
         "(the 32B eval was paused to preserve chess-instructor credit). These are the FREE "
         "deterministic axes only — no council spend. OURS-v4 = RAW drafts (ungated); OURS-4B = "
         "fully gated shipped pipeline. Instructiveness (blinded frontier council) + the final "
         "showcase come with the full resume on the new workspace.\n",
         "## Deterministic regression: did the 32B (v4) regress vs the 4B?\n",
         "| axis | OURS-v4 (32B) | OURS-4B | Δ (v4−4b) | better |",
         "|---|---:|---:|---:|:--:|"]
    for k, (a, b, d, dirn) in reg.items():
        L.append(f"| {k} | {f(a)} | {f(b)} | {f(d)} | {dirn} |")
    L += ["",
          f"- tier-fit by tier — OURS-v4: {dv4['tier_fit_by_tier']}; OURS-4B: {d4b['tier_fit_by_tier']}.",
          f"- untuned Qwen3-4B base (`base_4b`) tier-fit {f(dbase['tier_fit_mean'])}, distinct {f(dbase['distinct_rate'])} (reference).",
          "",
          "## vs-frontier + distinct-tier signal (deterministic)\n",
          f"Of {len(C)} complete positions, OURS-v4 gives distinct, sound, correctly-graded per-tier "
          f"moves on **{ndistinct}**; also diverges from the best frontier move on **{ndiv}**. On that "
          f"set (objective tier-fit+soundness vs best frontier): **{wins} wins / {losses} losses / "
          f"{ties} ties**.\n"]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"\nwrote -> {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
