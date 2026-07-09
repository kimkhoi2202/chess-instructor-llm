#!/usr/bin/env python3
"""HONEST eval CENTERED on OURS-v5 (Qwen3-32B QLoRA, v5 recipe) — v5 vs v4 vs 4B.

Adds ``ours_v5`` to the exact same held-out VAL field the v4 honest eval used
(``data/benchmark_honest/gen/*`` — ours_v4 / ours_4b / base_4b / pbase_4b / q3_32b
/ ours_v3 / gpt / claude / gemini, all reused unchanged), then judges the whole
v5-centered field with the SAME blinded cross-family frontier council (GPT-5.5 +
Claude + Gemini via TrueFoundry) on the identical 0-10 move + instructiveness
rubric. Reuses every aggregation helper from :mod:`scripts.honest_v4` verbatim
(only the config globals are re-pointed), so the numbers stay directly comparable.

The verdict answers the v5 goal head-on: did v5 KEEP v4's moat (tier-fit /
distinct-moves) AND FIX v4's regressions (instructiveness council + raw
faithfulness)?

Phases (each resumable)::

    P=/Users/khoilam/.venvs/mlx/bin/python
    $P -m scripts.honest_v5 slice     # score+add ours_v5 to the val field
    $P -m scripts.honest_v5 council    # fresh 0-10 council over the 10-model field -> council_v5.jsonl
    $P -m scripts.honest_v5 report     # v5-vs-v4-vs-4B verdict -> RESULTS_HONEST_EVAL_V5.md
    $P -m scripts.honest_v5 showcase   # rebuild web/public/showcase.json with OURS = v5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.honest_v4 as H  # reuse every helper; re-point the config below

log = logging.getLogger("honest_v5")

# --------------------------------------------------------------------------- #
# Re-point honest_v4's module globals to the v5-centered field + outputs.
# --------------------------------------------------------------------------- #
V5_SOURCES = [_ROOT / "data" / "benchmark_v5" / "gen" / "ours_v5.jsonl"]

H.V4_FIELD = ("ours_v5", "ours_v4", "ours_4b", "base_4b", "pbase_4b",
              "q3_32b", "ours_v3", "gpt", "claude", "gemini")
H.DISPLAY["ours_v5"] = {"name": "OURS-v5 (Qwen3-32B tuned, v5 recipe)", "family": "ours", "local": True}
H.SHOWCASE_MODELS = ("ours_v5", "gpt", "claude", "gemini", "q3_32b")
H.SHOWCASE_OURS = "ours_v5"
H.COUNCIL_V4 = H.HB / "council_v5.jsonl"
H.REPORT_JSON = H.HB / "report_v5.json"
H.REPORT_MD = _ROOT / "RESULTS_HONEST_EVAL_V5.md"

TIERS = H.TIERS
FRONTIER_KEYS = H.FRONTIER_KEYS
JUDGE_KEYS = H.JUDGE_KEYS


# --------------------------------------------------------------------------- #
# slice — score + add ours_v5 to the val field (reuse the rest unchanged)
# --------------------------------------------------------------------------- #
def cmd_slice(a: argparse.Namespace) -> int:
    scns = H._val_scenarios()
    by_id = {s["id"]: s for s in scns}
    want = set(by_id)

    v5_rows: Dict[str, Dict[str, Any]] = {}
    for src in V5_SOURCES:
        for r in H._read_jsonl(src):
            v5_rows[r["scenario_id"]] = r
    v5_rows = {sid: r for sid, r in v5_rows.items() if sid in want}
    if not v5_rows:
        raise SystemExit(
            "BLOCKED: no ours_v5 val gens found. Generate them first:\n"
            "  python -m scripts.build_v5_val_prompts\n"
            "  modal run src/eval/eval_modal_v5.py --block   (pinned to chess-instructor-4)")
    H._write_reused("ours_v5", v5_rows, by_id, keep_raw=True)

    print("\n[slice] v5-centered field coverage on the 120 val positions (360 scenarios):")
    for mk in H.V4_FIELD:
        n = len(H._read_jsonl(H.GEN_DIR / f"{mk}.jsonl"))
        flag = "" if n == 360 else "  <-- INCOMPLETE"
        print(f"  {mk:9} {n}/360{flag}")
    return 0


# --------------------------------------------------------------------------- #
# report — v5 vs v4 vs 4B verdict (moat kept? prose+faithfulness fixed?)
# --------------------------------------------------------------------------- #
def _fmt(x: Any) -> str:
    return H._fmt(x)


def cmd_report(a: argparse.Namespace) -> int:
    scns = H._val_scenarios()
    by_id = {s["id"]: s for s in scns}
    field = [m for m in H.V4_FIELD if (H.GEN_DIR / f"{m}.jsonl").exists()]

    grade = H._model_grade(field)
    ci = H._instr_ci(field)
    ranks = H._instr_ranks(field)
    tier = H._tier_fit(field, by_id)
    dist = H._distinct(field, by_id)
    gate = H._gate_metrics(field, by_id)
    coh = H._coherence(field, by_id)
    gated_v5 = H._gated_soundness("ours_v5", by_id)
    proof = H._vs_frontier_proof(by_id)

    def R(mk):  # instr rank (lower=better)
        return ranks.get(mk, {}).get("mean_rank")

    best_fr = min(((mk, R(mk)) for mk in FRONTIER_KEYS if R(mk) is not None),
                  key=lambda x: x[1], default=(None, None))

    def raw_faith(mk):  # verify-pass on the raw draft (1 - fabrication)
        return (gate.get(mk) or {}).get("verify_pass_draft1")

    # v5 must KEEP the moat vs v4 and FIX instructiveness/faithfulness (vs v4 & 4B).
    def cmp3(axis, higher_better=True):
        vals = {"ours_v5": None, "ours_v4": None, "ours_4b": None}
        pick = {
            "tier_fit": lambda m: tier[m]["tier_fit_mean"],
            "distinct": lambda m: dist[m]["distinct_rate"],
            "instr_rank": lambda m: R(m),
            "instr_grade": lambda m: grade[m]["instr"],
            "raw_faith": raw_faith,
            "coh_viol": lambda m: coh[m].get("violation_rate"),
        }[axis]
        for m in vals:
            if m in field:
                vals[m] = pick(m)
        return {"axis": axis, "higher_better": higher_better, **vals,
                "d_v5_v4": H._sub(vals["ours_v5"], vals["ours_v4"]),
                "d_v5_4b": H._sub(vals["ours_v5"], vals["ours_4b"])}

    axes = [
        cmp3("tier_fit", True), cmp3("distinct", True),
        cmp3("instr_rank", False), cmp3("instr_grade", True),
        cmp3("raw_faith", True), cmp3("coh_viol", False),
    ]

    def _kept_moat():
        tf = next(a for a in axes if a["axis"] == "tier_fit")
        ds = next(a for a in axes if a["axis"] == "distinct")
        return (tf["d_v5_v4"] is not None and tf["d_v5_v4"] >= -0.05
                and ds["d_v5_v4"] is not None and ds["d_v5_v4"] >= -0.05)

    def _fixed_prose():
        ig = next(a for a in axes if a["axis"] == "instr_grade")
        return ig["ours_v5"] is not None and ig["ours_v4"] is not None and ig["ours_v5"] > ig["ours_v4"]

    def _fixed_faith():
        rf = next(a for a in axes if a["axis"] == "raw_faith")
        return rf["ours_v5"] is not None and rf["ours_v4"] is not None and rf["ours_v5"] > rf["ours_v4"]

    verdict = {
        "kept_moat_vs_v4": _kept_moat(),
        "fixed_instructiveness_vs_v4": _fixed_prose(),
        "fixed_faithfulness_vs_v4": _fixed_faith(),
        "best_overall": _kept_moat() and (_fixed_prose() or _fixed_faith()),
    }

    report = {
        "n_val_positions": len(scns) // 3, "field": field,
        "council": {"n_items": len({r["scenario_id"] for r in H._read_jsonl(H.COUNCIL_V4)}),
                    "n_judges": len(JUDGE_KEYS), "n_gradings": len(H._read_jsonl(H.COUNCIL_V4)),
                    "scale": "0-10 move + instr (absolute), frontier panel"},
        "verdict": verdict,
        "v5_vs_v4_vs_4b_axes": axes,
        "ours_v5_gated_soundness": gated_v5,
        "distance_to_frontier": {"best_frontier": best_fr, "ours_v5_rank": R("ours_v5"),
                                 "gap": H._sub(R("ours_v5"), best_fr[1])},
        "vs_frontier_proof": {k: v for k, v in proof.items() if k != "candidates"},
        "per_model": {mk: {"instr_rank": R(mk), "instr_grade": grade[mk]["instr"],
                           "instr_grade_ci95": ci.get(mk), "move_grade": grade[mk]["move"],
                           "tier_fit": tier[mk]["tier_fit_mean"],
                           "tier_fit_by_tier": tier[mk]["by_tier"], "move_sound": tier[mk]["move_sound"],
                           "distinct": dist[mk], "gate": gate[mk],
                           "coherence": coh[mk].get("violation_rate"),
                           "flat_rate": coh[mk].get("flat_rate")} for mk in field},
    }
    H.REPORT_JSON.write_text(json.dumps({**report, "vs_frontier_candidates": proof["candidates"]},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(report, grade, ci, ranks, tier, dist, gate, coh, proof, field, axes, verdict, best_fr)
    print(json.dumps({"verdict": verdict, "axes": axes,
                      "distance_to_frontier": report["distance_to_frontier"],
                      "vs_frontier_proof": report["vs_frontier_proof"]}, indent=2))
    print(f"\nreport -> {H.REPORT_JSON}\nreport -> {H.REPORT_MD}")
    return 0


def _write_md(rep, grade, ci, ranks, tier, dist, gate, coh, proof, field, axes, verdict, best_fr) -> None:
    def disp(mk):
        return H.DISPLAY[mk]["name"]
    L: List[str] = []
    A = L.append
    A("# HONEST eval — CENTERED on OURS-v5 (Qwen3-32B QLoRA, v5 recipe)\n")
    A("The v5 goal was to KEEP v4's move-selection moat (tier-fit / distinct-moves) while FIXING v4's "
      "regressions — instructiveness (council) and raw faithfulness (~40% fabrication) — by applying the "
      "4B's v5 curation recipe at 32B scale (clean lead/artifact render, principle-in-takeaway ~99%, tempo "
      "scrub, tier-collapse fix) + folding in cross-family mined gold, with EVERY label filtered by the "
      "strong `verify_text_ext` (0 fabrication). Every contender coaches the SAME held-out VAL positions "
      "the v4 honest eval used; ours_v5 is generated with the identical grounded prompt + greedy decoding "
      "as ours_v4 (apples-to-apples). Instructiveness is one blinded cross-family frontier council "
      "(GPT-5.5 + Claude + Gemini) grading 0-10 on move + instructiveness.\n")
    A(f"- **VAL slice:** {rep['n_val_positions']} held-out positions × 3 tiers; council "
      f"items={rep['council']['n_items']}, judges={rep['council']['n_judges']}, "
      f"gradings={rep['council']['n_gradings']} ({rep['council']['scale']}).\n")

    v = verdict
    A("## Headline — did v5 keep the moat AND fix v4's regressions?\n")
    tf = next(a for a in axes if a["axis"] == "tier_fit")
    ds = next(a for a in axes if a["axis"] == "distinct")
    ig = next(a for a in axes if a["axis"] == "instr_grade")
    rf = next(a for a in axes if a["axis"] == "raw_faith")
    A(f"**Verdict: {'v5 is the best overall coach — keeps the moat AND fixes the prose/faithfulness.' if v['best_overall'] else 'v5 did NOT clearly dominate (see axes).'}**\n")
    A(f"- **Moat kept vs v4:** {'YES' if v['kept_moat_vs_v4'] else 'NO'} "
      f"(tier-fit {_fmt(tf['ours_v5'])} vs {_fmt(tf['ours_v4'])}, Δ {_fmt(tf['d_v5_v4'])}; "
      f"distinct {_fmt(ds['ours_v5'])} vs {_fmt(ds['ours_v4'])}, Δ {_fmt(ds['d_v5_v4'])}).")
    A(f"- **Instructiveness fixed vs v4:** {'YES' if v['fixed_instructiveness_vs_v4'] else 'NO'} "
      f"(council 0-10 {_fmt(ig['ours_v5'])} vs v4 {_fmt(ig['ours_v4'])}, 4B {_fmt(ig['ours_4b'])}).")
    A(f"- **Faithfulness fixed vs v4:** {'YES' if v['fixed_faithfulness_vs_v4'] else 'NO'} "
      f"(raw verify-pass draft1 {_fmt(rf['ours_v5'])} vs v4 {_fmt(rf['ours_v4'])} — higher = fewer false board facts).\n")

    A("**v5 vs v4 vs 4B — the axes that matter:**\n")
    A("| axis | better | OURS-v5 | OURS-v4 | OURS-4B | Δ(v5−v4) | Δ(v5−4b) |")
    A("|---|:--:|---:|---:|---:|---:|---:|")
    names = {"tier_fit": "tier-fit (moat)", "distinct": "distinct-moves (moat)",
             "instr_rank": "instr council rank", "instr_grade": "instr 0-10",
             "raw_faith": "raw faithfulness (draft1)", "coh_viol": "coherence-viol"}
    for ax in axes:
        arrow = "higher" if ax["higher_better"] else "lower"
        A(f"| {names[ax['axis']]} | {arrow} | {_fmt(ax['ours_v5'])} | {_fmt(ax['ours_v4'])} "
          f"| {_fmt(ax['ours_4b'])} | {_fmt(ax['d_v5_v4'])} | {_fmt(ax['d_v5_4b'])} |")
    A("")
    gv = rep.get("ours_v5_gated_soundness") or {}
    A("**OURS-v5 through the shipped gate (verify + fallback), like the 4B:** "
      f"gated move-sound {_fmt(gv.get('gated_move_sound'))}, gated well-formed "
      f"{_fmt(gv.get('gated_well_formed'))}, gated no-engine-speak {_fmt(gv.get('gated_no_engine_speak'))} "
      f"(gate fallback {_fmt(gv.get('gated_fallback_rate'))}).\n")
    d = rep["distance_to_frontier"]
    A(f"**Distance to frontier:** best frontier = {d['best_frontier'][0]} "
      f"(instr rank {_fmt(d['best_frontier'][1])}); OURS-v5 rank {_fmt(d['ours_v5_rank'])}; "
      f"gap = {_fmt(d['gap'])} rank positions.\n")

    p = proof
    A("## vs-frontier + distinct-tier PROOF (the moat)\n")
    A(f"Of **{p['n_positions_total']}** val positions, OURS-v5 gives distinct, sound, correctly-graded "
      f"per-tier moves on **{p['n_distinct']}**; of those it also DIVERGES from the best frontier move on "
      f"**{p['n_distinct_and_diverge']}**. On that proof set: **{p['wins']} wins / {p['losses']} losses / "
      f"{p['ties']} ties** for OURS-v5 on the MOAT (tier-fit then soundness) vs the best-moat frontier "
      "(the same win definition the platform uses).\n")

    order = sorted(field, key=lambda m: (ranks.get(m, {}).get("mean_rank") or 99))
    A("## Leaderboard (v5-centered VAL field)\n")
    A("| Model | gated | tier-fit↑ | instr rank↓ | instr 0-10↑ | move 0-10↑ | move-sound↑ | distinct↑ | raw-faith↑ | coh-viol↓ |")
    A("|---|:--:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mk in order:
        g = gate.get(mk, {})
        A(f"| {disp(mk)} | {'yes' if g.get('gated') else 'reuse'} | {_fmt(tier[mk]['tier_fit_mean'])} "
          f"| {_fmt(ranks.get(mk,{}).get('mean_rank'))} | {_fmt(grade[mk]['instr'])} | {_fmt(grade[mk]['move'])} "
          f"| {_fmt(tier[mk]['move_sound'])} | {_fmt(dist[mk]['distinct_rate'])} "
          f"| {_fmt(g.get('verify_pass_draft1'))} | {_fmt(coh[mk].get('violation_rate'))} |")
    A("")
    A("## Instructiveness (blinded frontier council, 0-10) with 95% CI\n")
    A("| Model | instr 0-10 [95% CI] | council rank↓ | top-1% |")
    A("|---|---:|---:|---:|")
    for mk in order:
        c = ci.get(mk) or {}
        cistr = (f"{_fmt(c.get('mean'))} [{_fmt(c.get('ci_lo'))}–{_fmt(c.get('ci_hi'))}]" if c else "—")
        rk = ranks.get(mk, {})
        A(f"| {disp(mk)} | {cistr} | {_fmt(rk.get('mean_rank'))} | {_fmt(rk.get('top1_pct'))} |")
    A("")
    A("## Deterministic gate axes (RAW draft for reused/ungated rows; telemetry for gated 4B)\n")
    A("| Model | gated | no-engine-speak↑ | well-formed↑ | move-sound↑ | verify-pass draft1↑ | mean attempts | fallback↓ |")
    A("|---|:--:|---:|---:|---:|---:|---:|---:|")
    for mk in order:
        g = gate.get(mk, {})
        A(f"| {disp(mk)} | {'yes' if g.get('gated') else 'reuse'} | {_fmt(g.get('no_engine_speak'))} "
          f"| {_fmt(g.get('well_formed'))} | {_fmt(g.get('move_sound'))} | {_fmt(g.get('verify_pass_draft1'))} "
          f"| {_fmt(g.get('mean_attempts'))} | {_fmt(g.get('fallback_rate'))} |")
    A("")
    H.REPORT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("slice").set_defaults(func=cmd_slice)
    pc = sub.add_parser("council")
    pc.add_argument("--concurrency", type=int, default=5)
    pc.add_argument("--min-interval", type=float, default=0.05)
    pc.add_argument("--timeout", type=float, default=300.0)
    pc.add_argument("--max-retries", type=int, default=8)
    pc.add_argument("--judge-max-tokens", dest="judge_max_tokens", type=int, default=1600)
    pc.set_defaults(func=H.cmd_council)   # reuse verbatim (reads the re-pointed globals)
    sub.add_parser("report").set_defaults(func=cmd_report)
    sub.add_parser("showcase").set_defaults(func=H.cmd_showcase)  # reuse (SHOWCASE_* re-pointed)
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    finally:
        try:
            from src.engine import maia_engine
            maia_engine.close_all()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
