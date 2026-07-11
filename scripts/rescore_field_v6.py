#!/usr/bin/env python3
"""Re-score the CACHED full-field gap803 generations on the CORRECTED v6 labels.

Local, free, NO model calls, NO GPU, NO network. The prior definitive field eval
(``RESULTS_FULL_EVAL_803.md``) scored the 15-model field against the OLD (v4-era)
benchmark labels. The v6 rebuild re-derived those labels on the SAME 803 positions
(``data/benchmark_gap803/scenarios_v6.jsonl``: 45.2% of canonical targets moved,
28.9% of the old "sound" moves removed) — the models' move CHOICES are unchanged,
only their tier-policy SCORES must be recomputed.

Method fidelity
---------------
Each model's recommended move is extracted ONCE from its cached raw ``output`` with
the SAME vendored extractor the shipped v4 headline uses and
``scripts/reproduce_v4.py`` asserts against
(:func:`src.eval.evaluate.extract_recommended_move`). Because extraction depends only
on ``(text, fen, student_uci)`` — all label-independent, and the v6 rebuild changed
0/2409 FENs and 0/2409 student moves — the extracted pick is identical whether it is
then scored against the OLD or the CORRECTED labels, so the ONLY thing that moves the
score is the label correction. This isolates the correction's effect on the
field-wide tier-appropriate-selection moat.

Metric definitions are byte-identical to
``scripts/stage4_rescore_committed.score_committed`` /
``scripts/stage4_eval.score_condition``:
  * tier-policy match  = mean over the 3 tiers of (pick == canonical_uci)
  * move-sound         = pick in the tier's sound pool
  * distinct-per-level = of positions whose canonical beginner!=advanced, the share
                         where the model's beginner!=advanced picks (honest
                         all-opportunities denominator, scoped to model coverage)
  * names-a-move       = extractor returned a legal move  (label-independent)
  * format             = names a move AND closes with "Takeaway:" (label-independent)

Coverage is whatever each cached model has: 12 open/OURS models at 803x3 (2409) and
the 3 frontier references at the 150-position subset x3 (450) — exactly the
``n(det)`` split of the historical table.

Run::

    ~/.venvs/mlx/bin/python scripts/rescore_field_v6.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.eval.evaluate import extract_recommended_move  # noqa: E402

TIERS: Tuple[str, ...] = ("beginner", "intermediate", "advanced")
GAP = _ROOT / "data" / "benchmark_gap803"
GEN_DIR = GAP / "gen"
SCEN_OLD = GAP / "scenarios.jsonl"
SCEN_V6 = GAP / "scenarios_v6.jsonl"
OUT = GAP / "field_v6_rescore.json"

# key -> (display, family), authoritative order from scripts/gap803_report.py.
MODELS: Tuple[Tuple[str, str, str], ...] = (
    ("ours_v3", "OURS-v3 (Qwen3-32B tuned)", "ours"),
    ("ours", "OURS-v2 (Qwen3-1.7B tuned)", "ours"),
    ("base", "BASE (Qwen3-1.7B untuned)", "base"),
    ("gpt", "GPT-5.5", "frontier"),
    ("claude", "Claude Opus 4.8", "frontier"),
    ("gemini", "Gemini 3.1 Pro", "frontier"),
    ("q3_32b", "Qwen3-32B (untuned v3 base)", "open"),
    ("gemma3_27b", "Gemma-3-27B-it", "open"),
    ("q3_next80b", "Qwen3-Next-80B-A3B", "open"),
    ("llama33_70b", "Llama-3.3-70B", "open"),
    ("dsv32", "DeepSeek-V3.2", "open"),
    ("glm5", "GLM-5", "open"),
    ("mistral3", "Mistral-Large-3 (675B)", "open"),
    ("kimi25", "Kimi-K2.5", "open"),
    ("dsr1", "DeepSeek-R1", "open"),
)


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _index(path: Path) -> Dict[str, dict]:
    return {s["id"]: s for s in _load_jsonl(path)}


def _sound_set(scn: dict) -> set:
    """Tier sound pool as a UCI set, tolerant of both schemas (v6 ``sound_pool``
    list-of-dicts; v4-era ``sound_uci`` list)."""
    pool = scn.get("sound_pool")
    if isinstance(pool, list):
        return {p.get("uci") for p in pool if isinstance(p, dict) and p.get("uci")}
    return set(scn.get("sound_uci") or [])


def _extract_all(gen_rows: List[dict], scen: Dict[str, dict]) -> List[Tuple[str, str, Optional[str], str]]:
    """(pos_id, tier, extracted_uci, text) per gen row — extraction is label-free."""
    out: List[Tuple[str, str, Optional[str], str]] = []
    for g in gen_rows:
        sid = g.get("scenario_id") or g.get("id")
        s = scen.get(sid)
        if s is None:
            continue
        text = g.get("output", "") or ""
        _san, uci = extract_recommended_move(text, s["fen"], s["student_move"].get("uci") or "")
        out.append((s["pos_id"], s["tier"], uci, text))
    return out


def _score(picks: List[Tuple[str, str, Optional[str], str]], scen: Dict[str, dict]) -> Dict[str, Any]:
    """Deterministic metrics of the extracted picks against ``scen``'s labels."""
    by_tier: Dict[str, List[int]] = {t: [0, 0] for t in TIERS}
    sound = [0, 0]
    preds: Dict[str, Dict[str, Optional[str]]] = {}
    covered_pos: set = set()
    for pos_id, tier, uci, _text in picks:
        s = scen.get(f"{pos_id}#{tier}")
        if s is None:
            continue
        covered_pos.add(pos_id)
        if tier in by_tier:
            by_tier[tier][1] += 1
            if uci and uci == s.get("canonical_uci"):
                by_tier[tier][0] += 1
        sound[1] += 1
        if uci and uci in _sound_set(s):
            sound[0] += 1
        preds.setdefault(pos_id, {})[tier] = uci

    per_tier = {t: (by_tier[t][0] / by_tier[t][1]) for t in TIERS if by_tier[t][1]}
    tier_policy = mean(per_tier.values()) if per_tier else 0.0

    # canonical beginner/advanced per covered position, honest all-opportunities denom.
    canon_pos: Dict[str, Dict[str, Optional[str]]] = {}
    for pid in covered_pos:
        for t in ("beginner", "advanced"):
            s = scen.get(f"{pid}#{t}")
            if s is not None:
                canon_pos.setdefault(pid, {})[t] = s.get("canonical_uci")
    diff = dist = 0
    for pid, cd in canon_pos.items():
        cb, ca = cd.get("beginner"), cd.get("advanced")
        if not (cb and ca and cb != ca):
            continue
        diff += 1
        mb = preds.get(pid, {}).get("beginner")
        ma = preds.get(pid, {}).get("advanced")
        if mb and ma and mb != ma:
            dist += 1
    return {
        "tier_policy_match": round(tier_policy, 4),
        "per_tier": {t: round(v, 4) for t, v in per_tier.items()},
        "per_tier_counts": {t: by_tier[t] for t in TIERS if by_tier[t][1]},
        "move_sound": round(sound[0] / sound[1], 4) if sound[1] else 0.0,
        "distinct_rate": round(dist / diff, 4) if diff else 0.0,
        "distinct_counts": [dist, diff],
    }


def _named_format(picks: List[Tuple[str, str, Optional[str], str]]) -> Dict[str, Any]:
    """Label-independent name/format rates."""
    named = [0, 0]
    fmt = [0, 0]
    for _pos, _tier, uci, text in picks:
        named[1] += 1
        if uci:
            named[0] += 1
        fmt[1] += 1
        if uci and ("I'd play" in text or "I\u2019d play" in text) and "Takeaway:" in text:
            fmt[0] += 1
    return {
        "named_rate": round(named[0] / named[1], 4) if named[1] else 0.0,
        "format_rate": round(fmt[0] / fmt[1], 4) if fmt[1] else 0.0,
    }


def main() -> int:
    old = _index(SCEN_OLD)
    new = _index(SCEN_V6)
    results: Dict[str, Any] = {}
    for key, display, family in MODELS:
        path = GEN_DIR / f"{key}.jsonl"
        if not path.exists():
            print(f"SKIP {key}: missing {path}")
            continue
        rows = _load_jsonl(path)
        picks = _extract_all(rows, new)  # extraction uses v6 fen == old fen (verified)
        n_pos = len({p for p, _t, _u, _x in picks})
        results[key] = {
            "display": display,
            "family": family,
            "n_scenarios": len(picks),
            "n_positions": n_pos,
            **_named_format(picks),
            "old_labels": _score(picks, old),
            "v6_labels": _score(picks, new),
        }
        results[key]["delta_tier_policy"] = round(
            results[key]["v6_labels"]["tier_policy_match"]
            - results[key]["old_labels"]["tier_policy_match"], 4)

    OUT.write_text(json.dumps({
        "benchmark": "gap803 full field re-scored on corrected v6 labels (free, cached gens)",
        "extractor": "src.eval.evaluate.extract_recommended_move (vendored; reproduce_v4 asserts)",
        "note": "extraction label-free; 0/2409 FEN & student-move changes v4->v6, so old vs "
                "v6 score differs ONLY by the label correction",
        "results": results,
    }, indent=2), encoding="utf-8")

    hdr = (f"{'model':30} {'fam':8} {'n':>5} "
           f"{'old_tier':>8} {'v6_tier':>8} {'Δ':>7} {'v6_sound':>8} {'v6_dist':>8} {'named':>6} {'fmt':>6}")
    print("=== gap803 full field, re-scored on CORRECTED v6 labels (cached gens, free) ===")
    print(hdr)
    for key, _d, _f in MODELS:
        r = results.get(key)
        if not r:
            continue
        o, v = r["old_labels"], r["v6_labels"]
        print(f"{r['display']:30} {r['family']:8} {r['n_scenarios']:>5} "
              f"{o['tier_policy_match']:>8.4f} {v['tier_policy_match']:>8.4f} "
              f"{r['delta_tier_policy']:>+7.4f} {v['move_sound']:>8.4f} {v['distinct_rate']:>8.4f} "
              f"{r['named_rate']:>6.3f} {r['format_rate']:>6.3f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
