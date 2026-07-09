#!/usr/bin/env python3
"""Assemble the **v5 (32B)** SFT set: v5-curated coaching + the contrastive moat +
mined gold, then REPORT row counts + gate pass-rates before training.

Goal (from the v5 task): keep v4's move-selection moat (tier-fit / distinct-moves)
while fixing v4's regressions (instructiveness + ~40% raw fabrication). It does this
by applying, at 32B scale, the SAME v5 curation recipe that cleaned the 4B's prose:

* **bulk** — re-derive rows from the Stockfish-verified v3 teacher labels
  (``candidates_v3.jsonl``) through :func:`src.teacher.build_4b_dataset._gather`
  (the exact v5 gate + clean render: lead/artifact stripper, tempo scrub,
  ``select_tier_move`` collapse fix, deterministic principle-in-takeaway gate,
  narrow faithfulness, no engine-speak, ply cap, format), then apply the
  **v5_moat** MIX (down-weight non-differentiating boards; UP-WEIGHT the
  contrastive full-gradient triads + beginner-discriminating rows) so the
  "distinct move per level" moat is preserved/strengthened — the exact thing v4
  was strong at and we must not drop.
* **gold** — fold in the separately-mined DISCRIMINATING multi-tier positions
  (``data/curate/labeled.jsonl``), each cross-family best-of-N and gated with the
  STRONG widened faithfulness checker (``faithfulness_ext``, 0 fabrication). These
  are NEW positions (deduped vs the eval), adding moat without duplication.

The final targets are re-checked against every gate — including the strong
``verify_text_ext`` faithfulness checker — and the pass-rates are reported per
source (bulk / gold) and overall, so the dataset's quality is transparent BEFORE
we spend on the one-shot 32B QLoRA.

CLI::

    python -m scripts.build_v5_32b --recipe v5_moat
    python -m scripts.build_v5_32b --recipe v5_moat --no-gold   # bulk only
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import settings  # noqa: E402
import src.teacher.build_4b_dataset as B  # noqa: E402
from src.orchestrate.data_recipes import RECIPES, _drop_allsame  # noqa: E402
from src.curate.gates import bad_heuristic, has_named_principle  # noqa: E402
from src.engine.faithfulness_ext import verify_text_ext  # noqa: E402
from src.filter.filter import detect_engine_speak, longest_san_run  # noqa: E402

SEED = 3407
GOLD_LABELED = settings.DATA / "curate" / "labeled.jsonl"
GOLD_SNAP = settings.DATA / "curate" / "labeled_v5snap.jsonl"
TRAIN_OUT = settings.DATASET / "train_v5.jsonl"
VALID_OUT = settings.DATASET / "valid_v5.jsonl"
MANIFEST = settings.GENERATED / "v5_32b_manifest.json"


# --------------------------------------------------------------------------- #
# Bulk (v5_moat) — reuse the exact v5 gate/render, apply the recipe MIX
# --------------------------------------------------------------------------- #
def build_bulk(recipe_name: str, ext_filter: bool) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
    recipe = RECIPES[recipe_name]
    fams = B._principle_families()
    rows, reason_hist, picks_by_base, n_cands = B._gather(fams)
    board_class = {b: B._board_class(p) for b, p in picks_by_base.items()}

    # v5 faithfulness lever: drop the bulk rows the STRONG (calibrated, both-board)
    # widened checker flags. Directly targets v4's ~40% inference fabrication by
    # never teaching a relational/move-consequence false board claim. Applied at
    # the UNIQUE-row level (before the moat oversample) so it is cheap + never
    # multiplies a fabrication. Gold is already 100% ext-faithful by its own gate.
    ext_dropped = 0
    if ext_filter:
        kept: List[dict] = []
        for r in rows:
            m = r["_meta"]
            target = r["messages"][-1]["content"]
            if verify_text_ext(target, m["fen"], m.get("rec_uci") or "").violations:
                ext_dropped += 1
                continue
            kept.append(r)
        rows = kept

    by_base: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_base[r["_meta"]["base_id"]].append(r)
    base_ids = sorted(by_base)
    rng = random.Random(SEED)
    rng.shuffle(base_ids)
    n_valid = max(1, int(len(base_ids) * recipe.valid_frac))
    valid_bases = set(base_ids[:n_valid])

    train: List[dict] = []
    valid: List[dict] = []
    drops = Counter()
    for bid in base_ids:
        cls = board_class.get(bid)
        is_valid = bid in valid_bases
        for r in by_base[bid]:
            m = r["_meta"]
            m["source"] = "bulk"
            tier = m["tier"]
            if recipe.drop_collapse_intermediate and cls == "collapse_BA" and tier == "intermediate":
                drops["collapse_BA_intermediate"] += 1
                continue
            if recipe.require_native_principle and m.get("augmented_takeaway"):
                drops["non_native_principle"] += 1
                continue
            if is_valid:
                valid.append(r)
                continue
            if cls == "all_same" and _drop_allsame(bid, recipe.allsame_drop):
                drops["allsame_downweight"] += 1
                continue
            copies = 1
            if cls == "full":
                copies += recipe.full_triad_boost
            if m["discriminating"]:
                copies += recipe.discriminating_boost
            train.extend([r] * copies)

    info = {
        "recipe": recipe.name,
        "recipe_params": {
            "allsame_drop": recipe.allsame_drop,
            "full_triad_boost": recipe.full_triad_boost,
            "discriminating_boost": recipe.discriminating_boost,
            "drop_collapse_intermediate": recipe.drop_collapse_intermediate,
            "valid_frac": recipe.valid_frac,
        },
        "candidates": n_cands,
        "kept_unique_rows": len(rows),
        "ext_faithfulness_filter": ext_filter,
        "ext_dropped_unique_rows": ext_dropped,
        "reject_reasons": dict(reason_hist.most_common()),
        "board_coherence": dict(Counter(v for v in board_class.values() if v)),
        "mix_drops": dict(drops),
        "unique_positions": len(base_ids),
    }
    return train, valid, info


# --------------------------------------------------------------------------- #
# Gold — fold in the mined, cross-family best-of-N, faithfulness_ext-gated rows
# --------------------------------------------------------------------------- #
def _read_jsonl_safe(path: Path) -> List[dict]:
    out: List[dict] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a partial final line (worker still writing)
    return out


def build_gold(snapshot: Path) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
    from src.curate.label import scenario_for, _base_id
    from src.curate.mine import MINED_OUT
    from src.eval.benchmark.prompts import build_grounded_user, load_system_prompt

    winners = _read_jsonl_safe(snapshot)
    if not winners:
        return [], [], {"gold_rows": 0, "note": "no gold labeled yet"}
    mined = {str(m["id"]): m for m in _read_jsonl_safe(MINED_OUT)}
    system_prompt = load_system_prompt()

    seen: set = set()
    by_base: Dict[str, List[dict]] = defaultdict(list)
    for w in winners:
        key = (str(w["id"]), w["tier"])
        if key in seen:
            continue
        seen.add(key)
        m = mined.get(str(w["id"]))
        if m is None:
            continue
        user = build_grounded_user(scenario_for(m, w["tier"]))
        row = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
                {"role": "assistant", "content": w["target"]},
            ],
            "_meta": {
                "cand_id": w["id"], "base_id": _base_id(w["id"]), "tier": w["tier"],
                "fen": m["fen"], "rec_uci": w["recommended_uci"], "rec_san": w["recommended_san"],
                "is_engine_best": w.get("is_engine_best", True),
                "discriminating": (w["tier"] == "beginner" and not w.get("is_engine_best", True)),
                "augmented_takeaway": w.get("augmented", False),
                "winner_family": w.get("family"), "source": "gold",
            },
        }
        by_base[_base_id(w["id"])].append(row)

    base_ids = sorted(by_base)
    rng = random.Random(SEED)
    rng.shuffle(base_ids)
    n_valid = max(1, int(len(base_ids) * 0.06))
    valid_bases = set(base_ids[:n_valid])
    train: List[dict] = []
    valid: List[dict] = []
    for bid in base_ids:
        (valid if bid in valid_bases else train).extend(by_base[bid])

    info = {
        "gold_rows": len(train) + len(valid),
        "gold_positions": len(base_ids),
        "gold_train": len(train), "gold_valid": len(valid),
        "winner_by_family": dict(Counter(r["_meta"]["winner_family"] for r in (train + valid))),
    }
    return train, valid, info


# --------------------------------------------------------------------------- #
# Gate pass-rate audit on the FINAL assembled rows (transparency before training)
# --------------------------------------------------------------------------- #
def audit_gates(rows: List[dict]) -> Dict[str, Any]:
    """Re-run every quality gate on the assembled targets and report pass-rates.

    Includes the STRONG widened faithfulness checker (``verify_text_ext``) so the
    real faithfulness level of the training labels is visible, not just the narrow
    checker used to build the bulk.
    """
    def _bucket() -> Dict[str, int]:
        return {"n": 0, "principle": 0, "faith_narrow": 0, "faith_ext": 0,
                "no_engine_speak": 0, "format": 0, "ply_cap": 0, "no_bad_heuristic": 0}

    from src.engine.faithfulness import verify_text
    agg: Dict[str, Dict[str, int]] = {"overall": _bucket(), "bulk": _bucket(), "gold": _bucket()}
    cache: Dict[tuple, Dict[str, bool]] = {}

    def _checks(target: str, fen: str, rec_uci: str, tier: str) -> Dict[str, bool]:
        ck = (fen, rec_uci, tier, target)
        hit = cache.get(ck)
        if hit is not None:
            return hit
        takeaway = target.split("Takeaway:", 1)[1] if "Takeaway:" in target else ""
        res = {
            "principle": has_named_principle(takeaway),
            "faith_narrow": not verify_text(target, fen).violations,
            "faith_ext": not verify_text_ext(target, fen, rec_uci).violations,
            "no_engine_speak": not detect_engine_speak(target),
            "format": target.startswith("I'd play ") and "Takeaway:" in target,
            "ply_cap": longest_san_run(target) <= settings.TIERS[tier]["ply_cap"],
            "no_bad_heuristic": bad_heuristic(target) is None,
        }
        cache[ck] = res
        return res

    for r in rows:
        m = r["_meta"]
        tier = m["tier"]
        target = r["messages"][-1]["content"]
        fen = m["fen"]
        rec_uci = m.get("rec_uci") or ""
        c = _checks(target, fen, rec_uci, tier)
        principle = c["principle"]; faith_narrow = c["faith_narrow"]; faith_ext = c["faith_ext"]
        no_es = c["no_engine_speak"]; fmt = c["format"]; cap = c["ply_cap"]; no_bad = c["no_bad_heuristic"]
        for key in ("overall", m["source"]):
            b = agg[key]
            b["n"] += 1
            b["principle"] += int(principle)
            b["faith_narrow"] += int(faith_narrow)
            b["faith_ext"] += int(faith_ext)
            b["no_engine_speak"] += int(no_es)
            b["format"] += int(fmt)
            b["ply_cap"] += int(cap)
            b["no_bad_heuristic"] += int(no_bad)

    def _rate(b: Dict[str, int]) -> Dict[str, Any]:
        n = max(1, b["n"])
        return {"n": b["n"], **{k: round(100.0 * b[k] / n, 2) for k in b if k != "n"}}

    return {k: _rate(v) for k, v in agg.items() if v["n"]}


def _write_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _compose(rows: List[dict], label: str) -> Dict[str, Any]:
    return {
        "rows": len(rows),
        "by_tier": dict(Counter(r["_meta"]["tier"] for r in rows)),
        "by_source": dict(Counter(r["_meta"]["source"] for r in rows)),
        "discriminating": sum(1 for r in rows if r["_meta"]["discriminating"]),
        "augmented_takeaway": sum(1 for r in rows if r["_meta"].get("augmented_takeaway")),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", default="v5_moat", choices=sorted(RECIPES))
    ap.add_argument("--no-gold", dest="gold", action="store_false", default=True)
    ap.add_argument("--no-ext-filter", dest="ext_filter", action="store_false", default=True,
                    help="Disable the strong verify_text_ext faithfulness drop on the bulk.")
    ap.add_argument("--snapshot", action="store_true", default=True,
                    help="Freeze the live gold labeled.jsonl to a snapshot first (avoid races).")
    args = ap.parse_args(argv)

    print(f"=== build_v5_32b: recipe={args.recipe} gold={args.gold} ext_filter={args.ext_filter} ===")
    bulk_train, bulk_valid, bulk_info = build_bulk(args.recipe, args.ext_filter)
    print(f"[bulk] recipe={args.recipe} kept_unique={bulk_info['kept_unique_rows']} "
          f"train={len(bulk_train)} valid={len(bulk_valid)}")

    gold_train: List[dict] = []
    gold_valid: List[dict] = []
    gold_info: Dict[str, Any] = {"gold_rows": 0}
    if args.gold:
        snap = GOLD_LABELED
        if args.snapshot and GOLD_LABELED.exists():
            shutil.copyfile(GOLD_LABELED, GOLD_SNAP)
            snap = GOLD_SNAP
        gold_train, gold_valid, gold_info = build_gold(snap)
        print(f"[gold] positions={gold_info.get('gold_positions')} "
              f"train={len(gold_train)} valid={len(gold_valid)}")

    train = bulk_train + gold_train
    valid = bulk_valid + gold_valid
    random.Random(SEED).shuffle(train)

    print("\n[audit] re-running every gate on the assembled TRAIN rows "
          "(incl. strong verify_text_ext)…")
    gate_rates = audit_gates(train)

    _write_jsonl(train, TRAIN_OUT)
    _write_jsonl(valid, VALID_OUT)

    manifest = {
        "name": "chess-coach-v5-32b",
        "recipe": args.recipe,
        "goal": "keep v4 moat (tier-fit/distinct) + fix v4 regressions (instructiveness + faithfulness) "
                "by applying the 4B's v5 clean-render recipe at 32B scale + folding in mined gold",
        "bulk": bulk_info,
        "gold": gold_info,
        "train": _compose(train, "train"),
        "valid": _compose(valid, "valid"),
        "gate_pass_rates_pct": gate_rates,
        "outputs": {"train": str(TRAIN_OUT), "valid": str(VALID_OUT)},
        "seed": SEED,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== v5 (32B) dataset built ===")
    print(json.dumps({"train": manifest["train"], "valid": manifest["valid"],
                      "gate_pass_rates_pct": gate_rates}, indent=2))
    print(f"\nwrote train -> {TRAIN_OUT} ({len(train)} rows)")
    print(f"wrote valid -> {VALID_OUT} ({len(valid)} rows)")
    print(f"wrote manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
