#!/usr/bin/env python3
"""Drive the shared benchmark harness for chess-coach-**v2** (no edits to it).

The 2x2x5 benchmark package (``src/eval/benchmark``) is owned by another worker
and already implements exactly what we need: 5 models x 2 conditions
(ungrounded/grounded), a blinded cross-family council ranking on *instructiveness
for the tier*, objective fabrication/soundness metrics, a blind human-label
export, cost tracking, and full resumability.

This driver **reuses** it verbatim, changing only two things at runtime:

1. Point the ``ours`` competitor at ``models/mlx/chess-coach-v2`` (instead of v1).
2. Build the held-out scenario set from the **reserve** (``data/analysis/heldout_v2.json``)
   — positions excluded from BOTH ``train.jsonl`` (v1) and ``train_v2.jsonl`` — so
   no model is evaluated on data it trained on.

Outputs go to ``data/benchmark_v2/`` and ``RESULTS_BENCHMARK_v2.md`` (v2-suffixed;
v1 benchmark artifacts, if any, are untouched).

CLI::

    python scripts/run_benchmark_v2.py scenarios --n 90
    python scripts/run_benchmark_v2.py all --n 90 --concurrency 6
    python scripts/run_benchmark_v2.py status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set the v2 output locations BEFORE importing the benchmark config (its paths are
# resolved from these env vars at import time).
os.environ.setdefault("BENCH_DIR", str(ROOT / "data" / "benchmark_v2"))
os.environ.setdefault("BENCH_REPORT", str(ROOT / "RESULTS_BENCHMARK_v2.md"))

from config import settings  # noqa: E402
from src.eval.benchmark import config as bcfg  # noqa: E402

# Point OURS at a chosen local model (default v2). Overridable via BENCH_OURS_MODEL
# so the SAME driver produces the v1 benchmark (for an apples-to-apples v1->v2
# delta on identical scenarios) without editing the shared harness.
_OURS_PATH = os.environ.get("BENCH_OURS_MODEL", str(settings.MODELS / "mlx" / "chess-coach-v2"))
_OURS_TAG = Path(_OURS_PATH).name  # e.g. "chess-coach-v2"
bcfg.MODELS["ours"] = replace(
    bcfg.MODELS["ours"],
    display=f"OURS ({_OURS_TAG}, 1.7B tuned)",
    ident=_OURS_PATH,
)

from src.eval.benchmark import scenarios as scen_mod  # noqa: E402
from src.eval.benchmark import generate as gen_mod  # noqa: E402
from src.eval.benchmark import objective as obj_mod  # noqa: E402
from src.eval.benchmark import council as coun_mod  # noqa: E402
from src.eval.benchmark import report as rep_mod  # noqa: E402
from src.eval.benchmark.io_utils import append_jsonl, read_jsonl  # noqa: E402
from src.engine import maia_engine  # noqa: E402

log = logging.getLogger("benchmark_v2")

RESERVE_PATH = ROOT / "data" / "analysis" / "heldout_v2.json"
V1_CANDIDATES = settings.GENERATED / "candidates_v1.jsonl"


# --------------------------------------------------------------------------- #
# Held-out scenarios from the reserve (disjoint from train_v1 AND train_v2)
# --------------------------------------------------------------------------- #


def _v1_student_moves() -> Dict[str, Dict[str, str]]:
    """id -> {uci, san} of the student's played move, from candidates_v1."""
    out: Dict[str, Dict[str, str]] = {}
    with V1_CANDIDATES.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            sm = (d.get("teacher_input") or {}).get("student_move") or {}
            if sm.get("uci"):
                out[str(d.get("id"))] = {"uci": sm["uci"], "san": sm.get("san", "")}
    return out


def build_scenarios_v2(n: int, *, movetime: int, tolerance: int, multipv: int,
                       maia_top_k: int) -> int:
    """Compute ground truth for reserve positions -> benchmark scenarios (resumable)."""
    if not RESERVE_PATH.exists():
        raise SystemExit(f"BLOCKED: reserve not found at {RESERVE_PATH}. Run the v2 "
                         "generator plan first (it writes the held-out reserve).")
    reserve = json.loads(RESERVE_PATH.read_text(encoding="utf-8"))
    records = reserve["records"]
    student_moves = _v1_student_moves()

    existing = read_jsonl(bcfg.SCENARIOS_PATH)
    existing_ids = {s["id"] for s in existing}
    # Round-robin over tiers for a balanced benchmark spread.
    by_tier: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_tier[r["tier"]].append(r)
    order: List[Dict[str, Any]] = []
    i = 0
    tiers = ("beginner", "intermediate", "advanced")
    while any(i < len(by_tier[t]) for t in tiers):
        for t in tiers:
            if i < len(by_tier[t]):
                order.append(by_tier[t][i])
        i += 1

    accepted = len(existing_ids)
    evaluated = 0
    for r in order:
        if accepted >= n:
            break
        rid = str(r["id"])
        if rid in existing_ids:
            continue
        sm = student_moves.get(rid)
        if not sm:
            continue
        pos = {
            "id": rid, "fen": r["fen"], "tier": r["tier"],
            "played_move_uci": sm["uci"], "played_move_san": sm["san"],
        }
        evaluated += 1
        try:
            scn = scen_mod.compute_ground_truth(
                pos, movetime_ms=movetime, tolerance_cp=tolerance,
                multipv=multipv, maia_top_k=maia_top_k,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: ground-truth failed (%s)", rid, exc)
            continue
        if scn is None:
            continue
        append_jsonl(bcfg.SCENARIOS_PATH, scn)
        existing_ids.add(rid)
        accepted += 1
        if accepted % 10 == 0:
            log.info("  scenarios: %d/%d", accepted, n)
    log.info("scenarios ready: %d (evaluated %d reserve positions)", accepted, evaluated)
    return accepted


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #


def _load_scn(limit: Optional[int]):
    scns = scen_mod.load_scenarios()
    if not scns:
        raise SystemExit("BLOCKED: no scenarios. Run `scenarios` first.")
    return scns[:limit] if limit else scns


def cmd_scenarios(a: argparse.Namespace) -> int:
    build_scenarios_v2(a.n, movetime=a.movetime, tolerance=a.tolerance,
                       multipv=a.multipv, maia_top_k=a.maia_top_k)
    dist = scen_mod.distribution(scen_mod.load_scenarios())
    for axis, counts in dist.items():
        print(f"  {axis}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


def cmd_generate(a: argparse.Namespace) -> int:
    models = ([m.strip() for m in a.models.split(",") if m.strip()]
              if getattr(a, "models", None) else list(bcfg.MODEL_ORDER))
    gen_mod.run_generation(
        _load_scn(a.limit), models, list(bcfg.CONDITIONS),
        concurrency=a.concurrency, min_interval=a.min_interval,
        timeout=a.timeout, max_retries=a.max_retries,
    )
    return 0


def cmd_objective(a: argparse.Namespace) -> int:
    obj_mod.run_objective(_load_scn(a.limit))
    return 0


def cmd_judge(a: argparse.Namespace) -> int:
    coun_mod.run_council(
        _load_scn(a.limit), list(bcfg.CONDITIONS), list(bcfg.JUDGE_KEYS),
        concurrency=a.concurrency, min_interval=a.min_interval,
        timeout=a.timeout, max_retries=a.max_retries,
    )
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    res = rep_mod.run_report(list(bcfg.CONDITIONS))
    print(f"\nWrote {bcfg.REPORT_MD_PATH}")
    print(f"Blind export: {bcfg.BLIND_LABEL_JSONL} + {bcfg.BLIND_LABEL_HTML}")
    print(f"Total estimated council/gen cost: ${res['cost']['total_cost_usd']:.2f}")
    return 0


def cmd_all(a: argparse.Namespace) -> int:
    cmd_scenarios(a)
    cmd_generate(a)
    cmd_objective(a)
    cmd_judge(a)
    cmd_report(a)
    return 0


def cmd_status(_a: argparse.Namespace) -> int:
    scns = read_jsonl(bcfg.SCENARIOS_PATH)
    gens = read_jsonl(bcfg.GENERATIONS_PATH)
    coun = read_jsonl(bcfg.COUNCIL_PATH)
    n = len(scns)
    print("=== benchmark_v2 status ===")
    print(f"BENCH_DIR: {bcfg.BENCH_DIR}")
    print(f"ours -> {bcfg.MODELS['ours'].ident}")
    print(f"scenarios:   {n}")
    print(f"generations: {len(gens)} / {n * len(bcfg.MODEL_ORDER) * len(bcfg.CONDITIONS)} expected")
    print(f"council:     {len(coun)} / {n * len(bcfg.CONDITIONS) * len(bcfg.JUDGE_KEYS)} expected")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the 2x2x5 benchmark for chess-coach-v2.")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common(sp):
        sp.add_argument("--n", type=int, default=90)
        sp.add_argument("--limit", type=int, default=None)
        sp.add_argument("--concurrency", type=int, default=6)
        sp.add_argument("--min-interval", dest="min_interval", type=float, default=0.05)
        sp.add_argument("--timeout", type=float, default=300.0)
        sp.add_argument("--max-retries", dest="max_retries", type=int, default=4)
        sp.add_argument("--movetime", type=int, default=settings.DEFAULT_MOVETIME_MS)
        sp.add_argument("--tolerance", type=int, default=settings.SOUND_TOLERANCE_CP)
        sp.add_argument("--multipv", type=int, default=settings.MULTIPV)
        sp.add_argument("--maia-top-k", dest="maia_top_k", type=int, default=6)
        sp.add_argument("--models", default=None,
                        help="Comma list of model keys to generate (default: all 5).")

    for name, fn in [("scenarios", cmd_scenarios), ("generate", cmd_generate),
                     ("objective", cmd_objective), ("judge", cmd_judge),
                     ("report", cmd_report), ("all", cmd_all), ("status", cmd_status)]:
        sp = sub.add_parser(name)
        _common(sp)
        sp.set_defaults(func=fn)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    finally:
        try:
            maia_engine.close_all()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
