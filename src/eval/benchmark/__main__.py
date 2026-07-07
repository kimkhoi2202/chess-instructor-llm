"""CLI for the 2x2x5 coaching benchmark.

Run phases independently (each is resumable) or all at once::

    python -m src.eval.benchmark scenarios --n 100
    python -m src.eval.benchmark generate            # 5 models x 2 conditions
    python -m src.eval.benchmark objective
    python -m src.eval.benchmark judge               # blinded council
    python -m src.eval.benchmark report              # tables + blind export
    python -m src.eval.benchmark all --n 100         # everything, in order
    python -m src.eval.benchmark status              # progress snapshot

Do NOT point this at the running web platform; it does all generation itself and
touches nothing on ports 8000/3000.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Sequence

from config import settings
from src.engine import maia_engine

from . import config as bcfg
from .io_utils import read_jsonl

log = logging.getLogger("benchmark")


def _csv(s: Optional[str], default: Sequence[str]) -> List[str]:
    if not s:
        return list(default)
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_scenarios(limit: Optional[int]):
    from . import scenarios as scen_mod

    scns = scen_mod.load_scenarios()
    if not scns:
        raise SystemExit(
            "BLOCKED: no scenarios found. Run `python -m src.eval.benchmark scenarios` first."
        )
    if limit:
        scns = scns[:limit]
    return scns


# --------------------------------------------------------------------------- #
# Sub-commands
# --------------------------------------------------------------------------- #


def cmd_scenarios(args: argparse.Namespace) -> int:
    from . import scenarios as scen_mod

    scns = scen_mod.build_scenarios(
        positions_path=(settings.POSITIONS / args.positions)
        if not args.positions.startswith("/")
        else __import__("pathlib").Path(args.positions),
        n_target=args.n,
        per_cell_cap=args.per_cell_cap,
        movetime_ms=args.movetime,
        tolerance_cp=args.tolerance,
        multipv=args.multipv,
        maia_top_k=args.maia_top_k,
        max_eval=args.max_eval,
    )
    dist = scen_mod.distribution(scns)
    print(f"\nscenarios: {len(scns)}")
    for axis, counts in dist.items():
        print(f"  {axis}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .generate import run_generation

    scns = _load_scenarios(args.limit)
    models = _csv(args.models, bcfg.MODEL_ORDER)
    conds = _csv(args.conditions, bcfg.CONDITIONS)
    run_generation(
        scns, models, conds,
        concurrency=args.concurrency, min_interval=args.min_interval,
        timeout=args.timeout, max_retries=args.max_retries,
    )
    return 0


def cmd_objective(args: argparse.Namespace) -> int:
    from .objective import run_objective

    scns = _load_scenarios(args.limit)
    run_objective(scns)
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    from .council import run_council

    scns = _load_scenarios(args.limit)
    conds = _csv(args.conditions, bcfg.CONDITIONS)
    judges = _csv(args.judges, bcfg.JUDGE_KEYS)
    run_council(
        scns, conds, judges,
        concurrency=args.concurrency, min_interval=args.min_interval,
        timeout=args.timeout, max_retries=args.max_retries,
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import run_report

    conds = _csv(args.conditions, bcfg.CONDITIONS)
    res = run_report(conds)
    print(f"\nWrote {bcfg.REPORT_MD_PATH}")
    print(f"Wrote {bcfg.RESULTS_JSON_PATH}")
    print(f"Blind export: {bcfg.BLIND_LABEL_JSONL} ({res['meta'].get('blind_items', 0)} items)")
    print(f"Total estimated cost: ${res['cost']['total_cost_usd']:.2f}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    cmd_scenarios(args)
    cmd_generate(args)
    cmd_objective(args)
    cmd_judge(args)
    cmd_report(args)
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    scns = read_jsonl(bcfg.SCENARIOS_PATH)
    gens = read_jsonl(bcfg.GENERATIONS_PATH)
    objs = read_jsonl(bcfg.OBJECTIVE_PATH)
    coun = read_jsonl(bcfg.COUNCIL_PATH)
    n = len(scns)
    exp_gen = n * len(bcfg.MODEL_ORDER) * len(bcfg.CONDITIONS)
    exp_coun = n * len(bcfg.CONDITIONS) * len(bcfg.JUDGE_KEYS)
    print("=== benchmark status ===")
    print(f"scenarios:   {n}")
    print(f"generations: {len(gens)} / {exp_gen} expected")
    print(f"objective:   {len(objs)} / {exp_gen} expected")
    print(f"council:     {len(coun)} / {exp_coun} expected")
    if gens:
        by_model = {}
        for g in gens:
            by_model[g["model"]] = by_model.get(g["model"], 0) + 1
        print("  gen by model: " + ", ".join(f"{k}={v}" for k, v in sorted(by_model.items())))
    if coun:
        by_judge = {}
        for c in coun:
            by_judge[c["judge"]] = by_judge.get(c["judge"], 0) + 1
        print("  council by judge: " + ", ".join(f"{k}={v}" for k, v in sorted(by_judge.items())))
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def _add_common_gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--concurrency", type=int, default=6, help="Parallel frontier calls.")
    p.add_argument("--min-interval", dest="min_interval", type=float, default=0.05,
                   help="Min seconds between frontier request starts.")
    p.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout (s).")
    p.add_argument("--max-retries", dest="max_retries", type=int, default=4,
                   help="Retries per frontier call on transient errors.")
    p.add_argument("--limit", type=int, default=None, help="Use only first N scenarios (smoke).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.eval.benchmark",
                                description="2x2x5 chess-coaching benchmark.")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("scenarios", help="Build the held-out scenario set.")
    ps.add_argument("--positions", default="positions_v1.jsonl",
                    help="Positions file (name under data/positions/ or an absolute path).")
    ps.add_argument("--n", type=int, default=100, help="Target scenario count.")
    ps.add_argument("--per-cell-cap", dest="per_cell_cap", type=int, default=6,
                    help="Max scenarios per tier×phase×severity cell in the balanced pass.")
    ps.add_argument("--movetime", type=int, default=settings.DEFAULT_MOVETIME_MS)
    ps.add_argument("--tolerance", type=int, default=settings.SOUND_TOLERANCE_CP)
    ps.add_argument("--multipv", type=int, default=settings.MULTIPV)
    ps.add_argument("--maia-top-k", dest="maia_top_k", type=int, default=6)
    ps.add_argument("--max-eval", dest="max_eval", type=int, default=2500,
                    help="Cap on positions evaluated with Stockfish while sampling.")
    ps.set_defaults(func=cmd_scenarios)

    pg = sub.add_parser("generate", help="Generate coaching for all model×condition units.")
    pg.add_argument("--models", default=None, help="Comma list (default: all 5).")
    pg.add_argument("--conditions", default=None, help="Comma list (default: both).")
    _add_common_gen_args(pg)
    pg.set_defaults(func=cmd_generate)

    po = sub.add_parser("objective", help="Score objective metrics for all generations.")
    po.add_argument("--limit", type=int, default=None)
    po.set_defaults(func=cmd_objective)

    pj = sub.add_parser("judge", help="Run the blinded cross-family council.")
    pj.add_argument("--judges", default=None, help="Comma list (default: gpt,claude,gemini).")
    pj.add_argument("--conditions", default=None, help="Comma list (default: both).")
    _add_common_gen_args(pj)
    pj.set_defaults(func=cmd_judge)

    pr = sub.add_parser("report", help="Aggregate + write RESULTS_BENCHMARK.md + blind export.")
    pr.add_argument("--conditions", default=None)
    pr.set_defaults(func=cmd_report)

    pa = sub.add_parser("all", help="scenarios → generate → objective → judge → report.")
    pa.add_argument("--positions", default="positions_v1.jsonl")
    pa.add_argument("--n", type=int, default=100)
    pa.add_argument("--per-cell-cap", dest="per_cell_cap", type=int, default=6)
    pa.add_argument("--movetime", type=int, default=settings.DEFAULT_MOVETIME_MS)
    pa.add_argument("--tolerance", type=int, default=settings.SOUND_TOLERANCE_CP)
    pa.add_argument("--multipv", type=int, default=settings.MULTIPV)
    pa.add_argument("--maia-top-k", dest="maia_top_k", type=int, default=6)
    pa.add_argument("--max-eval", dest="max_eval", type=int, default=2500)
    pa.add_argument("--models", default=None)
    pa.add_argument("--judges", default=None)
    pa.add_argument("--conditions", default=None)
    _add_common_gen_args(pa)
    pa.set_defaults(func=cmd_all)

    pst = sub.add_parser("status", help="Print progress snapshot.")
    pst.set_defaults(func=cmd_status)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
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
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


if __name__ == "__main__":
    raise SystemExit(main())
