"""Build a curated 'Position Library' for the Analysis Room demo.

Samples a diverse, balanced set of real positions from the generated candidate
pool (varied tiers + mistake severities, deduped by FEN), then runs each one
through the *running* coach backend (``/api/coach``) so we cache the **tuned
model's** actual coaching for instant browsing in the UI.

Run (backend must be up on :8000, pointed at the tuned model)::

    ~/.venvs/mlx/bin/python -m src.demo.build_library \
        --candidates data/generated/candidates_v1.jsonl \
        --out web/public/library.json --per-tier 15

The output is a JSON list the API serves at ``/api/library`` and the front end
renders as a scrollable gallery.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]

TIER_ORDER = ["beginner", "intermediate", "advanced"]
# Prefer instructive mistakes first, but keep some sound moves for variety.
SEVERITY_ORDER = ["blunder", "mistake", "inaccuracy", "good", "none", None]


def _phase(fen: str) -> str:
    """Coarse game phase from piece count on the board."""
    board = fen.split(" ", 1)[0]
    pieces = sum(1 for c in board if c.isalpha())
    if pieces >= 26:
        return "opening"
    if pieces >= 12:
        return "middlegame"
    return "endgame"


def _load_records(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ti = d.get("teacher_input") or {}
            fen = ti.get("fen")
            if not fen or fen in seen:
                continue
            sm = ti.get("student_move") or {}
            sev = sm.get("severity")
            out.append(
                {
                    "id": d.get("id"),
                    "tier": d.get("tier") or ti.get("tier"),
                    "fen": fen,
                    "student_san": sm.get("san"),
                    "student_uci": sm.get("uci"),
                    "severity": sev,
                    "cp_loss": sm.get("cp_loss"),
                    "phase": _phase(fen),
                }
            )
            seen.add(fen)
    return out


def _balanced_sample(records: List[Dict[str, Any]], per_tier: int) -> List[Dict[str, Any]]:
    """Round-robin over severities within each tier for maximum variety."""
    by_tier_sev: Dict[str, Dict[Any, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_tier_sev[r["tier"]][r["severity"]].append(r)

    picked: List[Dict[str, Any]] = []
    for tier in TIER_ORDER:
        buckets = by_tier_sev.get(tier, {})
        # Order severity buckets by our preference, skipping empties.
        ordered = [buckets[s] for s in SEVERITY_ORDER if buckets.get(s)]
        ordered += [v for k, v in buckets.items() if k not in SEVERITY_ORDER and v]
        take: List[Dict[str, Any]] = []
        idx = 0
        while len(take) < per_tier and any(ordered):
            b = ordered[idx % len(ordered)]
            if b:
                take.append(b.pop(0))
            ordered = [x for x in ordered if x]
            if not ordered:
                break
            idx += 1
        picked.extend(take[:per_tier])
    return picked


def _coach(api: str, fen: str, tier: str, student_move: Optional[str]) -> Optional[Dict[str, Any]]:
    body = json.dumps({"fen": fen, "tier": tier, "student_move": student_move}).encode()
    req = urllib.request.Request(
        f"{api.rstrip('/')}/api/coach",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001 - skip a bad position, keep going
        print(f"    ! coach failed: {exc}", file=sys.stderr)
        return None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", default="data/generated/candidates_v1.jsonl")
    p.add_argument("--out", default="web/public/library.json")
    p.add_argument("--per-tier", type=int, default=15)
    p.add_argument("--api", default="http://127.0.0.1:8000")
    args = p.parse_args(argv)

    cand_path = (_ROOT / args.candidates) if not Path(args.candidates).is_absolute() else Path(args.candidates)
    out_path = (_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    records = _load_records(cand_path)
    print(f"[lib] loaded {len(records)} unique positions from {cand_path.name}")
    sample = _balanced_sample(records, args.per_tier)
    print(f"[lib] sampled {len(sample)} positions ({args.per_tier}/tier target)")

    library: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, r in enumerate(sample, 1):
        label = f"{r['tier'].capitalize()} · {r['phase']}"
        if r.get("student_san"):
            label += f" · played {r['student_san']}"
        print(f"  [{i:>2}/{len(sample)}] {r['tier'][:3]} {r['phase'][:3]} "
              f"{r.get('student_san') or '(no move)'} … ", end="", flush=True)
        coach = _coach(args.api, r["fen"], r["tier"], r.get("student_san"))
        if not coach:
            print("skip")
            continue
        library.append(
            {
                "id": r["id"],
                "label": label,
                "fen": r["fen"],
                "tier": r["tier"],
                "phase": r["phase"],
                "severity": r["severity"],
                "student_move": r.get("student_san"),
                "coach": coach,
            }
        )
        print(f"ok → {coach.get('recommended_move_san')}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(library, indent=2), encoding="utf-8")
    print(f"[lib] wrote {len(library)} entries → {out_path}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
