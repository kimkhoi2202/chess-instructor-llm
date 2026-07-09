#!/usr/bin/env python3
"""Build ``prompts_v5_val.jsonl`` — the 120 held-out VAL positions × 3 tiers only.

Same render as ``gap803_prompts_v4`` (the SAME system + grounded
``build_user_prompt`` prompt every benchmark model gets), but restricted to the
``data/benchmark_honest/val_ids.txt`` slice, so the v5 eval-gen generates ONLY the
360 val scenarios (cheap: ~$2-3 on an A100) that ``scripts/honest_v5.py`` compares
against the already-generated v4 / 4B / frontier val gens.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("BENCH_DIR", str(_ROOT / "data" / "benchmark_gap803"))

from src.eval.benchmark.prompts import build_user_prompt, load_system_prompt  # noqa: E402

SCN = Path(os.environ["BENCH_DIR"]) / "scenarios.jsonl"
VAL_IDS = _ROOT / "data" / "benchmark_honest" / "val_ids.txt"
OUT_DIR = _ROOT / "data" / "benchmark_v5"
OUT = OUT_DIR / "prompts_v5_val.jsonl"


def main() -> int:
    if not SCN.exists():
        raise SystemExit(f"missing {SCN}")
    keep = set(VAL_IDS.read_text(encoding="utf-8").split())
    scns = [json.loads(l) for l in SCN.read_text(encoding="utf-8").splitlines() if l.strip()]
    val = [s for s in scns if s.get("pos_id") in keep]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    system = load_system_prompt()
    with OUT.open("w", encoding="utf-8") as fh:
        for s in val:
            fh.write(json.dumps({
                "id": s["id"], "pos_id": s["pos_id"], "tier": s["tier"],
                "phase": s["phase"], "severity": s["severity"],
                "system": system, "user": build_user_prompt(s, "grounded"),
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(val)} val prompts ({len(keep)} positions x 3 tiers) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
