"""Objective scoring: deterministic, free, and run over every generation.

Reuses the exact functions the project already trusts (``src/eval/evaluate.py``
for move extraction / engine-speak / ply-cap, and ``src/engine/faithfulness.py``
for the truth gate) so the benchmark's objective numbers are consistent with the
rest of the pipeline. Per generation it records:

* ``move_sound``      — recommended move is in this position's Stockfish pool
* ``no_engine_speak`` — no centipawns / eval / engine words leaked
* ``ply_cap_ok``      — narrated line within the tier's ply cap
* ``fabricated``      — the faithfulness verifier found >=1 false board fact
                        (plus ``n_violations`` — the honest hallucination metric)

Resumable: keyed by ``(scenario_id, model, condition)``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

from config import settings
from src.engine.faithfulness import verify_text
from src.eval.evaluate import (
    extract_recommended_move,
    find_engine_speak,
    longest_narrated_line,
)

from . import config as bcfg
from .io_utils import append_jsonl, done_keys, read_jsonl

log = logging.getLogger("benchmark.objective")


def score_one(scn: Dict[str, Any], output: str) -> Dict[str, Any]:
    """Run all deterministic checks for one output against its scenario."""
    fen = scn["fen"]
    tier = scn["tier"]
    student_uci = scn["student_move"]["uci"]
    sound_uci = set(scn["sound_uci"])
    ply_cap = settings.TIERS[tier]["ply_cap"]

    rec_san, rec_uci = extract_recommended_move(output, fen, student_uci)
    speak_hits = find_engine_speak(output)
    verdict = verify_text(output, fen)

    return {
        "rec_san": rec_san,
        "rec_uci": rec_uci,
        "produced_nonempty": bool(output.strip()),
        "move_parseable": rec_uci is not None,
        "move_sound": rec_uci is not None and rec_uci in sound_uci,
        "no_engine_speak": len(speak_hits) == 0,
        "ply_cap_ok": longest_narrated_line(output) <= ply_cap,
        "engine_speak_hits": speak_hits,
        "n_violations": len(verdict.violations),
        "fabricated": len(verdict.violations) >= 1,
        "violations": [
            {"sentence": v.sentence, "reason": v.reason} for v in verdict.violations[:5]
        ],
    }


def run_objective(scenarios: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Score every generation that has not yet been scored. Returns counts."""
    by_id = {s["id"]: s for s in scenarios}
    generations = read_jsonl(bcfg.GENERATIONS_PATH)
    done = done_keys(bcfg.OBJECTIVE_PATH, ["scenario_id", "model", "condition"])

    scored = skipped = 0
    for gen in generations:
        key = (gen["scenario_id"], gen["model"], gen["condition"])
        if key in done:
            continue
        scn = by_id.get(gen["scenario_id"])
        if scn is None:
            skipped += 1
            continue
        row = {
            "scenario_id": gen["scenario_id"],
            "model": gen["model"],
            "condition": gen["condition"],
            "tier": gen["tier"],
            "phase": gen["phase"],
            "severity": gen["severity"],
            **score_one(scn, gen.get("output", "")),
        }
        append_jsonl(bcfg.OBJECTIVE_PATH, row)
        done.add(key)
        scored += 1

    log.info("objective scoring: %d scored, %d skipped (no scenario)", scored, skipped)
    return {"scored": scored, "skipped": skipped}
