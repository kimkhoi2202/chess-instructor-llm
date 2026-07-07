"""Aggregate the raw benchmark artifacts into the numbers the report prints.

Produces, from ``scenarios/generations/objective/council`` JSONL:

* objective 2x2 tables (model x condition) for each deterministic metric,
* council mean-rank + top-1 win-rate (overall and per judge),
* the per-dimension rubric means,
* the self-preference check (does a judge rank its own lab's model better than
  the other judges do?),
* a cost summary from the token usage stored on every frontier row.

Pure functions over lists of dicts — no I/O except the final ``build_results``
convenience that reads the checkpoint files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import config as bcfg
from .io_utils import read_jsonl
from . import scenarios as scen_mod

OBJECTIVE_METRICS: Tuple[str, ...] = (
    "produced_nonempty",
    "move_parseable",
    "move_sound",
    "no_engine_speak",
    "ply_cap_ok",
)


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _blank_grid() -> Dict[str, Dict[str, Any]]:
    return {mk: {c: None for c in bcfg.CONDITIONS} for mk in bcfg.MODEL_ORDER}


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #


def aggregate_objective(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """model x condition rates for each check, fabrication rate + avg violations."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["condition"]), []).append(r)

    metrics: Dict[str, Dict[str, Dict[str, Any]]] = {
        m: _blank_grid() for m in OBJECTIVE_METRICS
    }
    metrics["fabrication_rate"] = _blank_grid()
    metrics["avg_violations"] = _blank_grid()
    n_grid = _blank_grid()

    for (mk, cond), g in groups.items():
        if mk not in metrics["move_sound"]:
            continue
        n_grid[mk][cond] = len(g)
        for metric in OBJECTIVE_METRICS:
            metrics[metric][mk][cond] = _mean([1.0 if r[metric] else 0.0 for r in g])
        metrics["fabrication_rate"][mk][cond] = _mean(
            [1.0 if r["fabricated"] else 0.0 for r in g]
        )
        metrics["avg_violations"][mk][cond] = _mean([float(r["n_violations"]) for r in g])

    return {"metrics": metrics, "n": n_grid}


def aggregate_objective_by_tier(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """move_sound + fabrication_rate per tier (model x condition) for the appendix."""
    out: Dict[str, Any] = {}
    for tier in scen_mod.TIER_ORDER:
        sub = [r for r in rows if r["tier"] == tier]
        if not sub:
            continue
        out[tier] = aggregate_objective(sub)["metrics"]
    return out


# --------------------------------------------------------------------------- #
# Council
# --------------------------------------------------------------------------- #


def _rank_of(model_key: str, ranking: List[str], label_to_model: Dict[str, str]) -> Optional[int]:
    """1-based rank the judge gave ``model_key`` in one item (or None)."""
    model_to_label = {m: l for l, m in label_to_model.items()}
    label = model_to_label.get(model_key)
    if label is None or label not in ranking:
        return None
    return ranking.index(label) + 1


def aggregate_council(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean rank, win-rate, per-judge ranks, rubric means, self-preference."""
    # Accumulators keyed appropriately.
    ranks: Dict[Tuple[str, str], List[int]] = {}            # (model, cond) -> ranks
    wins: Dict[Tuple[str, str], List[int]] = {}             # (model, cond) -> 1/0
    by_judge: Dict[str, Dict[Tuple[str, str], List[int]]] = {}  # judge -> (model,cond)->ranks
    rubric: Dict[str, Dict[Tuple[str, str], List[int]]] = {  # dim -> (model,cond)->scores
        d: {} for d in ("tier_calibration", "clarity", "correctness")
    }
    # For self-preference: per (judge, model) pooled across conditions.
    judge_model_ranks: Dict[Tuple[str, str], List[int]] = {}

    n_items: Dict[str, int] = {c: 0 for c in bcfg.CONDITIONS}

    for r in rows:
        cond = r["condition"]
        judge = r["judge"]
        ranking = r["ranking"]
        l2m = r["label_to_model"]
        scores = r.get("scores", {})
        n_items[cond] = n_items.get(cond, 0) + 1
        by_judge.setdefault(judge, {})
        for mk in bcfg.MODEL_ORDER:
            rank = _rank_of(mk, ranking, l2m)
            if rank is None:
                continue
            ranks.setdefault((mk, cond), []).append(rank)
            wins.setdefault((mk, cond), []).append(1 if rank == 1 else 0)
            by_judge[judge].setdefault((mk, cond), []).append(rank)
            judge_model_ranks.setdefault((judge, mk), []).append(rank)
        # rubric
        m2l = {m: l for l, m in l2m.items()}
        for mk in bcfg.MODEL_ORDER:
            lab = m2l.get(mk)
            cell = scores.get(lab, {}) if lab else {}
            for dim in rubric:
                if dim in cell:
                    rubric[dim].setdefault((mk, cond), []).append(int(cell[dim]))

    def _grid_from(acc: Dict[Tuple[str, str], List[int]]) -> Dict[str, Dict[str, Any]]:
        grid = _blank_grid()
        for (mk, cond), vals in acc.items():
            if mk in grid:
                grid[mk][cond] = _mean([float(v) for v in vals])
        return grid

    mean_rank = _grid_from(ranks)
    win_rate = _grid_from(wins)
    rubric_grids = {dim: _grid_from(acc) for dim, acc in rubric.items()}

    by_judge_grid: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for judge, acc in by_judge.items():
        by_judge_grid[judge] = _grid_from(acc)

    # Per-judge mean rank of each model, pooled across conditions (bias table).
    by_judge_pooled: Dict[str, Dict[str, Any]] = {}
    for (judge, mk), vals in judge_model_ranks.items():
        by_judge_pooled.setdefault(judge, {})[mk] = _mean([float(v) for v in vals])

    # Self-preference: compare a judge's rank of its own lab's model to the mean
    # rank the OTHER judges give that same model (pooled over conditions).
    self_pref: Dict[str, Any] = {}
    for judge in bcfg.JUDGE_KEYS:
        family = bcfg.MODELS[judge].family  # judge key == family competitor key
        own = judge_model_ranks.get((judge, judge), [])
        others = [
            v
            for (jj, mk), vals in judge_model_ranks.items()
            if mk == judge and jj != judge
            for v in vals
        ]
        own_mean = _mean([float(x) for x in own])
        others_mean = _mean([float(x) for x in others])
        delta = (others_mean - own_mean) if (own_mean is not None and others_mean is not None) else None
        self_pref[judge] = {
            "family_model": judge,
            "own_mean_rank": own_mean,
            "others_mean_rank": others_mean,
            "self_pref_delta": delta,  # +ve => judge ranks own family better than peers do
            "n_own": len(own),
            "n_others": len(others),
        }
    deltas = [v["self_pref_delta"] for v in self_pref.values() if v["self_pref_delta"] is not None]
    self_pref["_mean_abs_delta"] = _mean([abs(d) for d in deltas]) if deltas else None
    self_pref["_mean_signed_delta"] = _mean(deltas) if deltas else None

    return {
        "mean_rank": mean_rank,
        "win_rate": win_rate,
        "rubric": rubric_grids,
        "by_judge_mean_rank": by_judge_grid,
        "by_judge_pooled_mean_rank": by_judge_pooled,
        "self_preference": self_pref,
        "n_items": n_items,
    }


def aggregate_council_by_tier(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Council mean-rank per tier (model x condition) for the appendix."""
    out: Dict[str, Any] = {}
    for tier in scen_mod.TIER_ORDER:
        sub = [r for r in rows if r["tier"] == tier]
        if not sub:
            continue
        out[tier] = aggregate_council(sub)["mean_rank"]
    return out


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


def aggregate_cost(generations: Sequence[Dict[str, Any]],
                   council: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Token totals + estimated USD per model (generation) and per judge (council)."""
    def _acc(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for r in rows:
            k = r[key]
            d = out.setdefault(k, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
            d["prompt_tokens"] += int(r.get("prompt_tokens", 0))
            d["completion_tokens"] += int(r.get("completion_tokens", 0))
            d["calls"] += 1
        return out

    gen_tok = _acc(generations, "model")
    jud_tok = _acc(council, "judge")

    def _cost(model_key: str, tok: Dict[str, int]) -> float:
        pin, pout = bcfg.price_for(model_key)
        return tok["prompt_tokens"] / 1e6 * pin + tok["completion_tokens"] / 1e6 * pout

    per_model: Dict[str, Any] = {}
    total = 0.0
    for mk in bcfg.MODEL_ORDER:
        g = gen_tok.get(mk, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
        j = jud_tok.get(mk, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
        gen_cost = _cost(mk, g)
        jud_cost = _cost(mk, j)
        total += gen_cost + jud_cost
        per_model[mk] = {
            "gen_calls": g["calls"],
            "gen_prompt_tokens": g["prompt_tokens"],
            "gen_completion_tokens": g["completion_tokens"],
            "gen_cost_usd": round(gen_cost, 4),
            "judge_calls": j["calls"],
            "judge_prompt_tokens": j["prompt_tokens"],
            "judge_completion_tokens": j["completion_tokens"],
            "judge_cost_usd": round(jud_cost, 4),
            "total_cost_usd": round(gen_cost + jud_cost, 4),
            "price_in_per_m": bcfg.MODELS[mk].price_in,
            "price_out_per_m": bcfg.MODELS[mk].price_out,
        }
    return {"per_model": per_model, "total_cost_usd": round(total, 4)}


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #


def build_results() -> Dict[str, Any]:
    """Read all checkpoints and return the full aggregated results dict."""
    scenarios = scen_mod.load_scenarios()
    generations = read_jsonl(bcfg.GENERATIONS_PATH)
    objective = read_jsonl(bcfg.OBJECTIVE_PATH)
    council = read_jsonl(bcfg.COUNCIL_PATH)

    return {
        "meta": {
            "n_scenarios": len(scenarios),
            "distribution": scen_mod.distribution(scenarios),
            "conditions": list(bcfg.CONDITIONS),
            "models": {mk: bcfg.MODELS[mk].display for mk in bcfg.MODEL_ORDER},
            "judges": list(bcfg.JUDGE_KEYS),
            "counts": {
                "generations": len(generations),
                "objective": len(objective),
                "council": len(council),
            },
        },
        "objective": aggregate_objective(objective),
        "objective_by_tier": aggregate_objective_by_tier(objective),
        "council": aggregate_council(council),
        "council_by_tier": aggregate_council_by_tier(council),
        "cost": aggregate_cost(generations, council),
    }
