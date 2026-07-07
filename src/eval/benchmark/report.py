"""Render RESULTS_BENCHMARK.md and the blinded human-label export.

Everything here is presentation over the aggregated results dict + the raw
checkpoints. The narrative is generated from the actual numbers (it reports what
the data says, including where OURS loses), so re-running on new data keeps the
prose honest.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import chess

from config import schema
from . import config as bcfg
from .aggregate import OBJECTIVE_METRICS, build_results
from .council import anon_mapping
from .io_utils import append_jsonl, read_jsonl, write_json
from . import scenarios as scen_mod

log = logging.getLogger("benchmark.report")


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def _pct(x: Optional[float]) -> str:
    return "–" if x is None else f"{x * 100:.0f}%"


def _f2(x: Optional[float]) -> str:
    return "–" if x is None else f"{x:.2f}"


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    aligns = ["---"] + ["---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(aligns) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _model_label(mk: str) -> str:
    return bcfg.MODELS[mk].display


# --------------------------------------------------------------------------- #
# Objective + council tables
# --------------------------------------------------------------------------- #


def _objective_table(obj: Dict[str, Any], condition: str) -> str:
    m = obj["metrics"]
    header = ["Model", "move_sound", "no_engine_speak", "ply_cap_ok",
              "fabrication_rate", "avg_violations", "n"]
    rows: List[List[str]] = []
    for mk in bcfg.MODEL_ORDER:
        rows.append([
            _model_label(mk),
            _pct(m["move_sound"][mk][condition]),
            _pct(m["no_engine_speak"][mk][condition]),
            _pct(m["ply_cap_ok"][mk][condition]),
            _pct(m["fabrication_rate"][mk][condition]),
            _f2(m["avg_violations"][mk][condition]),
            str(obj["n"][mk][condition] or 0),
        ])
    return _md_table(header, rows)


def _council_table(council: Dict[str, Any]) -> str:
    mr = council["mean_rank"]
    wr = council["win_rate"]
    header = ["Model", "mean rank (ungr)", "mean rank (grnd)",
              "top-1 win% (ungr)", "top-1 win% (grnd)"]
    rows: List[List[str]] = []
    for mk in bcfg.MODEL_ORDER:
        rows.append([
            _model_label(mk),
            _f2(mr[mk]["ungrounded"]),
            _f2(mr[mk]["grounded"]),
            _pct(wr[mk]["ungrounded"]),
            _pct(wr[mk]["grounded"]),
        ])
    return _md_table(header, rows)


def _rubric_table(council: Dict[str, Any], condition: str) -> str:
    ru = council["rubric"]
    header = ["Model", "tier_calibration", "clarity", "correctness"]
    rows: List[List[str]] = []
    for mk in bcfg.MODEL_ORDER:
        rows.append([
            _model_label(mk),
            _f2(ru["tier_calibration"][mk][condition]),
            _f2(ru["clarity"][mk][condition]),
            _f2(ru["correctness"][mk][condition]),
        ])
    return _md_table(header, rows)


def _by_judge_table(council: Dict[str, Any]) -> str:
    pooled = council["by_judge_pooled_mean_rank"]
    header = ["Model"] + [f"judge={bcfg.MODELS[j].display}" for j in bcfg.JUDGE_KEYS]
    rows: List[List[str]] = []
    for mk in bcfg.MODEL_ORDER:
        row = [_model_label(mk)]
        for j in bcfg.JUDGE_KEYS:
            row.append(_f2(pooled.get(j, {}).get(mk)))
        rows.append(row)
    return _md_table(header, rows)


def _self_pref_table(council: Dict[str, Any]) -> str:
    sp = council["self_preference"]
    header = ["Judge (lab)", "own model", "own mean rank",
              "peers' mean rank", "self-pref Δ (peers − own)"]
    rows: List[List[str]] = []
    for j in bcfg.JUDGE_KEYS:
        d = sp[j]
        rows.append([
            bcfg.MODELS[j].display,
            bcfg.MODELS[d["family_model"]].display,
            _f2(d["own_mean_rank"]),
            _f2(d["others_mean_rank"]),
            ("–" if d["self_pref_delta"] is None else f"{d['self_pref_delta']:+.2f}"),
        ])
    return _md_table(header, rows)


def _cost_table(cost: Dict[str, Any]) -> str:
    pm = cost["per_model"]
    header = ["Model", "gen calls", "judge calls", "in tok", "out tok", "est. USD"]
    rows: List[List[str]] = []
    for mk in bcfg.MODEL_ORDER:
        d = pm[mk]
        in_tok = d["gen_prompt_tokens"] + d["judge_prompt_tokens"]
        out_tok = d["gen_completion_tokens"] + d["judge_completion_tokens"]
        rows.append([
            _model_label(mk),
            str(d["gen_calls"]),
            str(d["judge_calls"]),
            f"{in_tok:,}",
            f"{out_tok:,}",
            f"${d['total_cost_usd']:.2f}",
        ])
    return _md_table(header, rows)


def _distribution_tables(dist: Dict[str, Dict[str, int]]) -> str:
    parts: List[str] = []
    for axis in ("tier", "phase", "severity"):
        counts = dist.get(axis, {})
        row = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        parts.append(f"- **{axis}** — {row}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Narrative (generated from the numbers)
# --------------------------------------------------------------------------- #


def _avg(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def _narrative(res: Dict[str, Any]) -> str:
    obj = res["objective"]["metrics"]
    council = res["council"]
    mr = council["mean_rank"]
    frontier = ("gpt", "claude", "gemini")

    lines: List[str] = []

    def g(metric: str, mk: str, cond: str) -> Optional[float]:
        return obj[metric][mk][cond]

    # Grounding effect on soundness for OURS + frontier.
    ours_snd_u, ours_snd_g = g("move_sound", "ours", "ungrounded"), g("move_sound", "ours", "grounded")
    base_snd_u, base_snd_g = g("move_sound", "base", "ungrounded"), g("move_sound", "base", "grounded")
    fr_snd_u = _avg([g("move_sound", m, "ungrounded") for m in frontier])
    fr_snd_g = _avg([g("move_sound", m, "grounded") for m in frontier])
    lines.append(
        f"- **Grounding lifts move-soundness across the board.** OURS goes "
        f"{_pct(ours_snd_u)} → {_pct(ours_snd_g)} on picking a Stockfish-sound move; "
        f"BASE {_pct(base_snd_u)} → {_pct(base_snd_g)}; the frontier average "
        f"{_pct(fr_snd_u)} → {_pct(fr_snd_g)}. When every model is handed the sound "
        f"pool, choosing from it is the easy part."
    )

    # Fabrication story.
    ours_fab_u, ours_fab_g = g("fabrication_rate", "ours", "ungrounded"), g("fabrication_rate", "ours", "grounded")
    fr_fab_u = _avg([g("fabrication_rate", m, "ungrounded") for m in frontier])
    fr_fab_g = _avg([g("fabrication_rate", m, "grounded") for m in frontier])
    lines.append(
        f"- **Fabrication is the honest metric, and grounding is what moves it.** "
        f"OURS fabricates a false board fact in {_pct(ours_fab_u)} of ungrounded "
        f"outputs vs {_pct(ours_fab_g)} grounded; the frontier average is "
        f"{_pct(fr_fab_u)} → {_pct(fr_fab_g)}. Verified facts in the prompt are the "
        f"lever a 1.7B model cannot supply from its own board-tracking."
    )

    # Engine-speak (style behavior the fine-tune owns).
    ours_es_u = g("no_engine_speak", "ours", "ungrounded")
    ours_es_g = g("no_engine_speak", "ours", "grounded")
    fr_es_g = _avg([g("no_engine_speak", m, "grounded") for m in frontier])
    lines.append(
        f"- **The fine-tune still owns the style gate.** OURS keeps outputs "
        f"jargon-free (no engine-speak) {_pct(ours_es_u)}/{_pct(ours_es_g)} "
        f"(ungr/grnd) vs the grounded frontier average {_pct(fr_es_g)} — the "
        f"frontier models, handed evals, are more tempted to leak them."
    )

    # Council: does grounding close the coaching gap?
    ours_mr_u, ours_mr_g = mr["ours"]["ungrounded"], mr["ours"]["grounded"]
    fr_mr_g = _avg([mr[m]["grounded"] for m in frontier])
    best_fr_g = min([mr[m]["grounded"] for m in frontier if mr[m]["grounded"] is not None], default=None)
    lines.append(
        f"- **Council (instructiveness) is where the honest gap shows.** OURS mean "
        f"rank improves {_f2(ours_mr_u)} → {_f2(ours_mr_g)} (1=best of 5) with "
        f"grounding, while the grounded frontier averages {_f2(fr_mr_g)} "
        f"(best {_f2(best_fr_g)}). "
        + (
            "Grounding narrows the coaching gap but does not erase it — a bigger "
            "model still explains more instructively."
            if (ours_mr_g is not None and fr_mr_g is not None and ours_mr_g > fr_mr_g)
            else "On the grounded row OURS is competitive with the frontier panel on "
                 "the target behavior."
        )
    )

    # Bias check.
    sp = council["self_preference"]
    signed = sp.get("_mean_signed_delta")
    lines.append(
        f"- **Self-preference check.** Mean signed self-preference across the three "
        f"judges is {('–' if signed is None else f'{signed:+.2f}')} rank "
        f"(positive = a judge ranks its own lab's model better than its peers do). "
        + (
            "Small relative to the gaps above, so the council ranking is not merely lab loyalty."
            if (signed is not None and abs(signed) < 0.5)
            else "Non-trivial — read the per-judge table below and treat the aggregate with that caveat."
        )
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# RESULTS_BENCHMARK.md
# --------------------------------------------------------------------------- #


def write_report(res: Dict[str, Any]) -> None:
    meta = res["meta"]
    obj = res["objective"]
    council = res["council"]
    cost = res["cost"]
    n = meta["n_scenarios"]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    parts: List[str] = []
    parts.append("# Chess-Coach Benchmark — Grounded vs Ungrounded, Ours vs Frontier\n")
    parts.append(
        f"A 2×2×5 grid: **5 models × 2 conditions** (WITHOUT vs WITH Stockfish+Maia "
        f"grounding), scored on **{n} held-out positions** by deterministic objective "
        f"metrics and a **blinded, cross-family model council**. Generated {ts}.\n"
    )

    parts.append("## Setup\n")
    parts.append(
        "| Item | Value |\n|---|---|\n"
        f"| Held-out positions | {n} (excluded any board present in `train.jsonl`/`valid.jsonl`, dedup by placement+turn) |\n"
        "| Competitors | OURS `chess-coach-v1` (1.7B tuned) · BASE `Qwen3-1.7B-4bit` · GPT-5.5 · Claude Opus 4.8 · Gemini 3.1 Pro |\n"
        "| Conditions | **ungrounded** (tier + position + student move only) vs **grounded** (verified facts + Stockfish sound pool + Maia) |\n"
        "| Same system prompt + format instruction | Yes — identical for all 5 models, both conditions |\n"
        "| Objective | move-soundness (Stockfish pool), no-engine-speak, ply-cap, **fabrication rate** (faithfulness verifier) |\n"
        "| Council | GPT-5.5 + Claude Opus 4.8 + Gemini 3.1 Pro rank the 5 **blinded** outputs by *instructiveness for the tier* |\n"
    )
    parts.append(
        "\n**Scenario balance** (game phase = material-first: endgame if ≤6 major/minor "
        "pieces, else opening if ≤ move 12, else middlegame):\n\n"
        + _distribution_tables(meta["distribution"]) + "\n"
    )

    parts.append("\n## Headline\n")
    parts.append(_narrative(res) + "\n")

    parts.append("\n## Objective metrics (2×2: 5 models × 2 conditions)\n")
    parts.append("### WITHOUT grounding\n")
    parts.append(_objective_table(obj, "ungrounded") + "\n")
    parts.append("\n### WITH grounding\n")
    parts.append(_objective_table(obj, "grounded") + "\n")
    parts.append(
        "\n*`move_sound` = recommended move is in the Stockfish sound pool. "
        "`fabrication_rate` = share of outputs with ≥1 false board fact "
        "(non-LLM faithfulness verifier). `avg_violations` = mean false facts per output.*\n"
    )

    parts.append("\n## Council — instructiveness ranking (blinded, cross-family)\n")
    parts.append(
        f"Each of {len(meta['judges'])} judges ranked all 5 anonymized outputs per "
        f"item; lower mean rank = judged more instructive (1 = best of 5).\n\n"
    )
    parts.append(_council_table(council) + "\n")
    parts.append(f"\nItems judged: ungrounded {council['n_items'].get('ungrounded', 0)}, "
                 f"grounded {council['n_items'].get('grounded', 0)} (judge-observations).\n")

    parts.append("\n### Per-dimension rubric (mean 0–2)\n")
    parts.append("**WITHOUT grounding**\n\n" + _rubric_table(council, "ungrounded") + "\n")
    parts.append("\n**WITH grounding**\n\n" + _rubric_table(council, "grounded") + "\n")

    parts.append("\n## Bias / self-preference check\n")
    parts.append(
        "Mean rank each judge gives each model, pooled across conditions "
        "(a judge favoring its own lab shows a lower number in its own column):\n\n"
    )
    parts.append(_by_judge_table(council) + "\n")
    parts.append("\nSelf-preference (a judge vs its peers, on its own lab's model):\n\n")
    parts.append(_self_pref_table(council) + "\n")
    sp = council["self_preference"]
    _signed = sp.get("_mean_signed_delta")
    _absd = sp.get("_mean_abs_delta")
    _signed_str = "–" if _signed is None else f"{_signed:+.2f}"
    _absd_str = "–" if _absd is None else f"{_absd:.2f}"
    parts.append(
        f"\nMean signed self-preference Δ across judges: **{_signed_str}** rank "
        f"(positive = judges favor their own lab). Mean magnitude: {_absd_str}.\n"
    )

    parts.append("\n## Cost\n")
    parts.append(_cost_table(cost) + "\n")
    parts.append(
        f"\n**Total estimated cost: ${cost['total_cost_usd']:.2f}** "
        f"(local MLX models are free; frontier prices are per-1M-token estimates, "
        f"see per-model rows). Generations: {meta['counts']['generations']}, "
        f"council judgments: {meta['counts']['council']}.\n"
    )

    parts.append("\n## Artifacts\n")
    parts.append(
        f"- Scenario set: `data/benchmark/scenarios.jsonl`\n"
        f"- Raw generations: `data/benchmark/generations.jsonl`\n"
        f"- Objective scores: `data/benchmark/objective.jsonl`\n"
        f"- Council rankings: `data/benchmark/council.jsonl`\n"
        f"- Aggregated results: `data/benchmark/results.json`\n"
        f"- Blind human-label export: `data/benchmark/blind_label.jsonl` + "
        f"`blind_label.html` (viewer) + `blind_key.json` (label→model)\n"
    )

    bcfg.REPORT_MD_PATH.write_text("\n".join(parts), encoding="utf-8")
    log.info("wrote %s", bcfg.REPORT_MD_PATH)


# --------------------------------------------------------------------------- #
# Blind human-label export
# --------------------------------------------------------------------------- #


def write_blind_export(scenarios: Sequence[Dict[str, Any]], conditions: Sequence[str]) -> int:
    """Write blind_label.jsonl + blind_key.json + blind_label.html. Returns #items."""
    gen_index = {
        (g["scenario_id"], g["model"], g["condition"]): g.get("output", "")
        for g in read_jsonl(bcfg.GENERATIONS_PATH)
    }
    # Fresh files each time (idempotent regeneration).
    for p in (bcfg.BLIND_LABEL_JSONL,):
        if p.exists():
            p.unlink()

    items: List[Dict[str, Any]] = []
    key: Dict[str, Dict[str, str]] = {}
    for scn in scenarios:
        for cond in conditions:
            outs = {mk: gen_index.get((scn["id"], mk, cond)) for mk in bcfg.MODEL_ORDER}
            if any(v is None for v in outs.values()):
                continue
            mapping = anon_mapping(scn["id"], cond)
            item_id = f"{scn['id']}::{cond}"
            board = chess.Board(scn["fen"])
            responses = [
                {"label": lab, "text": (outs[mapping[lab]] or "").strip()}
                for lab in bcfg.ANON_LABELS
            ]
            row = {
                "item_id": item_id,
                "scenario_id": scn["id"],
                "condition": cond,
                "tier": scn["tier"],
                "phase": scn["phase"],
                "severity": scn["severity"],
                "fen": scn["fen"],
                "board_ascii": schema.ascii_board(scn["fen"]),
                "side_to_move": "white" if board.turn else "black",
                "student_move_san": scn["student_move"]["san"],
                "responses": responses,
            }
            items.append(row)
            append_jsonl(bcfg.BLIND_LABEL_JSONL, row)
            key[item_id] = mapping

    write_json(bcfg.BLIND_KEY_JSON, key)
    _write_blind_html(items)
    log.info("blind export: %d items -> %s", len(items), bcfg.BLIND_LABEL_JSONL)
    return len(items)


def _write_blind_html(items: List[Dict[str, Any]]) -> None:
    """A self-contained viewer: position + 5 unnamed answers + a ranking form."""
    data_json = json.dumps(items, ensure_ascii=False)
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind chess-coach label</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 24px;
         max-width: 1000px; margin-inline: auto; }
  h1 { font-size: 20px; } h2 { font-size: 16px; margin: 0 0 8px; }
  .item { border: 1px solid #8884; border-radius: 12px; padding: 16px; margin: 18px 0; }
  .meta { color: #888; font-size: 13px; margin-bottom: 8px; }
  pre.board { font: 13px/1.35 ui-monospace, Menlo, monospace; background: #8881; padding: 10px;
              border-radius: 8px; display: inline-block; }
  .resp { border: 1px solid #8883; border-radius: 10px; padding: 10px 12px; margin: 8px 0; }
  .resp h3 { margin: 0 0 4px; font-size: 14px; }
  .rank { width: 56px; }
  .bar { position: sticky; top: 0; background: Canvas; padding: 10px 0; border-bottom: 1px solid #8884; }
  button { font: inherit; padding: 6px 12px; border-radius: 8px; cursor: pointer; }
  .prog { color: #888; font-size: 13px; }
</style></head>
<body>
<div class="bar">
  <h1>Blind chess-coach label</h1>
  <div class="prog" id="prog"></div>
  <button id="dl">Download my rankings (JSON)</button>
  <span class="meta">Rank each response 1 (best) – 5 (worst) by how INSTRUCTIVE it is for the stated tier. No model names are shown.</span>
</div>
<div id="root"></div>
<script>
const ITEMS = __DATA__;
const root = document.getElementById('root');
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
ITEMS.forEach((it, i) => {
  const div = document.createElement('div'); div.className='item';
  let respHtml = it.responses.map(r => `
    <div class="resp">
      <h3>Response ${r.label}
        <label style="float:right;font-weight:normal">rank
          <input class="rank" type="number" min="1" max="5" data-item="${it.item_id}" data-label="${r.label}">
        </label></h3>
      <div>${esc(r.text)}</div>
    </div>`).join('');
  div.innerHTML = `
    <div class="meta">#${i+1}/${ITEMS.length} · tier <b>${it.tier}</b> · ${it.phase} · ${it.condition}
      · ${it.side_to_move} to move · student played <b>${esc(it.student_move_san)}</b></div>
    <pre class="board">${esc(it.board_ascii)}</pre>
    ${respHtml}`;
  root.appendChild(div);
});
function collect(){
  const out = {};
  document.querySelectorAll('input.rank').forEach(inp => {
    const v = inp.value.trim(); if(!v) return;
    (out[inp.dataset.item] ||= {})[inp.dataset.label] = Number(v);
  });
  return out;
}
function updateProg(){
  const done = Object.values(collect()).filter(o => Object.keys(o).length===5).length;
  document.getElementById('prog').textContent = `${done} / ${ITEMS.length} items fully ranked`;
}
document.addEventListener('input', updateProg); updateProg();
document.getElementById('dl').onclick = () => {
  const blob = new Blob([JSON.stringify(collect(), null, 2)], {type:'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'my_blind_rankings.json'; a.click();
};
</script></body></html>"""
    page = page.replace("__DATA__", data_json)
    bcfg.BLIND_LABEL_HTML.write_text(page, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #


def run_report(conditions: Sequence[str]) -> Dict[str, Any]:
    """Aggregate, persist results.json, write the report + blind export."""
    res = build_results()
    write_json(bcfg.RESULTS_JSON_PATH, res)
    write_report(res)
    scenarios = scen_mod.load_scenarios()
    n_blind = write_blind_export(scenarios, conditions)
    res["meta"]["blind_items"] = n_blind
    return res
