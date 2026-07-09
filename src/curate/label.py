#!/usr/bin/env python3
"""LABEL mined discriminating positions — cross-family, independent, best-of-N.

For every mined (position, tier) the three frontier teachers write coaching
**independently** (no cross-talk / no debate — independence avoids herding),
each label is **gated** for correctness (faithfulness = 0 fabrication,
tier-appropriateness, a named principle in the takeaway, no engine-speak,
well-formed, no wrong heuristics), and the most instructive SURVIVOR is selected
best-of-N (a light deterministic rubric, with an optional single blinded
cross-family judge tiebreak). The engine + the gates are the arbiter.

All teacher calls go through the org-funded TrueFoundry gateway (no GPU). The
move to teach is pre-selected (deterministically, from the engine + Maia in the
mine step), so each teacher only has to phrase the lesson — the cross-family
variable is explanation quality, which is exactly what best-of-N optimizes.

CLI
---
    python -m src.curate.label label --limit 6 --smoke      # tiny end-to-end
    python -m src.curate.label label --concurrency 6        # full labeling
    python -m src.curate.label build --valid-frac 0.06      # train/valid + manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

from config import schema, settings  # noqa: E402
from src.curate import gates as G  # noqa: E402
from src.curate.mine import CURATE_DIR, MINED_OUT, TIER_ORDER  # noqa: E402
from src.engine.position_facts import render_pool_facts  # noqa: E402
from src.eval.benchmark.backends import TFYChat, make_tfy_client  # noqa: E402
from src.eval.benchmark.prompts import build_grounded_user, load_system_prompt  # noqa: E402
from src.teacher.generate import (  # noqa: E402
    RateLimiter,
    _extract_tier_guide,
    _parse_teacher_json,
    _read_text,
)

log = logging.getLogger("curate.label")

# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
LABELED_OUT = CURATE_DIR / "labeled.jsonl"        # one best-of-N winner per (id,tier)
RAW_OUT = CURATE_DIR / "label_raw.jsonl"          # every teacher label (audit trail)
TRAIN_OUT = settings.DATASET / "train_curated_32b.jsonl"
VALID_OUT = settings.DATASET / "valid_curated_32b.jsonl"
MANIFEST = CURATE_DIR / "manifest.json"

SEED = 20260708


# The three cross-family teachers (ids + tuned reasoning effort from the working
# benchmark config; independent — no shared context between them).
def _teacher_specs() -> List[Dict[str, Any]]:
    from src.eval.benchmark.config import MODELS
    out = []
    for key in ("gpt", "claude", "gemini"):
        m = MODELS[key]
        out.append({"family": key, "ident": m.ident,
                    "effort": m.reasoning_effort,
                    "price_in": m.price_in, "price_out": m.price_out})
    return out


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #
# Compact, correctness-checked principle vocabulary (distilled from
# principle_library_v5.md §A/§B/§D). Injected instead of the full 9.6 KB library
# to cut teacher INPUT tokens ~40% (the dominant cost) while preserving the named
# vocabulary + the "do not parrot" caveats. The gates enforce the rest.
COMPACT_PRINCIPLES = """\
NAMED TRANSFERABLE PRINCIPLES (name one of these in the takeaway):
- Develop your pieces before you attack; develop each piece once, with a purpose.
- King safety first — get your king out of the center (but never castle into an attack).
- Fight for / control the center (with pawns OR pieces).
- Don't leave pieces hanging — count attackers vs defenders before every move.
- Make a threat while you develop (gain time / for free — not "tempo" for beginners).
- Don't bring the queen out early to be chased.
- A move that does two jobs at once is usually best.
- Put your rooks on open (or half-open) files — if the file has a target/entry.
- A rook on the 7th rank is powerful; double rooks to support each other.
- Improve your worst-placed piece — give your saddest piece a job.
- Trade pieces (not pawns) when you are AHEAD; keep pieces when BEHIND for counterplay.
- Weigh every trade by what it changes (files, activity, structure) — don't auto-simplify.
- Target the opponent's weak/isolated/backward pawn or weak square.
- Outpost: plant a knight on a protected square no enemy pawn can chase (intermediate+).
- Good vs bad bishop — improve, reroute, or trade your bad bishop (intermediate+).
- Support or blockade a passed pawn; push it only when it safely gains ground.
- Prophylaxis — take away the opponent's idea before pursuing your own (advanced).
- Bishop pair is an asset in OPEN positions; use space only if you have a break.
- In the endgame, activate your king.

DO NOT PARROT THESE WRONG HEURISTICS:
- NOT "trade/simplify when you are losing/behind" (that is a beginner mistake; keep pieces when behind).
- NOT "passed pawns must always be pushed" (support/blockade first; push only when safe).
- NOT "always castle" / "the bishop pair is always better" / "space is always good" / "always grab the free pawn".
- NOT "bring the queen out early to be active"."""


def _principle_library_text() -> str:
    return COMPACT_PRINCIPLES


def build_teacher_system(tier: str) -> str:
    """teacher_curate_v5 with the tier guide + v5 principle library injected."""
    template = _read_text(str(settings.PROMPTS / "teacher_curate_v5.md"))
    guide = _extract_tier_guide(
        _read_text(str(settings.PROMPTS / "tier_guides.md")), tier)
    return (template.replace("{TIER_GUIDE}", guide or "(none provided)")
            .replace("{PRINCIPLES}", _principle_library_text()))


_JSON_SCHEMA_HINT = (
    "\n\nReturn ONE JSON object, nothing else:\n"
    '{"tier": "%s", "recommended_move_san": "%s", "recommended_move_uci": "%s", '
    '"coaching": "...", "method": "...", "takeaway": "...", '
    '"concepts_used": ["...", "..."]}'
)


def ti_for(mined: Dict[str, Any], tier: str) -> schema.TeacherInput:
    return {
        "tier": tier,
        "fen": mined["fen"],
        "move_history_san": None,
        "student_move": mined["student_by_tier"][tier],
        "sound_pool": mined["sound_pool"],
        "maia_human_moves": mined["maia_by_tier"][tier],
    }  # type: ignore[return-value]


def scenario_for(mined: Dict[str, Any], tier: str) -> Dict[str, Any]:
    return {
        "fen": mined["fen"], "tier": tier,
        "student_move": mined["student_by_tier"][tier],
        "sound_pool": mined["sound_pool"],
        "maia": mined["maia_by_tier"][tier],
    }


def build_teacher_user(mined: Dict[str, Any], tier: str) -> str:
    """Grounded facts + pool + Maia + the pre-selected move to teach + JSON hint."""
    ti = ti_for(mined, tier)
    facts = render_pool_facts(mined["fen"], ti["sound_pool"])
    body = schema.render_user_prompt(ti)
    pick = mined["tier_picks"][tier]
    teach = (f"\n\nTEACH THIS MOVE (pre-selected as sound AND findable for a "
             f"{tier} player): {pick['san']}  ({pick['uci']}). Recommend exactly "
             f"this move and coach it per the rubric.")
    schema_hint = _JSON_SCHEMA_HINT % (tier, pick["san"], pick["uci"])
    return f"{facts}\n\n{body}{teach}{schema_hint}"


# --------------------------------------------------------------------------- #
# Teacher output normalization
# --------------------------------------------------------------------------- #
def _normalize_output(raw: Dict[str, Any], fen: str, tier: str) -> Dict[str, Any]:
    """Coerce a teacher JSON into the candidate shape (normalize move SAN/UCI)."""
    import chess
    board = chess.Board(fen)
    san = str(raw.get("recommended_move_san", "") or "").strip()
    uci = str(raw.get("recommended_move_uci", "") or "").strip()
    move = None
    if uci:
        try:
            c = chess.Move.from_uci(uci)
            if c in board.legal_moves:
                move = c
        except ValueError:
            move = None
    if move is None and san:
        try:
            move = board.parse_san(san)
        except ValueError:
            move = None
    if move is not None:
        san, uci = board.san(move), move.uci()
    concepts = raw.get("concepts_used", [])
    if isinstance(concepts, str):
        concepts = [concepts]
    if not isinstance(concepts, list):
        concepts = []
    return {
        "tier": tier,
        "recommended_move_san": san,
        "recommended_move_uci": uci,
        "coaching": str(raw.get("coaching", "") or "").strip(),
        "method": str(raw.get("method", "") or "").strip(),
        "takeaway": str(raw.get("takeaway", "") or "").strip(),
        "concepts_used": [str(c).strip() for c in concepts if str(c).strip()],
    }


# --------------------------------------------------------------------------- #
# Cost accounting (thread-safe)
# --------------------------------------------------------------------------- #
class Cost:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: Counter = Counter()
        self.tok_in: Counter = Counter()
        self.tok_out: Counter = Counter()

    def add(self, family: str, usage: Dict[str, int]) -> None:
        with self._lock:
            self.calls[family] += 1
            self.tok_in[family] += int(usage.get("prompt_tokens", 0))
            self.tok_out[family] += int(usage.get("completion_tokens", 0))

    def summary(self, specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        price = {s["family"]: (s["price_in"], s["price_out"]) for s in specs}
        out: Dict[str, Any] = {"by_family": {}, "total_usd": 0.0, "total_calls": 0}
        with self._lock:
            for fam in set(list(self.calls) + list(price)):
                pin, pout = price.get(fam, (0.0, 0.0))
                usd = self.tok_in[fam] / 1e6 * pin + self.tok_out[fam] / 1e6 * pout
                out["by_family"][fam] = {
                    "calls": self.calls[fam], "tok_in": self.tok_in[fam],
                    "tok_out": self.tok_out[fam], "usd": round(usd, 4)}
                out["total_usd"] += usd
                out["total_calls"] += self.calls[fam]
        out["total_usd"] = round(out["total_usd"], 2)
        return out


# --------------------------------------------------------------------------- #
# Best-of-N per (position, tier)
# --------------------------------------------------------------------------- #
def _blinded_judge(client, mined: Dict[str, Any], tier: str,
                   cands: List[Tuple[str, G.GateResult]], cost: Cost,
                   specs: List[Dict[str, Any]], max_retries: int) -> Optional[int]:
    """One blinded cross-family judge call: return the winning index or None.

    The judge family is rotated by hash so no single lab dominates; candidate
    authorship is hidden (labelled A/B/…), which removes self-preference.
    """
    key = f"{mined['id']}|{tier}"
    # Rotate the judge among the CHEAP cross-family models (gpt/gemini), never the
    # priciest (claude), so the light best-of-N tiebreak stays cheap + fair
    # (blinded => no self-preference). Falls back to any spec if none are present.
    judge_pool = [s for s in specs if s["family"] in ("gpt", "gemini")] or specs
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    judge = judge_pool[h % len(judge_pool)]
    letters = [chr(ord("A") + i) for i in range(len(cands))]
    blocks = []
    for L, (_fam, gr) in zip(letters, cands):
        blocks.append(f"[{L}]\n{gr.target}")
    sys_p = ("You are a strict chess-coaching editor. Pick the SINGLE most "
             "instructive explanation for a " + tier + " student: it must be "
             "correct, grounded in the position, teach a reusable method, and end "
             "with a named transferable principle. Reply with ONLY the letter.")
    usr = ("Position (FEN): " + mined["fen"] + "\nMove being taught: "
           + mined["tier_picks"][tier]["san"] + "\n\nCandidates:\n\n"
           + "\n\n".join(blocks) + "\n\nBest candidate letter:")
    chat = TFYChat(client, model_id=judge["ident"], max_tokens=2000,
                   max_retries=max_retries, limiter=RateLimiter(0.02),
                   reasoning_effort=judge["effort"])
    try:
        txt, usage = chat.complete(sys_p, usr)
        cost.add(judge["family"], usage)
    except Exception as exc:  # noqa: BLE001
        log.debug("judge failed (%s); falling back to score", exc)
        return None
    for i, L in enumerate(letters):
        if L.lower() in txt.strip().lower()[:4]:
            return i
    return None


def process_example(
    mined: Dict[str, Any], tier: str, clients: Dict[str, Any],
    specs: List[Dict[str, Any]], cost: Cost, *, max_retries: int,
    use_judge: bool, judge_gap: float,
) -> Dict[str, Any]:
    """Generate (3 teachers) -> gate -> best-of-N for one (position, tier)."""
    fen = mined["fen"]
    canonical_uci = mined["tier_picks"][tier]["uci"]
    sound_ucis = [m["uci"] for m in mined["sound_pool"]]
    system = build_teacher_system(tier)
    user = build_teacher_user(mined, tier)

    raw_records: List[Dict[str, Any]] = []
    gated: List[Tuple[str, G.GateResult, Dict[str, Any]]] = []  # (family, gr, to)
    reason_hist: Counter = Counter()

    def _run_teacher(spec: Dict[str, Any]) -> Dict[str, Any]:
        """One teacher's independent generation + first-pass gate (no augment)."""
        fam = spec["family"]
        chat = TFYChat(clients[fam], model_id=spec["ident"], max_tokens=3000,
                       max_retries=max_retries, limiter=RateLimiter(0.02),
                       reasoning_effort=spec["effort"])
        rec: Dict[str, Any] = {"id": mined["id"], "tier": tier, "family": fam}
        try:
            txt, usage = chat.complete(system, user)
            cost.add(fam, usage)
            parsed = _parse_teacher_json(txt)
            to = _normalize_output(parsed, fen, tier)
            rec["output"] = to
        except Exception as exc:  # noqa: BLE001 - a bad teacher must not kill the row
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            return {"spec": spec, "rec": rec, "gr": None, "to": None}
        gr = G.gate_candidate(to, fen, tier, canonical_uci, sound_ucis,
                              allow_augment=False)
        rec["gate_ok"] = gr.ok
        rec["gate_reasons"] = gr.reasons
        rec["native_principle"] = gr.native_principle
        return {"spec": spec, "rec": rec, "gr": gr, "to": to}

    # The three teachers write INDEPENDENTLY and in parallel (no cross-talk).
    with ThreadPoolExecutor(max_workers=len(specs)) as tpool:
        outs = list(tpool.map(_run_teacher, specs))
    for o in outs:  # aggregate deterministically in spec order
        raw_records.append(o["rec"])
        if o["gr"] is None:
            reason_hist[f"gen_error:{o['spec']['family']}"] += 1
            continue
        if o["gr"].ok:
            gated.append((o["spec"]["family"], o["gr"], o["to"]))
        else:
            for r in o["gr"].reasons:
                reason_hist[r] += 1

    # Best-of-N selection among survivors.
    winner: Optional[Dict[str, Any]] = None
    winner_family: Optional[str] = None
    selection = "none"

    if not gated:
        # Last resort: allow a sanctioned principle-splice IF the only failure was
        # "no_principle_in_takeaway" (keeps a correct, grounded label instead of
        # discarding the position). Any other failure is a real reject.
        salvage: List[Tuple[str, G.GateResult, Dict[str, Any]]] = []
        for spec in specs:
            rr = next((r for r in raw_records
                       if r["family"] == spec["family"] and "output" in r), None)
            if rr is None:
                continue
            gr2 = G.gate_candidate(rr["output"], fen, tier, canonical_uci,
                                   sound_ucis, allow_augment=True)
            if gr2.ok:
                salvage.append((spec["family"], gr2, rr["output"]))
        gated = salvage
        if gated:
            selection = "augmented_salvage"

    if len(gated) == 1:
        winner_family, gr, to = gated[0]
        winner = {"gr": gr, "to": to}
        selection = selection if selection == "augmented_salvage" else "sole_survivor"
    elif len(gated) >= 2:
        gated.sort(key=lambda x: x[1].score, reverse=True)
        top_i = 0
        if use_judge and (gated[0][1].score - gated[1][1].score) <= judge_gap:
            ji = _blinded_judge(clients[specs[0]["family"]], mined, tier,
                                [(f, gr) for f, gr, _ in gated], cost, specs,
                                max_retries)
            if ji is not None:
                top_i = ji
                selection = "judge"
            else:
                selection = "score"
        else:
            selection = "score"
        winner_family, gr, to = gated[top_i]
        winner = {"gr": gr, "to": to}

    result: Dict[str, Any] = {
        "id": mined["id"], "tier": tier, "raw": raw_records,
        "reason_hist": dict(reason_hist), "n_gated": len(gated),
        "selection": selection, "winner_family": winner_family,
    }
    if winner is not None:
        gr = winner["gr"]
        result["winner"] = {
            "fen": fen,
            "family": winner_family, "selection": selection,
            "target": gr.target, "score": gr.score, "words": gr.words,
            "native_principle": gr.native_principle, "augmented": gr.augmented,
            "recommended_uci": canonical_uci,
            "recommended_san": mined["tier_picks"][tier]["san"],
            "board_class": mined["board_class"], "motif": mined["motif"],
            "bucket": mined["bucket"], "rating": mined["rating"],
            "is_engine_best": mined["tier_picks"][tier]["is_engine_best"],
        }
    return result


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _load_mined() -> List[Dict[str, Any]]:
    if not MINED_OUT.exists():
        raise SystemExit(f"BLOCKED: no mined positions at {MINED_OUT}; run mine first")
    rows = []
    with MINED_OUT.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _done_keys(path: Path) -> set:
    done: set = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                done.add((str(o.get("id")), str(o.get("tier"))))
            except json.JSONDecodeError:
                continue
    return done


def cmd_label(args: argparse.Namespace) -> int:
    load_dotenv(settings.ROOT / ".env")
    specs = _teacher_specs()
    mined = _load_mined()
    if args.limit:
        mined = mined[: args.limit]

    CURATE_DIR.mkdir(parents=True, exist_ok=True)
    fresh = args.fresh or args.smoke
    done = set() if fresh else _done_keys(LABELED_OUT)
    tasks: List[Tuple[Dict[str, Any], str]] = []
    for m in mined:
        for tier in TIER_ORDER:
            if (str(m["id"]), tier) not in done:
                tasks.append((m, tier))
    if args.smoke:
        tasks = tasks[: args.smoke_tasks]
    if not tasks:
        log.info("nothing to label (all done)")
        return 0

    # One shared OpenAI-compatible client per family (thread-safe; reused).
    clients = {s["family"]: make_tfy_client(args.timeout) for s in specs}
    cost = Cost()

    log.info("labeling %d (position,tier) tasks across %d teachers, concurrency=%d",
             len(tasks), len(specs), args.concurrency)

    labeled_mode = "w" if fresh else "a"
    raw_mode = "w" if fresh else "a"
    out_lock = threading.Lock()
    t0 = time.time()
    n_done = n_win = 0
    sel_hist: Counter = Counter()
    win_family: Counter = Counter()
    reason_total: Counter = Counter()
    native_by_family: Counter = Counter()
    gen_by_family: Counter = Counter()

    labeled_fh = LABELED_OUT.open(labeled_mode, encoding="utf-8")
    raw_fh = RAW_OUT.open(raw_mode, encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {
                pool.submit(process_example, m, tier, clients, specs, cost,
                            max_retries=args.max_retries, use_judge=args.judge,
                            judge_gap=args.judge_gap): (m, tier)
                for (m, tier) in tasks
            }
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log.error("example failed: %s", exc)
                    continue
                n_done += 1
                for r in res["reason_hist"].items():
                    reason_total[r[0]] += r[1]
                for rec in res["raw"]:
                    if "output" in rec:
                        gen_by_family[rec["family"]] += 1
                        if rec.get("native_principle"):
                            native_by_family[rec["family"]] += 1
                sel_hist[res["selection"]] += 1
                with out_lock:
                    for rec in res["raw"]:
                        raw_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if res.get("winner"):
                        n_win += 1
                        win_family[res["winner_family"]] += 1
                        row = {"id": res["id"], "tier": res["tier"], **res["winner"]}
                        labeled_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    raw_fh.flush()
                    labeled_fh.flush()
                if n_done % 25 == 0:
                    rate = 100.0 * n_win / max(1, n_done)
                    log.info("  %d/%d tasks, kept %d (%.0f%%), $%.2f, %.0fs",
                             n_done, len(tasks), n_win, rate,
                             cost.summary(specs)["total_usd"], time.time() - t0)
    finally:
        labeled_fh.close()
        raw_fh.close()

    csum = cost.summary(specs)
    summary = {
        "tasks_this_run": len(tasks), "completed": n_done, "kept": n_win,
        "keep_rate_pct": round(100.0 * n_win / max(1, n_done), 2),
        "selection": dict(sel_hist), "winner_by_family": dict(win_family),
        "gen_by_family": dict(gen_by_family),
        "native_principle_by_family": dict(native_by_family),
        "reject_reasons": dict(reason_total.most_common()),
        "cost": csum, "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2))
    print(f"\nlabeled winners -> {LABELED_OUT}\nraw audit -> {RAW_OUT}")
    (CURATE_DIR / "label_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- #
# resummarize: rebuild label_run_summary.json from the on-disk audit trail.
# Lets a checkpoint (or an early-stopped run) be finalized with real gate/quality
# stats; total cost is read from the run log (per-family cost only exists in a
# fully-completed run's own summary, so it is preserved if already present).
# --------------------------------------------------------------------------- #
def cmd_resummarize(args: argparse.Namespace) -> int:
    raw: List[Dict[str, Any]] = []
    if RAW_OUT.exists():
        with RAW_OUT.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        raw.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    winners: List[Dict[str, Any]] = []
    if LABELED_OUT.exists():
        with LABELED_OUT.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        winners.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    tasks = {(r.get("id"), r.get("tier")) for r in raw}
    gen_by_family: Counter = Counter()
    native_by_family: Counter = Counter()
    reason_total: Counter = Counter()
    for r in raw:
        fam = r.get("family")
        if "output" in r:
            gen_by_family[fam] += 1
            if r.get("native_principle"):
                native_by_family[fam] += 1
        else:
            reason_total[f"gen_error:{fam}"] += 1
        if r.get("gate_ok") is False:
            for reason in r.get("gate_reasons", []):
                reason_total[reason] += 1
    win_family = Counter(w.get("family") for w in winners)
    sel = Counter(w.get("selection") for w in winners)

    # Total cost from the run log (cumulative "$X.XX" on the last progress line).
    total_usd = None
    logp = settings.DATA / "curate_label.log"
    if logp.exists():
        import re
        vals = re.findall(r"\$([0-9]+\.[0-9]+)", logp.read_text(encoding="utf-8"))
        if vals:
            total_usd = float(vals[-1])

    prior = _read_json(CURATE_DIR / "label_run_summary.json") or {}
    cost = prior.get("cost") if isinstance(prior.get("cost"), dict) else {}
    if total_usd is not None:
        cost = dict(cost or {})
        cost["total_usd"] = total_usd
        cost.setdefault("note", "total from run log; per-family from last completed summary if present")

    summary = {
        "tasks_this_run": len(tasks), "completed": len(tasks), "kept": len(winners),
        "keep_rate_pct": round(100.0 * len(winners) / max(1, len(tasks)), 2),
        "selection": dict(sel), "winner_by_family": dict(win_family),
        "gen_by_family": dict(gen_by_family),
        "native_principle_by_family": dict(native_by_family),
        "reject_reasons": dict(reason_total.most_common()),
        "cost": cost, "note": "reconstructed from label_raw.jsonl + labeled.jsonl",
    }
    (CURATE_DIR / "label_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# build: labeled.jsonl -> train/valid + manifest
# --------------------------------------------------------------------------- #
def _base_id(example_id: str) -> str:
    return str(example_id).split("#", 1)[0]


def cmd_build(args: argparse.Namespace) -> int:
    if not LABELED_OUT.exists():
        raise SystemExit(f"BLOCKED: no labeled winners at {LABELED_OUT}")
    system_prompt = load_system_prompt()
    mined = {str(m["id"]): m for m in _load_mined()}

    winners: List[Dict[str, Any]] = []
    with LABELED_OUT.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                winners.append(json.loads(line))

    # Dedup by (id, tier); build chat rows grouped by base position id.
    seen: set = set()
    by_base: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
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
                "id": w["id"], "base_id": _base_id(w["id"]), "tier": w["tier"],
                "fen": m["fen"], "rec_uci": w["recommended_uci"],
                "rec_san": w["recommended_san"], "winner_family": w["family"],
                "selection": w["selection"], "board_class": w.get("board_class"),
                "motif": w.get("motif"), "bucket": w.get("bucket"),
                "is_engine_best": w.get("is_engine_best"),
                "augmented": w.get("augmented", False),
                "discriminating": (w["tier"] == "beginner" and not w.get("is_engine_best", True)),
            },
        }
        by_base[_base_id(w["id"])].append(row)

    base_ids = sorted(by_base)
    rng = random.Random(SEED)
    rng.shuffle(base_ids)
    n_valid = max(1, int(len(base_ids) * args.valid_frac))
    valid_bases = set(base_ids[:n_valid])

    train_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    for bid in base_ids:
        (valid_rows if bid in valid_bases else train_rows).extend(by_base[bid])
    rng.shuffle(train_rows)

    def _clean(r): return {"messages": r["messages"]}
    _write_jsonl([_clean(r) for r in train_rows], TRAIN_OUT)
    _write_jsonl([_clean(r) for r in valid_rows], VALID_OUT)

    all_rows = train_rows + valid_rows
    tier_ct = Counter(r["_meta"]["tier"] for r in all_rows)
    train_tier = Counter(r["_meta"]["tier"] for r in train_rows)
    valid_tier = Counter(r["_meta"]["tier"] for r in valid_rows)
    winfam = Counter(r["_meta"]["winner_family"] for r in all_rows)
    cls_ct = Counter(r["_meta"].get("board_class") for r in all_rows)
    motif_ct = Counter(r["_meta"].get("motif") for r in all_rows)
    aug = sum(1 for r in all_rows if r["_meta"].get("augmented"))
    disc = sum(1 for r in all_rows if r["_meta"].get("discriminating"))

    # distinct positions actually covered at all three tiers (full triads)
    tiers_per_base: Dict[str, set] = defaultdict(set)
    for r in all_rows:
        tiers_per_base[r["_meta"]["base_id"]].add(r["_meta"]["tier"])
    full_triads = sum(1 for s in tiers_per_base.values() if len(s) == 3)

    mine_meta = _read_json(CURATE_DIR / "mine_meta.json")
    sample_meta = _read_json(settings.DATA / "bank" / "puzzle_sample_meta.json")
    label_summary = _read_json(CURATE_DIR / "label_run_summary.json")

    manifest = {
        "name": "chess-coach-curated-32b",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "purpose": ("Highest-quality curated SFT set of DISCRIMINATING multi-tier "
                    "chess-coaching positions (grounded + cross-family best-of-N + "
                    "gate-verified), ready for the next 32B QLoRA."),
        "pipeline": {
            "mine": "Stockfish sound pool + Maia per-tier policy + tier_select; "
                    "kept iff beginner_pick != advanced_pick (the moat).",
            "label": "independent cross-family teachers (gpt-5.5 / claude-opus-4-8 / "
                     "gemini-3.1-pro), gated, best-of-N by instructiveness"
                     + (" (+ blinded judge tiebreak)" if args.note_judge else ""),
            "gates": ["tier_move_match", "soundness", "legality",
                      "no_engine_speak", "ply_cap", "well_formed",
                      "named_principle_in_takeaway(v5 vocab)",
                      "no_rejected_heuristic(principle_library_v5)",
                      "faithfulness_ext(0 fabrication)"],
        },
        "counts": {
            "mined_discriminating": mine_meta.get("kept_total"),
            "label_tasks": label_summary.get("completed"),
            "kept_after_gates": label_summary.get("kept"),
            "gate_keep_rate_pct": label_summary.get("keep_rate_pct"),
            "train_rows": len(train_rows), "valid_rows": len(valid_rows),
            "total_rows": len(all_rows),
            "distinct_positions": len(base_ids),
            "full_triads_all_3_tiers": full_triads,
            "augmented_takeaway": aug,
            "beginner_discriminating_rows": disc,
        },
        "balance": {
            "by_tier": dict(tier_ct), "train_by_tier": dict(train_tier),
            "valid_by_tier": dict(valid_tier),
            "distinct_tier_pct": round(100.0 * full_triads / max(1, len(base_ids)), 1),
            "winner_by_family": dict(winfam),
            "by_board_class": dict(cls_ct.most_common()),
            "by_motif": dict(motif_ct.most_common()),
        },
        "gate_and_selection": {
            "reject_reasons": (label_summary or {}).get("reject_reasons"),
            "selection": (label_summary or {}).get("selection"),
            "native_principle_by_family": (label_summary or {}).get("native_principle_by_family"),
        },
        "teacher_cost_tfy": (label_summary or {}).get("cost"),
        "sample_meta": sample_meta, "mine_meta": mine_meta,
        "seed": SEED, "valid_frac": args.valid_frac,
        "outputs": {"train": str(TRAIN_OUT), "valid": str(VALID_OUT)},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote train -> {TRAIN_OUT} ({len(train_rows)} rows)")
    print(f"wrote valid -> {VALID_OUT} ({len(valid_rows)} rows)")
    print(f"wrote manifest -> {MANIFEST}")
    return 0


def _write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("label", help="Cross-family gen + gate + best-of-N.")
    pl.add_argument("--limit", type=int, default=None,
                    help="Only the first N mined positions.")
    pl.add_argument("--smoke", action="store_true",
                    help="Fresh tiny run (few tasks) + full summary.")
    pl.add_argument("--smoke-tasks", type=int, default=6)
    pl.add_argument("--concurrency", type=int, default=6)
    pl.add_argument("--max-retries", type=int, default=4)
    pl.add_argument("--timeout", type=float, default=120.0)
    pl.add_argument("--judge", action="store_true", default=True,
                    help="Blinded cross-family judge tiebreak when scores are close.")
    pl.add_argument("--no-judge", dest="judge", action="store_false")
    pl.add_argument("--judge-gap", type=float, default=1.5,
                    help="Only judge when top-2 instructiveness scores are within this.")
    pl.add_argument("--fresh", action="store_true")
    pl.set_defaults(func=cmd_label)

    pb = sub.add_parser("build", help="Assemble train/valid + manifest.")
    pb.add_argument("--valid-frac", type=float, default=0.06)
    pb.add_argument("--note-judge", action="store_true", default=True)
    pb.set_defaults(func=cmd_build)

    pr = sub.add_parser("resummarize",
                        help="Rebuild label_run_summary.json from the audit trail.")
    pr.set_defaults(func=cmd_resummarize)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
