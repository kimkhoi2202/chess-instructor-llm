"""v2 chess-coaching dataset generator — tier-aware, grounded, method-teaching.

This is the v2 upgrade of ``src/teacher/generate.py``. It regenerates the
coaching labels to fix the two measured v1 gaps (see ``RESULTS.md`` and
``data/analysis/DIVERGENCE_REPORT.md``) and to add the v2 refinement (teach the
METHOD, not just the answer). Concretely, per job it:

1. Reuses the coachable positions + Stockfish sound pools already computed in
   ``candidates_v1.jsonl`` (no re-run of Stockfish) and computes fresh **Maia**
   per-tier policy for the sound pool.
2. **Deterministically selects the teaching move per tier**
   (:mod:`src.teacher.tier_select`): beginner -> most human-findable sound move,
   advanced -> sharpest sound move (engine best), intermediate -> a blend. This
   fixes the "differentiation is weak and mis-directed" finding.
3. Builds a **grounded** teacher prompt (``render_pool_facts`` VERIFIED FACTS +
   the shared user prompt) and a **forced-move** directive, so GPT-5.5 explains
   the pre-selected move truthfully.
4. Requires a **3-part** coaching output — (a) the move, (b) WHY (concepts), and
   (c) the explicit METHOD (how a player at this tier should THINK to FIND it).
5. Emits BOTH single-tier rows (coverage) and **contrastive triples** — the SAME
   position taught at all three tiers — which is the supervised signal for
   "same position -> different move/method by tier" that v1 had 0% of.

Robustness: a held-out **reserve** of positions is excluded so the divergence +
benchmark evals stay clean; append-only **checkpoint/resume** by job id; a persisted
**cost ledger** with a ``--max-cost`` guard; a cached **plan** so a restart skips
the (local) Stockfish/Maia analysis.

Nothing here touches v1 artifacts, the running servers, or ``web/src``. All output
is v2-suffixed.

CLI
---
    python -m src.teacher.generate_v2 plan                 # build + cache the plan (no spend)
    python -m src.teacher.generate_v2 smoke                # 3 real generations (measures cost)
    python -m src.teacher.generate_v2 generate             # run all pending jobs (resumable)
    python -m src.teacher.generate_v2 generate --max-cost 120
"""

from __future__ import annotations

import argparse
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess
from dotenv import load_dotenv
from openai import OpenAI

from config import schema, settings
from src.engine.faithfulness import verify_text
from src.engine.position_facts import render_pool_facts
from src.teacher import tier_select
from src.teacher.generate import (
    RateLimiter,
    TeacherClient,
    _extract_tier_guide,
    _load_principles,
    _read_text,
)

log = logging.getLogger("teacher.generate_v2")

TIER_ORDER: Tuple[str, ...] = ("beginner", "intermediate", "advanced")
CORE_SEVERITIES: Tuple[str, ...] = ("inaccuracy", "mistake", "blunder")

# --- Paths (all v2-suffixed) ----------------------------------------------- #
V1_CANDIDATES = settings.GENERATED / "candidates_v1.jsonl"
V2_CANDIDATES = settings.GENERATED / "candidates_v2.jsonl"
PLAN_PATH = settings.GENERATED / "plan_v2.jsonl"
COST_PATH = settings.GENERATED / "cost_v2.json"
RESERVE_PATH = settings.DATA / "analysis" / "heldout_v2.json"
TRAIN_V1 = settings.DATASET / "train.jsonl"
VALID_V1 = settings.DATASET / "valid.jsonl"

SEED = 3407


# --------------------------------------------------------------------------- #
# Held-out reserve (kept OUT of v2 training so evals are clean for both models)
# --------------------------------------------------------------------------- #


def board_key(fen: str) -> Optional[str]:
    """Placement + side-to-move key (matches the benchmark + divergence harness)."""
    try:
        b = chess.Board(fen)
    except ValueError:
        return None
    return f"{b.board_fen()} {'w' if b.turn else 'b'}"


def _placement_from_ascii(user_content: str) -> Optional[str]:
    """Reconstruct the placement FEN from a rendered ``Board:`` ASCII grid."""
    lines = user_content.splitlines()
    grid: List[str] = []
    for ln in lines:
        toks = ln.strip().split()
        if len(toks) == 8 and all(t == "." or (len(t) == 1 and t.isalpha()) for t in toks):
            grid.append("".join(toks))
        elif grid:
            break
    if len(grid) != 8:
        return None
    rows: List[str] = []
    for row in grid:
        out, empties = "", 0
        for ch in row:
            if ch == ".":
                empties += 1
            else:
                if empties:
                    out += str(empties)
                    empties = 0
                out += ch
        if empties:
            out += str(empties)
        rows.append(out)
    return "/".join(rows)


def train_board_keys(paths: Sequence[Path]) -> set:
    """Every placement+turn key present in the given chat-format JSONL files."""
    keys: set = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                user = ""
                for msg in row.get("messages", []):
                    if msg.get("role") == "user":
                        user = str(msg.get("content", ""))
                        break
                placement = _placement_from_ascii(user)
                turn = "w" if "White to move" in user else ("b" if "Black to move" in user else None)
                if placement and turn:
                    keys.add(f"{placement} {turn}")
    return keys


def _phase(fen: str) -> str:
    board = fen.split(" ", 1)[0]
    pieces = sum(1 for c in board if c.isalpha())
    if pieces >= 26:
        return "opening"
    if pieces >= 12:
        return "middlegame"
    return "endgame"


def load_coachable() -> List[Dict[str, Any]]:
    """Load coachable records from candidates_v1 (fen/tier/student/pool), deduped by FEN."""
    out: List[Dict[str, Any]] = []
    seen: set = set()
    with V1_CANDIDATES.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ti = d.get("teacher_input") or {}
            fen = ti.get("fen")
            pool = ti.get("sound_pool") or []
            sm = ti.get("student_move") or {}
            if not fen or fen in seen or not pool:
                continue
            seen.add(fen)
            out.append(
                {
                    "id": str(d.get("id")),
                    "fen": fen,
                    "tier": d.get("tier") or ti.get("tier"),
                    "student_move": sm,
                    "sound_pool": pool,
                    "severity": sm.get("severity"),
                    "phase": _phase(fen),
                }
            )
    return out


def _balanced_pick(
    records: List[Dict[str, Any]], n: int, seed: int
) -> List[Dict[str, Any]]:
    """Round-robin over (phase, severity) buckets for a balanced sub-sample."""
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("severity") in CORE_SEVERITIES:
            buckets[(r["phase"], r["severity"])].append(r)
    for b in buckets.values():
        rng.shuffle(b)
    order = [(ph, sev) for ph in ("opening", "middlegame", "endgame") for sev in CORE_SEVERITIES]
    picked: List[Dict[str, Any]] = []
    i = 0
    while len(picked) < n and any(buckets[k] for k in order):
        k = order[i % len(order)]
        if buckets[k]:
            picked.append(buckets[k].pop())
        i += 1
    return picked


def compute_reserve(coachable: List[Dict[str, Any]], n: int) -> Dict[str, Any]:
    """Reserve ~n balanced held-out positions (excluded from v1 train, and from v2)."""
    heldin_v1 = train_board_keys([TRAIN_V1, VALID_V1])
    pool = [r for r in coachable if board_key(r["fen"]) not in heldin_v1]
    reserve = _balanced_pick(pool, n, SEED)
    keys = sorted({board_key(r["fen"]) for r in reserve if board_key(r["fen"])})
    payload = {
        "n": len(reserve),
        "board_keys": keys,
        "records": [
            {"id": r["id"], "fen": r["fen"], "tier": r["tier"], "phase": r["phase"],
             "severity": r["severity"]}
            for r in reserve
        ],
        "note": "Held-out for BOTH v1 and v2: excluded from v1 train and never put "
                "into candidates_v2/train_v2. Used by the divergence + benchmark evals.",
    }
    RESERVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESERVE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------- #
# Tier-aware selection (uses fresh Maia over the stored sound pool)
# --------------------------------------------------------------------------- #


def _maia_for(fen: str, tier: str) -> List[Dict[str, Any]]:
    """Maia moves (uci/san/policy) for ``tier``, large top_k so the whole pool is scored."""
    from src.engine import maia_engine

    try:
        return list(maia_engine.human_moves(fen, tier, top_k=64)["moves"])
    except Exception as exc:  # noqa: BLE001 - Maia is a helpful signal, not required
        log.warning("Maia unavailable for %s/%s: %s", fen[:24], tier, exc)
        return []


def _pick_for(fen: str, tier: str, pool: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return (pick_dict, maia_top6) for one (fen, tier)."""
    maia = _maia_for(fen, tier)
    policy = {m["uci"]: float(m["policy"]) for m in maia}
    pick = tier_select.select_tier_move(tier, pool, policy)
    maia6 = [{"uci": m["uci"], "san": m["san"], "policy": float(m["policy"])} for m in maia[:6]]
    return (
        {
            "uci": pick.uci, "san": pick.san, "pool_rank": pick.pool_rank,
            "policy": pick.policy, "is_engine_best": pick.is_engine_best,
            "weight": pick.weight,
        },
        maia6,
    )


def _engine_block(pool: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "best_san": pool[0]["san"],
        "best_cp": int(pool[0]["cp"]),
        "sound_ucis": [m["uci"] for m in pool],
    }


def _job(kind: str, rec: Dict[str, Any], tier: str, pick: Dict[str, Any],
         maia6: List[Dict[str, Any]]) -> Dict[str, Any]:
    base_id = rec["id"]
    job_id = base_id if kind == "single" else f"{base_id}#ctr-{tier}"
    return {
        "job_id": job_id,
        "base_id": base_id,
        "kind": kind,
        "fen": rec["fen"],
        "tier": tier,
        "phase": rec["phase"],
        "severity": rec.get("severity"),
        "student_move": rec["student_move"],
        "sound_pool": rec["sound_pool"],
        "engine": _engine_block(rec["sound_pool"]),
        "maia6": maia6,
        "pick": pick,
    }


# --------------------------------------------------------------------------- #
# Plan building (contrastive selection + single-tier), cached to disk
# --------------------------------------------------------------------------- #


def build_plan(
    *, n_contrastive: int, reserve_n: int, single_limit: Optional[int]
) -> List[Dict[str, Any]]:
    """Build (and persist) the full job plan. Local only — no teacher calls."""
    coachable = load_coachable()
    log.info("coachable positions (from candidates_v1): %d", len(coachable))

    reserve = compute_reserve(coachable, reserve_n)
    reserve_keys = set(reserve["board_keys"])
    log.info("reserved held-out: %d positions -> %s", reserve["n"], RESERVE_PATH)

    pool_records = [r for r in coachable if board_key(r["fen"]) not in reserve_keys]
    rng = random.Random(SEED)
    rng.shuffle(pool_records)

    # Contrastive selection: scan positions with >=2 sound moves, compute all-3-tier
    # picks, keep the DIFFERENTIATING ones (>=2 distinct picks), balanced by cell,
    # preferring 3-distinct. This is the "same position -> different move" signal.
    contrastive_jobs: List[Dict[str, Any]] = []
    contrastive_fens: set = set()
    cell_cap: Dict[Tuple[str, str], int] = defaultdict(int)
    per_cell = max(1, n_contrastive // 9 + 2)
    diff_hist = Counter()
    scanned = 0

    multi = [r for r in pool_records if len(r["sound_pool"]) >= 2]
    log.info("scanning up to %d multi-move positions for contrastive triples ...", len(multi))
    for rec in multi:
        if len(contrastive_fens) >= n_contrastive:
            break
        scanned += 1
        picks: Dict[str, Dict[str, Any]] = {}
        maia_by_tier: Dict[str, List[Dict[str, Any]]] = {}
        for tier in TIER_ORDER:
            pick, maia6 = _pick_for(rec["fen"], tier, rec["sound_pool"])
            picks[tier] = pick
            maia_by_tier[tier] = maia6
        distinct = len({picks[t]["uci"] for t in TIER_ORDER})
        diff_hist[distinct] += 1
        cell = (rec["phase"], rec.get("severity"))
        if distinct >= 2 and cell_cap[cell] < per_cell:
            cell_cap[cell] += 1
            contrastive_fens.add(rec["fen"])
            for tier in TIER_ORDER:
                contrastive_jobs.append(_job("contrastive", rec, tier, picks[tier], maia_by_tier[tier]))
        if scanned % 200 == 0:
            log.info("  scanned %d, selected %d contrastive FENs", scanned, len(contrastive_fens))

    log.info("contrastive: %d FENs (%d jobs); distinct-pick histogram over scanned=%s",
             len(contrastive_fens), len(contrastive_jobs), dict(diff_hist))

    # Single-tier jobs: every remaining coachable position at its native tier.
    single_records = [r for r in pool_records if r["fen"] not in contrastive_fens]
    if single_limit is not None:
        single_records = single_records[:single_limit]
    single_jobs: List[Dict[str, Any]] = []
    for i, rec in enumerate(single_records, 1):
        pick, maia6 = _pick_for(rec["fen"], rec["tier"], rec["sound_pool"])
        single_jobs.append(_job("single", rec, rec["tier"], pick, maia6))
        if i % 300 == 0:
            log.info("  single picks computed: %d/%d", i, len(single_records))

    plan = contrastive_jobs + single_jobs
    _write_plan(plan)
    log.info("plan built: %d jobs (%d contrastive + %d single) -> %s",
             len(plan), len(contrastive_jobs), len(single_jobs), PLAN_PATH)
    return plan


def _write_plan(plan: List[Dict[str, Any]]) -> None:
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PLAN_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for job in plan:
            fh.write(json.dumps(job, ensure_ascii=False) + "\n")
    os.replace(tmp, PLAN_PATH)


def load_plan() -> List[Dict[str, Any]]:
    if not PLAN_PATH.exists():
        return []
    out: List[Dict[str, Any]] = []
    with PLAN_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def plan_summary(plan: List[Dict[str, Any]]) -> str:
    kinds = Counter(j["kind"] for j in plan)
    tiers = Counter(j["tier"] for j in plan)
    diff = sum(1 for j in plan if not j["pick"]["is_engine_best"])
    by_tier_rank = defaultdict(list)
    for j in plan:
        by_tier_rank[j["tier"]].append(j["pick"]["pool_rank"])
    ranks = {t: round(sum(v) / len(v), 2) if v else 0 for t, v in by_tier_rank.items()}
    lines = [
        f"jobs: {len(plan)} ({dict(kinds)})",
        f"by tier: {dict(tiers)}",
        f"picks that are NOT the engine best: {diff} ({100*diff/max(1,len(plan)):.0f}%)",
        f"mean pick pool-rank by tier (want beginner > advanced): {ranks}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Teacher generation for one job (forced move, 3-part method, faithfulness retry)
# --------------------------------------------------------------------------- #


def build_system_prompt_v2(tier: str) -> str:
    """v2 teacher system prompt (teacher_system_v2.md) with TIER_GUIDE + PRINCIPLES."""
    template = _read_text(str(settings.PROMPTS / "teacher_system_v2.md"))
    guide = _extract_tier_guide(_read_text(str(settings.PROMPTS / "tier_guides.md")), tier)
    principles = _load_principles()
    return (
        template.replace("{TIER_GUIDE}", guide or "(none provided)")
        .replace("{PRINCIPLES}", principles or "(none provided)")
    )


def _forced_directive(tier: str, pick_san: str, pick_uci: str) -> str:
    return (
        f"\n\nPRE-SELECTED TEACHING MOVE for this {tier} player: {pick_san} ({pick_uci}). "
        f"You MUST recommend exactly this move — set recommended_move_san={pick_san} and "
        f"recommended_move_uci={pick_uci}. Do NOT recommend any other move as the main choice. "
        f"Teach it in three explicit parts: (a) name {pick_san} and address the student's move, "
        f"(b) WHY it is good using {tier}-level concepts, and (c) the METHOD — the concrete "
        f"thinking routine a {tier} player runs to FIND this move themselves. Put part (c) in "
        f"the required 'method' field."
    )


def _build_user_prompt(job: Dict[str, Any]) -> str:
    ti: schema.TeacherInput = {
        "tier": job["tier"],
        "fen": job["fen"],
        "move_history_san": None,
        "student_move": job["student_move"],  # type: ignore[typeddict-item]
        "sound_pool": job["sound_pool"],       # type: ignore[typeddict-item]
        "maia_human_moves": job["maia6"],      # type: ignore[typeddict-item]
    }
    facts = render_pool_facts(job["fen"], list(job["sound_pool"]))
    base = f"{facts}\n\n{schema.render_user_prompt(ti)}"
    return base + _forced_directive(job["tier"], job["pick"]["san"], job["pick"]["uci"])


def _coerce_v2(raw: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    """Force the pre-selected move; require coaching + method + takeaway."""
    coaching = str(raw.get("coaching", "") or "").strip()
    method = str(raw.get("method", "") or "").strip()
    takeaway = str(raw.get("takeaway", "") or "").strip()
    if not coaching:
        raise ValueError("empty coaching")
    if not method:
        raise ValueError("empty method")
    if not takeaway:
        raise ValueError("empty takeaway")
    concepts_raw = raw.get("concepts_used", [])
    if isinstance(concepts_raw, str):
        concepts_raw = [concepts_raw]
    concepts = [str(c).strip() for c in (concepts_raw or []) if str(c).strip()]
    return {
        "tier": job["tier"],
        "recommended_move_san": job["pick"]["san"],  # forced (deterministic)
        "recommended_move_uci": job["pick"]["uci"],
        "coaching": coaching,
        "method": method,
        "takeaway": takeaway,
        "concepts_used": concepts,
    }


def generate_one(job: Dict[str, Any], teacher: TeacherClient) -> Dict[str, Any]:
    """Generate the v2 candidate row for one job (1 faithfulness retry).

    Thread-safe: all state is local (no shared globals), so the ThreadPoolExecutor
    can run many jobs at once without cross-talk on the corrective retry note.
    """
    system = build_system_prompt_v2(job["tier"])
    user = _build_user_prompt(job)

    to: Optional[Dict[str, Any]] = None
    retries = 0
    retry_note = ""
    for attempt in range(2):  # first try + one corrective retry
        raw = teacher.complete(system, user + retry_note)
        to = _coerce_v2(raw, job)
        vr = verify_text(f"{to['coaching']}\n{to['method']}\n{to['takeaway']}", job["fen"])
        if vr.ok:
            break
        retries = attempt + 1
        bad = "; ".join(v.reason for v in vr.violations[:3])
        retry_note = (
            f"\n\nYour previous draft stated a board fact that is FALSE for this "
            f"position ({bad}). Rewrite using ONLY the VERIFIED FACTS; if unsure, "
            f"speak about the plan/method instead of a concrete claim."
        )
    assert to is not None
    fabricated_final = not verify_text(
        f"{to['coaching']}\n{to['method']}\n{to['takeaway']}", job["fen"]
    ).ok

    return {
        "id": job["job_id"],
        "tier": job["tier"],
        "teacher_input": {
            "tier": job["tier"],
            "fen": job["fen"],
            "move_history_san": None,
            "student_move": job["student_move"],
            "sound_pool": job["sound_pool"],
            "maia_human_moves": job["maia6"],
        },
        "teacher_output": to,
        "engine": job["engine"],
        "maia_top": job["maia6"],
        "meta": {
            "model": teacher.model,
            "reasoning_effort": teacher.reasoning_effort,
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": "v2_" + job["kind"],
            "base_id": job["base_id"],
            "pick_pool_rank": job["pick"]["pool_rank"],
            "pick_is_engine_best": job["pick"]["is_engine_best"],
            "faith_retries": retries,
            "fabricated_final": fabricated_final,
        },
    }


# --------------------------------------------------------------------------- #
# Cost ledger
# --------------------------------------------------------------------------- #


def write_cost(teacher: TeacherClient, price_in: float, price_out: float,
               extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with teacher._usage_lock:  # noqa: SLF001 - reading our own client's counters
        cin = teacher.prompt_tokens / 1_000_000 * price_in
        cout = teacher.completion_tokens / 1_000_000 * price_out
        payload = {
            "calls": teacher.calls,
            "prompt_tokens": teacher.prompt_tokens,
            "completion_tokens": teacher.completion_tokens,
            "reasoning_tokens": teacher.reasoning_tokens,
            "est_cost_usd": round(cin + cout, 4),
            "price_in_per_m": price_in,
            "price_out_per_m": price_out,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    if extra:
        payload.update(extra)
    COST_PATH.parent.mkdir(parents=True, exist_ok=True)
    COST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _done_ids(path: Path) -> set:
    ids: set = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(str(json.loads(line).get("id")))
            except json.JSONDecodeError:
                continue
    return ids


# --------------------------------------------------------------------------- #
# Generation driver
# --------------------------------------------------------------------------- #


def run_generation(args: argparse.Namespace) -> int:
    load_dotenv(settings.ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not found in env or %s/.env", settings.ROOT)
        return 2
    model = args.model or os.environ.get("TEACHER_MODEL") or settings.TEACHER_MODEL

    plan = load_plan()
    if not plan:
        log.info("no plan found; building it now ...")
        plan = build_plan(n_contrastive=args.contrastive, reserve_n=args.reserve,
                          single_limit=args.single_limit)

    done = _done_ids(V2_CANDIDATES)
    pending = [j for j in plan if j["job_id"] not in done]
    if args.smoke:
        pending = pending[: args.smoke_n]
    log.info("generation: %d pending of %d planned jobs (%d already done)",
             len(pending), len(plan), len(done))
    if not pending:
        log.info("nothing to do — all planned jobs generated.")
        return 0

    client = OpenAI(api_key=api_key, timeout=args.timeout, max_retries=0)
    limiter = RateLimiter(args.min_interval)
    teacher = TeacherClient(client, model=model, reasoning_effort=args.reasoning_effort,
                            max_retries=args.max_retries, limiter=limiter)

    stop = threading.Event()
    write_lock = threading.Lock()
    out_fh = V2_CANDIDATES.open("a", encoding="utf-8")
    written = failed = skipped = 0
    t0 = time.time()

    def worker(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if stop.is_set():
            return None
        return generate_one(job, teacher)

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {pool.submit(worker, j): j for j in pending}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001 - one bad job must not abort
                    failed += 1
                    log.error("job %s failed: %s", job["job_id"], exc)
                    continue
                if row is None:
                    skipped += 1
                    continue
                with write_lock:
                    out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    written += 1
                    if written % 25 == 0:
                        cost = write_cost(teacher, args.price_in, args.price_out,
                                          {"written": written, "failed": failed})
                        log.info("  wrote %d (fail=%d skip=%d) est_cost=$%.2f (%.1fs)",
                                 written, failed, skipped, cost["est_cost_usd"],
                                 time.time() - t0)
                        if args.max_cost and cost["est_cost_usd"] >= args.max_cost:
                            log.warning("MAX-COST $%.2f reached; stopping new work.",
                                        args.max_cost)
                            stop.set()
    finally:
        out_fh.close()

    cost = write_cost(teacher, args.price_in, args.price_out,
                      {"written": written, "failed": failed, "skipped": skipped})
    log.info("done: wrote=%d failed=%d skipped=%d -> %s", written, failed, skipped, V2_CANDIDATES)
    log.info("teacher usage: %s", teacher.usage_summary(args.price_in, args.price_out))
    log.info("cost ledger -> %s (est $%.2f)", COST_PATH, cost["est_cost_usd"])
    if args.smoke and written:
        _print_smoke(pending[:written])
    return 0


def _print_smoke(jobs: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 78 + "\nSMOKE — first generated rows\n" + "=" * 78)
    for line in V2_CANDIDATES.read_text(encoding="utf-8").splitlines()[-len(jobs):]:
        d = json.loads(line)
        to = d["teacher_output"]
        print(f"\n# {d['id']}  tier={d['tier']}  pick_rank={d['meta']['pick_pool_rank']} "
              f"engine_best={d['meta']['pick_is_engine_best']}  retries={d['meta']['faith_retries']}")
        print(f"  MOVE: {to['recommended_move_san']}")
        print(f"  COACH: {to['coaching'][:220]}")
        print(f"  METHOD: {to['method'][:220]}")
        print(f"  TAKEAWAY: {to['takeaway'][:160]}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _add_gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None)
    p.add_argument("--reasoning-effort", dest="reasoning_effort",
                   default=settings.TEACHER_REASONING_EFFORT)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--min-interval", dest="min_interval", type=float, default=0.05)
    p.add_argument("--max-retries", dest="max_retries", type=int, default=4)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--contrastive", type=int, default=450, help="Target contrastive FENs (x3 rows).")
    p.add_argument("--reserve", type=int, default=200, help="Held-out reserve positions.")
    p.add_argument("--single-limit", dest="single_limit", type=int, default=None,
                   help="Cap single-tier jobs (cost control).")
    p.add_argument("--max-cost", dest="max_cost", type=float, default=None,
                   help="Stop submitting new work once est cost reaches this USD.")
    p.add_argument("--price-in", dest="price_in", type=float, default=1.25)
    p.add_argument("--price-out", dest="price_out", type=float, default=10.0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v2 tier-aware, grounded, method-teaching generator.")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("plan", help="Build + cache the plan (local; no teacher calls).")
    pp.add_argument("--contrastive", type=int, default=450)
    pp.add_argument("--reserve", type=int, default=200)
    pp.add_argument("--single-limit", dest="single_limit", type=int, default=None)
    pp.add_argument("--price-in", dest="price_in", type=float, default=1.25)
    pp.add_argument("--price-out", dest="price_out", type=float, default=10.0)
    pp.add_argument("--est-in", type=int, default=2200, help="Est input tokens/call for cost preview.")
    pp.add_argument("--est-out", type=int, default=1400, help="Est output tokens/call for cost preview.")
    pp.set_defaults(func=cmd_plan)

    pg = sub.add_parser("generate", help="Run all pending jobs (resumable, costed).")
    _add_gen_args(pg)
    pg.set_defaults(func=lambda a: run_generation(_with_smoke(a, False)))

    psk = sub.add_parser("smoke", help="Generate a few real rows to measure cost/quality.")
    _add_gen_args(psk)
    psk.add_argument("--n", dest="smoke_n", type=int, default=3)
    psk.set_defaults(func=lambda a: run_generation(_with_smoke(a, True)))

    return p


def _with_smoke(args: argparse.Namespace, smoke: bool) -> argparse.Namespace:
    args.smoke = smoke
    if not hasattr(args, "smoke_n"):
        args.smoke_n = 3
    return args


def cmd_plan(args: argparse.Namespace) -> int:
    plan = build_plan(n_contrastive=args.contrastive, reserve_n=args.reserve,
                      single_limit=args.single_limit)
    print("\n=== PLAN ===")
    print(plan_summary(plan))
    est = len(plan) * (args.est_in / 1e6 * args.price_in + args.est_out / 1e6 * args.price_out)
    print(f"\nestimated teacher cost @ ~{args.est_in} in / {args.est_out} out tokens/call: "
          f"${est:.2f} (verify with `smoke`)")
    return 0


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
            from src.engine import maia_engine
            maia_engine.close_all()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
