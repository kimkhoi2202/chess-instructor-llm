#!/usr/bin/env python3
"""Measure THE GAP: do frontier models reliably do the narrow target behavior?

The instructor's framing (the crux): before we can claim training moved OUR
model into a valuable behavior gap, we must first prove the *frontier* models
are **not reliably good** at that narrow behavior. If a prompted GPT-5.5 / Claude
Opus / Gemini already nails it, there is no gap to train into and we would need a
different thesis. This script measures that, honestly, on data.

The target behavior (the "pass" definition)
--------------------------------------------
For a stated ELO tier, recommend the move that is
  (a) SOUND      — inside Stockfish's tolerance pool (doesn't throw the game),
  (b) FINDABLE   — high Maia-likelihood *at that tier* (a move a human at that
                   level would actually consider), not the sharpest engine-only
                   line, and
  (c) INSTRUCTIVE — then coach WHY it's good and HOW a player at that ELO should
                   think to find it.
Serving a beginner the engine-only best move with a GM-level line is a FAIL.

What this script does
---------------------
It takes a balanced, held-out subset of the SAME positions the v1 tuned model was
already evaluated on (``data/analysis/divergence.jsonl`` — every FEN verified
absent from ``train.jsonl``/``valid.jsonl``), re-verifies the held-out property
itself, and for EACH frontier model (GPT-5.5, Claude Opus 4.8, Gemini 3.1 Pro via
the TrueFoundry gateway) generates coaching at ALL THREE tiers using the **exact**
grounded prompt the live app + the v1 divergence run used (``render_pool_facts`` +
``render_user_prompt``, system = ``coach_system.md`` + grounding + format suffix).

Grounding (the Stockfish sound pool + the tier Maia block) is reused verbatim from
``divergence.jsonl`` so every model — frontier and v1 — sees byte-identical input.
Only the model changes. That is what makes the frontier-vs-v1 comparison fair.

Per (position, tier, model) it records the recommended move (extracted with the
live API's own logic, instrumented to separate a genuine named pick from the
``pool[0]`` fallback), plus the move's **Maia rank inside the sound pool at that
tier** (0 = the most human-findable sound move) and a deterministic **faithfulness**
count (sentences making a demonstrably-false board claim, via the repo verifier).

Output: ``data/analysis/frontier_gap.jsonl`` (one rich row per position, all three
frontier models + the v1 pick side-by-side). ``scripts/frontier_gap_report.py``
turns it into ``GAP_REPORT.md`` (rates + the honest verdict).

Run (from repo root, pinned interpreter; secrets come from ROOT/.env)::

    ~/.venvs/mlx/bin/python -m scripts.frontier_gap --num 50
    ~/.venvs/mlx/bin/python -m scripts.frontier_gap --num 50 --limit 2   # smoke

It does NOT load or touch the v1 MLX model, the running servers (8000/3000),
``web/src``, or ``src/eval/benchmark.py``. It only READS ``divergence.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import schema, settings  # noqa: E402
from src.engine import maia_engine  # noqa: E402
from src.engine.faithfulness import verify_text  # noqa: E402
from src.engine.position_facts import render_pool_facts  # noqa: E402

# Reuse the divergence run's EXACT methodology so the comparison is apples-to-apples.
# (Imported read-only; importing this module does not load the MLX model.)
from scripts.divergence_analysis import (  # noqa: E402
    SYSTEM_PROMPT,
    TIER_ORDER,
    _split_coaching,
    _strip_think,
    build_heldin_keys,
    extract_recommended_mode,
    pos_key,
)

CORE_SEVERITIES: Tuple[str, ...] = ("inaccuracy", "mistake", "blunder")
PHASES: Tuple[str, ...] = ("opening", "middlegame", "endgame")

#: Frontier models under test (TrueFoundry gateway ids). Claude: omit temperature.
FRONTIER_MODELS: Dict[str, str] = {
    "gpt-5.5": "openai-group/gpt-5.5",
    "claude-opus-4.8": "claude-group/claude-opus-4-8",
    "gemini-3.1-pro": "gemini-group/gemini-3.1-pro",
}

#: Per-model reasoning effort (empirically required by the gateway, matching the
#: benchmark backend): GPT-5.5 and Gemini otherwise spend the whole token budget
#: reasoning and return empty/truncated content; Claude is clean without it.
REASONING_EFFORT: Dict[str, Optional[str]] = {
    "openai-group/gpt-5.5": "low",
    "gemini-group/gemini-3.1-pro": "low",
    "claude-group/claude-opus-4-8": None,
}

#: Generous headroom so reasoning models are not truncated mid-coaching (their
#: reasoning tokens count toward output). Billed on actual usage, so this is safe.
GEN_MAX_TOKENS: int = 4000


# --------------------------------------------------------------------------- #
# TrueFoundry frontier client (resilient to per-model param quirks)
# --------------------------------------------------------------------------- #


class FrontierClient:
    """OpenAI-compatible TFY client that coaxes a plain-prose reply from any model.

    Different frontier families reject different knobs (reasoning models often
    reject ``temperature``; some want ``max_completion_tokens`` not ``max_tokens``).
    We try a small ladder of kwarg variants and fall back to the Responses API,
    exactly like the teacher client does, so one model's quirk never aborts a run.
    Transient errors (rate-limit / timeout / 5xx / gateway "unpaid invoice") are
    retried with backoff — they are recoverable, not fatal.
    """

    def __init__(self, *, timeout: float, max_retries: int, min_interval: float) -> None:
        from openai import OpenAI

        key = os.environ.get("TFY_API_KEY")
        base = os.environ.get("TFY_BASE_URL")
        if not key or not base:
            raise SystemExit("TFY_API_KEY / TFY_BASE_URL missing (looked in env + ROOT/.env).")
        self._client = OpenAI(api_key=key, base_url=base, timeout=timeout, max_retries=0)
        self.max_retries = max(0, max_retries)
        self._min_interval = max(0.0, min_interval)
        self._rl_lock = threading.Lock()
        self._next_slot = 0.0
        self._usage_lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        with self._rl_lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._min_interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    @staticmethod
    def _is_transient(exc: BaseException) -> bool:
        name = type(exc).__name__.lower()
        if any(k in name for k in ("ratelimit", "timeout", "connection", "internalserver", "apistatus")):
            return True
        blob = f"{getattr(exc, 'status_code', '')} {exc}".lower()
        return any(k in blob for k in ("invoice", "unpaid", "429", "500", "502", "503", "504", "overloaded"))

    def _record_usage(self, resp: Any) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        pt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0
        with self._usage_lock:
            self.calls += 1
            self.prompt_tokens += int(pt)
            self.completion_tokens += int(ct)

    def _kwarg_ladder(self, model: str, system: str, user: str) -> List[Dict[str, Any]]:
        """Kwarg variants tried in order; a param rejection advances to the next.

        Honors "temperature=0 for determinism except Claude" and the gateway's
        per-model ``reasoning_effort`` requirement, while degrading gracefully if
        either param is rejected (reasoning models often reject ``temperature``).
        """
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        common = {"model": model, "messages": msgs}
        temp = {} if "claude" in model.lower() else {"temperature": 0}
        eff = REASONING_EFFORT.get(model)
        effort = {"reasoning_effort": eff} if eff else {}
        return [
            {**common, **temp, **effort, "max_tokens": GEN_MAX_TOKENS},
            {**common, **effort, "max_tokens": GEN_MAX_TOKENS},         # drop temp
            {**common, **temp, "max_tokens": GEN_MAX_TOKENS},           # drop effort
            {**common, "max_tokens": GEN_MAX_TOKENS},                   # drop temp+effort
            {**common, **effort, "max_completion_tokens": GEN_MAX_TOKENS},
            {**common, "max_completion_tokens": GEN_MAX_TOKENS},
            {**common},
        ]

    def _via_responses(self, model: str, system: str, user: str) -> str:
        resp = self._client.responses.create(
            model=model, instructions=system, input=user, max_output_tokens=GEN_MAX_TOKENS
        )
        self._record_usage(resp)
        return resp.output_text or ""

    def complete(self, model: str, system: str, user: str) -> str:
        """Return plain-text coaching from ``model`` (retries transient errors)."""
        import openai

        ladder = self._kwarg_ladder(model, system, user)
        last: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            # Walk the kwarg ladder; a param rejection advances to the next rung.
            for kw in ladder:
                try:
                    resp = self._client.chat.completions.create(**kw)
                    self._record_usage(resp)
                    content = resp.choices[0].message.content or ""
                    if content.strip():
                        return content
                    last = RuntimeError("empty content")
                    break  # empty -> not a param problem; retry whole attempt
                except (openai.BadRequestError, TypeError) as exc:
                    last = exc
                    continue  # unsupported param -> try next rung
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    if not self._is_transient(exc):
                        break  # genuine error for these kwargs -> try the ladder? no, break
                    # transient -> break inner, back off, retry the whole attempt
                    break
            # Try the Responses API once as a last resort on the final attempt.
            if attempt == self.max_retries and last is not None:
                try:
                    txt = self._via_responses(model, system, user)
                    if txt.strip():
                        return txt
                except Exception as exc:  # noqa: BLE001
                    last = exc
            delay = min(2.0 ** attempt, 30.0) + random.uniform(0.0, 1.0)
            print(f"      ~ {model}: {type(last).__name__ if last else '?'} "
                  f"(attempt {attempt + 1}/{self.max_retries + 1}); retrying in {delay:.1f}s",
                  file=sys.stderr)
            time.sleep(delay)
        raise last if last is not None else RuntimeError(f"{model}: all attempts failed")


# --------------------------------------------------------------------------- #
# Maia full-pool ranking (findability)
# --------------------------------------------------------------------------- #


def maia_pool_ranking(fen: str, tier: str, pool: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rank the SOUND POOL by Maia policy at ``tier`` (0 = most human-findable).

    Returns ``{"policy": {uci: p}, "rank": {uci: r}, "order": [uci...]}`` where
    ``rank``/``order`` cover only the sound-pool moves (the moves the coach is
    allowed to recommend). This is the findability yardstick: a tier-appropriate
    pick is a *findable* sound move (low Maia rank), not the sharp engine best.
    """
    board = chess.Board(fen)
    n_legal = board.legal_moves.count()
    try:
        res = maia_engine.human_moves(fen, tier, top_k=max(n_legal, 8))["moves"]
    except Exception as exc:  # noqa: BLE001
        print(f"    ! maia rank failed ({tier}): {exc}", file=sys.stderr)
        res = []
    policy = {m["uci"]: float(m["policy"]) for m in res}
    pool_ucis = [m["uci"] for m in pool]
    # Sort pool moves by Maia policy desc; stable tie-break by uci (matches engine).
    order = sorted(pool_ucis, key=lambda u: (-policy.get(u, 0.0), u))
    rank = {u: i for i, u in enumerate(order)}
    return {"policy": {u: policy.get(u, 0.0) for u in pool_ucis}, "rank": rank, "order": order}


def _pool_rank(pool: List[Dict[str, Any]], uci: Optional[str]) -> Optional[int]:
    """Engine rank of ``uci`` in the sound pool (0 = engine best), or None."""
    if not uci:
        return None
    for i, m in enumerate(pool):
        if m["uci"] == uci:
            return i
    return None


# --------------------------------------------------------------------------- #
# Held-out balanced subsample of divergence.jsonl
# --------------------------------------------------------------------------- #


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def balanced_subsample(rows: List[Dict[str, Any]], num: int, seed: int) -> List[Dict[str, Any]]:
    """Round-robin over (phase x severity) buckets for a balanced held-out set."""
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        sev = r.get("student_move", {}).get("severity")
        ph = r.get("phase")
        if sev in CORE_SEVERITIES and ph in PHASES:
            buckets[(ph, sev)].append(r)
    for b in buckets.values():
        rng.shuffle(b)
    order = [(ph, sev) for ph in PHASES for sev in CORE_SEVERITIES]
    picked: List[Dict[str, Any]] = []
    idx = 0
    while len(picked) < num and any(buckets[k] for k in order):
        k = order[idx % len(order)]
        if buckets[k]:
            picked.append(buckets[k].pop())
        idx += 1
        if idx > len(order) * 100000:
            break
    return picked


# --------------------------------------------------------------------------- #
# Per (model, tier) coaching + scoring
# --------------------------------------------------------------------------- #


def _score_pick(
    rec_uci: Optional[str],
    pool: List[Dict[str, Any]],
    maia_rank: Dict[str, int],
    maia_top_uci: Optional[str],
) -> Dict[str, Any]:
    best_uci = pool[0]["uci"] if pool else None
    return {
        "eq_sf_best": rec_uci == best_uci,
        "eq_maia_top": bool(maia_top_uci and rec_uci == maia_top_uci),
        "engine_pool_rank": _pool_rank(pool, rec_uci),   # 0 = engine best
        "maia_pool_rank": maia_rank.get(rec_uci) if rec_uci else None,  # 0 = most findable
    }


def run_model_tier(
    client: FrontierClient,
    model_id: str,
    fen: str,
    board: chess.Board,
    pool: List[Dict[str, Any]],
    student_uci: str,
    ti: schema.TeacherInput,
    maia_rank: Dict[str, int],
    maia_top_uci: Optional[str],
) -> Dict[str, Any]:
    """One frontier call for one tier: coach, extract move, score findability + faith."""
    facts = render_pool_facts(fen, list(pool))
    user_prompt = f"{facts}\n\n{schema.render_user_prompt(ti)}"
    raw = _strip_think(client.complete(model_id, SYSTEM_PROMPT, user_prompt))
    rec_san, rec_uci, mode = extract_recommended_mode(raw, board, pool, student_uci)
    body, takeaway = _split_coaching(raw)
    faith = verify_text(body, fen)
    scored = _score_pick(rec_uci, pool, maia_rank, maia_top_uci)
    return {
        "rec_san": rec_san,
        "rec_uci": rec_uci,
        "mode": mode,
        "genuine": mode in ("cue", "prose"),
        **scored,
        "faith_violations": len(faith.violations),
        "faith_ok": faith.ok,
        "coaching": body,
        "takeaway": takeaway,
        "raw": raw,
    }


def _v1_pick_record(
    drow: Dict[str, Any], tier: str, pool: List[Dict[str, Any]],
    maia_rank: Dict[str, int], maia_top_uci: Optional[str],
) -> Dict[str, Any]:
    """Re-score the v1 tuned model's stored pick with the SAME findability/faith lens."""
    td = drow["tiers"][tier]
    rec_uci = td.get("rec_uci")
    body = td.get("coaching") or ""
    faith = verify_text(body, drow["fen"])
    scored = _score_pick(rec_uci, pool, maia_rank, maia_top_uci)
    return {
        "rec_san": td.get("rec_san"),
        "rec_uci": rec_uci,
        "mode": td.get("mode"),
        "genuine": td.get("genuine", td.get("mode") in ("cue", "prose")),
        **scored,
        "faith_violations": len(faith.violations),
        "faith_ok": faith.ok,
        "coaching": body,
        "takeaway": td.get("takeaway"),
    }


def analyze_position(
    drow: Dict[str, Any], client: FrontierClient, models: Dict[str, str]
) -> Dict[str, Any]:
    """Build identical grounding, then coach with every frontier model at all tiers."""
    fen = drow["fen"]
    board = chess.Board(fen)
    pool = drow["sound_pool"]                # reuse v1's exact pool (byte-identical grounding)
    student = drow["student_move"]
    student_uci = student.get("uci") or ""

    # Per-tier grounding: Maia top-6 (== what v1 saw) + full-pool findability ranking.
    tier_ground: Dict[str, Any] = {}
    for tier in TIER_ORDER:
        maia_top6 = drow["maia_by_tier"][tier]["moves"]  # stored (deterministic) top-6
        maia_top_uci = (drow["maia_by_tier"][tier].get("top") or {}).get("uci")
        ranking = maia_pool_ranking(fen, tier, pool)
        ti: schema.TeacherInput = {
            "tier": tier,
            "fen": fen,
            "move_history_san": None,
            "student_move": student,
            "sound_pool": pool,
            "maia_human_moves": maia_top6,
        }
        tier_ground[tier] = {
            "ti": ti,
            "maia_top_uci": maia_top_uci,
            "maia_rank": ranking["rank"],
            "maia_policy": ranking["policy"],
            "engine_best_maia_rank": ranking["rank"].get(pool[0]["uci"]) if pool else None,
        }

    # Fan out every (model, tier) call concurrently for this position.
    def _task(model_name: str, model_id: str, tier: str) -> Tuple[str, str, Dict[str, Any]]:
        g = tier_ground[tier]
        rec = run_model_tier(
            client, model_id, fen, board, pool, student_uci,
            g["ti"], g["maia_rank"], g["maia_top_uci"],
        )
        return model_name, tier, rec

    models_out: Dict[str, Dict[str, Any]] = {m: {} for m in models}
    with ThreadPoolExecutor(max_workers=len(models) * len(TIER_ORDER)) as ex:
        futs = [
            ex.submit(_task, mn, mid, t)
            for mn, mid in models.items()
            for t in TIER_ORDER
        ]
        for fut in as_completed(futs):
            mn, t, rec = fut.result()
            models_out[mn][t] = rec

    # The v1 tuned model, re-scored on the same lens (from divergence.jsonl).
    v1_out: Dict[str, Any] = {}
    for tier in TIER_ORDER:
        g = tier_ground[tier]
        v1_out[tier] = _v1_pick_record(drow, tier, pool, g["maia_rank"], g["maia_top_uci"])
    models_out["v1-tuned"] = v1_out

    # Compact per-tier grounding summary saved alongside the picks.
    ground_summary = {
        t: {
            "maia_top_uci": tier_ground[t]["maia_top_uci"],
            "engine_best_maia_rank": tier_ground[t]["engine_best_maia_rank"],
            "maia_rank": tier_ground[t]["maia_rank"],
            "maia_policy": tier_ground[t]["maia_policy"],
        }
        for t in TIER_ORDER
    }

    return {
        "id": drow.get("id"),
        "fen": fen,
        "phase": drow.get("phase"),
        "severity": student.get("severity"),
        "source_tier": drow.get("source_tier"),
        "student_move": student,
        "sound_pool": pool,
        "stockfish_best": drow.get("stockfish_best"),
        "grounding": ground_summary,
        "models": models_out,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _load_done_ids(path: Path) -> set:
    done: set = set()
    if path.exists():
        for l in path.read_text(encoding="utf-8").splitlines():
            l = l.strip()
            if not l:
                continue
            try:
                done.add(json.loads(l)["id"])
            except Exception:  # noqa: BLE001
                continue
    return done


def main(argv: Optional[Sequence[str]] = None) -> int:
    from dotenv import load_dotenv

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--divergence", default="data/analysis/divergence.jsonl",
                   help="v1 held-out records to reuse as the identical-grounding position set.")
    p.add_argument("--train", default="data/dataset/train.jsonl")
    p.add_argument("--valid", default="data/dataset/valid.jsonl")
    p.add_argument("--out", default="data/analysis/frontier_gap.jsonl")
    p.add_argument("--num", type=int, default=50)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--limit", type=int, default=0, help="Smoke cap on positions (0 = no cap).")
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--min-interval", type=float, default=0.05)
    p.add_argument("--resume", action="store_true", help="Skip ids already in --out.")
    p.add_argument("--only-model", default="", help="Smoke: restrict to one model key.")
    args = p.parse_args(argv)

    load_dotenv(settings.ROOT / ".env")

    def _abs(x: str) -> Path:
        pp = Path(x)
        return pp if pp.is_absolute() else _ROOT / pp

    div_path, out_path = _abs(args.divergence), _abs(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not div_path.exists():
        print(f"missing {div_path} — run scripts.divergence_analysis first.", file=sys.stderr)
        return 1

    models = dict(FRONTIER_MODELS)
    if args.only_model:
        models = {k: v for k, v in FRONTIER_MODELS.items() if k == args.only_model}
        if not models:
            print(f"unknown --only-model {args.only_model!r}; choices: {list(FRONTIER_MODELS)}",
                  file=sys.stderr)
            return 1

    print("[1/4] Loading v1 held-out records + re-verifying held-out property ...", file=sys.stderr)
    rows = _load_jsonl(div_path)
    heldin = build_heldin_keys(_abs(args.train), _abs(args.valid))
    kept, leaked = [], 0
    for r in rows:
        if pos_key(r["fen"]) in heldin:
            leaked += 1
            continue
        kept.append(r)
    print(f"      divergence rows: {len(rows)}; leaked into train/valid: {leaked}; "
          f"clean held-out: {len(kept)}", file=sys.stderr)

    print("[2/4] Balanced subsample ...", file=sys.stderr)
    sample = balanced_subsample(kept, args.num, args.seed)
    if args.limit:
        sample = sample[: args.limit]
    dist = defaultdict(int)
    for r in sample:
        dist[(r["phase"], r["student_move"]["severity"])] += 1
    print(f"      sampled {len(sample)} positions; (phase,severity) dist:", file=sys.stderr)
    for k in sorted(dist):
        print(f"        {k}: {dist[k]}", file=sys.stderr)

    done = _load_done_ids(out_path) if args.resume else set()
    if args.resume and done:
        print(f"      resuming: {len(done)} already done", file=sys.stderr)

    print(f"[3/4] Frontier client (models: {list(models)}) ...", file=sys.stderr)
    client = FrontierClient(
        timeout=args.timeout, max_retries=args.max_retries, min_interval=args.min_interval
    )

    print(f"[4/4] Coaching {len(sample)} positions x {len(TIER_ORDER)} tiers x "
          f"{len(models)} frontier models ...", file=sys.stderr)
    mode_open = "a" if (args.resume and done) else "w"
    t0 = time.time()
    n_done = 0
    with out_path.open(mode_open, encoding="utf-8") as fh:
        for i, drow in enumerate(sample, 1):
            if drow.get("id") in done:
                continue
            ts = time.time()
            try:
                row = analyze_position(drow, client, models)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! [{i}/{len(sample)}] {drow.get('id')} FAILED: {exc}", file=sys.stderr)
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += 1
            # Terse progress: beginner picks per model + engine best.
            sfb = (row["stockfish_best"] or {}).get("san")
            picks = " ".join(
                f"{mn.split('-')[0][:3]}:{row['models'][mn]['beginner']['rec_san']}"
                for mn in models
            )
            print(f"  + [{i}/{len(sample)}] {drow.get('id')} [{row['phase'][:3]}/{row['severity'][:4]}] "
                  f"SFbest:{sfb} B[{picks}]  ({time.time() - ts:.1f}s)", file=sys.stderr)

    dt = time.time() - t0
    print(f"DONE — wrote {n_done} rows to {out_path} in {dt:.0f}s "
          f"({dt / max(1, n_done):.1f}s/pos); frontier calls={client.calls} "
          f"tokens(in/out)={client.prompt_tokens:,}/{client.completion_tokens:,}", file=sys.stderr)
    try:
        maia_engine.close_all()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
