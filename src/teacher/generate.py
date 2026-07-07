"""Chess-coaching dataset generator -- the core teacher integrator.

Given a bank of tiered chess positions (a student's move + the position), this
CLI turns *verified engine analysis* into *level-appropriate coaching* using a
frontier teacher model (GPT-5.5, max reasoning), and writes one candidate
training row per coachable position.

Per position it:
  1. Uses Stockfish to classify the student's mistake and build a "sound-move
     pool", then analyzes the best move. Positions with nothing to coach are
     skipped (empty sound pool, or mistake severity ``"none"``).
  2. Uses Maia to estimate which moves a human at the student's tier would
     actually consider (best-effort; non-fatal if Maia is unavailable).
  3. Assembles a :class:`config.schema.TeacherInput`.
  4. Builds the teacher prompt (``teacher_system.md`` + the per-tier guide +
     optional ``principles.md`` / ``fewshots.json`` if another stage produced
     them) and renders the user message via
     :func:`config.schema.render_user_prompt`.
  5. Calls the teacher model, requesting a STRICT JSON object matching
     :class:`config.schema.TeacherOutput` (``chat.completions`` with
     ``response_format={"type": "json_object"}`` and ``reasoning_effort``;
     falls back to the Responses API if those params/model are rejected).
     Parses + validates the result.
  6. Appends a candidate row to ``data/generated/candidates.jsonl``.

Robustness: retries with exponential backoff on transient API errors, never
crashes the run on a single bad row (logs + skips it), small thread-pool
concurrency with basic rate limiting, and ``--limit`` / ``--smoke`` for quick
inspection.

Secrets: ``OPENAI_API_KEY`` is loaded from ``ROOT/.env`` via python-dotenv and
is never printed or logged.

CLI
---
    python -m src.teacher.generate --smoke
    python -m src.teacher.generate --limit 20 --concurrency 4
    python -m src.teacher.generate                 # full position bank
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chess
import openai
from dotenv import load_dotenv
from openai import OpenAI

from config import schema, settings
from src.engine import maia_engine, stockfish_engine

log = logging.getLogger("teacher.generate")


# --------------------------------------------------------------------------- #
# Prompt assembly (teacher_system.md + tier guide + optional principles/fewshots)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def _read_text(path_str: str) -> str:
    """Read and cache a UTF-8 text file (used for the always-present prompts)."""
    return Path(path_str).read_text(encoding="utf-8")


def _extract_tier_guide(md: str, tier: str) -> str:
    """Return the ``## <tier> ...`` section from ``tier_guides.md``.

    Captures from the tier's ``##`` heading up to (but excluding) the next
    ``##`` heading. Returns ``""`` if the tier section is not found.
    """
    tier_l = tier.strip().lower()
    out: List[str] = []
    capturing = False
    for line in md.splitlines():
        if line.startswith("## "):
            head = line[3:].strip().lower()
            if head.startswith(tier_l):
                capturing = True
                out = [line]
                continue
            if capturing:
                break
        if capturing:
            out.append(line)
    return "\n".join(out).strip()


def _format_fewshots(data: Any) -> str:
    """Best-effort render of an unknown-shaped ``fewshots.json`` payload.

    Another pipeline stage owns that file's schema, so we handle the common
    shapes (list of {user/input, assistant/output} dicts, list of strings, or a
    bare object) and fall back to pretty-printed JSON otherwise.
    """
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, list):
        blocks: List[str] = []
        for i, ex in enumerate(data, 1):
            if isinstance(ex, dict):
                user = ex.get("user") or ex.get("input") or ex.get("prompt") or ex.get("question")
                assistant = (
                    ex.get("assistant")
                    or ex.get("output")
                    or ex.get("coaching")
                    or ex.get("answer")
                )
                if user or assistant:
                    parts = [f"Example {i}:"]
                    if user:
                        parts.append(
                            "Situation: "
                            + (user if isinstance(user, str) else json.dumps(user, ensure_ascii=False))
                        )
                    if assistant:
                        parts.append(
                            "Coach: "
                            + (
                                assistant
                                if isinstance(assistant, str)
                                else json.dumps(assistant, ensure_ascii=False)
                            )
                        )
                    blocks.append("\n".join(parts))
                else:
                    blocks.append(f"Example {i}:\n" + json.dumps(ex, ensure_ascii=False, indent=2))
            else:
                blocks.append(f"Example {i}:\n{ex}")
        return "\n\n".join(blocks)
    return json.dumps(data, ensure_ascii=False, indent=2)


@lru_cache(maxsize=None)
def _load_principles() -> str:
    """Load ``prompts/principles.md`` if present, else ``""`` (optional stage)."""
    path = settings.PROMPTS / "principles.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=None)
def _load_fewshots() -> str:
    """Load + render ``prompts/fewshots.json`` if present, else ``""`` (optional)."""
    path = settings.PROMPTS / "fewshots.json"
    if not path.exists():
        return ""
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()
    return _format_fewshots(data)


@lru_cache(maxsize=None)
def build_system_prompt(tier: str) -> str:
    """Assemble the teacher system prompt for ``tier`` (cached per tier).

    Injects the per-tier guide plus optional principles / few-shots into the
    ``{TIER_GUIDE}`` / ``{PRINCIPLES}`` / ``{FEWSHOTS}`` placeholders. Absent
    optional blocks are filled with an explicit ``"(none provided)"`` marker so
    the prompt never contains a dangling placeholder.
    """
    template = _read_text(str(settings.PROMPTS / "teacher_system.md"))
    guide = _extract_tier_guide(_read_text(str(settings.PROMPTS / "tier_guides.md")), tier)
    principles = _load_principles()
    fewshots = _load_fewshots()
    return (
        template.replace("{TIER_GUIDE}", guide or "(none provided)")
        .replace("{PRINCIPLES}", principles or "(none provided)")
        .replace("{FEWSHOTS}", fewshots or "(none provided)")
    )


# --------------------------------------------------------------------------- #
# Teacher-output parsing + validation
# --------------------------------------------------------------------------- #


def _strip_code_fences(s: str) -> str:
    """Strip a leading/trailing Markdown code fence if the model added one."""
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _parse_teacher_json(content: str) -> Dict[str, Any]:
    """Parse the model's response into a JSON object.

    Tolerates stray code fences and leading/trailing prose by falling back to
    the outermost ``{...}`` span. Raises ``ValueError`` / ``JSONDecodeError`` on
    unrecoverable output so the caller can retry or skip.
    """
    if not content or not content.strip():
        raise ValueError("empty teacher response")
    s = _strip_code_fences(content)
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j != -1 and j > i:
            data = json.loads(s[i : j + 1])
        else:
            raise
    if not isinstance(data, dict):
        raise ValueError("teacher response is not a JSON object")
    return data


def _coerce_teacher_output(raw: Dict[str, Any], tier: str, fen: str) -> schema.TeacherOutput:
    """Validate + normalize a raw teacher dict into a :class:`schema.TeacherOutput`.

    Normalizes the recommended move's SAN/UCI against the actual board (deriving
    whichever is missing), forces the true ``tier``, and requires non-empty
    coaching/takeaway. Raises ``ValueError`` if the recommended move is
    illegal/unparseable or a required field is empty.
    """
    board = chess.Board(fen)
    san = str(raw.get("recommended_move_san", "") or "").strip()
    uci = str(raw.get("recommended_move_uci", "") or "").strip()

    move: Optional[chess.Move] = None
    if uci:
        try:
            candidate = chess.Move.from_uci(uci)
        except ValueError:
            candidate = None
        if candidate is not None and candidate in board.legal_moves:
            move = candidate
    if move is None and san:
        try:
            move = board.parse_san(san)
        except ValueError:
            move = None
    if move is None:
        raise ValueError(f"recommended move not legal/parseable (san={san!r}, uci={uci!r})")

    coaching = str(raw.get("coaching", "") or "").strip()
    takeaway = str(raw.get("takeaway", "") or "").strip()
    if not coaching:
        raise ValueError("empty 'coaching' field")
    if not takeaway:
        raise ValueError("empty 'takeaway' field")

    concepts_raw = raw.get("concepts_used", [])
    if isinstance(concepts_raw, str):
        concepts_raw = [concepts_raw]
    if not isinstance(concepts_raw, list):
        concepts_raw = []
    concepts = [str(c).strip() for c in concepts_raw if str(c).strip()]

    return schema.TeacherOutput(
        tier=tier,
        recommended_move_san=board.san(move),
        recommended_move_uci=move.uci(),
        coaching=coaching,
        takeaway=takeaway,
        concepts_used=concepts,
    )


# --------------------------------------------------------------------------- #
# Rate limiting + teacher API client
# --------------------------------------------------------------------------- #


class RateLimiter:
    """A minimal thread-safe limiter enforcing a min interval between starts."""

    def __init__(self, min_interval: float) -> None:
        self._min = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        """Block (if needed) so successive callers are spaced by ``min_interval``."""
        if self._min <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self._min
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


def _backoff(attempt: int) -> float:
    """Exponential backoff (capped) with jitter, in seconds."""
    base = min(2.0 ** attempt, 30.0)
    return base + random.uniform(0.0, 0.5 * base)


#: Transient errors worth retrying with backoff.
_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class TeacherClient:
    """Thin wrapper around the OpenAI SDK for strict-JSON teacher calls.

    Prefers ``chat.completions`` with ``response_format=json_object`` +
    ``reasoning_effort``; if the model/params are rejected it switches (once,
    for the whole run) to the Responses API. Transient API errors and malformed
    JSON are retried with exponential backoff.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        reasoning_effort: str,
        max_retries: int,
        limiter: RateLimiter,
    ) -> None:
        self._client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_retries = max(0, max_retries)
        self._limiter = limiter
        self._use_responses = False
        self._lock = threading.Lock()
        # Token accounting (thread-safe) for a post-run cost estimate.
        self._usage_lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0

    def complete(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Return a parsed JSON object from the teacher model (with retries)."""
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            self._limiter.acquire()
            try:
                content = self._request(system_prompt, user_prompt)
                return _parse_teacher_json(content)
            except _RETRYABLE as exc:
                last_exc = exc
                delay = _backoff(attempt)
                log.warning(
                    "transient API error (%s), attempt %d/%d; retrying in %.1fs",
                    type(exc).__name__,
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                )
                time.sleep(delay)
            except (json.JSONDecodeError, ValueError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = _backoff(attempt)
                log.warning(
                    "malformed teacher JSON (%s), attempt %d/%d; retrying in %.1fs",
                    exc,
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                )
                time.sleep(delay)
        raise last_exc if last_exc is not None else RuntimeError("teacher call failed")

    def _record_usage(self, resp: Any) -> None:
        """Accumulate token usage from a chat or responses result (best-effort)."""
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", None)
        if prompt is None:
            prompt = getattr(usage, "input_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", None)
        if completion is None:
            completion = getattr(usage, "output_tokens", 0) or 0
        reasoning = 0
        details = getattr(usage, "completion_tokens_details", None) or getattr(
            usage, "output_tokens_details", None
        )
        if details is not None:
            reasoning = getattr(details, "reasoning_tokens", 0) or 0
        with self._usage_lock:
            self.calls += 1
            self.prompt_tokens += int(prompt)
            self.completion_tokens += int(completion)
            self.reasoning_tokens += int(reasoning)

    def usage_summary(self, price_in_per_m: float, price_out_per_m: float) -> str:
        """Human-readable token totals + an estimated cost at the given prices."""
        with self._usage_lock:
            cin = self.prompt_tokens / 1_000_000 * price_in_per_m
            cout = self.completion_tokens / 1_000_000 * price_out_per_m
            return (
                f"calls={self.calls} prompt_tokens={self.prompt_tokens:,} "
                f"completion_tokens={self.completion_tokens:,} "
                f"(reasoning={self.reasoning_tokens:,}) "
                f"est_cost=${cin + cout:,.2f} "
                f"(in ${cin:,.2f} @ ${price_in_per_m}/M, out ${cout:,.2f} @ ${price_out_per_m}/M)"
            )

    # -- request paths ----------------------------------------------------- #

    def _request(self, system_prompt: str, user_prompt: str) -> str:
        if self._use_responses:
            return self._via_responses(system_prompt, user_prompt)
        try:
            return self._via_chat(system_prompt, user_prompt)
        except (openai.BadRequestError, TypeError) as exc:
            with self._lock:
                self._use_responses = True
            log.warning(
                "chat.completions rejected (%s); falling back to Responses API",
                type(exc).__name__,
            )
            return self._via_responses(system_prompt, user_prompt)

    def _via_chat(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            reasoning_effort=self.reasoning_effort,
            response_format={"type": "json_object"},
        )
        self._record_usage(resp)
        return resp.choices[0].message.content or ""

    def _via_responses(self, system_prompt: str, user_prompt: str) -> str:
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            reasoning={"effort": self.reasoning_effort},
        )
        try:
            resp = self._client.responses.create(
                text={"format": {"type": "json_object"}}, **kwargs
            )
        except (openai.BadRequestError, TypeError):
            resp = self._client.responses.create(**kwargs)
        self._record_usage(resp)
        return resp.output_text or ""


# --------------------------------------------------------------------------- #
# Per-position processing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EngineParams:
    """Stockfish / Maia knobs shared across all positions in a run."""

    movetime_ms: int
    tolerance_cp: int
    multipv: int
    maia_top_k: int


def _san_from_uci(fen: str, uci: str) -> str:
    """SAN for a UCI move on ``fen`` (falls back to the raw UCI on error)."""
    try:
        return chess.Board(fen).san(chess.Move.from_uci(uci))
    except (ValueError, AssertionError):
        return uci


def _best_line(fen: str, pool: List[Dict[str, Any]], movetime_ms: int) -> Dict[str, Any]:
    """Best move for the ``engine`` block via ``analyze`` (pool[0] as fallback)."""
    try:
        best = stockfish_engine.analyze(fen, multipv=1, movetime_ms=movetime_ms)["best"][0]
        return {"san": best.get("san") or pool[0]["san"], "cp": int(best["cp"])}
    except Exception as exc:  # noqa: BLE001 - defensive; pool[0] is a valid best
        log.debug("analyze(best) failed (%s); using sound_pool[0]", exc)
        return {"san": pool[0]["san"], "cp": int(pool[0]["cp"])}


def _maia_best_effort(
    fen: str, tier: str, top_k: int, pid: str
) -> List[Dict[str, Any]]:
    """Maia human-move likelihoods; empty list (with a warning) if unavailable."""
    try:
        return list(maia_engine.human_moves(fen, tier, top_k=top_k)["moves"])
    except Exception as exc:  # noqa: BLE001 - Maia is a helpful signal, not required
        log.warning("%s: Maia unavailable (%s); continuing without human-move signal", pid, exc)
        return []


def process_position(
    pos: Dict[str, Any], ep: EngineParams, teacher: TeacherClient
) -> Optional[Dict[str, Any]]:
    """Turn one position into a candidate row, or ``None`` if not coachable.

    Returns a dict ``{"id", "row", "user_prompt", "teacher_output"}`` on success
    (``row`` is the object written to ``candidates.jsonl``); ``None`` when the
    position has nothing to coach. Raises on genuine failures (engine/API/parse)
    so the caller can log + skip that single row.
    """
    pid = str(pos.get("id", "?"))
    fen = pos["fen"]
    tier = pos["tier"]

    # 1. Stockfish: how bad was the move, and what moves are sound?
    mistake = stockfish_engine.classify_mistake(
        fen, pos["played_move_uci"], movetime_ms=ep.movetime_ms
    )
    if mistake["severity"] == "none":
        log.info("skip %s: nothing to coach (severity 'none')", pid)
        return None

    pool = stockfish_engine.sound_pool(
        fen, tolerance_cp=ep.tolerance_cp, multipv=ep.multipv, movetime_ms=ep.movetime_ms
    )
    if not pool:
        log.info("skip %s: empty sound pool", pid)
        return None

    best = _best_line(fen, pool, ep.movetime_ms)

    # 2. Maia: which sound moves a human at this tier would actually consider.
    maia_moves = _maia_best_effort(fen, tier, ep.maia_top_k, pid)

    # 3. Assemble the teacher input contract.
    student_move: schema.StudentMove = schema.StudentMove(
        san=pos.get("played_move_san") or _san_from_uci(fen, pos["played_move_uci"]),
        uci=pos["played_move_uci"],
        cp_loss=int(mistake["cp_loss"]),
        severity=str(mistake["severity"]),
    )
    teacher_input: schema.TeacherInput = schema.TeacherInput(
        tier=tier,
        fen=fen,
        move_history_san=pos.get("move_history_san"),
        student_move=student_move,
        sound_pool=pool,  # type: ignore[typeddict-item]  # {uci,san,cp,pv} matches SoundMove
        maia_human_moves=maia_moves,  # type: ignore[typeddict-item]  # {uci,san,policy}
    )

    # 4. Build prompts (system per tier + rendered user message).
    system_prompt = build_system_prompt(tier)
    user_prompt = schema.render_user_prompt(teacher_input)

    # 5. Teacher call -> strict JSON -> validated TeacherOutput.
    raw = teacher.complete(system_prompt, user_prompt)
    teacher_output = _coerce_teacher_output(raw, tier, fen)

    # Soft spec check: the recommendation should come from the sound pool. We
    # keep it either way (the dedicated filter stage enforces the spec) but flag
    # it so it is easy to spot during review.
    if teacher_output["recommended_move_uci"] not in {m["uci"] for m in pool}:
        log.warning(
            "%s: recommended %s is outside the sound pool (kept for filter stage)",
            pid,
            teacher_output["recommended_move_san"],
        )

    # 6. Candidate row.
    row: Dict[str, Any] = {
        "id": pid,
        "tier": tier,
        "teacher_input": teacher_input,
        "teacher_output": teacher_output,
        "engine": {
            "best_san": best["san"],
            "best_cp": best["cp"],
            "sound_ucis": [m["uci"] for m in pool],
        },
        "maia_top": maia_moves,
        "meta": {
            "model": teacher.model,
            "reasoning_effort": teacher.reasoning_effort,
            "ts": datetime.now(timezone.utc).isoformat(),
        },
    }
    return {"id": pid, "row": row, "user_prompt": user_prompt, "teacher_output": teacher_output}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def load_positions(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL position bank, skipping (and logging) any unparseable lines."""
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("positions.jsonl line %d unparseable: %s", lineno, exc)
    return rows


def _existing_ids(path: Path) -> set:
    """Return the set of candidate ids already present in an output JSONL file."""
    ids: set = set()
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(str(json.loads(line).get("id")))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return ids


def _print_smoke_report(results: List[Dict[str, Any]], out_path: Path, written: int) -> None:
    """Print one rendered user prompt and up to 3 parsed outputs for eyeballing."""
    results = sorted(results, key=lambda r: r["id"])
    bar = "=" * 78
    print("\n" + bar)
    print("SMOKE REPORT")
    print(bar)
    print(f"candidates written: {written}  ->  {out_path}")
    if not results:
        print("(no successful rows to display)")
        print(bar)
        return

    first = results[0]
    print(f"\n--- RENDERED USER PROMPT (example id={first['id']}) ---\n")
    print(first["user_prompt"])

    print("\n--- PARSED TEACHER OUTPUT (up to 3 examples) ---")
    for r in results[:3]:
        print(f"\n# id={r['id']}  tier={r['row']['tier']}")
        print(json.dumps(r["teacher_output"], ensure_ascii=False, indent=2))
    print("\n" + bar)


def run(args: argparse.Namespace) -> int:
    """Execute a generation run; returns a process exit code."""
    load_dotenv(settings.ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not found (looked in environment and %s/.env)", settings.ROOT)
        return 2

    model = args.model or os.environ.get("TEACHER_MODEL") or settings.TEACHER_MODEL
    reasoning = args.reasoning_effort

    positions = load_positions(Path(args.positions))
    if not positions:
        log.error("no positions loaded from %s", args.positions)
        return 2

    # Resume: in append mode, skip positions already present in the output file so a
    # re-run after a crash never re-spends on completed rows or duplicates them.
    out_path = Path(args.out)
    if not (args.fresh or args.smoke) and out_path.exists():
        done_ids = _existing_ids(out_path)
        if done_ids:
            before = len(positions)
            positions = [p for p in positions if str(p.get("id")) not in done_ids]
            log.info("resume: %d already done, %d remaining", before - len(positions), len(positions))

    limit = 5 if args.smoke else args.limit
    if limit is not None:
        positions = positions[:limit]

    if not positions:
        log.info("nothing to do (all positions already generated)")
        return 0

    ep = EngineParams(
        movetime_ms=args.movetime,
        tolerance_cp=args.tolerance,
        multipv=args.multipv,
        maia_top_k=args.maia_top_k,
    )

    # We manage our own retries/backoff, so disable the SDK's built-in ones.
    client = OpenAI(api_key=api_key, timeout=args.timeout, max_retries=0)
    limiter = RateLimiter(args.min_interval)
    teacher = TeacherClient(
        client,
        model=model,
        reasoning_effort=reasoning,
        max_retries=args.max_retries,
        limiter=limiter,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fresh = args.fresh or args.smoke  # smoke is reproducible: always a clean file
    mode = "w" if fresh else "a"
    workers = max(1, min(args.concurrency, len(positions)))

    log.info(
        "generating: model=%s effort=%s positions=%d workers=%d out=%s (%s)",
        model,
        reasoning,
        len(positions),
        workers,
        out_path,
        "fresh" if fresh else "append",
    )

    results: List[Dict[str, Any]] = []
    written = skipped = failed = 0

    with out_path.open(mode, encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_position, pos, ep, teacher): pos for pos in positions}
            for fut in as_completed(futures):
                pid = str(futures[fut].get("id", "?"))
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
                    failed += 1
                    log.error("row %s failed: %s", pid, exc)
                    continue
                if result is None:
                    skipped += 1
                    continue
                out_fh.write(json.dumps(result["row"], ensure_ascii=False) + "\n")
                out_fh.flush()
                written += 1
                results.append(result)

    log.info("done: wrote=%d skipped=%d failed=%d -> %s", written, skipped, failed, out_path)
    log.info("teacher usage: %s", teacher.usage_summary(args.price_in, args.price_out))

    if args.smoke:
        _print_smoke_report(results, out_path, written)

    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate chess-coaching candidate rows with a teacher LLM.",
    )
    p.add_argument(
        "--positions",
        default=str(settings.POSITIONS / "positions.jsonl"),
        help="Input JSONL position bank (default: %(default)s).",
    )
    p.add_argument(
        "--out",
        default=str(settings.GENERATED / "candidates.jsonl"),
        help="Output JSONL candidates file (default: %(default)s).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Teacher model id (default: env TEACHER_MODEL, else settings.TEACHER_MODEL).",
    )
    p.add_argument(
        "--reasoning-effort",
        dest="reasoning_effort",
        default=settings.TEACHER_REASONING_EFFORT,
        help="Reasoning effort for the teacher model (default: %(default)s).",
    )
    p.add_argument("--limit", type=int, default=None, help="Process only the first N positions.")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Process only 5 positions, write a fresh file, and print samples.",
    )
    p.add_argument("--concurrency", type=int, default=4, help="Thread-pool size (default: 4).")
    p.add_argument(
        "--min-interval",
        dest="min_interval",
        type=float,
        default=0.1,
        help="Minimum seconds between API request starts (basic rate limit).",
    )
    p.add_argument(
        "--max-retries",
        dest="max_retries",
        type=int,
        default=4,
        help="Max retries per teacher call on transient/JSON errors (default: 4).",
    )
    p.add_argument(
        "--timeout", type=float, default=600.0, help="Per-request timeout in seconds."
    )
    p.add_argument("--movetime", type=int, default=settings.DEFAULT_MOVETIME_MS, help="Stockfish ms/pos.")
    p.add_argument(
        "--tolerance", type=int, default=settings.SOUND_TOLERANCE_CP, help="Sound-pool cp tolerance."
    )
    p.add_argument("--multipv", type=int, default=settings.MULTIPV, help="Stockfish MultiPV.")
    p.add_argument(
        "--maia-top-k", dest="maia_top_k", type=int, default=6, help="Maia moves to keep."
    )
    p.add_argument(
        "--fresh", action="store_true", help="Truncate the output file before writing."
    )
    p.add_argument(
        "--price-in", dest="price_in", type=float, default=1.25,
        help="Est. USD per 1M input tokens for the cost readout (verify for your model).",
    )
    p.add_argument(
        "--price-out", dest="price_out", type=float, default=10.0,
        help="Est. USD per 1M output tokens for the cost readout (verify for your model).",
    )
    p.add_argument("--log-level", dest="log_level", default="INFO", help="Logging level.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return run(args)
    finally:
        # Tear down any cached lc0 processes (also registered via atexit).
        try:
            maia_engine.close_all()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


if __name__ == "__main__":
    raise SystemExit(main())
