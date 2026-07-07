#!/usr/bin/env python3
"""Hard QUALITY FILTER for the chess-coaching SFT dataset.

Each teacher-generated *candidate* is passed through a stack of hard gates; only
rows that survive every gate become training examples. The gates encode the
behavior spec (``config.settings.BEHAVIOR_SPEC``): recommend only sound moves,
never leak engine numbers, never narrate past the tier's ply cap, stay valid, and
don't duplicate a (position, move) lesson.

Gates
-----
1. SOUNDNESS  - ``teacher_output.recommended_move_uci`` must be listed in
                ``engine.sound_ucis`` (i.e. inside the sound pool, never a blunder).
2. NO-ENGINE-SPEAK - the coaching + takeaway text must not leak centipawns / evals
                / engine identity / mate-count or signed-advantage numbers.
3. PLY-CAP    - the coaching must not narrate a concrete line longer than the
                tier's ``ply_cap`` (heuristic: longest run of consecutive SAN-like
                tokens must not exceed the cap).
4. VALIDITY   - valid JSON, a legal recommended move in the position's FEN, and
                non-empty coaching + takeaway.
5. DEDUP      - at most one kept row per ``(fen, recommended move)`` pair.

KEEP  -> a chat training row via ``schema.build_chat_example`` into ``train.jsonl``.
REJECT -> the original candidate plus a ``reasons`` list into ``rejects.jsonl``.

The module always runs a self-contained SELF-TEST on synthetic candidates (it does
NOT depend on real candidates existing) and, if a real ``candidates.jsonl`` is
present, filters it too and prints a kept/rejected summary.

Run from the project root with the pinned interpreter::

    /Users/khoilam/.venvs/mlx/bin/python src/filter/filter.py
    /Users/khoilam/.venvs/mlx/bin/python src/filter/filter.py --candidates data/generated/candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

# --- make the project root importable regardless of how we're launched -------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess  # noqa: E402  (import after sys.path bootstrap)

from config import schema, settings  # noqa: E402
from src.engine.faithfulness import verify_text  # noqa: E402


# --------------------------------------------------------------------------- #
# Reason codes (stable strings written into rejects.jsonl + asserted in tests)
# --------------------------------------------------------------------------- #

R_INVALID_JSON = "invalid_json"
R_MISSING_TIER = "missing_tier"
R_MISSING_FEN = "missing_fen"
R_MISSING_REC = "missing_recommendation"
R_SOUNDNESS = "soundness"
R_ENGINE_SPEAK = "engine_speak"
R_PLY_CAP = "ply_cap"
R_ILLEGAL_MOVE = "illegal_move"
R_EMPTY_COACHING = "empty_coaching"
R_EMPTY_TAKEAWAY = "empty_takeaway"
R_RENDER_ERROR = "render_error"
R_DUPLICATE = "duplicate"
#: v2 requires the explicit "how to find it" method clause on every example.
R_MISSING_METHOD = "missing_method"

#: Training-target format. ``v1`` = move + coaching + takeaway. ``v2`` = adds the
#: explicit method clause (and requires it).
TARGET_V1 = "v1"
TARGET_V2 = "v2"
#: v2 truth gate: the coaching/takeaway states a board fact that is demonstrably
#: false for the position (a named piece is not on the named square, etc.). This
#: is the fix for RESULTS.md's flat truthfulness — labels were filtered for
#: format/soundness but never for FAITHFULNESS, so the student learned the
#: teacher's occasional fabrication. Verified with :func:`verify_text`.
R_FAITHFULNESS = "faithfulness"

#: Faithfulness-gate modes (``--faithfulness``). ``off`` reproduces v1 exactly.
FAITH_OFF = "off"
FAITH_REJECT = "reject"   # drop any candidate with a false board claim (v2 default)
FAITH_STRIP = "strip"     # drop only the false sentence(s); keep the rest if usable

#: Fallback cap used only when a candidate has no valid tier (it's rejected anyway).
MAX_PLY_CAP = max(t["ply_cap"] for t in settings.TIERS.values())


# --------------------------------------------------------------------------- #
# NO-ENGINE-SPEAK detection
# --------------------------------------------------------------------------- #

#: Patterns whose presence in coaching/takeaway leaks engine internals. All are
#: matched case-insensitively. The signed-number patterns use a lookbehind so a
#: sign glued to a word (e.g. "top-3", "2-1 majority") is not misread as an eval.
_ENGINE_SPEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\d+\s*cp\b", re.IGNORECASE),          # "150cp", "30 cp"
    re.compile(r"\bcentipawns?\b", re.IGNORECASE),     # centipawn(s)
    re.compile(r"\beval", re.IGNORECASE),              # eval / evaluation / evaluate
    re.compile(r"\bengines?\b", re.IGNORECASE),        # engine / engines
    re.compile(r"\bstockfish\b", re.IGNORECASE),       # stockfish
    re.compile(r"\bcomputers?\b", re.IGNORECASE),      # computer(s)
    re.compile(r"#\d"),                                # mate marker "#3"
    re.compile(r"(?<!\w)[+-]\d+\.\d+"),                # signed decimal "+1.3" / "-0.5"
    re.compile(r"(?<!\w)[+-]\d"),                       # "+2" / "-1" advantage
)


def detect_engine_speak(text: str) -> list[str]:
    """Return the distinct engine-speak snippets found in ``text`` (empty if clean)."""
    hits: list[str] = []
    for pat in _ENGINE_SPEAK_PATTERNS:
        for m in pat.finditer(text):
            snippet = m.group(0)
            if snippet not in hits:
                hits.append(snippet)
    return hits


# --------------------------------------------------------------------------- #
# PLY-CAP heuristic (longest run of consecutive SAN-like tokens)
# --------------------------------------------------------------------------- #

#: A single Standard Algebraic Notation move, e.g. "e4", "exd5", "Nbd2", "e8=Q+",
#: "O-O", "O-O-O", optionally suffixed with a check/mate marker.
_SAN_TOKEN = re.compile(
    r"^(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?$"
)
#: A standalone move-number token, e.g. "1.", "12.", "3...".
_MOVE_NUMBER = re.compile(r"^\d+\.+$")
#: A move-number prefix glued to a move, e.g. "1.e4" -> "e4", "3...Nf6" -> "Nf6".
_LEADING_MOVE_NUMBER = re.compile(r"^\d+\.+")


def longest_san_run(text: str) -> int:
    """Return the length of the longest run of consecutive SAN-like tokens.

    Move-number tokens ("1.", "12...") are transparent: they neither count nor
    break a run, so "1. e4 e5 2. Nf3" reads as a 3-ply line. Any non-move word
    resets the run. This approximates "how many plies of a concrete line does the
    coaching narrate" without a full parser.
    """
    best = run = 0
    for raw in text.split():
        tok = raw.strip(" \t\r\n()[]{}\"'")
        if not tok or _MOVE_NUMBER.match(tok):
            continue  # blank or pure move number -> transparent
        tok = tok.rstrip(".,;:!?")            # drop sentence punctuation, keep + / #
        m = _LEADING_MOVE_NUMBER.match(tok)
        if m:
            tok = tok[m.end():]
        if not tok:
            continue
        if _SAN_TOKEN.match(tok):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


# --------------------------------------------------------------------------- #
# Field resolution + move legality
# --------------------------------------------------------------------------- #

def _resolve_tier(candidate: dict[str, Any]) -> Optional[str]:
    """Return the candidate's tier from the top level or the teacher_input."""
    ti = candidate.get("teacher_input") or {}
    return candidate.get("tier") or ti.get("tier")


def _resolve_fen(candidate: dict[str, Any]) -> Optional[str]:
    """Return the position FEN from the teacher_input or the top level."""
    ti = candidate.get("teacher_input") or {}
    return ti.get("fen") or candidate.get("fen")


def move_is_legal(fen: str, uci: str, san: str) -> tuple[bool, Optional[str]]:
    """Return ``(is_legal, normalized_uci)`` for a recommended move in ``fen``.

    Prefers the UCI string; falls back to SAN. Returns ``(False, None)`` for a
    malformed FEN/UCI/SAN or an illegal move.
    """
    try:
        board = chess.Board(fen)
    except ValueError:
        return False, None
    if uci:
        try:
            mv = chess.Move.from_uci(uci)
            if mv in board.legal_moves:
                return True, mv.uci()
        except ValueError:
            pass
    if san:
        try:
            mv = board.parse_san(san)  # parse_san only returns legal moves
            return True, mv.uci()
        except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            return False, None
    return False, None


# --------------------------------------------------------------------------- #
# Candidate evaluation
# --------------------------------------------------------------------------- #

@dataclass
class EvalResult:
    """Outcome of running the (non-dedup) gates over one candidate."""

    reasons: list[str] = field(default_factory=list)
    train_row: Optional[dict] = None
    dedup_key: Optional[tuple[str, str, str]] = None
    tier: str = "unknown"
    engine_speak_hits: list[str] = field(default_factory=list)
    san_run: int = 0
    #: Number of demonstrably-false board-fact sentences found (0 = faithful).
    faith_violations: int = 0
    #: True when ``strip`` mode removed >=1 false sentence but kept the row.
    stripped: bool = False

    @property
    def kept(self) -> bool:
        return not self.reasons


def _faithfulness_violations(coaching: str, takeaway: str, fen: str) -> int:
    """Count demonstrably-false board-fact sentences across coaching + takeaway."""
    combined = f"{coaching}\n{takeaway}".strip()
    if not combined:
        return 0
    return len(verify_text(combined, fen).violations)


def evaluate_candidate(
    candidate: dict[str, Any],
    system_prompt: str,
    *,
    faithfulness: str = FAITH_OFF,
    target_format: str = TARGET_V1,
) -> EvalResult:
    """Apply gates 1-4 (+ optional faithfulness) and, if clean, build the row.

    Deduplication (gate 5) is intentionally left to the caller because it needs
    state shared across the whole stream; this function is otherwise pure.

    ``faithfulness`` (v2 truth gate; default ``off`` reproduces v1 exactly):
      * ``reject`` — reject any candidate whose coaching/takeaway states a false
        board fact (verified against the FEN with :func:`verify_text`).
      * ``strip``  — remove only the false sentence(s) and keep the row if a
        non-empty coaching + takeaway survive (otherwise reject).
    """
    reasons: list[str] = []
    tier = _resolve_tier(candidate)
    fen = _resolve_fen(candidate)
    ti = candidate.get("teacher_input") or {}
    to = candidate.get("teacher_output") or {}
    engine = candidate.get("engine") or {}

    coaching = str(to.get("coaching") or "").strip()
    takeaway = str(to.get("takeaway") or "").strip()
    method = str(to.get("method") or "").strip()  # v2: the "how to find it" clause
    rec_uci = str(to.get("recommended_move_uci") or "").strip()
    rec_san = str(to.get("recommended_move_san") or "").strip()

    tier_key = tier if tier in settings.TIERS else "unknown"
    if tier not in settings.TIERS:
        reasons.append(R_MISSING_TIER)
    ply_cap = settings.TIERS.get(tier, {}).get("ply_cap", MAX_PLY_CAP)

    # Gate 6 (v2, runs first so it can STRIP before the other text gates): the
    # coaching/takeaway must not state a false board fact. In ``strip`` mode the
    # false sentence(s) are removed here so the remaining gates + the training
    # row all see the cleaned, faithful text.
    faith_violations = 0
    stripped = False
    if faithfulness != FAITH_OFF and fen and (coaching or takeaway or method):
        # The whole learned target is checked: coaching + method + takeaway.
        combined = "\n".join(x for x in (coaching, method, takeaway) if x)
        faith_violations = len(verify_text(combined, fen).violations)
        if faith_violations:
            if faithfulness == FAITH_STRIP:
                coaching = verify_text(coaching, fen).clean.strip()
                method = verify_text(method, fen).clean.strip() if method else method
                takeaway = verify_text(takeaway, fen).clean.strip()
                stripped = True  # will still reject below if nothing usable remains
            else:  # FAITH_REJECT
                reasons.append(R_FAITHFULNESS)

    # Gate 1: SOUNDNESS -- recommended move must be in the engine's sound pool.
    sound_set = {str(u).lower() for u in (engine.get("sound_ucis") or [])}
    if not rec_uci and not rec_san:
        reasons.append(R_MISSING_REC)
    elif not rec_uci or rec_uci.lower() not in sound_set:
        reasons.append(R_SOUNDNESS)

    # Gate 2: NO-ENGINE-SPEAK over coaching + takeaway.
    engine_hits = detect_engine_speak(f"{coaching}\n{takeaway}")
    if engine_hits:
        reasons.append(R_ENGINE_SPEAK)

    # Gate 3: PLY-CAP -- the coaching must not narrate a line longer than the cap.
    san_run = longest_san_run(coaching)
    if san_run > ply_cap:
        reasons.append(R_PLY_CAP)

    # Gate 4: VALIDITY -- fen present, move legal, non-empty coaching + takeaway.
    if not fen:
        reasons.append(R_MISSING_FEN)
    elif rec_uci or rec_san:
        legal, _ = move_is_legal(fen, rec_uci, rec_san)
        if not legal:
            reasons.append(R_ILLEGAL_MOVE)
    if not coaching:
        reasons.append(R_EMPTY_COACHING)  # e.g. strip removed every sentence
    if not takeaway:
        reasons.append(R_EMPTY_TAKEAWAY)
    if target_format == TARGET_V2 and not method:
        reasons.append(R_MISSING_METHOD)  # v2 must teach the "how to find it" method

    # Dedup key carries the tier so a caller can choose (fen, move) — the v1
    # default — OR (fen, tier, move), which KEEPS the v2 contrastive triples
    # (same position taught at each tier) instead of collapsing them.
    dedup_key: Optional[tuple[str, str, str]] = None
    if fen and (rec_uci or rec_san):
        dedup_key = (fen, tier or "", (rec_uci or rec_san).lower())

    # Build the training row only for otherwise-clean candidates. In strip mode
    # the (possibly cleaned) coaching/takeaway are folded back into the output.
    train_row: Optional[dict] = None
    if not reasons:
        ti_full: dict[str, Any] = dict(ti)
        ti_full.setdefault("fen", fen)
        ti_full.setdefault("tier", tier)
        ti_full.setdefault("move_history_san", ti.get("move_history_san"))
        to_out: dict[str, Any] = dict(to)
        to_out["coaching"] = coaching
        to_out["takeaway"] = takeaway
        to_out["method"] = method
        builder = (
            schema.build_chat_example_v2 if target_format == TARGET_V2
            else schema.build_chat_example
        )
        try:
            train_row = builder(system_prompt, ti_full, to_out)  # type: ignore[arg-type]
        except Exception:  # malformed teacher_input the renderer can't consume
            reasons.append(R_RENDER_ERROR)
            train_row = None

    return EvalResult(
        reasons=reasons,
        train_row=train_row,
        dedup_key=dedup_key,
        tier=tier_key,
        engine_speak_hits=engine_hits,
        san_run=san_run,
        faith_violations=faith_violations,
        stripped=stripped and not reasons,
    )


# --------------------------------------------------------------------------- #
# Stream processing (adds gate 5: dedup) + stats
# --------------------------------------------------------------------------- #

@dataclass
class FilterStats:
    """Roll-up counts for the run summary."""

    total: int = 0
    kept: int = 0
    rejected: int = 0
    by_reason: Counter[str] = field(default_factory=Counter)
    kept_by_tier: Counter[str] = field(default_factory=Counter)
    rejected_by_tier: Counter[str] = field(default_factory=Counter)
    #: v2 faithfulness telemetry.
    faith_mode: str = FAITH_OFF
    faith_flagged: int = 0   # candidates with >=1 false board claim (pre-gate)
    faith_stripped: int = 0  # rows kept after removing false sentence(s)


#: Dedup key modes. ``fen_move`` is the v1 behavior (one lesson per position+move);
#: ``fen_tier_move`` additionally keeps per-tier variants of the same position, so
#: the v2 contrastive triples are not collapsed.
DEDUP_FEN_MOVE = "fen_move"
DEDUP_FEN_TIER_MOVE = "fen_tier_move"


def _effective_dedup_key(
    key: tuple[str, str, str], mode: str
) -> tuple[str, ...]:
    """Reduce the (fen, tier, move) key according to ``mode``."""
    fen, tier, move = key
    if mode == DEDUP_FEN_TIER_MOVE:
        return (fen, tier, move)
    return (fen, move)


def process_candidates(
    records: Iterator[tuple[Optional[dict[str, Any]], str]],
    system_prompt: str,
    *,
    faithfulness: str = FAITH_OFF,
    dedup_mode: str = DEDUP_FEN_MOVE,
    target_format: str = TARGET_V1,
) -> tuple[list[dict], list[dict], FilterStats]:
    """Filter a stream of ``(candidate_or_None, raw_line)`` records.

    Returns ``(train_rows, reject_rows, stats)``. ``candidate_or_None`` is ``None``
    for lines that failed to parse as JSON; those are rejected as ``invalid_json``.
    ``faithfulness`` selects the v2 truth gate (see :func:`evaluate_candidate`).
    ``dedup_mode`` selects whether same-position-different-tier rows are kept.
    """
    train_rows: list[dict] = []
    reject_rows: list[dict] = []
    stats = FilterStats(faith_mode=faithfulness)
    seen_keys: set[tuple[str, ...]] = set()

    for candidate, raw in records:
        stats.total += 1

        if candidate is None or not isinstance(candidate, dict):
            stats.rejected += 1
            stats.by_reason[R_INVALID_JSON] += 1
            stats.rejected_by_tier["unknown"] += 1
            reject_rows.append({"raw": raw, "reasons": [R_INVALID_JSON]})
            continue

        result = evaluate_candidate(
            candidate, system_prompt,
            faithfulness=faithfulness, target_format=target_format,
        )
        if result.faith_violations:
            stats.faith_flagged += 1
        if result.stripped:
            stats.faith_stripped += 1
        reasons = list(result.reasons)

        # Gate 5: DEDUP -- only among otherwise-valid rows, first occurrence wins.
        if not reasons and result.dedup_key is not None:
            eff_key = _effective_dedup_key(result.dedup_key, dedup_mode)
            if eff_key in seen_keys:
                reasons.append(R_DUPLICATE)
            else:
                seen_keys.add(eff_key)

        if reasons:
            stats.rejected += 1
            stats.rejected_by_tier[result.tier] += 1
            for r in reasons:
                stats.by_reason[r] += 1
            reject = dict(candidate)
            reject["reasons"] = reasons
            reject_rows.append(reject)
        else:
            stats.kept += 1
            stats.kept_by_tier[result.tier] += 1
            assert result.train_row is not None  # guaranteed when reasons is empty
            train_rows.append(result.train_row)

    return train_rows, reject_rows, stats


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def iter_jsonl(path: Path) -> Iterator[tuple[Optional[dict[str, Any]], str]]:
    """Yield ``(obj_or_None, raw_line)`` for each non-blank line of a JSONL file."""
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line), line
            except json.JSONDecodeError:
                yield None, line


def write_jsonl(rows: list[dict], out_path: Path) -> None:
    """Write ``rows`` to ``out_path`` atomically (temp file + replace)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)


def load_system_prompt(path: Path) -> str:
    """Read the coach system-prompt text used for every training row."""
    return path.read_text(encoding="utf-8").strip()


def print_summary(stats: FilterStats, *, title: str) -> None:
    """Print kept/rejected counts overall, by reason, and by tier."""
    print(f"\n=== {title} ===")
    print(f"candidates: {stats.total}")
    print(f"kept:       {stats.kept}")
    print(f"rejected:   {stats.rejected}")

    if stats.faith_mode != FAITH_OFF:
        print(f"\nfaithfulness gate: mode={stats.faith_mode}")
        print(f"  candidates with a false board claim: {stats.faith_flagged}")
        if stats.faith_mode == FAITH_STRIP:
            print(f"  rows kept after stripping false line(s): {stats.faith_stripped}")

    if stats.by_reason:
        print("\nrejected by reason:")
        width = max(len(r) for r in stats.by_reason)
        for reason, count in stats.by_reason.most_common():
            print(f"  {reason:<{width}} : {count}")

    tiers = [t for t in settings.TIERS] + ["unknown"]
    print("\nby tier (kept / rejected):")
    for tier in tiers:
        k = stats.kept_by_tier.get(tier, 0)
        r = stats.rejected_by_tier.get(tier, 0)
        if k or r or tier in settings.TIERS:
            print(f"  {tier:<12} : {k} / {r}")


# --------------------------------------------------------------------------- #
# SELF-TEST (synthetic in-memory candidates; no dependency on real data)
# --------------------------------------------------------------------------- #

def _synthetic_candidates() -> list[dict[str, Any]]:
    """Three candidates: (a) clean/valid, (b) engine-speak leak, (c) unsound move.

    All share one position (Black to move after 1.e4). The recommended moves are
    legal so that (b) fails ONLY on engine-speak and (c) fails ONLY on soundness.
    """
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    sound_pool = [
        {"san": "c5", "uci": "c7c5", "cp": 18, "pv": ["c7c5", "g1f3", "d7d6"]},
        {"san": "e5", "uci": "e7e5", "cp": 12, "pv": ["e7e5", "g1f3", "b8c6"]},
        {"san": "Nf6", "uci": "g8f6", "cp": 22, "pv": ["g8f6", "e4e5", "f6d5"]},
    ]
    maia = [
        {"san": "e5", "uci": "e7e5", "policy": 0.50},
        {"san": "c5", "uci": "c7c5", "policy": 0.25},
        {"san": "Nf6", "uci": "g8f6", "policy": 0.15},
    ]
    student = {"san": "a5", "uci": "a7a5", "cp_loss": 110, "severity": "mistake"}
    engine = {"best_san": "e5", "best_cp": 12, "sound_ucis": ["c7c5", "e7e5", "g8f6"]}

    def base_input() -> dict[str, Any]:
        return {
            "tier": "beginner",
            "fen": fen,
            "move_history_san": "1. e4",
            "student_move": dict(student),
            "sound_pool": [dict(m) for m in sound_pool],
            "maia_human_moves": [dict(m) for m in maia],
        }

    coaching_clean = (
        "When your opponent grabs space in the middle, meet them head on by "
        "claiming your own share of the center. This keeps your position active and "
        "gives your minor pieces natural squares to develop toward. Play with a "
        "plan: fight for the middle first, then bring out your knights and bishops."
    )
    takeaway_clean = (
        "Answer a push in the center by fighting for the center yourself, then develop."
    )
    coaching_leak = (
        "You are clearly on top here; the engine says this is about +1.3 for you, "
        "so just keep developing and press your advantage in the center."
    )

    clean = {
        "id": "selftest-a-clean",
        "tier": "beginner",
        "teacher_input": base_input(),
        "teacher_output": {
            "tier": "beginner",
            "recommended_move_san": "c5",
            "recommended_move_uci": "c7c5",
            "coaching": coaching_clean,
            "takeaway": takeaway_clean,
            "concepts_used": ["center control", "development"],
        },
        "engine": dict(engine),
    }
    leak = {
        "id": "selftest-b-enginespeak",
        "tier": "beginner",
        "teacher_input": base_input(),
        "teacher_output": {
            "tier": "beginner",
            "recommended_move_san": "e5",
            "recommended_move_uci": "e7e5",
            "coaching": coaching_leak,
            "takeaway": "Keep developing and convert your edge.",
            "concepts_used": ["center control"],
        },
        "engine": dict(engine),
    }
    unsound = {
        "id": "selftest-c-unsound",
        "tier": "beginner",
        "teacher_input": base_input(),
        "teacher_output": {
            "tier": "beginner",
            "recommended_move_san": "d6",
            "recommended_move_uci": "d7d6",  # legal, but NOT in sound_ucis
            "coaching": coaching_clean,
            "takeaway": takeaway_clean,
            "concepts_used": ["development"],
        },
        "engine": dict(engine),
    }
    return [clean, leak, unsound]


def run_self_test(system_prompt: str) -> bool:
    """Filter three synthetic candidates and assert the expected verdicts."""
    print("=== self-test ===")
    clean, leak, unsound = _synthetic_candidates()

    res_clean = evaluate_candidate(clean, system_prompt)
    res_leak = evaluate_candidate(leak, system_prompt)
    res_unsound = evaluate_candidate(unsound, system_prompt)

    for label, res in (("(a) clean", res_clean), ("(b) engine-speak", res_leak),
                       ("(c) unsound", res_unsound)):
        verdict = "KEEP" if res.kept else "REJECT"
        extra = ""
        if res.engine_speak_hits:
            extra = f"  hits={res.engine_speak_hits}"
        print(f"  {label:<18} -> {verdict:<6} reasons={res.reasons or '[]'}{extra}")

    ok = True
    try:
        assert res_clean.reasons == [], f"(a) expected KEEP, got {res_clean.reasons}"
        assert res_clean.train_row is not None, "(a) should build a training row"
        assert list(res_clean.train_row["messages"][0].values())  # system present
        assert res_leak.reasons == [R_ENGINE_SPEAK], \
            f"(b) expected [{R_ENGINE_SPEAK}], got {res_leak.reasons}"
        assert res_unsound.reasons == [R_SOUNDNESS], \
            f"(c) expected [{R_SOUNDNESS}], got {res_unsound.reasons}"
    except AssertionError as exc:
        ok = False
        print(f"  SELF-TEST FAILED: {exc}")

    print(f"  self-test: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hard quality filter for chess-coaching SFT candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates", type=Path, default=settings.GENERATED / "candidates.jsonl",
        help="Input candidates JSONL (filtered only if it exists).",
    )
    parser.add_argument(
        "--train-out", type=Path, default=settings.DATASET / "train.jsonl",
        help="Output JSONL of kept training rows.",
    )
    parser.add_argument(
        "--rejects-out", type=Path, default=settings.GENERATED / "rejects.jsonl",
        help="Output JSONL of rejected candidates (with a 'reasons' list).",
    )
    parser.add_argument(
        "--system-prompt", type=Path, default=settings.PROMPTS / "coach_system.md",
        help="Coach system-prompt text embedded in every training row.",
    )
    parser.add_argument(
        "--faithfulness", choices=(FAITH_OFF, FAITH_REJECT, FAITH_STRIP),
        default=FAITH_OFF,
        help="v2 truth gate: 'reject' drops candidates with a false board claim, "
             "'strip' removes only the false sentence(s). 'off' reproduces v1.",
    )
    parser.add_argument(
        "--dedup-key", dest="dedup_key", choices=(DEDUP_FEN_MOVE, DEDUP_FEN_TIER_MOVE),
        default=DEDUP_FEN_MOVE,
        help="'fen_move' (v1) keeps one lesson per position+move; 'fen_tier_move' "
             "additionally keeps the v2 contrastive per-tier variants of a position.",
    )
    parser.add_argument(
        "--target-format", dest="target_format", choices=(TARGET_V1, TARGET_V2),
        default=TARGET_V1,
        help="'v2' renders the training target with the explicit 'how to find it' "
             "method clause and requires it on every kept row.",
    )
    parser.add_argument(
        "--skip-self-test", action="store_true",
        help="Skip the synthetic self-test (not recommended).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    system_prompt = load_system_prompt(args.system_prompt)

    self_test_ok = True
    if not args.skip_self_test:
        self_test_ok = run_self_test(system_prompt)
        if not self_test_ok:
            print("\nBLOCKED: self-test failed.")
            return 1

    if args.candidates.exists():
        records = iter_jsonl(args.candidates)
        train_rows, reject_rows, stats = process_candidates(
            records, system_prompt,
            faithfulness=args.faithfulness, dedup_mode=args.dedup_key,
            target_format=args.target_format,
        )
        write_jsonl(train_rows, args.train_out)
        write_jsonl(reject_rows, args.rejects_out)
        print_summary(stats, title="quality filter summary")
        print(f"\nwrote {stats.kept} training rows -> {args.train_out}")
        print(f"wrote {stats.rejected} rejects       -> {args.rejects_out}")
    else:
        print(f"\n(no real candidates at {args.candidates} — self-test only.)")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
