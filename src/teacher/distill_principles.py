#!/usr/bin/env python3
"""Distill reusable chess *coaching* guidance from GM commentary transcripts.

This is a one-time, internal pre-processing step for the ``chess-instructor-llm``
pipeline. It reads cleaned YouTube commentary transcripts (GothamChess "Win At
Chess" + Naroditsky speedruns), samples a manageable set across tiers, and uses
``gpt-5.5`` to turn the messy spoken pedagogy into two artifacts that are baked
into the teacher system prompt:

    prompts/principles.md   distilled, DEDUPED coaching principles
    prompts/fewshots.json   2-3 leveled coaching exemplars per tier

Design (map -> reduce, cost-bounded):
    1. Sample ~6-9 transcripts, balanced across beginner/intermediate/advanced
       using ``manifest.json`` tier hints (falling back to a per-playlist
       default when a transcript is not in the manifest).
    2. MAP: for each transcript, build a representative digest by sampling a few
       evenly-spaced word-windows ("chunks"), then make ONE ``gpt-5.5`` call to
       extract PARAPHRASED coaching principles (never verbatim quotes).
    3. REDUCE: one synthesis call merges/dedupes all candidate principles into a
       clean set, emphasizing the behaviors central to this project.
    4. FEWSHOTS: one call writes original, board-agnostic leveled exemplars.

Everything the model produces is paraphrased/original: the transcripts are a
*pedagogy reference only*; nothing verbatim leaves this stage and the training
dataset itself stays 100% synthetic (see ``README.md``).

Secrets: ``OPENAI_API_KEY`` is loaded from ``ROOT/.env`` via ``python-dotenv``
and never logged.

Example:
    python src/teacher/distill_principles.py                 # full run
    python src/teacher/distill_principles.py --dry-run       # plan only, no API
    python src/teacher/distill_principles.py --max-transcripts 6
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #

# Project root = two levels up from src/teacher/distill_principles.py
ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS = ROOT / "data" / "transcripts"
CLEAN_DIR = TRANSCRIPTS / "clean"
MANIFEST = TRANSCRIPTS / "manifest.json"
PROMPTS = ROOT / "prompts"
TIER_GUIDES = PROMPTS / "tier_guides.md"
PRINCIPLES_OUT = PROMPTS / "principles.md"
FEWSHOTS_OUT = PROMPTS / "fewshots.json"

DEFAULT_MODEL = os.environ.get("TEACHER_MODEL", "gpt-5.5")
TIER_ORDER: tuple[str, ...] = ("beginner", "intermediate", "advanced")

# When a transcript is absent from the manifest (no title-embedded ELO), fall
# back to a coarse per-playlist tier so sampling still spans the tiers.
PLAYLIST_DEFAULT_TIER: dict[str, str] = {
    "gothamchess-win-at-chess": "beginner",
    "naroditsky-beginner-to-master-speedrun": "intermediate",
    "naroditsky-master-class-speedrun": "advanced",
}

# The four behaviors this project cares about most. These headings are fixed so
# principles.md always foregrounds them; the model fills their bullet points.
CENTRAL_HEADINGS: tuple[str, ...] = (
    "Teach the instructive move, not the engine's top choice",
    "Calibrate the explanation to the student's rating",
    "Tie every recommendation to a concrete plan",
    "Coach with an encouraging, human voice",
)

PROJECT_THESIS = (
    "We are building a chess move-review COACH. Given a position, the student's "
    "rating tier, the move they played, and verified engine analysis, the coach "
    "recommends ONE sound move and explains it in plain, level-appropriate human "
    "terms. The objectively best move does not change with rating, but the move "
    "worth TEACHING does: the coach often picks a sound, human, instructive move "
    "over the engine's #1. Coaching must be tied to a concrete plan and to the "
    "student's actual mistake, calibrated to the tier, encouraging in tone, and "
    "must NEVER use engine-speak (centipawns, evals, 'the computer') or dump long "
    "forcing lines."
)

# Words/phrases the coaching voice must never leak (behavioral guardrail).
ENGINE_SPEAK = re.compile(
    r"\b(centipawn|stockfish|engine|computer|eval(uation)?|\+\d|\-\d(?!\d*[a-z])"
    r"|winning by|mate in \d)\b",
    re.IGNORECASE,
)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Transcript discovery, tier inference, sampling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TranscriptRef:
    """A cleaned transcript on disk, tagged with an inferred coaching tier."""

    video_id: str
    playlist_slug: str
    tier: str
    title: str
    path: Path

    @property
    def label(self) -> str:
        """Short human label for logs/reports."""
        return f"{self.playlist_slug}/{self.video_id} [{self.tier}]"


_ELO_IN_TITLE = re.compile(r"\b(\d{3,4})\b")


def _tier_from_elo(elo: int) -> str:
    """Map a numeric ELO to a coarse tier (mirrors the ingest/config bands)."""
    if elo < 1300:
        return "beginner"
    if elo < 1700:
        return "intermediate"
    return "advanced"


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Load ``manifest.json`` into a ``{video_id: record}`` map (empty if absent)."""
    if not path.exists():
        return {}
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {r["video_id"]: r for r in records if isinstance(r, dict) and r.get("video_id")}


def infer_tier(video_id: str, playlist_slug: str, manifest: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """Return ``(tier, title)`` for a transcript.

    Precedence: an explicit ``tier_hint`` from the manifest, then a rating parsed
    from the manifest title, then the per-playlist default, then "intermediate".
    """
    rec = manifest.get(video_id, {})
    title = str(rec.get("title") or "")

    hint = rec.get("tier_hint")
    if isinstance(hint, str) and hint in TIER_ORDER:
        return hint, title

    if title:
        m = _ELO_IN_TITLE.search(title)
        if m:
            value = int(m.group(1))
            if 600 <= value <= 2800:
                return _tier_from_elo(value), title

    return PLAYLIST_DEFAULT_TIER.get(playlist_slug, "intermediate"), title


def discover_transcripts(clean_dir: Path, manifest: dict[str, dict[str, Any]]) -> list[TranscriptRef]:
    """Find every ``clean/<playlist>/<id>.txt`` and tag it with a tier."""
    refs: list[TranscriptRef] = []
    for txt in sorted(clean_dir.glob("*/*.txt")):
        video_id = txt.stem
        playlist_slug = txt.parent.name
        tier, title = infer_tier(video_id, playlist_slug, manifest)
        refs.append(
            TranscriptRef(
                video_id=video_id,
                playlist_slug=playlist_slug,
                tier=tier,
                title=title,
                path=txt,
            )
        )
    return refs


def sample_transcripts(
    refs: list[TranscriptRef],
    max_transcripts: int,
    per_tier: int,
) -> list[TranscriptRef]:
    """Round-robin sample across tiers for balance, deterministically.

    Picks at most ``per_tier`` from each tier and at most ``max_transcripts``
    total, cycling tiers so coverage stays even when one tier is data-rich.
    """
    buckets: dict[str, list[TranscriptRef]] = {t: [] for t in TIER_ORDER}
    for ref in refs:
        buckets.setdefault(ref.tier, []).append(ref)
    # Deterministic order within a tier: spread playlists, then by id.
    for tier in buckets:
        buckets[tier].sort(key=lambda r: (r.playlist_slug, r.video_id))

    selected: list[TranscriptRef] = []
    taken_per_tier: dict[str, int] = {t: 0 for t in buckets}
    cursor: dict[str, int] = {t: 0 for t in buckets}
    order = [t for t in TIER_ORDER if buckets.get(t)] + [
        t for t in buckets if t not in TIER_ORDER
    ]

    progress = True
    while len(selected) < max_transcripts and progress:
        progress = False
        for tier in order:
            if len(selected) >= max_transcripts:
                break
            if taken_per_tier[tier] >= per_tier:
                continue
            idx = cursor[tier]
            if idx < len(buckets[tier]):
                selected.append(buckets[tier][idx])
                cursor[tier] = idx + 1
                taken_per_tier[tier] += 1
                progress = True
    return selected


# --------------------------------------------------------------------------- #
# Chunking -> per-transcript digest
# --------------------------------------------------------------------------- #


def build_digest(text: str, chunk_words: int, max_chunks: int) -> tuple[str, int]:
    """Chunk ``text`` into word-windows and stitch a representative digest.

    Returns ``(digest, n_chunks_total)``. When the transcript has more windows
    than ``max_chunks``, evenly-spaced windows are sampled so the digest covers
    the whole video (opening ideas through endgame) instead of only the intro.
    """
    words = text.split()
    if not words:
        return "", 0
    chunks = [words[i : i + chunk_words] for i in range(0, len(words), chunk_words)]
    n_total = len(chunks)

    if n_total <= max_chunks:
        picked = chunks
    elif max_chunks <= 1:
        picked = [chunks[0]]
    else:
        idxs = sorted({round(k * (n_total - 1) / (max_chunks - 1)) for k in range(max_chunks)})
        picked = [chunks[i] for i in idxs]

    digest = "\n\n[...]\n\n".join(" ".join(c) for c in picked)
    return digest, n_total


# --------------------------------------------------------------------------- #
# gpt-5.5 client (chat + reasoning_effort, Responses API fallback)
# --------------------------------------------------------------------------- #


class GPTClient:
    """Thin ``gpt-5.5`` wrapper that prefers chat completions w/ reasoning_effort.

    A single preflight call decides the transport: ``chat.completions.create``
    with ``reasoning_effort`` if that is accepted, otherwise ``responses.create``
    with ``reasoning={"effort": ...}``. All subsequent calls reuse that mode.
    """

    def __init__(self, model: str, effort: str, verbose: bool = False) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("openai SDK not installed in this interpreter") from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found (checked ROOT/.env and env)")

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.effort = effort
        self.verbose = verbose
        self.mode: Optional[str] = None  # "chat" | "responses"

    # -- transports -------------------------------------------------------- #

    def _chat(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            reasoning_effort=self.effort,
        )
        return (resp.choices[0].message.content or "").strip()

    def _responses(self, system: str, user: str) -> str:
        resp = self._client.responses.create(
            model=self.model,
            reasoning={"effort": self.effort},
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = getattr(resp, "output_text", None)
        if text:
            return text.strip()
        # Fallback extraction if the SDK convenience attribute is unavailable.
        parts: list[str] = []
        for item in getattr(resp, "output", []) or []:
            for block in getattr(item, "content", []) or []:
                chunk = getattr(block, "text", None)
                if chunk:
                    parts.append(chunk)
        return "".join(parts).strip()

    # -- preflight + dispatch --------------------------------------------- #

    def preflight(self) -> str:
        """Make ONE tiny call to lock in a working transport. Returns the mode."""
        system = "You are a connectivity test."
        user = "Reply with the single word: ok"
        try:
            out = self._chat(system, user)
            self.mode = "chat"
            if self.verbose:
                print(f"  preflight: chat.completions OK (reply={out[:20]!r})")
            return self.mode
        except Exception as exc:  # noqa: BLE001 - probe both transports
            if self.verbose:
                print(f"  preflight: chat mode failed ({type(exc).__name__}); trying Responses")
        out = self._responses(system, user)  # let this raise if it also fails
        self.mode = "responses"
        if self.verbose:
            print(f"  preflight: responses OK (reply={out[:20]!r})")
        return self.mode

    def complete(self, system: str, user: str, tries: int = 3) -> str:
        """Run a completion in the selected mode with simple exponential retry."""
        if self.mode is None:
            self.preflight()
        call = self._chat if self.mode == "chat" else self._responses
        return _with_retries(lambda: call(system, user), tries=tries, verbose=self.verbose)

    def complete_json(self, system: str, user: str, tries: int = 3) -> Any:
        """Completion whose output is parsed as JSON (robust to prose/fences)."""
        raw = self.complete(system, user, tries=tries)
        return _extract_json(raw)


def _with_retries(fn: Callable[[], T], tries: int, verbose: bool) -> T:
    """Call ``fn`` up to ``tries`` times with 2/4/8s backoff on any exception."""
    last: Optional[Exception] = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - network/transient tolerance
            last = exc
            wait = 2 ** (attempt + 1)
            if verbose:
                print(f"    ! call failed ({type(exc).__name__}); retry in {wait}s")
            time.sleep(wait)
    assert last is not None
    raise last


def _extract_json(text: str) -> Any:
    """Parse a JSON object/array from a model reply, tolerating fences/prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the widest {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("Could not parse JSON from model reply")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_MAP_SYSTEM = (
    "You are a chess-pedagogy analyst. You read a messy, auto-captioned excerpt "
    "of a strong coach thinking aloud (GothamChess or GM Naroditsky) and extract "
    "the REUSABLE COACHING METHOD hidden in it: how they decide what to teach, "
    "how they explain a move, how they calibrate to a student's level, their tone, "
    "and what they avoid. Extract teaching TECHNIQUE, not chess trivia, opening "
    "names, or move-by-move narration of one game. PARAPHRASE in your own words. "
    "Never quote verbatim and never include player names."
)


def build_map_user(ref: TranscriptRef, digest: str) -> str:
    """Prompt to extract paraphrased coaching principles from one transcript."""
    return (
        f"{PROJECT_THESIS}\n\n"
        f"Transcript source tier: {ref.tier}. This is a representative digest "
        "(non-contiguous excerpts joined by [...]).\n\n"
        "Extract 8-14 reusable coaching principles as JSON. Favor principles about: "
        "choosing which move to teach, explaining ideas in plain language, tying a "
        "move to a plan or to the student's mistake, leveling to a rating, tone/"
        "encouragement, and habits/mistakes to name. Skip anything specific to one "
        "opening or one game.\n\n"
        'Return ONLY JSON: {"principles":[{"theme":"<short>","principle":'
        '"<one paraphrased sentence, imperative voice>","tiers":["beginner"|'
        '"intermediate"|"advanced"]}]}\n\n'
        f"--- DIGEST ---\n{digest}"
    )


_SYNTH_SYSTEM = (
    "You are the editor assembling an internal COACHING PRINCIPLES reference that "
    "will be injected into a chess move-review teacher prompt. Merge and DEDUPE "
    "candidate principles distilled from several coaches into one clean, organized, "
    "paraphrased set. Original wording only; no verbatim quotes, no player/opening "
    "names. Be concrete and imperative. Foreground the project's core behaviors."
)


def build_synth_user(candidates: list[dict[str, Any]]) -> str:
    """Prompt to synthesize/dedupe pooled principles into the final structure."""
    headings = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(CENTRAL_HEADINGS))
    pool = json.dumps(candidates, ensure_ascii=False, indent=0)
    return (
        f"{PROJECT_THESIS}\n\n"
        f"Fill these FIXED central headings (use them verbatim as keys), each with "
        f"3-6 deduped, paraphrased bullet points drawn from the candidates:\n{headings}\n\n"
        "Then produce an 'avoid' list (5-8 bullets: engine-speak, dumping lines, "
        "wrong-tier reasoning, teaching a move to unlearn, fabricating tactics) and "
        "a 'by_theme' section (4-6 themes, each 3-6 bullets) covering the remaining "
        "reusable pedagogy (e.g. choosing what to teach, explaining moves, leveling, "
        "tone & psychology, student habits to name).\n\n"
        "Return ONLY JSON with this schema:\n"
        '{"central":[{"heading":"<one of the fixed headings>","points":["..."]}],'
        '"avoid":["..."],"by_theme":[{"theme":"<short>","points":["..."]}]}\n\n'
        f"--- CANDIDATE PRINCIPLES ({len(candidates)}) ---\n{pool}"
    )


_FEWSHOT_SYSTEM = (
    "You write short, original COACHING EXEMPLARS for a chess move-review coach. "
    "Each shows the ideal voice for one rating tier: name what the student's move "
    "allows, offer a better sound idea tied to a concrete plan, and end with a "
    "transferable takeaway. Plain human language only. Never use engine-speak "
    "(no centipawns, evals, 'the computer', mate counts) and never narrate lines "
    "longer than the tier's depth. Board-agnostic/generic situations are fine; do "
    "not copy any transcript."
)


def build_fewshot_user(central_md: str, tier_guides: str) -> str:
    """Prompt to write 2-3 leveled exemplars per tier as JSON."""
    return (
        f"{PROJECT_THESIS}\n\n"
        "Match the voice to these distilled central principles:\n"
        f"{central_md}\n\n"
        "Respect these per-tier leveling guides (vocabulary + depth):\n"
        f"{tier_guides}\n\n"
        "Write 2-3 exemplars for EACH tier. Keep 'situation' to one generic "
        "sentence about the student's move/position problem, and 'coaching' to "
        "2-4 sentences that end with a takeaway. Simpler ideas for beginner than "
        "advanced.\n\n"
        'Return ONLY JSON: {"beginner":[{"situation":"...","coaching":"..."}],'
        '"intermediate":[...],"advanced":[...]}'
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_principles_md(synth: dict[str, Any], used: list[TranscriptRef]) -> str:
    """Render the synthesis JSON into the final principles.md markdown."""
    central = {c.get("heading", ""): c.get("points", []) for c in synth.get("central", [])}
    lines: list[str] = []
    lines.append("# Chess Coaching Principles (distilled reference)")
    lines.append("")
    lines.append(
        "> Internal, **paraphrased** pedagogy distilled from strong-coach commentary "
        "(GothamChess \"Win At Chess\", Naroditsky speedruns). Original wording only — "
        "no verbatim quotes. Injected into the teacher system prompt as `{PRINCIPLES}`."
    )
    lines.append(">")
    lines.append(f"> Distilled from {len(used)} transcript(s) across tiers.")
    lines.append("")

    lines.append("## Central to this project")
    lines.append("")
    for i, heading in enumerate(CENTRAL_HEADINGS, start=1):
        lines.append(f"### {i}. {heading}")
        points = central.get(heading) or _match_points(central, heading)
        for pt in points:
            lines.append(f"- {str(pt).strip()}")
        if not points:
            lines.append("- _(no distilled points)_")
        lines.append("")

    lines.append("## What to avoid")
    lines.append("")
    for pt in synth.get("avoid", []):
        lines.append(f"- {str(pt).strip()}")
    lines.append("")

    lines.append("## Principles by theme")
    lines.append("")
    for block in synth.get("by_theme", []):
        theme = str(block.get("theme", "")).strip() or "General"
        lines.append(f"### {theme}")
        for pt in block.get("points", []):
            lines.append(f"- {str(pt).strip()}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _match_points(central: dict[str, list[Any]], heading: str) -> list[Any]:
    """Best-effort match if the model lightly reworded a fixed heading."""
    key = heading.lower()[:18]
    for h, pts in central.items():
        if h.lower()[:18] == key:
            return pts
    return []


def validate_fewshots(data: Any) -> dict[str, list[dict[str, str]]]:
    """Coerce/validate the few-shot payload into ``{tier: [{situation,coaching}]}``."""
    if not isinstance(data, dict):
        raise ValueError("few-shots payload is not a JSON object")
    out: dict[str, list[dict[str, str]]] = {}
    for tier in TIER_ORDER:
        items = data.get(tier) or []
        clean: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            situation = str(item.get("situation", "")).strip()
            coaching = str(item.get("coaching", "")).strip()
            if situation and coaching:
                clean.append({"situation": situation, "coaching": coaching})
        out[tier] = clean
    return out


def scan_engine_speak(fewshots: dict[str, list[dict[str, str]]]) -> list[str]:
    """Return warnings for any exemplar whose coaching leaks engine-speak."""
    warnings: list[str] = []
    for tier, items in fewshots.items():
        for i, item in enumerate(items):
            hit = ENGINE_SPEAK.search(item["coaching"])
            if hit:
                warnings.append(f"{tier}[{i}] contains forbidden term: {hit.group(0)!r}")
    return warnings


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass
class RunPlan:
    """The sampled + digested inputs, produced before any API spend."""

    used: list[TranscriptRef] = field(default_factory=list)
    digests: dict[str, str] = field(default_factory=dict)  # video_id -> digest
    chunk_counts: dict[str, int] = field(default_factory=dict)


def plan_run(args: argparse.Namespace) -> RunPlan:
    """Discover, sample, and digest transcripts (no network calls)."""
    manifest = load_manifest(MANIFEST)
    refs = discover_transcripts(CLEAN_DIR, manifest)
    if not refs:
        raise RuntimeError(f"No transcripts found under {CLEAN_DIR}")

    used = sample_transcripts(refs, args.max_transcripts, args.per_tier)
    plan = RunPlan(used=used)
    for ref in used:
        text = ref.path.read_text(encoding="utf-8", errors="replace")
        digest, n_chunks = build_digest(text, args.chunk_words, args.max_chunks_per_transcript)
        plan.digests[ref.video_id] = digest
        plan.chunk_counts[ref.video_id] = n_chunks
    return plan


def print_plan(plan: RunPlan, refs_total: int) -> None:
    """Log the sampling/chunking plan for transparency."""
    print(f"Discovered {refs_total} transcript(s); sampled {len(plan.used)}:")
    tier_counts: dict[str, int] = {}
    for ref in plan.used:
        tier_counts[ref.tier] = tier_counts.get(ref.tier, 0) + 1
        words = len(plan.digests[ref.video_id].split())
        print(
            f"  - {ref.label}  (digest ~{words} words from "
            f"{plan.chunk_counts[ref.video_id]} chunk(s))"
        )
    spread = ", ".join(f"{t}={tier_counts.get(t, 0)}" for t in TIER_ORDER)
    print(f"  tier spread: {spread}")


def run(args: argparse.Namespace) -> int:
    """Full pipeline: sample -> map -> synthesize -> few-shots -> write files."""
    plan = plan_run(args)
    refs_total = len(discover_transcripts(CLEAN_DIR, load_manifest(MANIFEST)))
    print_plan(plan, refs_total)

    if args.dry_run:
        print("\n[dry-run] no API calls made; no files written.")
        return 0

    client = GPTClient(model=args.model, effort=args.reasoning_effort, verbose=args.verbose)
    print(f"\nVerifying gpt-5.5 connectivity (model={args.model}, effort={args.reasoning_effort}) ...")
    mode = client.preflight()
    print(f"  transport: {mode}")

    # -- MAP: per-transcript principle extraction -------------------------- #
    print("\nExtracting principles per transcript ...")
    candidates: list[dict[str, Any]] = []
    used_ok: list[TranscriptRef] = []
    for ref in plan.used:
        digest = plan.digests[ref.video_id]
        if not digest.strip():
            print(f"  - {ref.label}: empty digest, skipped")
            continue
        try:
            data = client.complete_json(_MAP_SYSTEM, build_map_user(ref, digest))
            items = data.get("principles", []) if isinstance(data, dict) else []
        except Exception as exc:  # noqa: BLE001 - tolerate one bad transcript
            print(f"  - {ref.label}: extraction failed ({type(exc).__name__}), skipped")
            continue
        for it in items:
            if isinstance(it, dict) and str(it.get("principle", "")).strip():
                candidates.append(
                    {
                        "theme": str(it.get("theme", "")).strip(),
                        "principle": str(it["principle"]).strip(),
                        "tiers": it.get("tiers", []),
                        "source_tier": ref.tier,
                    }
                )
        used_ok.append(ref)
        print(f"  - {ref.label}: +{len(items)} principle(s)")

    if not candidates:
        print("\nBLOCKED: no principles extracted from any transcript.")
        return 3

    # -- REDUCE: synthesize + dedupe --------------------------------------- #
    print(f"\nSynthesizing {len(candidates)} candidate principles ...")
    synth = client.complete_json(_SYNTH_SYSTEM, build_synth_user(candidates))
    if not isinstance(synth, dict):
        raise RuntimeError("synthesis did not return a JSON object")
    principles_md = render_principles_md(synth, used_ok)
    PRINCIPLES_OUT.write_text(principles_md, encoding="utf-8")
    print(f"  wrote {PRINCIPLES_OUT.relative_to(ROOT)} ({len(principles_md)} chars)")

    # -- FEWSHOTS: leveled exemplars --------------------------------------- #
    print("\nGenerating leveled few-shot exemplars ...")
    central_md = _central_summary(synth)
    tier_guides = TIER_GUIDES.read_text(encoding="utf-8") if TIER_GUIDES.exists() else ""
    fewshot_raw = client.complete_json(_FEWSHOT_SYSTEM, build_fewshot_user(central_md, tier_guides))
    fewshots = validate_fewshots(fewshot_raw)
    for warn in scan_engine_speak(fewshots):
        print(f"  ! engine-speak warning: {warn}")
    FEWSHOTS_OUT.write_text(json.dumps(fewshots, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = ", ".join(f"{t}={len(fewshots[t])}" for t in TIER_ORDER)
    print(f"  wrote {FEWSHOTS_OUT.relative_to(ROOT)} ({counts})")

    # -- Report ------------------------------------------------------------ #
    print("\n" + "=" * 66)
    print("SUMMARY")
    print(f"  transcripts used: {len(used_ok)}")
    for ref in used_ok:
        print(f"    - {ref.label}")
    print(f"  candidate principles: {len(candidates)}")
    print(f"  files: {PRINCIPLES_OUT.relative_to(ROOT)}, {FEWSHOTS_OUT.relative_to(ROOT)}")
    print("\n--- principles.md (first 15 lines) ---")
    for line in principles_md.splitlines()[:15]:
        print(line)
    return 0


def _central_summary(synth: dict[str, Any]) -> str:
    """Compact text of the central section to steer the few-shot voice."""
    out: list[str] = []
    for c in synth.get("central", []):
        out.append(f"- {c.get('heading', '')}")
        for pt in c.get("points", [])[:3]:
            out.append(f"  * {str(pt).strip()}")
    return "\n".join(out)


def build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Teacher model id (default: {DEFAULT_MODEL}).")
    p.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"],
                   help="Reasoning effort for gpt-5.5 (default: high).")
    p.add_argument("--max-transcripts", type=int, default=9, help="Max transcripts to sample (default: 9).")
    p.add_argument("--per-tier", type=int, default=3, help="Max transcripts per tier (default: 3).")
    p.add_argument("--chunk-words", type=int, default=2600, help="Words per chunk window (default: 2600).")
    p.add_argument("--max-chunks-per-transcript", type=int, default=3,
                   help="Evenly-spaced chunks stitched into each digest (default: 3).")
    p.add_argument("--dry-run", action="store_true", help="Plan + digest only; no API calls, no writes.")
    p.add_argument("--verbose", action="store_true", help="Verbose diagnostics.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint. Loads .env (never logging the key) and runs the pipeline."""
    args = build_argparser().parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        print("WARNING: python-dotenv not installed; relying on process env", file=sys.stderr)

    try:
        return run(args)
    except RuntimeError as exc:
        print(f"\nBLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
