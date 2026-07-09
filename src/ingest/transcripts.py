#!/usr/bin/env python3
"""Harvest, clean, and catalog YouTube auto-caption transcripts for chess coaching.

This module drives ``yt-dlp`` (the Homebrew CLI, *not* the YouTube Data API) to
pull English captions from a fixed set of chess-coaching playlists, converts the
downloaded WebVTT files into de-duplicated plaintext, and writes a manifest that
downstream steps use to distill coaching *principles* (pedagogy reference only —
the training dataset itself stays 100% synthetic; see ``README.md``).

Pipeline per playlist:
    1. Enumerate video ids + titles with ``yt-dlp --flat-playlist -J``.
    2. For each video, download English subs *without* the media with
       ``--skip-download --write-auto-subs --write-subs --sub-langs "en.*"``.
    3. Parse the best available ``.vtt`` into clean plaintext (strip timestamps,
       cue numbers, inline word-timing tags, and collapse the rolling-caption
       repetition that YouTube auto-captions emit).
    4. Emit ``data/transcripts/manifest.json`` — a list of per-video records.

Outputs (all under ``data/transcripts/`` which is gitignored — internal use):
    raw/<playlist_id>/<video_id>.<lang>.vtt   original WebVTT (archival)
    clean/<playlist_slug>/<video_id>.txt      cleaned plaintext transcript
    manifest.json                             list[dict] catalog

Example:
    python src/ingest/transcripts.py --per-playlist-limit 3
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_YT_DLP = "/opt/homebrew/bin/yt-dlp"

# Project root = two levels up from this file (src/ingest/transcripts.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Playlist:
    """A source playlist to harvest.

    Attributes:
        name: Human-friendly name recorded in the manifest.
        slug: Filesystem-safe folder name under ``clean/``.
        url: Full YouTube playlist URL (must contain a ``list=`` query param).
    """

    name: str
    slug: str
    url: str

    @property
    def playlist_id(self) -> str:
        """Return the ``PL...`` id parsed from the playlist URL's ``list`` param."""
        qs = parse_qs(urlparse(self.url).query)
        ids = qs.get("list", [])
        if not ids:
            raise ValueError(f"Playlist URL missing 'list=' param: {self.url}")
        return ids[0]


# Curated chess-coaching playlists (verified playlist IDs). Chosen to maximize
# (a) TIER-APPROPRIATE coaching — the same ideas taught at many rating levels
# (Naroditsky speedruns, Chessbrah "Building Habits", GothamChess rating climbs),
# and (b) transferable PRINCIPLES/pedagogy (Hanging Pawns strategy series, tips
# and guides). Transcripts feed the coaching principle library + coaching STYLE
# reference only — the training dataset itself stays engine-grounded/synthetic.
PLAYLISTS: list[Playlist] = [
    # --- GothamChess (Levy Rozman) — rating-ladder climbs + principle guides --- #
    Playlist(
        name="GothamChess - Win At Chess",
        slug="gothamchess-win-at-chess",
        url="https://www.youtube.com/playlist?list=PLBRObSmbZluSo6h0AySyeZRdlQzEhr2XL",
    ),
    Playlist(
        name="GothamChess - Chess Tips",
        slug="gothamchess-chess-tips",
        url="https://www.youtube.com/playlist?list=PLBRObSmbZluSEeBkH17c72MaM4nFjqoi8",
    ),
    Playlist(
        name="GothamChess - Guide",
        slug="gothamchess-guide",
        url="https://www.youtube.com/playlist?list=PLBRObSmbZluRBQOO_6FzyxQUaFyzusSl0",
    ),
    # --- Daniel Naroditsky — tier-progressive speedruns + mastery lessons ------ #
    Playlist(
        name="Naroditsky - Beginner to Master Speedrun",
        slug="naroditsky-beginner-to-master-speedrun",
        url="https://www.youtube.com/playlist?list=PLT1F2nOxLHOfQ-eoJTpyvKkQFwYewDduj",
    ),
    Playlist(
        name="Naroditsky - Master Class Speedrun",
        slug="naroditsky-master-class-speedrun",
        url="https://www.youtube.com/playlist?list=PLT1F2nOxLHOefj_z54LNBpnASnIROm43e",
    ),
    Playlist(
        name="Naroditsky - Develop Your Instincts Speedrun",
        slug="naroditsky-develop-your-instincts-speedrun",
        url="https://www.youtube.com/playlist?list=PLT1F2nOxLHOdrvOyOXb_l2yGJrkwLA72Z",
    ),
    Playlist(
        name="Naroditsky - Sensei Speedrun",
        slug="naroditsky-sensei-speedrun",
        url="https://www.youtube.com/playlist?list=PLT1F2nOxLHOeyyw85utYJpWtSmxvA-2WR",
    ),
    Playlist(
        name="Naroditsky - Chess Mastery Explained",
        slug="naroditsky-chess-mastery-explained",
        url="https://www.youtube.com/playlist?list=PLT1F2nOxLHOcZlKiT0J-ov5-RsM9taTvm",
    ),
    # --- Chessbrah (Aman Hambleton) — explicit rating-tier "Building Habits" --- #
    Playlist(
        name="Chessbrah - Building Chess Habits",
        slug="chessbrah-building-chess-habits",
        url="https://www.youtube.com/playlist?list=PLUjxDD7HNNThftJtE0OIRFRMMFf6AV_69",
    ),
    Playlist(
        name="Chessbrah - Building Habits v2",
        slug="chessbrah-building-habits-v2",
        url="https://www.youtube.com/playlist?list=PLUjxDD7HNNThwCNW3f36RZcMxPwQIjYae",
    ),
    Playlist(
        name="Chessbrah - 100 Tips from a GM",
        slug="chessbrah-100-tips-from-a-gm",
        url="https://www.youtube.com/playlist?list=PLUjxDD7HNNTj46EZKKxsU_WgeqENUJzYC",
    ),
    # --- Hanging Pawns — transferable strategy/endgame principles -------------- #
    Playlist(
        name="Hanging Pawns - Chess Lessons",
        slug="hangingpawns-chess-lessons",
        url="https://www.youtube.com/playlist?list=PLssNbVBYrGcBXTr66cBqHuQUecSSdzF87",
    ),
    Playlist(
        name="Hanging Pawns - Middlegame Strategy",
        slug="hangingpawns-middlegame-strategy",
        url="https://www.youtube.com/playlist?list=PLssNbVBYrGcAiV9arkX-uqWk0q9S38ETE",
    ),
    Playlist(
        name="Hanging Pawns - Endgame Strategy",
        slug="hangingpawns-endgame-strategy",
        url="https://www.youtube.com/playlist?list=PLssNbVBYrGcDvGO9P9mxJEqwMlVZom4YL",
    ),
    Playlist(
        name="Hanging Pawns - Instructive Game Analysis",
        slug="hangingpawns-instructive-game-analysis",
        url="https://www.youtube.com/playlist?list=PLssNbVBYrGcC-yPo5qoPqdsHFpWvcmvs3",
    ),
]


@dataclass
class VideoRecord:
    """One manifest row describing a harvested (or attempted) video."""

    video_id: str
    title: str
    playlist_name: str
    playlist_id: str
    url: str
    tier_hint: Optional[str]
    elo_hint: Optional[int]
    has_transcript: bool = False
    transcript_path: Optional[str] = None  # relative to data/transcripts/


# --------------------------------------------------------------------------- #
# Tier / ELO parsing
# --------------------------------------------------------------------------- #

# "1100 ELO", "1500 rated"
_NUM_THEN_KEYWORD = re.compile(r"(\d{3,4})\s*(?:elo|rated)\b", re.IGNORECASE)
# "ELO 1100", "rated: 1500", "Elo #1800"
_KEYWORD_THEN_NUM = re.compile(r"(?:elo|rated)\s*[:#]?\s*(\d{3,4})\b", re.IGNORECASE)
# ranges like "700-1200" (GothamChess "Win At Chess" episode titles)
_RANGE = re.compile(r"\b(\d{3,4})\s*[-\u2013\u2014]\s*(\d{3,4})\b")
# bare 3-4 digit number as a last resort
_BARE = re.compile(r"\b(\d{3,4})\b")


def elo_to_tier(elo: int) -> str:
    """Map a numeric ELO hint to a coarse tier label.

    Buckets follow the tiers locked in ``README.md`` (Beginner 1000-1200 /
    Intermediate 1300-1600 / Advanced 1700-2000), widened to cover the lower
    speedrun ratings and everything above 2000.
    """
    if elo < 1300:
        return "beginner"
    if elo < 1700:
        return "intermediate"
    return "advanced"


def parse_tier(title: str) -> tuple[Optional[int], Optional[str]]:
    """Extract an ELO hint and mapped tier from a video title.

    Precedence: an explicit "N ELO"/"ELO N" (or "rated") pattern wins; then a
    numeric range (midpoint); then any plausible bare rating (600-2800). Returns
    ``(None, None)`` when nothing rating-like is found (e.g. GothamChess
    "Episode 1" with no rating in the title).
    """
    elo: Optional[int] = None

    m = _NUM_THEN_KEYWORD.search(title) or _KEYWORD_THEN_NUM.search(title)
    if m:
        elo = int(m.group(1))

    if elo is None:
        m = _RANGE.search(title)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            elo = (lo + hi) // 2

    if elo is None:
        for bm in _BARE.finditer(title):
            value = int(bm.group(1))
            if 600 <= value <= 2800:  # plausible chess rating; skips "Episode 1"
                elo = value
                break

    if elo is None:
        return None, None
    return elo, elo_to_tier(elo)


# --------------------------------------------------------------------------- #
# VTT parsing
# --------------------------------------------------------------------------- #

_TAG = re.compile(r"<[^>]+>")  # <c>, </c>, <00:00:01.234>
_HEADER_PREFIXES = ("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE", "REGION")


def _clean_line(line: str) -> str:
    """Strip inline tags/entities from a single caption text line."""
    line = _TAG.sub("", line)
    line = html.unescape(line)  # &nbsp; -> \xa0, &amp; -> & etc.
    line = line.replace("\xa0", " ")
    return re.sub(r"\s+", " ", line).strip()


def parse_vtt(path: Path) -> str:
    """Parse a WebVTT file into clean, de-duplicated plaintext.

    Handles both YouTube caption flavors:
      * segmented tracks (``en``/``en-en``) — non-overlapping cues, and
      * the raw ASR track (``en-orig``) — rolling cues where each cue repeats the
        previously finalized line then appends new words with inline timestamps.

    Timestamp/cue-number/header lines are dropped, inline ``<...>`` tags removed,
    HTML entities unescaped, and consecutive duplicate lines collapsed (which is
    exactly what turns the rolling-caption repetition back into flowing text).
    Kept lines are joined with single spaces.
    """
    kept: list[str] = []
    last: Optional[str] = None

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith(_HEADER_PREFIXES):
                continue
            if "-->" in raw:  # cue timing line
                continue
            if raw.isdigit():  # cue sequence number
                continue

            text = _clean_line(raw)
            if not text or text == last:
                continue
            kept.append(text)
            last = text

    return " ".join(kept)


def select_vtt(video_id: str, raw_dir: Path) -> Optional[Path]:
    """Choose the best English ``.vtt`` for a video from the raw download dir.

    Prefers the clean segmented tracks (``en``, then ``en-en``, then regional
    ``en-US``/``en-GB``) over the verbose rolling ``en-orig`` track. Returns
    ``None`` when no caption file was produced.
    """
    candidates = sorted(raw_dir.glob(f"{video_id}*.vtt"))
    if not candidates:
        return None

    def lang_of(p: Path) -> str:
        stem = p.name[len(video_id):]
        if stem.startswith("."):
            stem = stem[1:]
        if stem.endswith(".vtt"):
            stem = stem[: -len(".vtt")]
        return stem.lower()

    priority = {"en": 0, "en-en": 1, "en-us": 2, "en-gb": 2}

    def score(p: Path) -> int:
        lang = lang_of(p)
        if lang in priority:
            return priority[lang]
        if "orig" in lang:
            return 9  # complete but verbose rolling track — last resort
        if lang.startswith("en"):
            return 5
        return 20

    return min(candidates, key=score)


# --------------------------------------------------------------------------- #
# yt-dlp helpers
# --------------------------------------------------------------------------- #


def _run(
    cmd: list[str],
    timeout: int,
    verbose: bool,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess from the project root, capturing text output."""
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=not verbose,
        text=True,
        timeout=timeout,
        check=False,
    )


def enumerate_playlist(
    yt_dlp: str,
    playlist: Playlist,
    limit: int,
    sleep_requests: float,
    timeout: int,
    verbose: bool,
) -> list[tuple[str, str]]:
    """Return ``[(video_id, title), ...]`` for up to ``limit`` playlist videos.

    Uses ``--flat-playlist -J`` and reads the ``entries`` array. On any failure
    (network, parse) an empty list is returned so the run can continue.
    """
    cmd = [
        yt_dlp,
        "--flat-playlist",
        "-J",
        "--playlist-end",
        str(limit),
        "--sleep-requests",
        str(sleep_requests),
        playlist.url,
    ]
    try:
        proc = _run(cmd, timeout=timeout, verbose=False)
    except subprocess.TimeoutExpired:
        print(f"  ! enumeration timed out for {playlist.name}", file=sys.stderr)
        return []
    if proc.returncode != 0:
        print(
            f"  ! enumeration failed for {playlist.name} (rc={proc.returncode})",
            file=sys.stderr,
        )
        if verbose and proc.stderr:
            print(proc.stderr, file=sys.stderr)
        return []

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"  ! could not parse playlist JSON: {exc}", file=sys.stderr)
        return []

    out: list[tuple[str, str]] = []
    for entry in data.get("entries") or []:
        vid = entry.get("id")
        if not vid:
            continue
        title = entry.get("title") or ""
        out.append((vid, title))
    return out


def download_subs(
    yt_dlp: str,
    video_id: str,
    playlist_id: str,
    sleep_requests: float,
    sleep_interval: float,
    max_sleep_interval: float,
    timeout: int,
    verbose: bool,
) -> None:
    """Download English captions for one video (no media) via yt-dlp.

    Files land at ``data/transcripts/raw/<playlist_id>/<id>.<lang>.vtt`` (the
    output template resolves relative to the project root). The playlist id is
    substituted literally rather than via ``%(playlist_id)s`` because that field
    resolves to ``NA`` when a bare ``watch?v=`` URL is downloaded outside a
    playlist context. Failures are logged but never raised — a captionless video
    simply yields no ``.vtt``.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_template = f"data/transcripts/raw/{playlist_id}/%(id)s.%(ext)s"
    cmd = [
        yt_dlp,
        "--skip-download",
        "--write-auto-subs",
        "--write-subs",
        "--sub-langs",
        "en.*",
        "--sub-format",
        "vtt",
        "--sleep-requests",
        str(sleep_requests),
        "--sleep-interval",
        str(sleep_interval),
        "--max-sleep-interval",
        str(max_sleep_interval),
        "-o",
        out_template,
        url,
    ]
    try:
        proc = _run(cmd, timeout=timeout, verbose=verbose)
    except subprocess.TimeoutExpired:
        print(f"  ! caption download timed out for {video_id}", file=sys.stderr)
        return
    if proc.returncode != 0 and verbose and proc.stderr:
        # Non-zero is common (e.g. no subs); keep going regardless.
        print(proc.stderr, file=sys.stderr)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def harvest_playlist(
    playlist: Playlist,
    args: argparse.Namespace,
    raw_root: Path,
    clean_root: Path,
) -> list[VideoRecord]:
    """Enumerate, download, and clean transcripts for a single playlist."""
    print(f"\n=== {playlist.name} ({playlist.playlist_id}) ===")
    videos = enumerate_playlist(
        args.yt_dlp,
        playlist,
        limit=args.per_playlist_limit,
        sleep_requests=args.sleep_requests,
        timeout=args.enumerate_timeout,
        verbose=args.verbose,
    )
    print(f"  found {len(videos)} video(s)")

    raw_dir = raw_root / playlist.playlist_id
    clean_dir = clean_root / playlist.slug
    clean_dir.mkdir(parents=True, exist_ok=True)

    records: list[VideoRecord] = []
    for idx, (video_id, title) in enumerate(videos, start=1):
        elo_hint, tier_hint = parse_tier(title)
        record = VideoRecord(
            video_id=video_id,
            title=title,
            playlist_name=playlist.name,
            playlist_id=playlist.playlist_id,
            url=f"https://www.youtube.com/watch?v={video_id}",
            tier_hint=tier_hint,
            elo_hint=elo_hint,
        )

        existing = sorted(raw_dir.glob(f"{video_id}*.vtt"))
        if existing and not args.force:
            print(f"  [{idx}/{len(videos)}] {video_id} — using cached captions")
        else:
            print(f"  [{idx}/{len(videos)}] {video_id} — downloading captions")
            download_subs(
                args.yt_dlp,
                video_id,
                playlist.playlist_id,
                sleep_requests=args.sleep_requests,
                sleep_interval=args.sleep_interval,
                max_sleep_interval=args.max_sleep_interval,
                timeout=args.download_timeout,
                verbose=args.verbose,
            )
            if args.sleep_interval:
                time.sleep(args.sleep_interval)  # extra politeness between videos

        vtt = select_vtt(video_id, raw_dir)
        if vtt is None:
            print(f"        no captions — recording success=false")
            records.append(record)
            continue

        try:
            text = parse_vtt(vtt)
        except OSError as exc:
            print(f"        failed to read {vtt.name}: {exc}", file=sys.stderr)
            records.append(record)
            continue

        if not text.strip():
            print(f"        parsed transcript empty — success=false")
            records.append(record)
            continue

        out_path = clean_dir / f"{video_id}.txt"
        out_path.write_text(text + "\n", encoding="utf-8")
        record.has_transcript = True
        record.transcript_path = f"clean/{playlist.slug}/{video_id}.txt"
        records.append(record)
        print(f"        wrote {out_path.relative_to(PROJECT_ROOT)} ({len(text)} chars)")

    return records


def build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Harvest English auto-caption transcripts from chess-coaching "
            "playlists, clean them to plaintext, and write a manifest."
        )
    )
    parser.add_argument(
        "--per-playlist-limit",
        type=int,
        default=15,
        help="Max videos to process per playlist (default: 15, polite first pass).",
    )
    parser.add_argument(
        "--yt-dlp",
        default=shutil.which("yt-dlp") or DEFAULT_YT_DLP,
        help=f"Path to the yt-dlp binary (default: {DEFAULT_YT_DLP}).",
    )
    parser.add_argument(
        "--sleep-requests",
        type=float,
        default=1.0,
        help="Seconds to sleep between yt-dlp requests (default: 1.0).",
    )
    parser.add_argument(
        "--sleep-interval",
        type=float,
        default=1.0,
        help="Min seconds to sleep before each download (default: 1.0).",
    )
    parser.add_argument(
        "--max-sleep-interval",
        type=float,
        default=5.0,
        help="Max seconds to sleep before each download (default: 5.0).",
    )
    parser.add_argument(
        "--enumerate-timeout",
        type=int,
        default=180,
        help="Per-playlist enumeration timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=180,
        help="Per-video caption download timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download captions even if raw .vtt files already exist.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream yt-dlp output instead of capturing it.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    args = build_argparser().parse_args(argv)

    data_root = PROJECT_ROOT / "data" / "transcripts"
    raw_root = data_root / "raw"
    clean_root = data_root / "clean"
    raw_root.mkdir(parents=True, exist_ok=True)
    clean_root.mkdir(parents=True, exist_ok=True)

    if not (Path(args.yt_dlp).exists() or shutil.which(args.yt_dlp)):
        print(f"ERROR: yt-dlp not found at {args.yt_dlp!r}", file=sys.stderr)
        return 2

    all_records: list[VideoRecord] = []
    summary: list[tuple[str, int, int]] = []  # (name, found, with_transcript)
    for playlist in PLAYLISTS:
        records = harvest_playlist(playlist, args, raw_root, clean_root)
        found = len(records)
        with_tx = sum(1 for r in records if r.has_transcript)
        summary.append((playlist.name, found, with_tx))
        all_records.extend(records)

    manifest_path = data_root / "manifest.json"
    manifest_path.write_text(
        json.dumps([r.__dict__ for r in all_records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # Summary report
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SUMMARY")
    for name, found, with_tx in summary:
        print(f"  {name}: {found} found / {with_tx} with transcript")
    total_found = sum(f for _, f, _ in summary)
    total_tx = sum(t for _, _, t in summary)
    print(f"  TOTAL: {total_found} found / {total_tx} with transcript")
    print(f"  manifest: {manifest_path.relative_to(PROJECT_ROOT)} ({len(all_records)} records)")

    sample = next((r for r in all_records if r.has_transcript), None)
    if sample and sample.transcript_path:
        sample_text = (data_root / sample.transcript_path).read_text(encoding="utf-8")
        print("\n--- sample cleaned transcript (first ~300 chars) ---")
        print(f"[{sample.playlist_name}] {sample.title}")
        print(sample_text[:300].strip())
    else:
        print("\n(no transcripts obtained)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
