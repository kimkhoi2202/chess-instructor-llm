#!/usr/bin/env python3
"""Sample real human chess *decision positions* from Lichess, bucketed by rating band.

This module pulls a SMALL sample of real, rated human games from the public
Lichess API and extracts middlegame "decision positions" into a JSONL file that
downstream steps (Stockfish soundness -> Maia likelihood -> GPT teacher) consume.

Output schema (one JSON object per line in ``data/positions/positions.jsonl``)::

    {
      "id": "<game_id>_<ply>",
      "fen": "<FEN before the played move>",
      "tier": "beginner|intermediate|advanced",
      "played_move_uci": "e2e4",
      "played_move_san": "e4",
      "side_to_move": "white|black",
      "mover_rating": 1523,
      "game_id": "<lichess id>",
      "ply": 17,
      "time_control": "blitz"
    }

Rating tiers (bands)::

    beginner      1000-1200
    intermediate  1300-1600
    advanced      1700-2000

How positions are attributed to a tier (v0, deliberately simple)
----------------------------------------------------------------
A position is attributed to a tier by the **actual in-game rating of the side to
move** (the ``WhiteElo``/``BlackElo`` recorded by Lichess for *that* game), not by
the identity of the seed user. This is the most honest per-position attribution we
can do without the full database dump: the mover's rating is exactly the label we
want for "what would a human at this level play here". Ratings between bands (e.g.
1250) fall in an intentional gap and are dropped; provisional ratings are dropped
by default because they are noisy.

How we find in-band humans without downloading the DB dump
----------------------------------------------------------
There is no public "give me games at rating X" endpoint, so we *discover* in-band
humans by crawling:

1. Seed with the Lichess **Maia bots** (``maia1``/``maia5``/``maia9``). These are
   real accounts that play thousands of rated games against humans whose ratings
   span every band we care about (observed opponents ~850-1900). We use the bots
   ONLY to discover human opponents -- bot moves are never recorded as human data.
2. For every game we see, we enqueue the (non-bot) opponents whose rating is in a
   broad discovery window, then mine *their* human-vs-human games. Beginners play
   beginners, so a breadth-first crawl quickly descends into each band.
3. We optionally also seed from the blitz leaderboard, but note that top players
   (~3000) sit far above the "advanced" band, so they mainly demonstrate the
   endpoint and contribute strong opponents to the crawl frontier.

Per-user ratings are additionally verified via ``GET /api/user/{username}`` for the
seed users so the run log documents roughly where each anchor sits.

Politeness / rate limits
------------------------
We send a descriptive ``User-Agent``, stream NDJSON, throttle between requests, and
respect ``Retry-After`` on HTTP 429. A ``LICHESS_TOKEN`` (loaded from ``.env`` only,
never hard-coded) is used if present to raise rate limits; it is optional.

Fallback
--------
If the crawl is blocked/rate-limited and collects nothing, the sampler falls back to
fetching a small list of specific public games via ``GET /game/export/{gameId}`` and
produces the same schema (see ``--mode`` and ``--fallback-game-ids``). The limitation
is reported to stderr.

Example::

    python src/ingest/lichess_sampler.py --count 60
    python src/ingest/lichess_sampler.py --count 90 --games-per-user 20 --use-leaderboard
    python src/ingest/lichess_sampler.py --mode fallback --fallback-game-ids abcd1234,wxyz5678
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import chess
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

API_BASE = "https://lichess.org"
USER_AGENT = (
    "chess-instructor-llm/0.1 (leveled chess-coach dataset research; "
    "polite small-sample API use)"
)

#: Strict rating bands used for *attribution* of a recorded position.
TIERS: dict[str, tuple[int, int]] = {
    "beginner": (1000, 1200),
    "intermediate": (1300, 1600),
    "advanced": (1700, 2000),
}
TIER_ORDER: tuple[str, ...] = ("beginner", "intermediate", "advanced")

#: Broad window used for *discovery* (crawl frontier). Wider than the union of the
#: bands so we can bridge the inter-band gaps and reach each band's edges.
DISCOVER_LO = 950
DISCOVER_HI = 2050

#: Maia bots make excellent discovery anchors: real, always-online opponents whose
#: human challengers span every band. Their own (bot) moves are never recorded.
DEFAULT_SEED_USERS: tuple[str, ...] = ("maia1", "maia5", "maia9")

#: Human-vs-human public games spanning the bands, used only by the fallback path.
#: These are best-effort defaults harvested from a live crawl (game IDs are stable
#: on Lichess); override with --fallback-game-ids if any 404 or you want others.
DEFAULT_FALLBACK_GAME_IDS: tuple[str, ...] = (
    "yTjIRay8",  # beginner-band movers (~1080)
    "KB0rBPfu",  # beginner-band movers
    "IKFxmHWR",  # intermediate-band movers
    "Q2IOLJ1C",  # intermediate/advanced-band movers
    "jEFIU8MJ",  # advanced-band movers
)

PERF_TYPES = "blitz,rapid"
REQUEST_TIMEOUT = (10, 60)  # (connect, read) seconds


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def band_of(rating: Optional[int]) -> Optional[str]:
    """Return the tier name whose band contains ``rating``, else ``None``."""
    if rating is None:
        return None
    for tier, (lo, hi) in TIERS.items():
        if lo <= rating <= hi:
            return tier
    return None


def balanced_targets(count: int) -> dict[str, int]:
    """Split ``count`` as evenly as possible across the three tiers."""
    base, extra = divmod(count, len(TIER_ORDER))
    return {t: base + (1 if i < extra else 0) for i, t in enumerate(TIER_ORDER)}


def sample_plies(n_plies: int, skip_open: int, skip_end: int, per_game: int) -> list[int]:
    """Pick up to ``per_game`` evenly spaced middlegame plies (1-based half-moves).

    Skips the first ``skip_open`` plies (opening book) and the last ``skip_end``
    plies (endgame/scramble). Returns a sorted list of 1-based ply indices.
    """
    lo = skip_open + 1
    hi = n_plies - skip_end
    if hi < lo:
        return []
    span = hi - lo + 1
    k = min(per_game, span)
    if k <= 0:
        return []
    step = span / k
    plies = {int(lo + step * (j + 0.5)) for j in range(k)}
    return sorted(min(max(p, lo), hi) for p in plies)


# --------------------------------------------------------------------------- #
# HTTP session
# --------------------------------------------------------------------------- #

def make_session(token: Optional[str]) -> requests.Session:
    """Build a requests session with a descriptive UA and optional bearer token."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})
    return session


@dataclass
class Budget:
    """Tracks the HTTP request budget and per-request throttle."""

    max_requests: int
    sleep: float
    used: int = 0

    def spend(self) -> None:
        self.used += 1

    def exhausted(self) -> bool:
        return self.used >= self.max_requests

    def throttle(self) -> None:
        if self.sleep > 0:
            time.sleep(self.sleep)


def _get_stream(
    session: requests.Session,
    url: str,
    params: dict,
    budget: Budget,
    max_retries: int,
) -> Iterator[dict]:
    """GET ``url`` as an NDJSON stream, yielding decoded objects.

    Handles HTTP 429 by honoring ``Retry-After`` (bounded) and retrying. Raises
    ``requests.HTTPError`` for other non-2xx responses.
    """
    headers = {"Accept": "application/x-ndjson"}
    attempt = 0
    while True:
        budget.spend()
        try:
            with session.get(
                url, params=params, headers=headers, stream=True, timeout=REQUEST_TIMEOUT
            ) as resp:
                if resp.status_code == 429:
                    attempt += 1
                    if attempt > max_retries:
                        raise requests.HTTPError("429 Too Many Requests (retries exhausted)")
                    wait = min(int(resp.headers.get("Retry-After", "60") or "60"), 90)
                    print(f"  [rate-limited] sleeping {wait}s then retrying…", file=sys.stderr)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    if raw:
                        yield json.loads(raw)
                return
        finally:
            budget.throttle()


def verify_user_band(session: requests.Session, username: str, budget: Budget) -> Optional[int]:
    """Return a user's best blitz/rapid rating via ``/api/user`` (for run logging)."""
    budget.spend()
    try:
        resp = session.get(f"{API_BASE}/api/user/{username}", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        perfs = resp.json().get("perfs", {})
    except (requests.RequestException, ValueError):
        return None
    finally:
        budget.throttle()
    ratings = [perfs.get(p, {}).get("rating") for p in ("blitz", "rapid")]
    ratings = [r for r in ratings if isinstance(r, int)]
    return max(ratings) if ratings else None


def fetch_top_blitz_users(session: requests.Session, n: int, budget: Budget) -> list[str]:
    """Fetch the top-``n`` blitz leaderboard usernames (strong-player seed source)."""
    budget.spend()
    try:
        resp = session.get(f"{API_BASE}/api/player/top/{n}/blitz", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        users = resp.json().get("users", [])
    except (requests.RequestException, ValueError) as exc:
        print(f"  [warn] leaderboard fetch failed: {exc}", file=sys.stderr)
        return []
    finally:
        budget.throttle()
    return [u["username"] for u in users if "username" in u]


def iter_user_games(
    session: requests.Session, username: str, max_games: int, budget: Budget, max_retries: int
) -> Iterator[dict]:
    """Yield a user's most recent rated blitz/rapid standard games as dicts."""
    params = {
        "max": max_games,
        "rated": "true",
        "perfType": PERF_TYPES,
        "moves": "true",
        "clocks": "false",
        "evals": "false",
        "pgnInJson": "false",
    }
    yield from _get_stream(
        session, f"{API_BASE}/api/games/user/{username}", params, budget, max_retries
    )


def fetch_game_by_id(
    session: requests.Session, game_id: str, budget: Budget, max_retries: int
) -> Optional[dict]:
    """Fetch a single public game by ID via ``/game/export`` (fallback path)."""
    params = {"moves": "true", "clocks": "false", "evals": "false", "pgnInJson": "false"}
    for obj in _get_stream(
        session, f"{API_BASE}/game/export/{game_id}", params, budget, max_retries
    ):
        return obj
    return None


# --------------------------------------------------------------------------- #
# Position extraction
# --------------------------------------------------------------------------- #

@dataclass
class Player:
    name: Optional[str]
    title: Optional[str]
    rating: Optional[int]
    provisional: bool

    @property
    def is_bot(self) -> bool:
        return self.title == "BOT"


def _player(side: dict) -> Player:
    user = side.get("user") or {}
    return Player(
        name=user.get("name"),
        title=user.get("title"),
        rating=side.get("rating"),
        provisional=bool(side.get("provisional", False)),
    )


def extract_positions(
    game: dict,
    *,
    targets: dict[str, int],
    counts: dict[str, int],
    seen_ids: set[str],
    skip_open: int,
    skip_end: int,
    per_game: int,
    allow_provisional: bool,
    require_human_vs_human: bool,
) -> list[dict]:
    """Extract in-band, human-mover decision positions from a single game dict.

    Only standard, rated blitz/rapid games are used. By default only genuine
    human-vs-human games are recorded (``require_human_vs_human``); games against a
    bot (e.g. the Maia discovery anchors) contribute no records here but are still
    mined for opponents by the caller. Each returned record follows the output
    schema. Positions whose tier is already full (``counts >= targets``) are skipped
    so the final file stays balanced across tiers.
    """
    if game.get("variant") != "standard" or not game.get("rated"):
        return []
    speed = game.get("speed")
    if speed not in ("blitz", "rapid"):
        return []

    moves_str = game.get("moves") or ""
    sans = moves_str.split()
    if not sans:
        return []

    game_id = game.get("id")
    if not game_id:
        return []

    white = _player(game["players"]["white"])
    black = _player(game["players"]["black"])

    # Real-human-games gate: skip the whole game if either side is a bot.
    if require_human_vs_human and (white.is_bot or black.is_bot):
        return []

    chosen = set(sample_plies(len(sans), skip_open, skip_end, per_game))
    if not chosen:
        return []

    records: list[dict] = []
    board = chess.Board()
    for i, san in enumerate(sans):
        ply = i + 1  # 1-based half-move index
        try:
            move = board.parse_san(san)
        except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            break  # corrupt move stream — stop replaying this game

        if ply in chosen:
            mover = white if board.turn == chess.WHITE else black
            tier = band_of(mover.rating)
            usable = (
                tier is not None
                and counts.get(tier, 0) < targets.get(tier, 0)
                and not mover.is_bot
                and (allow_provisional or not mover.provisional)
            )
            if usable:
                rec_id = f"{game_id}_{ply}"
                if rec_id not in seen_ids:
                    records.append(
                        {
                            "id": rec_id,
                            "fen": board.fen(),  # position BEFORE the played move
                            "tier": tier,
                            "played_move_uci": move.uci(),
                            "played_move_san": board.san(move),
                            "side_to_move": "white" if board.turn == chess.WHITE else "black",
                            "mover_rating": int(mover.rating),
                            "game_id": game_id,
                            "ply": ply,
                            "time_control": speed,
                        }
                    )
        board.push(move)

    return records


# --------------------------------------------------------------------------- #
# Crawl frontier
# --------------------------------------------------------------------------- #

@dataclass
class Frontier:
    """Discovery frontier that biases work toward the least-filled tier."""

    seeds: deque[str] = field(default_factory=deque)
    by_tier: dict[str, deque[str]] = field(
        default_factory=lambda: {t: deque() for t in TIER_ORDER}
    )
    general: deque[str] = field(default_factory=deque)
    seen: set[str] = field(default_factory=set)

    def add_seed(self, name: str) -> None:
        if name and name.lower() not in self.seen:
            self.seen.add(name.lower())
            self.seeds.append(name)

    def discover(self, name: Optional[str], rating: Optional[int]) -> None:
        if not name or name.lower() in self.seen or rating is None:
            return
        if not (DISCOVER_LO <= rating <= DISCOVER_HI):
            return
        self.seen.add(name.lower())
        tier = band_of(rating)
        (self.by_tier[tier] if tier else self.general).append(name)

    def next_user(self, counts: dict[str, int], targets: dict[str, int]) -> Optional[str]:
        if self.seeds:
            return self.seeds.popleft()
        # Prefer the tier furthest from its target that still has candidates.
        ranked = sorted(
            TIER_ORDER,
            key=lambda t: (counts.get(t, 0) / targets[t]) if targets[t] else 1.0,
        )
        for tier in ranked:
            if counts.get(tier, 0) < targets.get(tier, 0) and self.by_tier[tier]:
                return self.by_tier[tier].popleft()
        if self.general:
            return self.general.popleft()
        for tier in TIER_ORDER:
            if self.by_tier[tier]:
                return self.by_tier[tier].popleft()
        return None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _tiers_full(counts: dict[str, int], targets: dict[str, int]) -> bool:
    return all(counts.get(t, 0) >= targets.get(t, 0) for t in TIER_ORDER)


def run_crawl(session: requests.Session, args: argparse.Namespace, budget: Budget) -> list[dict]:
    """Discover in-band humans via the Maia anchors and extract balanced positions."""
    targets = balanced_targets(args.count)
    counts: dict[str, int] = {t: 0 for t in TIER_ORDER}
    seen_ids: set[str] = set()
    out: list[dict] = []

    frontier = Frontier()
    seed_users = list(dict.fromkeys(args.seed_users))  # de-dup, keep order
    if args.use_leaderboard:
        seed_users += fetch_top_blitz_users(session, args.leaderboard_n, budget)
    for name in seed_users:
        frontier.add_seed(name)

    # Verify + log where the (small) seed set sits.
    for name in list(dict.fromkeys(seed_users))[: args.verify_seeds]:
        rating = verify_user_band(session, name, budget)
        print(f"  seed {name!r}: best blitz/rapid rating = {rating}", file=sys.stderr)

    print(f"  targets per tier: {targets}", file=sys.stderr)

    bar = tqdm(total=args.count, desc="positions", unit="pos", disable=not args.progress)
    try:
        while not _tiers_full(counts, targets) and not budget.exhausted():
            user = frontier.next_user(counts, targets)
            if user is None:
                break
            try:
                games = list(iter_user_games(session, user, args.games_per_user, budget, args.max_retries))
            except requests.RequestException as exc:
                print(f"  [warn] games fetch failed for {user!r}: {exc}", file=sys.stderr)
                continue

            for game in games:
                # Discovery: enqueue non-bot opponents in the broad window.
                for side in ("white", "black"):
                    p = _player(game["players"][side])
                    if not p.is_bot:
                        frontier.discover(p.name, p.rating)
                # Extraction.
                recs = extract_positions(
                    game,
                    targets=targets,
                    counts=counts,
                    seen_ids=seen_ids,
                    skip_open=args.skip_open,
                    skip_end=args.skip_end,
                    per_game=args.positions_per_game,
                    allow_provisional=args.allow_provisional,
                    require_human_vs_human=not args.include_bot_games,
                )
                for rec in recs:
                    tier = rec["tier"]
                    if counts[tier] >= targets[tier]:
                        continue
                    seen_ids.add(rec["id"])
                    counts[tier] += 1
                    out.append(rec)
                    bar.update(1)
                if _tiers_full(counts, targets):
                    break
    finally:
        bar.close()

    print(f"  crawl finished: counts={counts} requests_used={budget.used}", file=sys.stderr)
    return out


def run_fallback(session: requests.Session, args: argparse.Namespace, budget: Budget) -> list[dict]:
    """Fetch a fixed list of public games by ID and extract positions (no discovery)."""
    game_ids = list(dict.fromkeys(args.fallback_game_ids or DEFAULT_FALLBACK_GAME_IDS))
    if not game_ids:
        print(
            "  [fallback] no game IDs available. Pass --fallback-game-ids id1,id2,…",
            file=sys.stderr,
        )
        return []

    # Fallback relaxes the balance target so a fixed ID list can still populate the
    # file; attribution is still by the mover's true in-game rating band.
    targets = {t: args.count for t in TIER_ORDER}
    counts: dict[str, int] = {t: 0 for t in TIER_ORDER}
    seen_ids: set[str] = set()
    out: list[dict] = []

    print(f"  [fallback] exporting {len(game_ids)} game(s) by ID", file=sys.stderr)
    for gid in game_ids:
        try:
            game = fetch_game_by_id(session, gid, budget, args.max_retries)
        except requests.RequestException as exc:
            print(f"  [warn] export failed for {gid!r}: {exc}", file=sys.stderr)
            continue
        if not game:
            print(f"  [warn] no data for game {gid!r}", file=sys.stderr)
            continue
        recs = extract_positions(
            game,
            targets=targets,
            counts=counts,
            seen_ids=seen_ids,
            skip_open=args.skip_open,
            skip_end=args.skip_end,
            per_game=args.positions_per_game,
            allow_provisional=True,  # be permissive in fallback
            require_human_vs_human=not args.include_bot_games,
        )
        for rec in recs:
            seen_ids.add(rec["id"])
            counts[rec["tier"]] += 1
            out.append(rec)

    print(f"  fallback finished: counts={counts}", file=sys.stderr)
    return out


def write_jsonl(records: list[dict], out_path: Path) -> None:
    """Write records to ``out_path`` (atomically via a temp file + replace)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(out_path)


def summarize(records: list[dict]) -> dict[str, int]:
    counts = {t: 0 for t in TIER_ORDER}
    for rec in records:
        counts[rec["tier"]] = counts.get(rec["tier"], 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample real human chess decision positions from Lichess by rating band.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count", type=int, default=60, help="Total positions (balanced across 3 tiers).")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/positions/positions.jsonl"),
        help="Output JSONL path (relative to the project root).",
    )
    parser.add_argument("--mode", choices=("auto", "crawl", "fallback"), default="auto",
                        help="'auto' crawls then falls back if it collects nothing.")
    parser.add_argument("--seed-users", nargs="+", default=list(DEFAULT_SEED_USERS),
                        help="Discovery anchors (Maia bots by default).")
    parser.add_argument("--use-leaderboard", action="store_true",
                        help="Also seed from the blitz leaderboard (strong players; mostly above bands).")
    parser.add_argument("--leaderboard-n", type=int, default=10, help="How many leaderboard users to seed.")
    parser.add_argument("--verify-seeds", type=int, default=3,
                        help="How many seed users to rating-verify via /api/user (logging).")
    parser.add_argument("--games-per-user", type=int, default=15, help="Games to fetch per crawled user.")
    parser.add_argument("--positions-per-game", type=int, default=3, help="Middlegame positions per game.")
    parser.add_argument("--skip-open", type=int, default=16, help="Opening plies to skip (~8 full moves).")
    parser.add_argument("--skip-end", type=int, default=6, help="Ending plies to skip.")
    parser.add_argument("--max-requests", type=int, default=150, help="Hard cap on HTTP requests.")
    parser.add_argument("--sleep", type=float, default=0.7, help="Seconds to sleep between requests.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries on HTTP 429 per request.")
    parser.add_argument("--allow-provisional", action="store_true",
                        help="Keep positions where the mover's rating is provisional.")
    parser.add_argument("--include-bot-games", action="store_true",
                        help="Also record the human side of human-vs-bot games "
                             "(default: real human-vs-human games only).")
    parser.add_argument("--fallback-game-ids", type=lambda s: [x for x in s.split(",") if x],
                        default=None, help="Comma-separated public game IDs for fallback mode.")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="Disable the tqdm progress bar.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    load_dotenv()  # token from .env only — never hard-coded
    token = os.getenv("LICHESS_TOKEN") or None
    print(f"Lichess token: {'present (raising rate limits)' if token else 'absent (public limits)'}",
          file=sys.stderr)

    session = make_session(token)
    budget = Budget(max_requests=args.max_requests, sleep=args.sleep)

    records: list[dict] = []
    if args.mode in ("auto", "crawl"):
        records = run_crawl(session, args, budget)
    if args.mode == "fallback" or (args.mode == "auto" and not records):
        if args.mode == "auto" and not records:
            print("  [auto] crawl produced 0 positions — switching to fallback.", file=sys.stderr)
        records += run_fallback(session, args, budget)

    write_jsonl(records, args.out)

    counts = summarize(records)
    print("\n=== summary ===")
    print(f"output: {args.out}")
    print(f"total positions: {len(records)}")
    for tier in TIER_ORDER:
        lo, hi = TIERS[tier]
        print(f"  {tier:<12} ({lo}-{hi}): {counts.get(tier, 0)}")
    for rec in records[:2]:
        print("example:", json.dumps(rec, ensure_ascii=False))

    if not records:
        print("BLOCKED: no positions collected.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
