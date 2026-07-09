#!/usr/bin/env python3
"""Acquire a large, DIVERSE, multi-source RAW chess data **BANK** onto a Modal
volume (``chess-data`` on the **chess-instructor-2** workspace) and mirror the
freely-redistributable (CC0 / self-derived) artifacts to a *private* Hugging
Face dataset repo for durability.

Framing (see the accompanying report): the goal is a big raw bank to **mine**
from — rich in *discriminating* multi-tier positions (where the tier-appropriate
move differs by level) and in *coaching pedagogy* — **not** a labeled training
set. Everything streams/downloads *directly onto the Modal volume* so local disk
stays clean.

Sources gathered
----------------
1. **Lichess puzzle database** (CC0) — the ~6M-row CSV with per-puzzle Glicko-2
   *rating* (a built-in difficulty/tier signal), *themes* (motif), FEN, solution
   moves, popularity, and opening tags. The single best mining source for
   discriminating, tier-labeled tactical positions.
2. **Lichess standard games** (CC0) — a representative *streamed slice* of a
   recent monthly rated-games dump (never the full ~28 GB). Every game carries
   ``WhiteElo``/``BlackElo`` so positions are rating-attributable per side.
3. **Curated Hugging Face datasets** — master/tournament PGN corpora
   (Caissabase mirror, Lichess tournament games), coaching-pedagogy exemplars
   (turning-point explanations), CoT reasoning, and rating-stratified SFT mixes.

Run (from repo root; kim-lam tokens unset; NEVER the kim-lam workspace)::

    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    MODAL_PROFILE=chess-instructor-2 modal run src/ingest/modal_data_bank.py                 # all (parallel)
    MODAL_PROFILE=chess-instructor-2 modal run src/ingest/modal_data_bank.py --stage puzzles
    MODAL_PROFILE=chess-instructor-2 modal run src/ingest/modal_data_bank.py --stage games
    MODAL_PROFILE=chess-instructor-2 modal run src/ingest/modal_data_bank.py --stage hf
    MODAL_PROFILE=chess-instructor-2 modal run src/ingest/modal_data_bank.py --stage manifest
    MODAL_PROFILE=chess-instructor-2 modal run src/ingest/modal_data_bank.py --stage mirror
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from typing import Any, Optional

import modal

# --------------------------------------------------------------------------- #
# Names / layout
# --------------------------------------------------------------------------- #
APP_NAME = "chess-data-bank"
VOLUME_NAME = "chess-data"
VOL = "/vol"

# Volume layout (one subtree per source; each writes a ``_meta.json`` sidecar).
DIR_PUZZLES = f"{VOL}/lichess/puzzles"
DIR_GAMES = f"{VOL}/lichess/games"
DIR_HF = f"{VOL}/hf"
DIR_TRANSCRIPTS = f"{VOL}/transcripts"  # populated via `modal volume put` from local

# Lichess open database (CC0). Static CDN files — no token needed.
PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
GAMES_URL_TMPL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"

# HF durability mirror (private — avoids re-hosting third-party licensed data).
HF_MIRROR_REPO = "khoilamalphaai/chess-data-bank"

# --------------------------------------------------------------------------- #
# Curated HF datasets to snapshot in full onto the volume.
# (repo_id, license, note) — chosen for mining value, sized to be cheap.
# Huge sets (laion 838 GB, chess-position-evaluations 42 GB, chess-puzzles-with-games
# 18 GB) are intentionally NOT snapshotted; they're recorded as streaming refs.
# --------------------------------------------------------------------------- #
HF_DATASETS: list[dict[str, str]] = [
    {"repo": "mapama247/chess_games_caissabase", "license": "apache-2.0",
     "note": "Caissabase master-game corpus mirror (~2.4M curated master games). Master PGN."},
    {"repo": "Lichess/tournament-chess-games", "license": "cc-by-sa-4.0",
     "note": "Lichess tournament/broadcast games (titled/strong play, many with evals)."},
    {"repo": "suman-kalavagunta/chess-coach-turningpoints", "license": "apache-2.0",
     "note": "Engine-grounded human-style COACHING explanations of turning points. Pedagogy gold."},
    {"repo": "suman-kalavagunta/chess-coach-turningpoints-gpt-v5", "license": "unknown",
     "note": "Newer GPT-v5 coaching turning-point explanations (coaching STYLE exemplars)."},
    {"repo": "open-chess/MetaChess-20k", "license": "apache-2.0",
     "note": "20k positions with structured Chain-of-Thought analysis (reasoning STYLE)."},
    {"repo": "cetusian/chess-sft-mix-200k", "license": "cc0-1.0",
     "note": "Rating-stratified puzzles + GM games + Stockfish labels (discriminating by construction)."},
    {"repo": "Thytu/ChessInstruct", "license": "cc-by-4.0",
     "note": "Instruction-formatted chess tasks (varied task framing)."},
    {"repo": "HardlySalty/annotated_chess_games", "license": "unknown",
     "note": "Annotated games with natural-language commentary (coaching commentary raw material)."},
    {"repo": "patrickfrank1/chess-pgn-games", "license": "cc0-1.0",
     "note": "CC0 PGN game corpus (extra raw games for position mining)."},
]

# Recorded in the manifest as references only (too big to snapshot; stream on demand).
HF_REFERENCES: list[dict[str, str]] = [
    {"repo": "Lichess/chess-puzzles", "license": "cc0-1.0", "approx": "871 MB",
     "note": "Parquet form of the puzzle DB (we store the canonical CSV instead)."},
    {"repo": "Lichess/standard-chess-games", "license": "cc0-1.0", "approx": "7.14B rows",
     "note": "Full rated-games dataset; we take a streamed PGN slice instead."},
    {"repo": "Lichess/chess-position-evaluations", "license": "cc0-1.0", "approx": "42 GB",
     "note": "Stockfish evals per position — stream for grounding, do not hoard."},
    {"repo": "Lichess/chess-puzzles-with-games", "license": "cc0-1.0", "approx": "18 GB",
     "note": "Puzzles WITH their source games (coaching narrative context) — stream a slice."},
    {"repo": "laion/strategic_game_chess", "license": "cc-by-4.0", "approx": "838 GB",
     "note": "Massive engine self-play corpus — reference only."},
]

# Rating buckets for the puzzle/game rating histograms (mining-tier signal).
# Aligned to the project tiers, plus coarse buckets spanning the full range.
COARSE_BUCKETS = [(0, 800), (800, 1200), (1200, 1600), (1600, 2000),
                  (2000, 2400), (2400, 3600)]
PROJECT_TIERS = {"beginner": (1000, 1200), "intermediate": (1300, 1600),
                 "advanced": (1700, 2000)}

# --------------------------------------------------------------------------- #
# Modal infra
# --------------------------------------------------------------------------- #
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("zstd")
    .pip_install("requests", "zstandard", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)

# Timeouts: HF snapshot is the long pole (~3.8 GB).
T_SHORT = 45 * 60
T_LONG = 90 * 60


# --------------------------------------------------------------------------- #
# Small helpers (run remotely)
# --------------------------------------------------------------------------- #
def _du_bytes(path: str) -> int:
    """Total size in bytes of a file or directory tree."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _write_meta(dir_path: str, meta: dict[str, Any]) -> None:
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)


def _bucketize(rating: int, buckets: list[tuple[int, int]]) -> Optional[str]:
    for lo, hi in buckets:
        if lo <= rating < hi:
            return f"{lo}-{hi}"
    return None


# --------------------------------------------------------------------------- #
# Stage 1: Lichess puzzle database (CC0)
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={VOL: volume}, timeout=T_SHORT)
def fetch_lichess_puzzles() -> dict:
    """Download the puzzle CSV.zst to the volume, decompress, and profile it.

    Keeps BOTH the canonical ``.csv.zst`` (small, for the HF mirror) and the
    decompressed ``.csv`` (for direct mining). Computes rating + theme
    histograms so the manifest shows exactly how discriminating the bank is.
    """
    import csv

    import requests

    os.makedirs(DIR_PUZZLES, exist_ok=True)
    zst_path = os.path.join(DIR_PUZZLES, "lichess_db_puzzle.csv.zst")
    csv_path = os.path.join(DIR_PUZZLES, "lichess_db_puzzle.csv")

    print(f"[puzzles] downloading {PUZZLE_URL}")
    t0 = time.time()
    with requests.get(PUZZLE_URL, stream=True, timeout=(10, 300)) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        nextpct = 10
        with open(zst_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total and done * 100 // total >= nextpct:
                    print(f"[puzzles]   {done/1e6:.0f}/{total/1e6:.0f} MB")
                    nextpct += 10
    print(f"[puzzles] downloaded {done/1e6:.1f} MB in {time.time()-t0:.0f}s")

    print("[puzzles] decompressing with zstd CLI …")
    subprocess.run(["zstd", "-d", "-f", "-q", zst_path, "-o", csv_path], check=True)

    # Single streaming pass: count rows + rating/theme histograms.
    print("[puzzles] profiling CSV …")
    n = 0
    n_with_opening = 0
    coarse: dict[str, int] = {}
    tiers: dict[str, int] = {k: 0 for k in PROJECT_TIERS}
    themes: dict[str, int] = {}
    rating_min, rating_max = 10_000, -1
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n += 1
            try:
                r = int(row["Rating"])
            except (KeyError, ValueError):
                r = -1
            if r >= 0:
                rating_min = min(rating_min, r)
                rating_max = max(rating_max, r)
                b = _bucketize(r, COARSE_BUCKETS)
                if b:
                    coarse[b] = coarse.get(b, 0) + 1
                for name, (lo, hi) in PROJECT_TIERS.items():
                    if lo <= r <= hi:
                        tiers[name] += 1
            if row.get("OpeningTags"):
                n_with_opening += 1
            for th in (row.get("Themes") or "").split():
                themes[th] = themes.get(th, 0) + 1

    top_themes = dict(sorted(themes.items(), key=lambda kv: -kv[1])[:25])
    meta = {
        "source": "Lichess puzzle database",
        "url": PUZZLE_URL,
        "license": "CC0-1.0 (public domain)",
        "path": DIR_PUZZLES.replace(VOL, ""),
        "files": ["lichess_db_puzzle.csv.zst", "lichess_db_puzzle.csv"],
        "count": n,
        "size_bytes": _du_bytes(DIR_PUZZLES),
        "fields": ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation",
                   "Popularity", "NbPlays", "Themes", "GameUrl", "OpeningTags"],
        "rating_range": [rating_min, rating_max],
        "rating_histogram_coarse": dict(sorted(coarse.items())),
        "project_tier_counts": tiers,
        "n_with_opening_tags": n_with_opening,
        "n_distinct_themes": len(themes),
        "top_themes": top_themes,
        "mining_value": "PRIMARY discriminating source: per-puzzle rating = difficulty/"
                        "tier signal; themes = motif labels; FEN+Moves = ground-truth line.",
    }
    _write_meta(DIR_PUZZLES, meta)
    volume.commit()
    print(f"[puzzles] done: {n:,} puzzles, {meta['size_bytes']/1e6:.0f} MB on volume")
    return meta


# --------------------------------------------------------------------------- #
# Stage 2: Lichess standard games — streamed representative slice (CC0)
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={VOL: volume}, timeout=T_LONG)
def fetch_lichess_games_slice(month: str = "2026-06",
                              max_games: int = 200_000,
                              max_mb: int = 350) -> dict:
    """Stream a monthly rated-games ``.pgn.zst`` and keep only the first slice.

    Decompresses on the fly and stops after ``max_games`` whole games or
    ``max_mb`` of decompressed PGN, whichever comes first — so we pull only a
    few tens of MB of compressed data, never the full ~28 GB. Records a rating
    histogram from the per-game Elo headers (positions are rating-attributable).
    """
    import io

    import requests
    import zstandard as zstd

    os.makedirs(DIR_GAMES, exist_ok=True)
    url = GAMES_URL_TMPL.format(month=month)
    out_path = os.path.join(DIR_GAMES, f"lichess_slice_{month}.pgn")
    max_bytes = max_mb * 1_000_000

    print(f"[games] streaming slice from {url} (<= {max_games:,} games / {max_mb} MB)")
    n_games = 0
    written = 0
    white_elos: dict[str, int] = {}
    black_elos: dict[str, int] = {}
    n_with_eval = 0
    re_welo = re.compile(rb'\[WhiteElo "(\d+)"\]')
    re_belo = re.compile(rb'\[BlackElo "(\d+)"\]')

    t0 = time.time()
    with requests.get(url, stream=True, timeout=(10, 300)) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = True
        dctx = zstd.ZstdDecompressor()
        game_buf: list[bytes] = []
        cur_has_eval = False

        with open(out_path, "wb") as out, dctx.stream_reader(resp.raw) as reader:
            bufr = io.BufferedReader(reader, buffer_size=1 << 20)
            for raw in bufr:
                # A new game starts at an [Event ...] header line.
                if raw.startswith(b"[Event ") and game_buf:
                    chunk = b"".join(game_buf)
                    out.write(chunk)
                    written += len(chunk)
                    n_games += 1
                    if cur_has_eval:
                        n_with_eval += 1
                    game_buf = []
                    cur_has_eval = False
                    if n_games >= max_games or written >= max_bytes:
                        break
                    if n_games % 20_000 == 0:
                        print(f"[games]   {n_games:,} games / {written/1e6:.0f} MB")
                game_buf.append(raw)
                m = re_welo.search(raw)
                if m:
                    b = _bucketize(int(m.group(1)), COARSE_BUCKETS)
                    if b:
                        white_elos[b] = white_elos.get(b, 0) + 1
                m = re_belo.search(raw)
                if m:
                    b = _bucketize(int(m.group(1)), COARSE_BUCKETS)
                    if b:
                        black_elos[b] = black_elos.get(b, 0) + 1
                if b"%eval" in raw:
                    cur_has_eval = True
            # flush a trailing complete game if we ended by EOF (not by break)
            if game_buf and n_games < max_games and written < max_bytes:
                chunk = b"".join(game_buf)
                out.write(chunk)
                written += len(chunk)
                n_games += 1

    meta = {
        "source": f"Lichess standard rated games (slice, {month})",
        "url": url,
        "license": "CC0-1.0 (public domain)",
        "path": DIR_GAMES.replace(VOL, ""),
        "files": [f"lichess_slice_{month}.pgn"],
        "count": n_games,
        "size_bytes": _du_bytes(DIR_GAMES),
        "n_with_engine_eval": n_with_eval,
        "white_elo_histogram": dict(sorted(white_elos.items())),
        "black_elo_histogram": dict(sorted(black_elos.items())),
        "mining_value": "Real human games across the full rating spectrum; per-side Elo "
                        "headers make middlegame positions rating-attributable for tier mining.",
        "note": "Representative prefix slice of the monthly dump (time-ordered ≈ random wrt rating). "
                "Full month is ~28 GB; only a few tens of MB compressed were transferred.",
    }
    _write_meta(DIR_GAMES, meta)
    volume.commit()
    print(f"[games] done: {n_games:,} games, {written/1e6:.0f} MB in {time.time()-t0:.0f}s")
    return meta


# --------------------------------------------------------------------------- #
# Stage 3: curated Hugging Face datasets → volume
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={VOL: volume}, timeout=T_LONG)
def fetch_hf_datasets() -> dict:
    """Snapshot the curated HF chess datasets onto the volume, pinned by SHA."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    os.makedirs(DIR_HF, exist_ok=True)
    results: list[dict] = []
    for spec in HF_DATASETS:
        repo = spec["repo"]
        local = os.path.join(DIR_HF, repo.replace("/", "__"))
        print(f"[hf] downloading dataset {repo} -> {local}")
        try:
            info = api.dataset_info(repo)
            sha = info.sha
        except Exception as exc:  # noqa: BLE001
            sha = None
            print(f"[hf]   dataset_info failed for {repo}: {exc}")
        try:
            snapshot_download(repo_id=repo, repo_type="dataset", local_dir=local,
                              revision=sha, max_workers=8)
        except Exception as exc:  # noqa: BLE001
            print(f"[hf]   FAILED {repo}: {type(exc).__name__} {exc}")
            results.append({"repo": repo, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        size = _du_bytes(local)
        nfiles = sum(len(fs) for _r, _d, fs in os.walk(local))
        rec = {"repo": repo, "ok": True, "sha": sha, "license": spec["license"],
               "note": spec["note"], "path": local.replace(VOL, ""),
               "size_bytes": size, "n_files": nfiles}
        results.append(rec)
        print(f"[hf]   ok {repo}: {size/1e6:.0f} MB, {nfiles} files, sha={sha}")
        volume.commit()

    meta = {
        "source": "Curated Hugging Face chess datasets",
        "path": DIR_HF.replace(VOL, ""),
        "datasets": results,
        "size_bytes": _du_bytes(DIR_HF),
        "n_ok": sum(1 for r in results if r.get("ok")),
    }
    _write_meta(DIR_HF, meta)
    volume.commit()
    print(f"[hf] done: {meta['n_ok']}/{len(HF_DATASETS)} datasets, {meta['size_bytes']/1e6:.0f} MB")
    return meta


# --------------------------------------------------------------------------- #
# Stage 4: manifest
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={VOL: volume}, timeout=T_SHORT)
def build_manifest() -> dict:
    """Aggregate every ``_meta.json`` on the volume into ``/vol/manifest.json``."""
    volume.reload()
    sources: list[dict] = []
    for sub in (DIR_PUZZLES, DIR_GAMES, DIR_HF, DIR_TRANSCRIPTS):
        mp = os.path.join(sub, "_meta.json")
        if os.path.exists(mp):
            with open(mp, "r", encoding="utf-8") as fh:
                sources.append(json.load(fh))
        elif os.path.isdir(sub):
            sources.append({"source": sub.replace(VOL, ""), "path": sub.replace(VOL, ""),
                            "size_bytes": _du_bytes(sub), "note": "present (no _meta.json)"})

    total = _du_bytes(VOL)
    manifest = {
        "name": "chess-data-bank",
        "workspace": "chess-instructor-2",
        "volume": VOLUME_NAME,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_size_bytes": total,
        "total_size_gb": round(total / 1e9, 3),
        "sources": sources,
        "hf_references_not_snapshotted": HF_REFERENCES,
        "purpose": "Raw multi-source bank to MINE for discriminating multi-tier positions "
                   "and coaching pedagogy. Not a labeled training set.",
    }
    with open(os.path.join(VOL, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    volume.commit()
    print(f"[manifest] wrote /vol/manifest.json — {total/1e9:.2f} GB total, "
          f"{len(sources)} sources")
    return manifest


# --------------------------------------------------------------------------- #
# Stage 5: mirror CC0 / self-derived artifacts to a private HF repo
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={VOL: volume},
              secrets=[modal.Secret.from_dotenv()], timeout=T_LONG)
def mirror_to_hf(private: bool = True) -> dict:
    """Upload the freely-redistributable artifacts + manifest to a private HF repo.

    Only mirrors CC0 Lichess artifacts (puzzle CSV.zst, games slice), our
    harvested transcripts, and the manifest. Third-party HF datasets are already
    durable on HF and are pinned by SHA in the manifest rather than re-hosted
    (respects their CC-BY-SA / unknown licenses).
    """
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("no HF_TOKEN in secret (.env)")
    volume.reload()
    api = HfApi(token=token)
    who = api.whoami()
    print(f"[mirror] authenticated as {who.get('name')}")
    api.create_repo(HF_MIRROR_REPO, repo_type="dataset", exist_ok=True, private=private)

    uploaded: list[dict] = []

    def _up_file(local: str, remote: str) -> None:
        if os.path.exists(local):
            api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                            repo_id=HF_MIRROR_REPO, repo_type="dataset")
            uploaded.append({"path": remote, "size_bytes": os.path.getsize(local)})
            print(f"[mirror]   + {remote} ({os.path.getsize(local)/1e6:.1f} MB)")

    def _up_dir(local: str, remote: str) -> None:
        if os.path.isdir(local):
            api.upload_folder(folder_path=local, path_in_repo=remote,
                              repo_id=HF_MIRROR_REPO, repo_type="dataset")
            uploaded.append({"path": remote + "/", "size_bytes": _du_bytes(local)})
            print(f"[mirror]   + {remote}/ ({_du_bytes(local)/1e6:.1f} MB)")

    _up_file(os.path.join(DIR_PUZZLES, "lichess_db_puzzle.csv.zst"),
             "lichess/puzzles/lichess_db_puzzle.csv.zst")
    for fn in sorted(os.listdir(DIR_GAMES)) if os.path.isdir(DIR_GAMES) else []:
        if fn.endswith(".pgn"):
            _up_file(os.path.join(DIR_GAMES, fn), f"lichess/games/{fn}")
    _up_dir(DIR_TRANSCRIPTS, "transcripts")
    _up_file(os.path.join(VOL, "manifest.json"), "manifest.json")

    card = _mirror_card()
    with open("/tmp/README.md", "w", encoding="utf-8") as fh:
        fh.write(card)
    api.upload_file(path_or_fileobj="/tmp/README.md", path_in_repo="README.md",
                    repo_id=HF_MIRROR_REPO, repo_type="dataset")

    print(f"[mirror] done -> https://huggingface.co/datasets/{HF_MIRROR_REPO} (private={private})")
    return {"repo": HF_MIRROR_REPO, "private": private, "uploaded": uploaded}


def _mirror_card() -> str:
    return (
        "---\nlicense: cc0-1.0\ntags:\n- chess\n- raw-data-bank\n- lichess\n"
        "pretty_name: Chess Data Bank (raw, multi-source)\n---\n\n"
        "# Chess Data Bank — raw, multi-source (durability mirror)\n\n"
        "Durability mirror of the freely-redistributable (CC0 / self-derived) parts of a "
        "raw chess data bank used to MINE discriminating multi-tier positions and coaching "
        "pedagogy for a small chess-coach model. The primary store is a Modal volume; this "
        "repo backs up the CC0 Lichess artifacts, a streamed games slice, harvested "
        "coaching-video transcripts, and the manifest.\n\n"
        "Third-party HF datasets that are part of the bank are **not** re-hosted here — they "
        "are already durable on the Hub and are pinned by revision SHA in `manifest.json`.\n\n"
        "See `manifest.json` for full source/count/size/license/path detail.\n"
    )


# --------------------------------------------------------------------------- #
# Local entrypoint
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main(stage: str = "all", month: str = "2026-06",
         max_games: int = 200_000, max_mb: int = 350, private: bool = True) -> None:
    print(f"=== {APP_NAME}: stage={stage} ===")
    if stage in ("all", "download"):
        # Parallel: puzzles + games slice + HF datasets write disjoint subtrees.
        calls = [
            ("puzzles", fetch_lichess_puzzles.spawn()),
            ("games", fetch_lichess_games_slice.spawn(month, max_games, max_mb)),
            ("hf", fetch_hf_datasets.spawn()),
        ]
        for name, call in calls:
            res = call.get()
            print(f"\n=== {name} result ===")
            print(json.dumps(res, indent=2, default=str)[:2000])
        m = build_manifest.remote()
        print("\n=== manifest ===")
        print(json.dumps(m, indent=2, default=str)[:2500])
    elif stage == "puzzles":
        print(json.dumps(fetch_lichess_puzzles.remote(), indent=2, default=str))
    elif stage == "games":
        print(json.dumps(fetch_lichess_games_slice.remote(month, max_games, max_mb),
                         indent=2, default=str))
    elif stage == "hf":
        print(json.dumps(fetch_hf_datasets.remote(), indent=2, default=str))
    elif stage == "manifest":
        print(json.dumps(build_manifest.remote(), indent=2, default=str)[:3000])
    elif stage == "mirror":
        print(json.dumps(mirror_to_hf.remote(private), indent=2, default=str))
    else:
        raise SystemExit(f"unknown stage {stage!r}")
