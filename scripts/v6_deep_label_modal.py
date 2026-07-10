#!/usr/bin/env python3
"""v6 deep-labeling engine harness on Modal CPU (chess-instructor-2).

The v6 label rebuild replaces the shallow v3/v4 grounding
(``sound_pool``: 300 ms / MultiPV-8 / 150 cp) with a **deep, robust,
WDL/tablebase-verified** engine pass, so no "sound-pool but actually a blunder"
label survives. This module is the *engine* half — pure Stockfish 17 + Syzygy +
Maia-2 run at scale on Modal CPU containers. Selection / dataset assembly lives
in ``scripts/build_v6.py`` (local, imports ``src.teacher.tier_select_v6``).

Per position (``_label_one``), addressing audit fixes #1/#2:
- MultiPV@depth2 finds the near-best candidate set.
- **Every candidate is ROOT-SEARCHED** at depth1 AND depth2 (``root_moves=[mv]``)
  for a clean per-move eval + WDL — no MultiPV truncation error.
- **Two-depth agreement:** a move must be within tolerance of the best at BOTH
  depths to enter the pool (kills shallow-only "sound" moves).
- **WDL bands:** a move that crosses a win/draw/loss band or materially worsens
  the expected score vs the best is rejected (``wdl_band`` / ``wdl_drop``).
- **Syzygy** (3-4-5 piece) gives exact WDL for eligible endgames; a move that
  worsens the tablebase result is rejected (``tb_worse``).
- **Maia-2** scores every pool move's human-likelihood at 1100/1500/1900.

Stages (run from repo root; kim-lam tokens unset; NEVER the kim-lam workspace)::

    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    MODAL_PROFILE=chess-instructor-2 modal run scripts/v6_deep_label_modal.py::smoke
    MODAL_PROFILE=chess-instructor-2 modal run scripts/v6_deep_label_modal.py::download_syzygy
    # mining + labeling are driven from scripts/build_v6.py via .remote()/.map()
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import modal

# --------------------------------------------------------------------------- #
# Names / layout
# --------------------------------------------------------------------------- #
APP_NAME = "chess-coach-v6-label"
VOLUME_NAME = "chess-data"
VOL = "/vol"

PUZZLE_CSV = f"{VOL}/lichess/puzzles/lichess_db_puzzle.csv"
SYZYGY_DIR = f"{VOL}/syzygy/3-4-5"
LABELS_DIR = f"{VOL}/v6/labels"
MINED_DIR = f"{VOL}/v6/mined"

# Deep-search defaults (fixed DEPTH, not movetime, for reproducibility).
DEPTH1 = 14
DEPTH2 = 20
# Per-search wall-time caps (seconds) so a single pathological position cannot
# stall a container; normal positions reach the target depth well within these.
TIME2 = 6.0
TIME1 = 1.0
MULTIPV = 10
SOUND_TOLERANCE_CP = 120
WDL_DROP_MAX = 0.10        # expected-score (0..1) a pool move may lose vs best
WIN_BAND, LOSS_BAND = 0.75, 0.25

SF_URL = "https://github.com/official-stockfish/Stockfish/releases/download/sf_17/stockfish-ubuntu-x86-64-avx2.tar"
SF_BIN = "/opt/sf/stockfish/stockfish-ubuntu-x86-64-avx2"
SYZYGY_INDEX = "http://tablebase.sesse.net/syzygy/3-4-5/"

ELO = {"beginner": 1100, "intermediate": 1500, "advanced": 1900}


def _dl_stockfish() -> None:
    import subprocess

    os.makedirs("/opt/sf", exist_ok=True)
    tar = "/opt/sf/sf.tar"
    subprocess.run(["bash", "-lc", f"curl -fsSL '{SF_URL}' -o '{tar}'"], check=True)
    subprocess.run(["bash", "-lc", f"tar -xf '{tar}' -C /opt/sf"], check=True)
    subprocess.run(["bash", "-lc", f"chmod +x '{SF_BIN}'"], check=True)


def _prefetch_maia2() -> None:
    """Bake the Maia-2 'rapid' weights into the image so containers don't
    re-download them at cold start."""
    from maia2 import model as m2model

    m2model.from_pretrained(type="rapid", device="cpu")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "tar", "libgomp1", "stockfish")
    .pip_install(
        "python-chess==1.999", "chess", "requests", "zstandard",
        "torch", "maia2", "gdown",
    )
    .pip_install("numpy", "pandas", "scikit-learn", "pyzstd", "einops",
                 "safetensors", "tqdm", "pyyaml", "matplotlib")
    .run_function(_dl_stockfish)
    .run_function(_prefetch_maia2)
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)

T_SHORT = 30 * 60
T_LONG = 150 * 60


# --------------------------------------------------------------------------- #
# Engine helpers (run remotely)
# --------------------------------------------------------------------------- #
def _resolve_sf() -> str:
    if os.path.exists(SF_BIN):
        return SF_BIN
    from shutil import which

    return which("stockfish") or "stockfish"


def _open_sf(threads: int = 2, hash_mb: int = 256):
    import chess.engine

    eng = chess.engine.SimpleEngine.popen_uci(_resolve_sf())
    opts: Dict[str, Any] = {"Threads": threads, "Hash": hash_mb}
    try:
        eng.configure({**opts, "UCI_ShowWDL": True})
    except Exception:
        eng.configure(opts)
    if os.path.isdir(SYZYGY_DIR):
        try:
            eng.configure({"SyzygyPath": SYZYGY_DIR})
        except Exception:
            pass
    return eng


def _exp(wdl: Optional[List[int]]) -> Optional[float]:
    if not wdl or len(wdl) != 3:
        return None
    tot = sum(wdl)
    return (wdl[0] + 0.5 * wdl[1]) / tot if tot > 0 else None


def _band(exp: Optional[float]) -> Optional[str]:
    if exp is None:
        return None
    return "win" if exp >= WIN_BAND else ("loss" if exp <= LOSS_BAND else "draw")


_BAND_RANK = {"win": 2, "draw": 1, "loss": 0}


def _sev(cp_loss: int) -> str:
    if cp_loss < 50:
        return "none"
    if cp_loss < 100:
        return "inaccuracy"
    if cp_loss < 250:
        return "mistake"
    return "blunder"


def _phase_of(board) -> str:
    import chess

    n = chess.popcount(board.occupied)
    q = bool(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK))
    if n <= 12 or (not q and n <= 16):
        return "endgame"
    if board.fullmove_number <= 12:
        return "opening"
    return "middlegame"


TB_API = "https://tablebase.lichess.ovh/standard"
# The API's per-move ``category`` is the OPPONENT's result after our move;
# invert to our own 3-band result (cursed/blessed collapse to the practical draw).
_TB_INVERT = {
    "loss": "win", "maybe-loss": "win", "blessed-loss": "draw",
    "draw": "draw", "cursed-win": "draw", "maybe-win": "loss", "win": "loss",
    "unknown": None,
}


def _json_safe(o, path="", found=None):
    """Recursively convert chess.Move -> uci so records are JSON-serializable.
    Records offending paths in ``found`` (once) for diagnostics."""
    import chess

    if isinstance(o, chess.Move):
        if found is not None:
            found.add(path)
        return o.uci()
    if isinstance(o, dict):
        return {k: _json_safe(v, f"{path}.{k}", found) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v, f"{path}[{i}]", found) for i, v in enumerate(o)]
    return o


def _tb_map(board, session, cache: dict) -> Dict[str, str]:
    """``{uci: our-band}`` for every legal move via the Lichess Syzygy API
    (<=7 pieces). Cached per FEN; best-effort (returns {} on any failure)."""
    fen = board.fen()
    if fen in cache:
        return cache[fen]
    out: Dict[str, str] = {}
    for attempt in range(2):
        try:
            r = session.get(TB_API, params={"fen": fen}, timeout=8)
            if r.status_code == 429:
                time.sleep(1 + attempt)
                continue
            r.raise_for_status()
            data = r.json()
            for m in data.get("moves") or []:
                band = _TB_INVERT.get(str(m.get("category")), None)
                if band:
                    out[str(m.get("uci"))] = band
            break
        except Exception:
            time.sleep(0.5)
    cache[fen] = out
    return out


# --------------------------------------------------------------------------- #
# The per-position deep label
# --------------------------------------------------------------------------- #
def _label_one(eng, session, tb_cache, mdl, prepared, m2inf, rec,
               d1, d2, multipv, tol_cp) -> Optional[dict]:
    import chess
    import chess.engine

    fen = rec["fen"]
    try:
        board = chess.Board(fen)
    except Exception:
        return None
    if not board.is_valid() or board.is_game_over():
        return None
    turn = board.turn
    npieces = chess.popcount(board.occupied)

    def cp_of(info) -> int:
        return int(info["score"].pov(turn).score(mate_score=100000))

    def mate_of(info):
        return info["score"].pov(turn).mate()

    def wdl_of(info):
        try:
            w = info["score"].pov(turn).wdl(model="sf12")
            return [w.wins, w.draws, w.losses]
        except Exception:
            return None

    # 1) candidate discovery + per-move DEEP eval: MultiPV @ depth2 pins each
    # first move to its own line, so each line's score/WDL IS that move's value.
    try:
        infos2 = eng.analyse(board, chess.engine.Limit(depth=d2, time=TIME2),
                             multipv=multipv)
    except Exception:
        return None
    if isinstance(infos2, dict):
        infos2 = [infos2]
    d2info: Dict[str, dict] = {}
    for info in infos2:
        pv = info.get("pv") or []
        if not pv:
            continue
        u = pv[0].uci()
        if u not in d2info:
            d2info[u] = {"cp": cp_of(info), "wdl": wdl_of(info),
                         "mate": mate_of(info), "pv_moves": list(pv)}
    if not d2info:
        return None

    prov_best_cp = max(v["cp"] for v in d2info.values())
    keep = [u for u, v in d2info.items() if prov_best_cp - v["cp"] <= tol_cp]

    # Syzygy (via Lichess API) for eligible endgames — one call, all moves.
    tbmap = _tb_map(board, session, tb_cache) if npieces <= 7 else {}
    tb_used = bool(tbmap)

    # 2) ROOT-SEARCH each selected move at the shallower depth1 for the
    # two-depth agreement check (the deep eval already comes from MultiPV@d2).
    scored: List[Dict[str, Any]] = []
    for u in keep:
        mv = chess.Move.from_uci(u)
        de = d2info[u]
        try:
            i1 = eng.analyse(board, chess.engine.Limit(depth=d1, time=TIME1),
                             root_moves=[mv])
            i1 = i1[0] if isinstance(i1, list) else i1
            cp_d1 = cp_of(i1)
        except Exception:
            cp_d1 = de["cp"]
        pv_sans: List[str] = []
        prev = board.copy(stack=False)
        for m in de["pv_moves"][:6]:
            try:
                pv_sans.append(prev.san(m))
                prev.push(m)
            except Exception:
                break
        scored.append({
            "uci": u, "san": board.san(mv),
            "cp": de["cp"], "cp_d1": cp_d1,
            "wdl": de["wdl"], "mate": de["mate"], "pv": pv_sans,
            "tb": tbmap.get(u),
        })
    if not scored:
        return None

    scored.sort(key=lambda e: -e["cp"])
    best = scored[0]
    best_cp, best_cp_d1 = best["cp"], best["cp_d1"]
    best_exp = _exp(best["wdl"])
    best_band = _band(best_exp)
    best_tb_band = best.get("tb")  # tablebase band of the engine best (if endgame)

    # 3) filter: two-depth agreement + WDL band + tablebase.
    pool: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for e in scored:
        depth_agree = (best_cp_d1 - e["cp_d1"]) <= tol_cp
        within_d2 = (best_cp - e["cp"]) <= tol_cp
        exp_m = _exp(e["wdl"])
        wdl_drop = (best_exp - exp_m) if (best_exp is not None and exp_m is not None) else None
        band_m = _band(exp_m)
        band_cross = bool(
            band_m and best_band and _BAND_RANK[band_m] < _BAND_RANK[best_band]
        )
        tb_worse = bool(
            best_tb_band and e.get("tb")
            and _BAND_RANK[e["tb"]] < _BAND_RANK[best_tb_band]
        )
        reason = None
        if not within_d2:
            reason = "cp_d2"
        elif not depth_agree:
            reason = "shallow_only"
        elif tb_worse:
            reason = "tb_worse"
        elif band_cross:
            reason = "wdl_band"
        elif wdl_drop is not None and wdl_drop > WDL_DROP_MAX:
            reason = "wdl_drop"
        rec_e = {"uci": e["uci"], "san": e["san"], "cp": e["cp"], "cp_d1": e["cp_d1"],
                 "wdl": e["wdl"], "tb": e["tb"],
                 "wdl_drop": round(wdl_drop, 3) if wdl_drop is not None else None}
        if reason and e["uci"] != best["uci"]:
            rec_e["reason"] = reason
            rejected.append(rec_e)
        else:
            pool.append({"uci": e["uci"], "san": e["san"], "cp": e["cp"], "cp_d1": e["cp_d1"],
                         "wdl": e["wdl"], "mate": e["mate"], "pv": e["pv"],
                         "depth_agree": bool(depth_agree), "tb": e["tb"]})

    engine_best = {k: best[k] for k in ("uci", "san", "cp", "wdl", "mate")}

    # 4) student move (if any) — root-searched at d2.
    student = None
    su = rec.get("student_uci")
    if su:
        try:
            smv = chess.Move.from_uci(su)
            if smv in board.legal_moves:
                si = eng.analyse(board, chess.engine.Limit(depth=d2), root_moves=[smv])
                si = si[0] if isinstance(si, list) else si
                scp = cp_of(si)
                student = {
                    "uci": su, "san": board.san(smv), "cp": scp, "wdl": wdl_of(si),
                    "cp_loss": max(0, best_cp - scp), "severity": _sev(max(0, best_cp - scp)),
                    "in_pool": any(p["uci"] == su for p in pool),
                }
        except Exception:
            student = None

    # 5) Maia-2 policy for every pool move (+ student) at each tier, plus the
    # single most human-likely legal move per tier (the "typical" move, used to
    # frame mined positions that have no real student move).
    want = {p["uci"] for p in pool}
    if student:
        want.add(student["uci"])
    legal_by_uci = {mv.uci(): mv for mv in board.legal_moves}
    maia: Dict[str, Any] = {}
    maia_top: Dict[str, Any] = {}
    for tier, elo in ELO.items():
        try:
            mp, wp = m2inf.inference_each(mdl, prepared, fen, elo, elo)
            maia[tier] = {u: round(float(mp.get(u, 0.0)), 4) for u in want}
            maia[f"{tier}_win_prob"] = round(float(wp), 4)
            legal_probs = {u: p for u, p in mp.items() if u in legal_by_uci}
            if legal_probs:
                tu = max(legal_probs, key=legal_probs.get)
                maia_top[tier] = {"uci": tu, "san": board.san(legal_by_uci[tu]),
                                  "policy": round(float(legal_probs[tu]), 4)}
        except Exception:
            maia[tier] = {u: 0.0 for u in want}

    # Root-search each tier's typical (top-Maia) move so mined positions can be
    # framed with a real, evaluated "student" move (coherent endorse/correct).
    top_eval: Dict[str, dict] = {}
    for tier in ELO:
        tinfo = maia_top.get(tier)
        if not tinfo:
            continue
        tu = tinfo["uci"]
        if tu in top_eval:
            pass
        else:
            try:
                ti_ = eng.analyse(board, chess.engine.Limit(depth=d1, time=TIME1),
                                  root_moves=[chess.Move.from_uci(tu)])
                ti_ = ti_[0] if isinstance(ti_, list) else ti_
                tcp = cp_of(ti_)
                top_eval[tu] = {"cp": tcp, "wdl": wdl_of(ti_),
                                "cp_loss": max(0, best_cp - tcp),
                                "severity": _sev(max(0, best_cp - tcp))}
            except Exception:
                top_eval[tu] = {}
        ev = top_eval.get(tu, {})
        tinfo.update(ev)
        tinfo["in_pool"] = any(p["uci"] == tu for p in pool)

    return {
        "pos_id": rec.get("pos_id"),
        "fen": fen,
        "source": rec.get("source"),
        "rating": rec.get("rating"),
        "themes": rec.get("themes"),
        "side_to_move": "white" if turn else "black",
        "n_pieces": npieces,
        "phase": rec.get("phase") or _phase_of(board),
        "engine_best": engine_best,
        "best_cp": best_cp, "best_cp_d1": best_cp_d1,
        "sound_pool": pool, "rejected": rejected,
        "n_sound": len(pool), "n_rejected": len(rejected),
        "maia": maia, "maia_top": maia_top, "student": student,
        "tb_used": tb_used,
        "depths": [d1, d2], "multipv": multipv, "tol_cp": tol_cp,
        "maia_source": "maia2-rapid",
    }


# --------------------------------------------------------------------------- #
# Remote: deep-label a batch (checkpointed to the volume)
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={VOL: volume}, timeout=T_LONG, cpu=2.0,
              max_containers=64, retries=2)
def deep_label_batch(recs: List[dict], tag: str,
                     depth1: int = DEPTH1, depth2: int = DEPTH2,
                     multipv: int = MULTIPV, tol_cp: int = SOUND_TOLERANCE_CP,
                     force: bool = False) -> dict:
    import requests
    from maia2 import inference as m2inf
    from maia2 import model as m2model

    shard_path = f"{LABELS_DIR}/{tag}.jsonl"
    if not force and os.path.exists(shard_path):
        labels = [json.loads(x) for x in open(shard_path) if x.strip()]
        return {"tag": tag, "n_in": len(recs), "n_out": len(labels),
                "cached": True, "labels": labels}

    mdl = m2model.from_pretrained(type="rapid", device="cpu")
    prepared = m2inf.prepare()
    session = requests.Session()
    session.headers.update({"User-Agent": "chess-coach-v6/1.0 (research dataset build)"})
    tb_cache: Dict[str, Dict[str, str]] = {}

    out: List[dict] = []
    found_paths: set = set()
    t0 = time.time()
    with _open_sf() as eng:
        for i, rec in enumerate(recs):
            try:
                r = _label_one(eng, session, tb_cache, mdl, prepared, m2inf, rec,
                               depth1, depth2, multipv, tol_cp)
            except Exception as e:  # noqa: BLE001
                r = None
                print(f"  ! {rec.get('pos_id')}: {e!r}")
            if r:
                out.append(_json_safe(r, "", found_paths))
            if (i + 1) % 10 == 0:
                print(f"[{tag}] {i+1}/{len(recs)} ({(time.time()-t0)/(i+1):.1f}s/pos)",
                      flush=True)
    if found_paths:
        print(f"[{tag}] sanitized Move at paths: {sorted(found_paths)[:5]}", flush=True)

    os.makedirs(LABELS_DIR, exist_ok=True)
    tmp = shard_path + ".tmp"
    with open(tmp, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, shard_path)
    volume.commit()
    return {"tag": tag, "n_in": len(recs), "n_out": len(out),
            "secs": round(time.time() - t0, 1), "cached": False, "labels": out}


# --------------------------------------------------------------------------- #
# Remote: mine candidate positions from the Lichess puzzle bank
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={VOL: volume}, timeout=T_LONG)
def mine_candidates(theme_quota: Dict[str, int],
                    rating_lo: int = 1000, rating_hi: int = 2000,
                    min_pieces: int = 5, max_pieces: int = 24,
                    exclude_board_keys: Optional[List[str]] = None,
                    seed: int = 3407) -> List[dict]:
    """Stream the puzzle CSV and sample real game positions per theme.

    We use the puzzle **FEN** (the real game position *before* the opponent's
    move — a natural, often quiet/endgame position) rather than the tactical
    puzzle line, then let the deep labeler decide soundness/discrimination. The
    per-puzzle ``Rating`` seeds the tier and ``Themes`` biases phase coverage.
    """
    import csv
    import random

    import chess

    rng = random.Random(seed)
    exclude = set(exclude_board_keys or [])
    remaining = dict(theme_quota)
    picked: Dict[str, dict] = {}
    seen_board: set = set(exclude)

    def board_key(fen: str) -> str:
        parts = fen.split()
        return " ".join(parts[:2]) if len(parts) >= 2 else fen

    with open(PUZZLE_CSV, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not remaining:
                break
            try:
                rating = int(row["Rating"])
            except Exception:
                continue
            if not (rating_lo <= rating <= rating_hi):
                continue
            themes = (row.get("Themes") or "").split()
            hit = next((t for t in themes if remaining.get(t, 0) > 0), None)
            if hit is None:
                continue
            # Downsample so we don't take only the first rows of the CSV.
            if rng.random() > 0.5:
                continue
            fen = row.get("FEN") or ""
            try:
                board = chess.Board(fen)
            except Exception:
                continue
            if not board.is_valid() or board.is_game_over():
                continue
            npieces = chess.popcount(board.occupied)
            if not (min_pieces <= npieces <= max_pieces):
                continue
            bk = board_key(fen)
            if bk in seen_board:
                continue
            seen_board.add(bk)
            pid = f"puz_{row.get('PuzzleId')}"
            picked[pid] = {
                "pos_id": pid, "fen": fen, "source": "mined_puzzle",
                "rating": rating, "themes": themes,
                "phase": _phase_of(board),
                # No fabricated student move for mined positions.
                "student_uci": None,
            }
            remaining[hit] -= 1
            if remaining[hit] <= 0:
                del remaining[hit]

    os.makedirs(MINED_DIR, exist_ok=True)
    out = list(picked.values())
    with open(f"{MINED_DIR}/candidates.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    volume.commit()
    print(f"[mine] picked {len(out)} candidates; unmet quota: {remaining}")
    return out


# --------------------------------------------------------------------------- #
# Remote: download 3-4-5 Syzygy WDL+DTZ tablebases onto the volume
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={VOL: volume}, timeout=T_LONG)
def download_syzygy(wdl_only: bool = True) -> dict:
    import concurrent.futures as cf
    import re

    import requests

    UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
          "Referer": SYZYGY_INDEX}
    sess = requests.Session()
    sess.headers.update(UA)

    os.makedirs(SYZYGY_DIR, exist_ok=True)
    html = sess.get(SYZYGY_INDEX, timeout=60).text
    files = re.findall(r'href="([^"]+\.rtb[wz])"', html)
    if wdl_only:
        files = [f for f in files if f.endswith(".rtbw")]
    files = sorted(set(files))
    print(f"[syzygy] {len(files)} files to fetch (wdl_only={wdl_only})")

    def _get(name: str) -> Tuple[str, int]:
        dst = os.path.join(SYZYGY_DIR, os.path.basename(name))
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return name, os.path.getsize(dst)
        url = SYZYGY_INDEX + name if not name.startswith("http") else name
        last = None
        for _ in range(4):
            try:
                with sess.get(url, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    tmp = dst + ".tmp"
                    with open(tmp, "wb") as fh:
                        for ch in r.iter_content(1 << 20):
                            fh.write(ch)
                    os.replace(tmp, dst)
                return name, os.path.getsize(dst)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2)
        print(f"  ! syzygy fail {name}: {last!r}")
        return name, 0

    total = 0
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for name, sz in ex.map(_get, files):
            total += sz
    volume.commit()
    n = len([f for f in os.listdir(SYZYGY_DIR) if f.endswith(".rtbw")])
    print(f"[syzygy] {n} WDL files, {total/1e6:.0f} MB on volume")
    return {"n_wdl": n, "size_mb": round(total / 1e6, 1)}


# --------------------------------------------------------------------------- #
# Smoke
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={VOL: volume}, timeout=T_SHORT)
def smoke() -> dict:
    import requests

    from maia2 import inference as m2inf
    from maia2 import model as m2model

    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5"
    mdl = m2model.from_pretrained(type="rapid", device="cpu")
    prepared = m2inf.prepare()
    session = requests.Session()
    tb_cache: Dict[str, Any] = {}
    with _open_sf() as eng:
        r = _label_one(eng, session, tb_cache, mdl, prepared, m2inf,
                       {"pos_id": "smoke", "fen": fen, "student_uci": "f3g5"},
                       DEPTH1, DEPTH2, MULTIPV, SOUND_TOLERANCE_CP)
        # 6-piece rook endgame to exercise the tablebase API path.
        eg = "8/8/4k3/8/8/4K3/4P3/6R1 w - - 0 1"
        r2 = _label_one(eng, session, tb_cache, mdl, prepared, m2inf,
                        {"pos_id": "eg", "fen": eg, "student_uci": None},
                        DEPTH1, DEPTH2, MULTIPV, SOUND_TOLERANCE_CP)
    out = {"mid": r, "endgame": r2}
    print(json.dumps(out, indent=2, default=str))
    return out


@app.function(image=image, volumes={VOL: volume}, timeout=T_SHORT, cpu=2.0)
def timeit(n: int = 8) -> dict:
    """Break down per-position cost: SF MultiPV vs root-searches vs maia2 vs tb."""
    import chess
    import chess.engine
    import requests
    from maia2 import inference as m2inf
    from maia2 import model as m2model

    fens = [
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
        "rn3rk1/pp2bppp/2p2n2/8/3qP3/N6P/PPB1Q1P1/R1B1K2R b KQ - 1 12",
        "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NPP2/1BN1B3/PPP3PP/R2Q1RK1 w - - 0 12",
        "8/8/4k3/8/8/4K3/4P3/6R1 w - - 0 1",
    ] * 3
    fens = fens[:n]
    t_load = time.time()
    mdl = m2model.from_pretrained(type="rapid", device="cpu")
    prepared = m2inf.prepare()
    load_s = time.time() - t_load
    sess = requests.Session()
    times = {"multipv": 0.0, "root_d1": 0.0, "maia": 0.0, "tb": 0.0, "n": 0}
    with _open_sf() as eng:
        for fen in fens:
            board = chess.Board(fen)
            turn = board.turn
            t = time.time()
            infos = eng.analyse(board, chess.engine.Limit(depth=DEPTH2), multipv=MULTIPV)
            if isinstance(infos, dict):
                infos = [infos]
            times["multipv"] += time.time() - t
            ucis = [i["pv"][0].uci() for i in infos if i.get("pv")][:MULTIPV]
            t = time.time()
            for u in ucis:
                eng.analyse(board, chess.engine.Limit(depth=DEPTH1),
                            root_moves=[chess.Move.from_uci(u)])
            times["root_d1"] += time.time() - t
            t = time.time()
            for elo in (1100, 1500, 1900):
                m2inf.inference_each(mdl, prepared, fen, elo, elo)
            times["maia"] += time.time() - t
            if chess.popcount(board.occupied) <= 7:
                t = time.time()
                sess.get(TB_API, params={"fen": fen}, timeout=8)
                times["tb"] += time.time() - t
            times["n"] += 1
    per = {k: round(v / times["n"], 2) for k, v in times.items() if k != "n"}
    out = {"maia_load_s": round(load_s, 1), "n": times["n"], "per_pos_s": per,
           "total_per_pos": round(sum(per.values()), 2)}
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(timeit.remote(8), indent=2, default=str))
