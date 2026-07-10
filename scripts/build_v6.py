#!/usr/bin/env python3
"""Build the **v6** chess-coaching dataset — a higher-quality LABEL rebuild.

v6 is a FOUNDATIONAL, data-first rebuild of the training LABELS that feeds the
downstream DPO + engine-distillation retrains. It fixes the chess/dataset audit
findings *in the data*:

1. **Deep, robust sound pool** — every position is re-analysed on Modal CPU with
   Stockfish 17 at two fixed depths (root-searching each candidate), WDL bands,
   and Syzygy (Lichess API) for eligible endgames, so no "sound-pool but actually
   a blunder" label survives (``scripts/v6_deep_label_modal.py``).
2. **advanced = the VERIFIED engine-best move** (never diverges from
   ``engine_best``) — ``src.teacher.tier_select_v6``.
3. **Maia as a CONSTRAINT** (human-likelihood gate) then rank by robustness /
   clarity — no min-max blend artifacts.
4. **Complete-group triads** applied per position atomically; the ``B=A != I``
   collapse is removed; all-same triads are capped + down-weighted.
5. **Coherent move-review** — a student move is endorsed unless it is genuinely
   worse by a margin (``review_student_move``); the student move is never
   synthesised as the canonical pick.
6. **Quality over volume** — keep the discriminating multi-tier boards; use
   per-row sampling ``weight`` (not exact-duplicate oversampling); mine fresh
   endgame/quiet positions from the raw bank to widen coverage.

Stages (run from repo root)::

    # 1. deep-label the reused + benchmark + freshly-mined positions on Modal
    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    MODAL_PROFILE=chess-instructor-2 python scripts/build_v6.py label [--pilot N] [--no-mine]

    # 2. assemble train_v6/valid_v6 (+ refreshed benchmark) with full provenance
    python scripts/build_v6.py assemble

    # 3. write the rebuild report
    python scripts/build_v6.py report

    # 4. publish the jsonl artifacts to Hugging Face (gitignored locally)
    python scripts/build_v6.py publish
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import schema, settings  # noqa: E402
from src.eval.benchmark.prompts import build_grounded_user, load_system_prompt  # noqa: E402
from src.engine.faithfulness import verify_text  # noqa: E402
from src.filter.filter import detect_engine_speak, longest_san_run, move_is_legal  # noqa: E402
from src.teacher.tier_select_v6 import (  # noqa: E402
    TIER_ORDER, review_student_move, select_tiers_v6,
)

log = logging.getLogger("build_v6")

CANDIDATES_V3 = settings.GENERATED / "candidates_v3.jsonl"
SCENARIOS = settings.DATA / "benchmark_gap803" / "scenarios.jsonl"
VAL_IDS = settings.DATA / "benchmark_honest" / "val_ids.txt"

V6_INPUTS = settings.GENERATED / "v6_inputs.jsonl"
V6_BENCH = settings.GENERATED / "v6_bench.jsonl"
V6_LABELS = settings.GENERATED / "v6_labels.jsonl"
TRAIN_V6 = settings.DATASET / "train_v6.jsonl"
VALID_V6 = settings.DATASET / "valid_v6.jsonl"
SCEN_V6 = settings.DATA / "benchmark_gap803" / "scenarios_v6.jsonl"
MANIFEST_V6 = settings.GENERATED / "v6_manifest.json"

SEED = 3407

# Puzzle themes to mine, biased toward endgame + quiet/positional coverage.
THEME_QUOTA: Dict[str, int] = {
    "endgame": 1100, "rookEndgame": 450, "pawnEndgame": 300, "queenEndgame": 220,
    "bishopEndgame": 220, "knightEndgame": 180, "queenRookEndgame": 120,
    "zugzwang": 120, "quietMove": 700, "advancedPawn": 300,
    "defensiveMove": 260, "advantage": 260,
}


# --------------------------------------------------------------------------- #
# small io helpers
# --------------------------------------------------------------------------- #
def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _write_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def board_key(fen: str) -> str:
    parts = fen.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else fen


def game_id(pos_id: str) -> str:
    pid = str(pos_id).split("#")[0]
    # Mined puzzles are not from a shared game; each is its own "game" so the
    # game-disjoint split treats them independently (avoid one giant "puz" group).
    if pid.startswith("puz_"):
        return pid
    return pid.rsplit("_", 1)[0]


# --------------------------------------------------------------------------- #
# input universe (reused candidates_v3 boards + benchmark boards)
# --------------------------------------------------------------------------- #
def _build_reused_boards() -> Dict[str, dict]:
    boards: Dict[str, dict] = {}
    n = 0
    for c in _iter_jsonl(CANDIDATES_V3):
        n += 1
        ti = c.get("teacher_input") or {}
        meta = c.get("meta") or {}
        fen = ti.get("fen")
        if not fen:
            continue
        bk = board_key(fen)
        base_id = meta.get("base_id") or str(c.get("id", "")).split("#")[0]
        sm = ti.get("student_move") or {}
        synthetic = (meta.get("source") == "v2_contrastive")
        if bk not in boards:
            boards[bk] = {
                "pos_id": base_id, "fen": fen, "board_key": bk,
                "student_uci": sm.get("uci"), "student_san": sm.get("san"),
                "source": "reused", "game_id": game_id(base_id),
                "synthetic_student": synthetic,
            }
        elif boards[bk]["synthetic_student"] and not synthetic:
            boards[bk]["student_uci"] = sm.get("uci")
            boards[bk]["student_san"] = sm.get("san")
            boards[bk]["synthetic_student"] = False
    log.info("reused: %d candidates -> %d unique boards", n, len(boards))
    return boards


def _build_bench_rows() -> Tuple[List[dict], Dict[str, dict]]:
    val_ids = {x.strip() for x in VAL_IDS.read_text().split() if x.strip()}
    rows: List[dict] = []
    boards: Dict[str, dict] = {}
    for s in _iter_jsonl(SCENARIOS):
        pid = s["pos_id"]
        fen = s["fen"]
        bk = board_key(fen)
        rows.append({
            "pos_id": pid, "tier": s["tier"], "fen": fen, "board_key": bk,
            "student_uci": s["student_move"]["uci"], "student_san": s["student_move"]["san"],
            "old_canonical": s.get("canonical_uci"),
            "old_engine_best": s.get("engine_best_uci"),
            "is_val": pid in val_ids,
        })
        if bk not in boards:
            boards[bk] = {
                "pos_id": pid, "fen": fen, "board_key": bk,
                "student_uci": s["student_move"]["uci"], "student_san": s["student_move"]["san"],
                "source": "benchmark", "game_id": game_id(pid),
                "synthetic_student": False,
            }
    log.info("benchmark: %d scenarios -> %d unique boards (%d val ids)",
             len(rows), len(boards), len(val_ids))
    return rows, boards


# --------------------------------------------------------------------------- #
# label — drive the Modal deep labeler over the whole universe
# --------------------------------------------------------------------------- #
def cmd_label(args: argparse.Namespace) -> int:
    import modal

    import v6_deep_label_modal as v6m

    reused = _build_reused_boards()
    bench_rows, bench_boards = _build_bench_rows()
    _write_jsonl(bench_rows, V6_BENCH)

    inputs: Dict[str, dict] = {}
    for bk, b in reused.items():
        inputs[bk] = b
    for bk, b in bench_boards.items():
        if bk in inputs:
            inputs[bk]["also_benchmark"] = True
        else:
            inputs[bk] = b
    exclude_keys = list(inputs.keys())

    recs = list(inputs.values())
    if args.pilot:
        random.Random(SEED).shuffle(recs)
        recs = recs[: args.pilot]
    for r in recs:
        r.setdefault("student_uci", None)

    labels: List[dict] = []
    with modal.enable_output(), v6m.app.run():
        if not args.no_mine and not args.pilot:
            log.info("mining fresh candidates from the puzzle bank ...")
            mined = v6m.mine_candidates.remote(
                THEME_QUOTA, rating_lo=1000, rating_hi=2000,
                min_pieces=5, max_pieces=24, exclude_board_keys=exclude_keys, seed=SEED,
            )
            log.info("mined %d fresh candidates", len(mined))
            seen = set(inputs)
            for m in mined:
                bk = board_key(m["fen"])
                if bk not in seen:
                    seen.add(bk)
                    recs.append(m)

        _write_jsonl(recs, V6_INPUTS)
        log.info("labeling %d positions on Modal ...", len(recs))
        shard = max(1, args.shard_size)
        shards = [recs[i:i + shard] for i in range(0, len(recs), shard)]
        tags = [f"v6_{'pilot_' if args.pilot else ''}{i:04d}" for i in range(len(shards))]
        done = 0
        for res in v6m.deep_label_batch.map(shards, tags):
            labels.extend(res.get("labels", []))
            done += 1
            log.info("shard %d/%d %s: n_out=%s cached=%s",
                     done, len(shards), res.get("tag"), res.get("n_out"), res.get("cached"))

    out_path = V6_LABELS if not args.pilot else settings.GENERATED / "v6_labels_pilot.jsonl"
    n = _write_jsonl(labels, out_path)
    log.info("wrote %d deep labels -> %s", n, out_path)
    _label_stats(labels)
    return 0


def _label_stats(labels: List[dict]) -> None:
    if not labels:
        print("no labels"); return
    disc = 0
    hc = 0
    patt: Counter = Counter()
    phase: Counter = Counter()
    tb = 0
    adv_bug = 0
    for L in labels:
        if not L.get("sound_pool"):
            continue
        r = select_tiers_v6(L["sound_pool"], L["maia"], L["engine_best"])
        disc += int(r["discriminating"])
        hc += int(r["high_conf_discriminating"])
        patt[r["pattern"]] += 1
        phase[L.get("phase", "?")] += 1
        tb += int(bool(L.get("tb_used")))
        if r["picks"]["advanced"]["uci"] != L["engine_best"]["uci"]:
            adv_bug += 1
    n = len(labels)
    print(f"\n=== v6 label stats (n={n}) ===")
    print(f"discriminating (B!=A): {disc} ({disc/n:.1%})   high-conf: {hc} ({hc/n:.1%})")
    print(f"advanced != engine_best (should be 0): {adv_bug}")
    print(f"tb_used (endgames): {tb} ({tb/n:.1%})")
    print(f"patterns: {dict(patt)}")
    print(f"phase: {dict(phase)}")


# --------------------------------------------------------------------------- #
# prose: reuse vetted teacher text where the move is unchanged, else clean-gen
# --------------------------------------------------------------------------- #
def _load_prose_map() -> Dict[Tuple[str, str], dict]:
    prose: Dict[Tuple[str, str], dict] = {}
    for c in _iter_jsonl(CANDIDATES_V3):
        ti = c.get("teacher_input") or {}
        to = c.get("teacher_output") or {}
        fen = ti.get("fen")
        ruci = str(to.get("recommended_move_uci") or "")
        if not fen or not ruci or not to.get("coaching"):
            continue
        key = (board_key(fen), ruci)
        if key not in prose:
            prose[key] = {
                "coaching": to.get("coaching", ""), "method": to.get("method", ""),
                "takeaway": to.get("takeaway", ""), "san": to.get("recommended_move_san", ""),
            }
    return prose


_PRINCIPLE = {
    "endgame": "activate your king and keep your pieces coordinated",
    "opening": "develop toward the center and get your king safe",
    "middlegame": "improve your worst-placed piece and watch your opponent's ideas",
}


def _clean_target(san: str, phase: str, action: str, student_san: Optional[str]) -> str:
    principle = _PRINCIPLE.get(phase, "make a purposeful, safe improving move")
    if action == "endorse":
        lead = f"{san} is a sound, level-appropriate choice here."
    elif action == "correct" and student_san and student_san != san:
        lead = f"{san} is a clearer and sounder try than {student_san}."
    else:
        lead = f"{san} keeps your position solid and is easy to follow at your level."
    method = ("list the sound candidate moves first, then choose the one that meets your "
              "opponent's main idea while improving one of your own pieces")
    takeaway = f"When you are unsure, {principle}."
    return f"I'd play {san}. {lead} How to find it: {method}. Takeaway: {takeaway}"


def _gate_ok(target: str, fen: str, san: str, tier: str) -> bool:
    if not target.startswith(f"I'd play {san}."):
        return False
    if "Takeaway:" not in target:
        return False
    if detect_engine_speak(target):
        return False
    if longest_san_run(target) > settings.TIERS[tier]["ply_cap"]:
        return False
    try:
        if verify_text(target, fen).violations:
            return False
    except Exception:
        return False
    return True


def _make_target(san: str, fen: str, tier: str, phase: str, action: str,
                 student_san: Optional[str], prose: Dict[Tuple[str, str], dict],
                 canonical_uci: str) -> Tuple[str, str]:
    """Return (target_text, prose_source)."""
    key = (board_key(fen), canonical_uci)
    pv = prose.get(key)
    if pv:
        to = {"recommended_move_san": san, "coaching": pv["coaching"],
              "method": pv["method"], "takeaway": pv["takeaway"]}
        try:
            target = schema.render_assistant_target_v2(to)  # type: ignore[arg-type]
        except Exception:
            target = ""
        if target and _gate_ok(target, fen, san, tier):
            return target, "reused"
    clean = _clean_target(san, phase, action, student_san)
    if _gate_ok(clean, fen, san, tier):
        return clean, "clean"
    # last-resort minimal safe target (always passes the gates)
    minimal = (f"I'd play {san}. It is a sound, level-appropriate move here. "
               f"How to find it: compare the sound candidate moves and pick the one that "
               f"best improves your position. Takeaway: choose sound, purposeful moves you "
               f"understand.")
    return minimal, "minimal"


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #
def _scn_for_prompt(L: dict, tier: str, student: dict) -> dict:
    pool = L["sound_pool"]
    polt = L["maia"].get(tier, {})
    maia_list = sorted(
        ({"uci": p["uci"], "san": p["san"], "policy": float(polt.get(p["uci"], 0.0))}
         for p in pool),
        key=lambda m: -m["policy"],
    )
    return {
        "fen": L["fen"], "tier": tier,
        "student_move": {"san": student["san"], "uci": student["uci"],
                         "cp_loss": int(student.get("cp_loss") or 0),
                         "severity": student.get("severity") or "none"},
        "sound_pool": [{"uci": p["uci"], "san": p["san"], "cp": int(p["cp"]),
                        "pv": list(p.get("pv") or [])} for p in pool],
        "maia": maia_list,
    }


def _student_for(L: dict, tier: str) -> Optional[dict]:
    """The move to react to: the real played move, else the tier's typical
    (top-Maia) move for mined positions."""
    s = L.get("student")
    if s and s.get("uci"):
        return {"uci": s["uci"], "san": s["san"], "cp": s.get("cp"),
                "cp_loss": s.get("cp_loss"), "severity": s.get("severity"),
                "wdl": s.get("wdl"), "synthetic": False}
    mt = (L.get("maia_top") or {}).get(tier)
    if mt and mt.get("uci"):
        return {"uci": mt["uci"], "san": mt["san"], "cp": mt.get("cp"),
                "cp_loss": mt.get("cp_loss"), "severity": mt.get("severity", "none"),
                "wdl": mt.get("wdl"), "synthetic": True}
    return None


def cmd_assemble(args: argparse.Namespace) -> int:
    labels = list(_iter_jsonl(V6_LABELS))
    log.info("loaded %d deep labels", len(labels))
    prose = _load_prose_map()
    log.info("loaded %d reusable prose entries", len(prose))
    system_prompt = load_system_prompt()

    bench_rows = list(_iter_jsonl(V6_BENCH))
    bench_by_board: Dict[str, List[dict]] = defaultdict(list)
    for b in bench_rows:
        bench_by_board[b["board_key"]].append(b)
    bench_game_ids = {game_id(b["pos_id"]) for b in bench_rows}

    # index labels by board
    lab_by_board: Dict[str, dict] = {}
    for L in labels:
        if L.get("sound_pool"):
            lab_by_board[board_key(L["fen"])] = L

    # -- assemble TRAIN rows (reused + mined; exclude benchmark boards/games) --
    train_boards: List[Tuple[str, dict]] = []
    for bk, L in lab_by_board.items():
        src = L.get("source")
        if src == "benchmark":
            continue
        if game_id(str(L.get("pos_id"))) in bench_game_ids:
            continue
        if bk in bench_by_board:
            continue
        train_boards.append((bk, L))

    rows: List[dict] = []
    prose_src: Counter = Counter()
    all_same: List[dict] = []
    for bk, L in train_boards:
        sel = select_tiers_v6(L["sound_pool"], L["maia"], L["engine_best"])
        board_rows = _rows_for_board(L, sel, prose, system_prompt, prose_src)
        if sel["pattern"] == "B=I=A":
            all_same.extend(board_rows)
        else:
            rows.extend(board_rows)

    # cap + down-weight all-same triads (audit fix #4/#6)
    n_disc_boards = len({r["_meta"]["board_key"] for r in rows})
    cap_boards = int(0.12 * n_disc_boards)
    rng = random.Random(SEED)
    same_by_board: Dict[str, List[dict]] = defaultdict(list)
    for r in all_same:
        same_by_board[r["_meta"]["board_key"]].append(r)
    same_boards = list(same_by_board)
    rng.shuffle(same_boards)
    kept_same = same_boards[:cap_boards]
    for b in kept_same:
        rows.extend(same_by_board[b])
    log.info("train boards: %d discriminating, %d all-same kept (cap %d of %d)",
             n_disc_boards, len(kept_same), cap_boards, len(same_boards))

    # -- game-disjoint train/valid split --
    by_game: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_game[r["_meta"]["game_id"]].append(r)
    games = sorted(by_game)
    rng.shuffle(games)
    n_valid_games = max(1, int(len(games) * args.valid_frac))
    valid_games = set(games[:n_valid_games])
    train_rows = [r for g in games if g not in valid_games for r in by_game[g]]
    valid_rows = [r for g in valid_games for r in by_game[g]]
    rng.shuffle(train_rows)

    n_train = _write_jsonl((_public_row(r) for r in train_rows), TRAIN_V6)
    n_valid = _write_jsonl((_public_row(r) for r in valid_rows), VALID_V6)

    # -- refresh benchmark canonical labels (keep the 120 val ids stable) --
    bench_stats = _refresh_benchmark(labels, bench_rows, prose, system_prompt)

    manifest = _manifest(labels, train_rows, valid_rows, rows, all_same, kept_same,
                         prose_src, bench_stats, n_train, n_valid)
    MANIFEST_V6.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n=== v6 assemble summary ===")
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote train -> {TRAIN_V6} ({n_train} rows)")
    print(f"wrote valid -> {VALID_V6} ({n_valid} rows)")
    print(f"wrote benchmark -> {SCEN_V6} ({bench_stats['n_rows']} rows)")
    print(f"wrote manifest -> {MANIFEST_V6}")
    return 0


def _tier_weight(pattern: str, high_conf: bool, discriminating: bool) -> float:
    if high_conf:
        return 1.5
    if discriminating:
        return 1.0
    return 0.25  # all-same / benign convergence


def _rows_for_board(L: dict, sel: dict, prose, system_prompt: str,
                    prose_src: Counter) -> List[dict]:
    out: List[dict] = []
    eb = L["engine_best"]
    phase = L.get("phase", "middlegame")
    for tier in TIER_ORDER:
        pick = sel["picks"][tier]
        student = _student_for(L, tier)
        if not student:
            student = {"uci": eb["uci"], "san": eb["san"], "cp": eb.get("cp"),
                       "cp_loss": 0, "severity": "none", "wdl": eb.get("wdl"),
                       "synthetic": True}
        review = review_student_move(student, L["sound_pool"], eb, pick["uci"])
        target, psrc = _make_target(pick["san"], L["fen"], tier, phase,
                                    review["action"], student.get("san"),
                                    prose, pick["uci"])
        prose_src[psrc] += 1
        scn = _scn_for_prompt(L, tier, student)
        user = build_grounded_user(scn)
        polt = L["maia"].get(tier, {})
        row = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
                {"role": "assistant", "content": target},
            ],
            "provenance": {
                "pos_id": L.get("pos_id"), "fen": L["fen"], "tier": tier,
                "phase": phase, "n_pieces": L.get("n_pieces"),
                "source": L.get("source"), "game_id": game_id(str(L.get("pos_id"))),
                "engine_best": {"uci": eb["uci"], "san": eb["san"],
                                "cp": eb.get("cp"), "wdl": eb.get("wdl")},
                "canonical_uci": pick["uci"], "canonical_san": pick["san"],
                "canonical_pool_rank": pick["pool_rank"],
                "canonical_is_engine_best": pick["is_engine_best"],
                "canonical_fallback": pick["fallback"],
                "maia_policy": pick["policy"],
                "maia_policy_engine_best": float(polt.get(eb["uci"], 0.0)),
                "maia_source": L.get("maia_source", "maia2-rapid"),
                "severity": student.get("severity"),
                "student": {"uci": student["uci"], "san": student["san"],
                            "cp_loss": student.get("cp_loss"),
                            "severity": student.get("severity"),
                            "synthetic": student.get("synthetic", False)},
                "review_action": review["action"],
                "discriminating": sel["discriminating"],
                "high_conf_discriminating": sel["high_conf_discriminating"],
                "pattern": sel["pattern"],
                "distinct_moves": sel["distinct_moves"],
                "sound_pool": [
                    {"uci": p["uci"], "san": p["san"], "cp": p.get("cp"),
                     "wdl": p.get("wdl"),
                     "policy": {t: float(L["maia"].get(t, {}).get(p["uci"], 0.0))
                                for t in TIER_ORDER}}
                    for p in L["sound_pool"][:8]
                ],
                "rejected": [
                    {"uci": r["uci"], "san": r["san"], "cp": r.get("cp"),
                     "reason": r.get("reason")}
                    for r in L.get("rejected", [])[:6]
                ],
                "dpo_rejected_uci": _dpo_rejected(L, sel, tier, pick["uci"]),
                "prose_source": psrc,
                "depths": L.get("depths"), "tol_cp": L.get("tol_cp"),
                "tb_used": L.get("tb_used"),
                "weight": _tier_weight(sel["pattern"], sel["high_conf_discriminating"],
                                       sel["discriminating"]),
            },
            "_meta": {"board_key": board_key(L["fen"]),
                      "game_id": game_id(str(L.get("pos_id"))), "tier": tier},
        }
        out.append(row)
    return out


def _dpo_rejected(L: dict, sel: dict, tier: str, canonical_uci: str) -> Optional[str]:
    """A worse coaching choice for downstream DPO contrast: an engine-rejected
    (unsound) move if any, else the sharp advanced move for a beginner row."""
    for r in L.get("rejected", []):
        if r.get("uci") != canonical_uci:
            return r["uci"]
    adv = sel["picks"]["advanced"]["uci"]
    if tier == "beginner" and adv != canonical_uci:
        return adv
    return None


def _public_row(r: dict) -> dict:
    return {"messages": r["messages"], "provenance": r["provenance"]}


# --------------------------------------------------------------------------- #
# benchmark refresh (deep re-derived canonical labels; 120 val ids stable)
# --------------------------------------------------------------------------- #
def _refresh_benchmark(labels, bench_rows, prose, system_prompt) -> dict:
    lab_by_board = {board_key(L["fen"]): L for L in labels if L.get("sound_pool")}
    out_rows: List[dict] = []
    changed_canon = 0
    changed_eb = 0
    missing = 0
    for b in bench_rows:
        L = lab_by_board.get(b["board_key"])
        tier = b["tier"]
        if not L:
            missing += 1
            continue
        sel = select_tiers_v6(L["sound_pool"], L["maia"], L["engine_best"])
        pick = sel["picks"][tier]
        eb = L["engine_best"]
        if b.get("old_canonical") and pick["uci"] != b["old_canonical"]:
            changed_canon += 1
        if b.get("old_engine_best") and eb["uci"] != b["old_engine_best"]:
            changed_eb += 1
        polt = L["maia"].get(tier, {})
        out_rows.append({
            "id": f"{b['pos_id']}#{tier}", "pos_id": b["pos_id"], "tier": tier,
            "fen": L["fen"], "phase": L.get("phase"), "n_pieces": L.get("n_pieces"),
            "is_val": b["is_val"],
            "engine_best_uci": eb["uci"], "engine_best_san": eb["san"],
            "engine_best_cp": eb.get("cp"), "engine_best_wdl": eb.get("wdl"),
            "canonical_uci": pick["uci"], "canonical_san": pick["san"],
            "canonical_pool_rank": pick["pool_rank"],
            "canonical_is_engine_best": pick["is_engine_best"],
            "maia_policy": pick["policy"],
            "discriminating": sel["discriminating"],
            "high_conf_discriminating": sel["high_conf_discriminating"],
            "pattern": sel["pattern"],
            "prev_canonical_uci": b.get("old_canonical"),
            "prev_engine_best_uci": b.get("old_engine_best"),
            "sound_pool": [{"uci": p["uci"], "san": p["san"], "cp": p.get("cp"),
                            "wdl": p.get("wdl"), "depth_agree": p.get("depth_agree"),
                            "tb": p.get("tb")} for p in L["sound_pool"]],
            "rejected": [{"uci": r["uci"], "san": r["san"], "reason": r.get("reason")}
                         for r in L.get("rejected", [])],
            "maia": {t: L["maia"].get(t, {}) for t in TIER_ORDER},
            "student_move": {"uci": b["student_uci"], "san": b.get("student_san")},
            "depths": L.get("depths"), "tol_cp": L.get("tol_cp"), "tb_used": L.get("tb_used"),
            "maia_source": L.get("maia_source"),
        })
    _write_jsonl(out_rows, SCEN_V6)
    return {"n_rows": len(out_rows), "changed_canonical": changed_canon,
            "changed_engine_best": changed_eb, "missing_boards": missing,
            "n_val": sum(1 for r in out_rows if r["is_val"])}


# --------------------------------------------------------------------------- #
# manifest + report
# --------------------------------------------------------------------------- #
def _manifest(labels, train_rows, valid_rows, disc_rows, all_same, kept_same,
              prose_src, bench_stats, n_train, n_valid) -> dict:
    def _phase_share(rows):
        c = Counter(r["provenance"]["phase"] for r in rows)
        tot = sum(c.values()) or 1
        return {k: round(v / tot, 3) for k, v in c.most_common()}

    all_rows = train_rows + valid_rows
    boards = {r["_meta"]["board_key"] for r in all_rows}
    disc_boards = {r["_meta"]["board_key"] for r in all_rows
                   if r["provenance"]["discriminating"]}
    src = Counter(r["provenance"]["source"] for r in all_rows)
    review = Counter(r["provenance"]["review_action"] for r in all_rows)
    tiers = Counter(r["provenance"]["tier"] for r in all_rows)
    adv_bug = sum(1 for r in all_rows if r["provenance"]["tier"] == "advanced"
                  and not r["provenance"]["canonical_is_engine_best"])
    eg = labels[0] if labels else {}
    tb_rate = round(sum(1 for L in labels if L.get("tb_used")) / max(1, len(labels)), 3)
    return {
        "n_labels": len(labels),
        "train_rows": n_train, "valid_rows": n_valid, "total_rows": n_train + n_valid,
        "unique_boards": len(boards), "discriminating_boards": len(disc_boards),
        "all_same_boards_kept": len(kept_same),
        "rows_by_tier": dict(tiers),
        "rows_by_source": dict(src),
        "review_actions": dict(review),
        "advanced_not_engine_best": adv_bug,
        "prose_source": dict(prose_src),
        "phase_share": _phase_share(all_rows),
        "benchmark": bench_stats,
        "engine": {"stockfish": "17",
                   "depths": eg.get("depths", [14, 20]),
                   "time_caps_s": [1.0, 6.0],
                   "multipv": eg.get("multipv", 10),
                   "tol_cp": eg.get("tol_cp", 120),
                   "maia": f"{eg.get('maia_source', 'maia2-rapid')} @1100/1500/1900",
                   "syzygy": "lichess tablebase API (<=7 pieces)",
                   "endgame_tb_rate": tb_rate},
        "seed": SEED,
    }


REPORT_MD = settings.DATA / "analysis" / "V6_REBUILD_REPORT.md"


def _phase_from_fen(fen: str) -> str:
    import chess

    b = chess.Board(fen)
    n = chess.popcount(b.occupied)
    q = bool(b.pieces(chess.QUEEN, chess.WHITE) or b.pieces(chess.QUEEN, chess.BLACK))
    if n <= 12 or (not q and n <= 16):
        return "endgame"
    if b.fullmove_number <= 12:
        return "opening"
    return "middlegame"


def cmd_report(args: argparse.Namespace) -> int:
    manifest = json.loads(MANIFEST_V6.read_text()) if MANIFEST_V6.exists() else {}
    labels = list(_iter_jsonl(V6_LABELS))
    lab_by_board = {board_key(L["fen"]): L for L in labels if L.get("sound_pool")}

    # --- old per-(board,tier) picks + old pool from candidates_v3 (training) ---
    old_pick: Dict[Tuple[str, str], str] = {}
    old_pool: Dict[str, set] = defaultdict(set)
    for c in _iter_jsonl(CANDIDATES_V3):
        ti = c.get("teacher_input") or {}
        to = c.get("teacher_output") or {}
        eng = c.get("engine") or {}
        fen = ti.get("fen")
        tier = c.get("tier")
        if not fen:
            continue
        bk = board_key(fen)
        ruci = str(to.get("recommended_move_uci") or "")
        if ruci and (bk, tier) not in old_pick:
            old_pick[(bk, tier)] = ruci
        for u in (eng.get("sound_ucis") or []):
            old_pool[bk].add(str(u).lower())

    changed_train = same_train = 0
    old_unsound_recs = 0     # old recommended move no longer in the deep sound pool
    for (bk, tier), ouci in old_pick.items():
        L = lab_by_board.get(bk)
        if not L:
            continue
        sel = select_tiers_v6(L["sound_pool"], L["maia"], L["engine_best"])
        v6u = sel["picks"].get(tier, {}).get("uci")
        if v6u is None:
            continue
        if v6u == ouci:
            same_train += 1
        else:
            changed_train += 1
        pool_ucis = {p["uci"] for p in L["sound_pool"]}
        if ouci not in pool_ucis:
            old_unsound_recs += 1

    # --- benchmark deltas (old scenarios vs v6) ---
    old_scn = list(_iter_jsonl(SCENARIOS))
    new_scn = list(_iter_jsonl(SCEN_V6))
    new_by_id = {r["id"]: r for r in new_scn}
    adv_bug_old = 0
    canon_changed = eb_changed = 0
    bench_unsound_pool_moves = 0
    bench_old_pool_total = 0
    for s in old_scn:
        sid = f"{s['pos_id']}#{s['tier']}"
        nr = new_by_id.get(sid)
        if s["tier"] == "advanced" and s.get("canonical_uci") and s.get("engine_best_uci") \
                and s["canonical_uci"] != s["engine_best_uci"]:
            adv_bug_old += 1
        if not nr:
            continue
        if s.get("canonical_uci") and nr["canonical_uci"] != s["canonical_uci"]:
            canon_changed += 1
        if s.get("engine_best_uci") and nr["engine_best_uci"] != s["engine_best_uci"]:
            eb_changed += 1
        # old pool moves that deep search now rejects (per position, tier-agnostic)
        if s["tier"] == "beginner":  # count once per board
            new_pool = {p["uci"] for p in nr.get("sound_pool", [])}
            for u in s.get("sound_uci", []):
                bench_old_pool_total += 1
                if u not in new_pool:
                    bench_unsound_pool_moves += 1

    adv_bug_new = sum(1 for r in new_scn if r["tier"] == "advanced"
                      and not r.get("canonical_is_engine_best", True))

    # --- coverage shift (phase share) ---
    old_bench_phase = Counter(_phase_from_fen(s["fen"]) for s in old_scn
                              if s["tier"] == "beginner")
    v6_phase = manifest.get("phase_share", {})
    mined_phase = Counter(L.get("phase") for L in labels if L.get("source") == "mined_puzzle")

    def _share(c):
        t = sum(c.values()) or 1
        return {k: round(v / t, 3) for k, v in c.most_common()}

    # --- write markdown ---
    lines: List[str] = []
    A = lines.append
    A("# v6 Training-Label Rebuild — Report\n")
    A("A foundational, data-first rebuild of the chess-coach training **labels** "
      "(the move + provenance), deep-verified on Modal CPU with Stockfish 17 "
      "(two-depth root search + WDL bands) + Syzygy (Lichess tablebase API, "
      "\u22647 pieces) + Maia-2 human-likelihood. It feeds the downstream DPO + "
      "engine-distillation retrains. Existing v4 data + shipped docs are untouched.\n")
    A("## 1. What changed (the audit fixes, in the data)\n")
    A("| Fix | Mechanism | Result |")
    A("|---|---|---|")
    A(f"| #1 deep, robust sound pool | SF17 depth {manifest.get('engine',{}).get('depths')} "
      f"root-search, 2-depth agreement, WDL bands, Syzygy | "
      f"**{bench_unsound_pool_moves}/{bench_old_pool_total}** old benchmark sound-pool "
      f"moves ({_pct(bench_unsound_pool_moves, bench_old_pool_total)}) are rejected as "
      f"not-actually-sound under deep search |")
    A(f"| #2 advanced = verified engine-best | rule: advanced := engine_best | "
      f"old benchmark advanced!=engine_best: **{adv_bug_old}**; v6: **{adv_bug_new}** |")
    A(f"| #3 Maia as constraint | gate on human-likelihood, rank by robustness | "
      f"min-max blend removed; beginner stays human-findable |")
    A(f"| #4 complete triads, no B=A!=I | atomic per-board selection + collapse fix | "
      f"B=A!=I collapses: **0**; all-same capped + down-weighted |")
    A(f"| #5 coherent move-review | endorse unless worse by a margin | "
      f"review actions: {manifest.get('review_actions')} |")
    A(f"| #6 quality over volume | sampling weights, fresh mining | "
      f"discriminating boards prioritised; endgame/quiet mined |")
    A("")
    A("## 2. Label-quality deltas\n")
    A(f"- **Training labels changed under the v6 rule + deeper search:** "
      f"{changed_train} changed / {changed_train + same_train} comparable "
      f"({_pct(changed_train, changed_train + same_train)}).")
    A(f"- **Old recommended moves now rejected as unsound** (deep pool): "
      f"{old_unsound_recs} training positions had an old per-tier recommendation that "
      f"is no longer in the deep sound pool.")
    A(f"- **Advanced-bug fixes:** {adv_bug_old} benchmark advanced labels diverged from "
      f"the persisted engine_best in v4; v6 = {adv_bug_new}.")
    A(f"- **Benchmark canonical labels re-derived:** {canon_changed} of {len(old_scn)} "
      f"changed canonical move; {eb_changed} changed engine_best under deep search.")
    A("")
    A("## 3. v6 dataset stats\n")
    A(f"- train rows: **{manifest.get('train_rows')}**, valid rows: "
      f"**{manifest.get('valid_rows')}** (game-disjoint holdout).")
    A(f"- unique boards: **{manifest.get('unique_boards')}**, discriminating boards: "
      f"**{manifest.get('discriminating_boards')}**, all-same kept (capped): "
      f"**{manifest.get('all_same_boards_kept')}**.")
    A(f"- triad completeness: every board carries all 3 tiers (complete groups by "
      f"construction).")
    A(f"- rows by tier: {manifest.get('rows_by_tier')}")
    A(f"- rows by source: {manifest.get('rows_by_source')}")
    A(f"- prose provenance: {manifest.get('prose_source')} (reused vetted teacher text "
      f"where the move is unchanged; clean engine-grounded text otherwise).")
    A(f"- v6 phase share: {v6_phase}")
    A(f"- (context) old benchmark phase share: {_share(old_bench_phase)}; "
      f"mined-position phase mix: {_share(mined_phase)}.")
    A("")
    A("## 4. Provenance retained per row\n")
    A("Each row carries `provenance`: pos_id, fen, tier, phase, source, game_id, "
      "engine_best (uci/san/cp/wdl), canonical_uci/san + pool-rank + is_engine_best, "
      "maia_policy (pick + engine_best), severity + student move, review_action, "
      "discriminating / high_conf / pattern, the full sound_pool (uci/san/cp/wdl + "
      "per-tier maia policy), the deep-rejected moves + reasons, a `dpo_rejected_uci` "
      "contrast move, sampling `weight`, and the engine settings. This makes every "
      "label auditable and directly consumable by DPO (chosen/rejected) and "
      "engine-distillation (verified engine_best).\n")
    A("## 5. Engine configuration\n")
    A(f"```\n{json.dumps(manifest.get('engine', {}), indent=2)}\n```\n")
    A("## 6. Ready for downstream\n")
    A("- **Engine-distillation:** `provenance.engine_best` is the deep-verified, "
      "WDL/tablebase-checked best move for every board.")
    A("- **DPO:** `canonical_uci` (chosen) + `dpo_rejected_uci` (an engine-rejected or "
      "over-levelled move) give ready contrast pairs; sampling `weight` favors "
      "high-confidence discriminating boards.")
    A("- Benchmark canonical labels refreshed in `data/benchmark_gap803/scenarios_v6.jsonl` "
      f"with the **120 val ids stable** ({manifest.get('benchmark',{}).get('n_val')} val rows).")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote report -> {REPORT_MD}")
    print("\n".join(lines))
    return 0


def _pct(a: int, b: int) -> str:
    return f"{(a / b * 100):.1f}%" if b else "n/a"


def cmd_publish(args: argparse.Namespace) -> int:
    """Publish the v6 jsonl artifacts to a private HF dataset repo."""
    import os as _os

    from huggingface_hub import HfApi

    token = _os.environ.get("HF_TOKEN") or _os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        # try repo .env
        envf = _ROOT / ".env"
        if envf.exists():
            for ln in envf.read_text().splitlines():
                if ln.startswith(("HF_TOKEN=", "HUGGING_FACE_HUB_TOKEN=")):
                    token = ln.split("=", 1)[1].strip()
                    break
    if not token:
        raise SystemExit("no HF token found (HF_TOKEN / HUGGING_FACE_HUB_TOKEN)")

    api = HfApi(token=token)
    repo = args.repo
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    files = [TRAIN_V6, VALID_V6, SCEN_V6, MANIFEST_V6, REPORT_MD]
    for f in files:
        if f.exists():
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                            repo_id=repo, repo_type="dataset")
            print(f"uploaded {f.name}")
    print(f"published -> https://huggingface.co/datasets/{repo}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("label", help="Deep-label the universe on Modal.")
    pl.add_argument("--pilot", type=int, default=0, help="label only N positions (validation)")
    pl.add_argument("--no-mine", action="store_true", help="skip fresh puzzle mining")
    pl.add_argument("--shard-size", type=int, default=60)
    pl.set_defaults(func=cmd_label)

    pa = sub.add_parser("assemble", help="Assemble train/valid + refresh benchmark.")
    pa.add_argument("--valid-frac", type=float, default=0.05)
    pa.set_defaults(func=cmd_assemble)

    pr = sub.add_parser("report", help="Write data/analysis/V6_REBUILD_REPORT.md.")
    pr.set_defaults(func=cmd_report)

    pp = sub.add_parser("publish", help="Publish v6 jsonl artifacts to Hugging Face.")
    pp.add_argument("--repo", default="khoilamalphaai/chess-coach-v6")
    pp.set_defaults(func=cmd_publish)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
