#!/usr/bin/env python3
"""MATCHED same-backend, spec-exact PROMPT CONTROL for the 32B coach.

Purpose
-------
Settle — honestly — the BrainLift's falsifiable hypothesis that *"a well-prompted
base cannot reproduce the tier-appropriate move selection, including at 32B."* The
earlier base comparison that motivated the hypothesis used a DIFFERENT backend
(aws-bedrock) than v4's Unsloth/Modal path, so it was confounded (the eval audit
flagged this). This script removes BOTH confounds at once and turns the hypothesis
into a RESULT:

  1. **Same backend as v4.** It serves the EXACT base v4 was trained on
     (``unsloth/Qwen3-32B-unsloth-bnb-4bit``) through the SAME Unsloth/Modal path,
     with the SAME tokenizer, 4-bit quant, greedy decoding, token cap, and
     post-processing (``_strip_think`` + ``_clean_lead``) as
     ``src/eval/eval_modal_v4.py`` / ``eval_modal_v4_val.py`` — but WITHOUT the v4
     LoRA adapter (base weights only).

  2. **A real, spec-exact prompt.** Instead of the shipped coach system prompt,
     the base is given a hand-authored system prompt that states the EXACT
     canonical tier-selection algorithm from ``src/teacher/tier_select.py``
     (per-tier human weight w = {beginner 1.0, intermediate 0.5, advanced 0.0};
     min-max normalized engine-eval and Maia-policy; the 50/50 intermediate blend;
     and the documented tie-break: higher score, then higher raw cp, then earlier
     in the best-first sound list). The USER prompt is the IDENTICAL grounded
     prompt v4 saw (``build_grounded_user`` = verified facts + the sound pool with
     internal evals + the per-tier Maia human-likelihoods), so the base gets the
     same grounded facts, no more and no less.

It runs on the 120 held-out VAL positions x 3 tiers (360 scenarios). Reusing VAL
is legitimate here because the prompt is HAND-AUTHORED, not tuned on VAL. Outputs
are scored with the SAME deterministic tier-policy-exact-match scorer used for v4
(extract the move with :func:`src.eval.evaluate.extract_recommended_move` ->
:func:`src.teacher.coach_gate.pick_recommendation`; compare to ``canonical_uci``,
averaged over the three tiers) so the number is directly comparable to the
published base (0.347) and v4 (0.767).

Two system-prompt variants are generated in ONE GPU session (one model load):
  * ``spec_exact``     — the key matched control (faithful statement of the rule).
  * ``spec_exact_opt`` — a lightly prompt-optimized variant (the same rule with
                         the per-tier shortcut surfaced first + crisper steps),
                         the optional "prompt-optimized" arm the task allows.

Commands
--------
    # build prompts + generate on Modal (spawned detached), then wait + download + score:
    MODAL_PROFILE=chess-instructor-3 modal run scripts/prompt_control_32b.py --block

    # spawn only (poll the Volume yourself), then later score:
    MODAL_PROFILE=chess-instructor-3 modal run scripts/prompt_control_32b.py
    MODAL_PROFILE=chess-instructor-3 modal run scripts/prompt_control_32b.py --score-only

The base ``unsloth/Qwen3-32B-unsloth-bnb-4bit`` is a PUBLIC repo, so no HF token /
Modal secret is required (it is pulled at runtime by Unsloth, hf_transfer-fast).
The image definition is byte-identical to ``eval_modal_v4_val.py`` so its cached
layers are reused on the v4 eval workspace (chess-instructor-3) — same environment
as the v4 eval, i.e. maximally matched.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import modal

# --------------------------------------------------------------------------- #
# Names / paths / decoding (mirrors eval_modal_v4_val.py)
# --------------------------------------------------------------------------- #
APP_NAME = "chess-coach-prompt-control-32b"
VOLUME_NAME = "chess-coach-lora"
VOL_MOUNT = "/vol"
REMOTE_PROMPTS = "/data/prompts_prompt_control.jsonl"
REMOTE_OUT = f"{VOL_MOUNT}/prompt_control_32b/gen.jsonl"

#: The EXACT base v4 was trained on (public unsloth dynamic-4bit checkpoint).
BASE_MODEL = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
GPU = "A100-80GB"
TIMEOUT_S = 2 * 3600
CUDA_TAG = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION = "3.11"
#: Byte-identical to the v4 eval decoding.
MAX_NEW_TOKENS = 512
BATCH_SIZE = 32

TIERS = ("beginner", "intermediate", "advanced")

# --------------------------------------------------------------------------- #
# The spec-exact system prompts (the whole point of the control).
# These mirror src/teacher/tier_select.py PRECISELY. The user prompt fed to the
# model is the UNCHANGED grounded prompt v4 saw (build_grounded_user), so these
# system prompts are the ONLY difference vs the shipped coach.
# --------------------------------------------------------------------------- #
SPEC_EXACT_SYSTEM = """\
You are a chess coach doing move review for a student at a stated rating tier \
(beginner, intermediate, or advanced).

For each position you are given, in the user message:
- the move the student played;
- a list of engine-SOUND candidate moves, listed best-first, each with an internal \
evaluation in centipawns (cp) from the side-to-move's point of view (higher cp = \
better for the mover);
- for THIS tier, how likely a human at that level is to play each move, under \
"Human-likelihood at this tier (Maia)", given as percentages.

Recommend exactly ONE move, chosen by the following EXACT rule. Do not substitute \
your own judgement; apply the rule mechanically.

Step 1 - Restrict to the sound list. Only the moves in the "Engine-sound candidate \
moves" list are eligible. Never recommend a move outside it.

Step 2 - Normalize two signals ACROSS the sound candidates:
- eval_norm: rescale the candidates' centipawn evaluations to the range 0..1, where \
the highest-cp sound move = 1 and the lowest-cp sound move = 0. (If every candidate \
has the same cp, set every eval_norm = 1.)
- human_norm: rescale the candidates' human-likelihood percentages to the range \
0..1, where the most human-likely sound move = 1 and the least human-likely = 0. If \
a sound move has no percentage listed, treat its percentage as 0. (If every listed \
percentage is equal, set every human_norm = 1.)

Step 3 - Blend with the tier weight w on the human term:
- beginner:     w = 1.0
- intermediate: w = 0.5
- advanced:     w = 0.0
score(move) = (1 - w) * eval_norm + w * human_norm

Step 4 - Pick the sound move with the HIGHEST score. Break ties in this EXACT \
order: (1) higher score; then (2) higher raw centipawn evaluation; then (3) it \
appears earlier in the best-first sound list.

Equivalently, the rule collapses per tier to:
- ADVANCED (w=0): the engine-best sound move = the FIRST move in the best-first \
sound list.
- BEGINNER (w=1): the sound move with the HIGHEST human-likelihood percentage (the \
most findable sound move for a human at this level) - often NOT the engine's top move.
- INTERMEDIATE (w=0.5): the sound move that best balances engine strength and \
human-likelihood under the 50/50 blended score above.

Then write your reply in exactly this shape: begin with `I'd play <MOVE>.` where \
<MOVE> is the chosen move in standard algebraic notation; give 2-4 sentences of \
encouraging coaching tied to the student's actual mistake and a concrete plan; and \
end with one line `Takeaway: <one transferable sentence>.` Use the centipawn \
numbers and percentages ONLY to make the selection - never quote them, and never \
write "engine", "Stockfish", or "computer" in your reply.\
"""

SPEC_EXACT_OPT_SYSTEM = """\
You are a chess move-review coach. The student's rating tier is stated in the user \
message (beginner, intermediate, or advanced). The user message also gives you a \
best-first list of engine-SOUND candidate moves (each with an internal centipawn \
evaluation, higher = stronger for the side to move) and, for this tier, the \
"Human-likelihood at this tier (Maia)" percentages for the most human moves.

Pick exactly ONE move by applying this exact tier rule to the SOUND list only:

- If the tier is ADVANCED: recommend the FIRST move in the best-first sound list \
(the strongest sound move).
- If the tier is BEGINNER: recommend the sound move with the HIGHEST \
human-likelihood percentage. Read the "Human-likelihood at this tier (Maia)" list, \
ignore any move that is not in the sound list, and among the sound moves take the \
one with the largest percentage (a sound move with no percentage listed counts as \
0%). This is usually NOT the engine's top move.
- If the tier is INTERMEDIATE: recommend the sound move that best balances the two \
signals. Concretely, rescale the sound candidates' centipawn evals to 0..1 (best=1, \
worst=0) and their human-likelihood percentages to 0..1 (most human=1, least=0; \
missing=0), then choose the move with the highest average of the two rescaled \
numbers. Break ties by higher raw eval, then by earlier position in the list.

The recommended move MUST be one of the sound candidates. Do the selection first, \
then write the coaching.

Reply format (exactly): start with `I'd play <MOVE>.` (standard algebraic \
notation), then 2-4 sentences of plain, encouraging coaching tied to the student's \
mistake and a concrete plan, then a final line `Takeaway: <one sentence>.` Never \
quote the centipawn numbers or the percentages, and never mention "engine", \
"Stockfish", or "computer" - use them only to choose the move.\
"""

SYSTEM_VARIANTS: Dict[str, str] = {
    "spec_exact": SPEC_EXACT_SYSTEM,
    "spec_exact_opt": SPEC_EXACT_OPT_SYSTEM,
}

# --------------------------------------------------------------------------- #
# Local paths + prompt build (only when running locally; guarded so the remote
# container never imports the repo / engines).
# --------------------------------------------------------------------------- #
if modal.is_local():
    REPO_ROOT: Optional[Path] = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    OUT_DIR: Optional[Path] = REPO_ROOT / "data" / "prompt_control_32b"
    LOCAL_PROMPTS: Optional[Path] = OUT_DIR / "prompts.jsonl"
    LOCAL_OUT: Optional[Path] = OUT_DIR / "gen.jsonl"
    LOCAL_SCORES: Optional[Path] = OUT_DIR / "scores.json"
    SCENARIOS: Optional[Path] = REPO_ROOT / "data" / "benchmark_gap803" / "scenarios.jsonl"
    VAL_IDS: Optional[Path] = REPO_ROOT / "data" / "benchmark_honest" / "val_ids.txt"
else:
    REPO_ROOT = OUT_DIR = LOCAL_PROMPTS = LOCAL_OUT = LOCAL_SCORES = SCENARIOS = VAL_IDS = None


def _read_jsonl(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _val_scenarios() -> List[dict]:
    keep = set(VAL_IDS.read_text(encoding="utf-8").split())
    return [s for s in _read_jsonl(SCENARIOS) if s.get("pos_id") in keep]


def _build_prompts_local() -> int:
    """Build the val-slice prompts (one row per variant x scenario) and write them
    to LOCAL_PROMPTS so they can be baked into the Modal image. USER prompt is the
    unchanged grounded prompt v4 saw; SYSTEM is the spec-exact variant."""
    from src.eval.benchmark.prompts import build_grounded_user

    scns = _val_scenarios()
    n_pos = len({s["pos_id"] for s in scns})
    rows: List[dict] = []
    for variant, system in SYSTEM_VARIANTS.items():
        for s in scns:
            rows.append({
                "id": f"{variant}::{s['id']}",
                "variant": variant,
                "scenario_id": s["id"],
                "pos_id": s["pos_id"],
                "tier": s["tier"],
                "phase": s.get("phase"),
                "severity": s.get("severity"),
                "system": system,
                "user": build_grounded_user(s),
            })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_PROMPTS.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"[build] {len(rows)} prompts ({n_pos} val positions x {len(TIERS)} tiers x "
          f"{len(SYSTEM_VARIANTS)} variants) -> {LOCAL_PROMPTS}")
    return len(rows)


# --------------------------------------------------------------------------- #
# Modal image — byte-identical to eval_modal_v4_val.py so the cached (heavy) pip
# layers are reused on the v4 eval workspace (chess-instructor-3).
# --------------------------------------------------------------------------- #
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .apt_install("git")
    .pip_install("unsloth", "trl", "peft", "bitsandbytes", "transformers", "datasets",
                 "accelerate", "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)

if modal.is_local():
    _build_prompts_local()
    image = image.add_local_file(LOCAL_PROMPTS.as_posix(), REMOTE_PROMPTS)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


# --------------------------------------------------------------------------- #
# Post-processing — byte-identical to eval_modal_v4_val.py.
# --------------------------------------------------------------------------- #
def _strip_think(text: str) -> str:
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.replace("<think>", "").replace("</think>", "").strip()


def _clean_lead(text: str) -> str:
    """Drop a leading garble/prompt-echo fragment before the first "I'd play"."""
    t = text.strip()
    if t.startswith("I'd play") or t.startswith("I\u2019d play"):
        return t
    idx = t.find("I'd play")
    if idx < 0:
        idx = t.find("I\u2019d play")
    if 0 < idx <= 160:
        return t[idx:].strip()
    return t


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={VOL_MOUNT: volume},
              retries=modal.Retries(max_retries=6, initial_delay=5.0, backoff_coefficient=1.0))
def generate(limit: int = 0) -> dict:
    """BASE-ONLY generation over the val prompts (both spec-exact variants).

    Identical load/decoding to eval_modal_v4_val.generate EXCEPT it loads the base
    ``unsloth/Qwen3-32B-unsloth-bnb-4bit`` directly (NO v4 LoRA adapter). Greedy,
    repetition_penalty=1.15, no_repeat_ngram_size=4, enable_thinking=False, 512 new
    tokens, then _strip_think + _clean_lead. Resumable by row id."""
    import time
    from datetime import datetime, timezone

    import torch
    from unsloth import FastLanguageModel

    volume.reload()
    print(f"[load] Unsloth BASE 4-bit (NO adapter) = {BASE_MODEL}")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=3072, load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    rows = [json.loads(l) for l in open(REMOTE_PROMPTS, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]

    done: set = set()
    Path(REMOTE_OUT).parent.mkdir(parents=True, exist_ok=True)
    if Path(REMOTE_OUT).exists():
        for l in open(REMOTE_OUT, encoding="utf-8"):
            if l.strip():
                try:
                    done.add(json.loads(l)["id"])
                except Exception:  # noqa: BLE001
                    pass
    todo = [r for r in rows if r["id"] not in done]
    print(f"[gen] {len(todo)} pending of {len(rows)} ({len(done)} done)")

    t0 = time.time()
    written = 0
    n_lead_cleaned = 0
    with open(REMOTE_OUT, "a", encoding="utf-8") as out:
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            texts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": r["system"]},
                     {"role": "user", "content": r["user"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                for r in batch
            ]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=3072).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                     repetition_penalty=1.15, no_repeat_ngram_size=4,
                                     pad_token_id=tok.pad_token_id)
            for r, g, inp in zip(batch, gen, enc["input_ids"]):
                raw = _strip_think(tok.decode(g[inp.shape[0]:], skip_special_tokens=True))
                cleaned = _clean_lead(raw)
                if cleaned != raw:
                    n_lead_cleaned += 1
                out.write(json.dumps({
                    "id": r["id"], "variant": r["variant"], "scenario_id": r["scenario_id"],
                    "model": "prompt_control_32b", "condition": "grounded",
                    "tier": r["tier"], "phase": r["phase"], "severity": r["severity"],
                    "pos_id": r["pos_id"], "output": cleaned, "output_raw": raw,
                    "lead_cleaned": cleaned != raw,
                    "prompt_tokens": int(inp.shape[0]), "completion_tokens": 0,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")
                written += 1
            out.flush()
            volume.commit()
            dt = time.time() - t0
            n = i + len(batch)
            print(f"  {n}/{len(todo)} ({dt/max(1,n):.2f}s/it, "
                  f"eta {dt/max(1,n)*(len(todo)-n)/60:.0f}m, lead_cleaned={n_lead_cleaned})")
    volume.commit()
    return {"written": written, "total": len(rows), "lead_cleaned": n_lead_cleaned,
            "secs": round(time.time() - t0, 1), "out": REMOTE_OUT}


# --------------------------------------------------------------------------- #
# Scoring (local) — the SAME tier-policy-exact-match scorer used for v4
# (reproduce_v4.py / honest_v4.py): extract the move with the strict any-legal
# extractor and compare to canonical_uci, averaged over tiers.
# --------------------------------------------------------------------------- #
def _score_local() -> dict:
    from statistics import mean

    from src.eval.evaluate import extract_recommended_move

    scns = _val_scenarios()
    by_id = {s["id"]: s for s in scns}
    n_pos = len({s["pos_id"] for s in scns})

    gens = _read_jsonl(LOCAL_OUT)
    by_variant: Dict[str, List[dict]] = {}
    for r in gens:
        by_variant.setdefault(r.get("variant", "spec_exact"), []).append(r)

    # canonical beginner!=advanced opportunities (distinct-moves denominator)
    canon: Dict[str, Dict[str, Optional[str]]] = {}
    for s in scns:
        canon.setdefault(s["pos_id"], {})[s["tier"]] = s.get("canonical_uci")
    diff_pos = [pid for pid, cd in canon.items()
                if cd.get("beginner") and cd.get("advanced") and cd["beginner"] != cd["advanced"]]

    results: Dict[str, dict] = {}
    for variant, rows in by_variant.items():
        rec: Dict[str, Dict[str, Optional[str]]] = {}
        by_tier = {t: [0, 0] for t in TIERS}   # [match, n]
        sound = [0, 0]
        for r in rows:
            scn = by_id.get(r["scenario_id"])
            if scn is None:
                continue
            _san, uci = extract_recommended_move(
                r.get("output", ""), scn["fen"], scn["student_move"].get("uci") or "")
            rec.setdefault(scn["pos_id"], {})[scn["tier"]] = uci
            t = scn["tier"]
            by_tier[t][1] += 1
            if uci and uci == scn.get("canonical_uci"):
                by_tier[t][0] += 1
            sound[1] += 1
            if uci and uci in set(scn.get("sound_uci", [])):
                sound[0] += 1
        per_tier = {t: (by_tier[t][0] / by_tier[t][1]) for t in TIERS if by_tier[t][1]}
        overall = mean(per_tier.values()) if per_tier else 0.0
        # distinct-moves-per-level over all canonical b!=a opportunities
        distinct = 0
        for pid in diff_pos:
            tp = rec.get(pid, {})
            mb, ma = tp.get("beginner"), tp.get("advanced")
            if mb and ma and mb != ma:
                distinct += 1
        results[variant] = {
            "n_rows": len(rows),
            "n_scored": sum(by_tier[t][1] for t in TIERS),
            "tier_policy_match_overall": round(overall, 4),
            "tier_policy_match_by_tier": {t: round(by_tier[t][0] / by_tier[t][1], 4)
                                          if by_tier[t][1] else None for t in TIERS},
            "per_tier_counts": {t: {"match": by_tier[t][0], "n": by_tier[t][1]} for t in TIERS},
            "move_sound": round(sound[0] / sound[1], 4) if sound[1] else None,
            "distinct_moves_per_level": round(distinct / len(diff_pos), 4) if diff_pos else None,
            "differentiating_n": len(diff_pos),
            "distinct_count": distinct,
        }

    summary = {
        "n_val_positions": n_pos,
        "scenarios_per_variant": n_pos * len(TIERS),
        "base_model": BASE_MODEL,
        "decoding": {"do_sample": False, "repetition_penalty": 1.15,
                     "no_repeat_ngram_size": 4, "max_new_tokens": MAX_NEW_TOKENS,
                     "enable_thinking": False, "postproc": "_strip_think + _clean_lead"},
        "reference": {"base_default_q3_32b_tier_fit": 0.3472, "ours_v4_tuned_tier_fit": 0.7667,
                      "ours_v4_by_tier": {"beginner": 0.725, "intermediate": 0.7333, "advanced": 0.8417},
                      "q3_32b_by_tier": {"beginner": 0.3, "intermediate": 0.3833, "advanced": 0.3583}},
        "variants": results,
    }
    LOCAL_SCORES.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 78)
    print("SPEC-EXACT PROMPT CONTROL — tier-policy exact match (vs canonical_uci)")
    print("=" * 78)
    print(f"base default (q3_32b, shipped coach prompt): 0.3472   "
          f"[B 0.300 / I 0.383 / A 0.358]")
    print(f"v4 tuned (same backend):                     0.7667   "
          f"[B 0.725 / I 0.733 / A 0.842]")
    print("-" * 78)
    for variant, r in results.items():
        bt = r["tier_policy_match_by_tier"]
        print(f"{variant:16} overall={r['tier_policy_match_overall']:.4f}  "
              f"[B {bt['beginner']} / I {bt['intermediate']} / A {bt['advanced']}]  "
              f"sound={r['move_sound']}  distinct={r['distinct_moves_per_level']}  "
              f"(scored {r['n_scored']}/{n_pos*len(TIERS)})")
    print("=" * 78)
    print(f"scores -> {LOCAL_SCORES}")
    return summary


def _download_out() -> bool:
    """Pull the Volume gen file to LOCAL_OUT. Returns True if present.

    Download into a PRE-EXISTING temp directory so ``modal volume get`` places the
    file inside it (``_dl/gen.jsonl``); if a modal version instead writes the file
    AS the destination path, handle that too."""
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOCAL_OUT.parent / "_dl"
    if tmp.is_file():          # a prior single-file download wrote the file AS _dl
        tmp.unlink()
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force",
                    VOLUME_NAME, "/prompt_control_32b/gen.jsonl", str(tmp)], check=True)
    got = tmp / "gen.jsonl"
    if not got.exists() and tmp.is_file():  # modal wrote the file AS the dest path
        got = tmp
    if got.exists():
        shutil.move(str(got), str(LOCAL_OUT))
        shutil.rmtree(LOCAL_OUT.parent / "_dl", ignore_errors=True)
        return True
    shutil.rmtree(tmp, ignore_errors=True)
    return LOCAL_OUT.exists()


@app.local_entrypoint()
def main(limit: int = 0, block: bool = False, score_only: bool = False) -> None:
    if score_only:
        if _download_out():
            _score_local()
        else:
            print("no gen file on the Volume yet; run without --score-only first.")
        return

    call = generate.spawn(limit=limit)
    print(f"SPAWNED generate call_id={call.object_id} — running detached on Modal; "
          f"poll {REMOTE_OUT} on the Volume, or re-run with --score-only to download + score.")
    if block:
        res = call.get()
        print(json.dumps(res, indent=2, default=str))
        if _download_out():
            _score_local()
        else:
            print("BLOCKED: generation returned but gen file missing on the Volume.")
