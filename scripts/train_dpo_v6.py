#!/usr/bin/env python3
"""DPO preference-tune the shipped **v4** 32B coach into **v6-dpo** on Modal.

Goal (the "moat"): SHARPEN tier-appropriate move selection on top of v4 WITHOUT
regressing v4's soundness/format. So policy AND reference both initialize from the
**v4 LoRA adapter** (never from the bare base): DPO only has to move the *move
choice*, and the reference/KL (beta) pressure holds the rest in place.

Data (v6 preference pairs)
--------------------------
Built LOCALLY from ``data/dataset/train_v6.jsonl``. Each usable row carries a
deep-verified tier-appropriate move (``provenance.canonical_uci`` = PREFERRED) and
an off-tier / off-spec contrast move (``provenance.dpo_rejected_uci`` = REJECTED).
For each pair we take the row's REAL v4-style assistant target as ``chosen`` and
build ``rejected`` by swapping ONLY the move SAN in that same text (token-boundary
replace of ``canonical_san`` -> ``rejected_san``). chosen and rejected are then
byte-identical except for the move, so DPO learns MOVE preference, not prose style.
Pairs are balanced across tier / phase / move-rank by weighted stratified sampling
(favoring the provenance ``weight``), so beginner (the richest moat) does not swamp
the scarcer intermediate/advanced contrasts.

Method (TRL DPO on the 32B QLoRA)
---------------------------------
* Same base as v4: ``unsloth/Qwen3-32B-unsloth-bnb-4bit`` (loaded 4-bit with
  Unsloth so the v4 adapter sits on the SAME quantized base it was trained on).
* Two PEFT adapters on that base: ``default`` (trainable, init = v4) and
  ``reference`` (frozen = v4). DPOConfig(model_adapter_name="default",
  ref_adapter_name="reference") -> the KL is measured against v4, not the base.
* Low LR (~1e-5), ~1 short pass, beta=0.1 reference pressure, checkpoint every few
  steps to the shared Volume (a timeout / credit-kill leaves a usable adapter).

Selection
---------
Every checkpoint (+ the v4 baseline) is generated over ``valid_v6.jsonl`` (the
game-disjoint v6 dev set) with the SAME greedy decode as the v4 eval, then scored
LOCALLY with the canonical ``extract_recommended_move`` for tier-policy exact match
(mean over tiers) vs ``provenance.canonical_uci``, tiebreak move soundness. The
best checkpoint is promoted to the run's ``adapter`` dir, pushed to a Modal Volume
and to HF (``khoilamalphaai/chess-coach-32b-v6-dpo``). The 120-position held-out
test set is NEVER touched here (that is Stage-4's job).

Commands (ALWAYS scrub the bare kim-lam tokens + pin a FUNDED workspace first)::

    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    export MODAL_PROFILE=chess-instructor-4
    P=/Users/khoilam/.venvs/mlx/bin/modal

    $P run scripts/train_dpo_v6.py --smoke              # tiny loop (proves 2-adapter DPO)
    $P run --detach scripts/train_dpo_v6.py --skip-eval # full short pass (resumable)
    $P run scripts/train_dpo_v6.py --skip-train         # eval every ckpt + select + push
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

# --------------------------------------------------------------------------- #
# Names / paths
# --------------------------------------------------------------------------- #
APP_NAME: str = "chess-coach-dpo-v6"
VOLUME_NAME: str = "chess-coach-lora"            # shared volume; v6-dpo uses its own run dir
RUN_NAME: str = "chess-coach-v6-dpo"

VOL_MOUNT: str = "/vol"
REMOTE_PAIRS: str = "/data/dpo_pairs_v6.jsonl"
REMOTE_VALID: str = "/data/valid_v6.jsonl"
V4_ADAPTER_REMOTE: str = "/v4_adapter"           # baked-in v4 LoRA (policy + reference init)
ADAPTER_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/adapter"
TRAINER_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/_trainer"

HF_ADAPTER_REPO: str = "khoilamalphaai/chess-coach-32b-v6-dpo"

# --------------------------------------------------------------------------- #
# Hyper-parameters
# --------------------------------------------------------------------------- #
BASE_MODEL: str = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
MAX_SEQ_LEN: int = 2048
MAX_PROMPT_LEN: int = 1600

LEARNING_RATE: float = 1e-5      # low end of 5e-6..2e-5: improve v4 without regressing it
DPO_BETA: float = 0.1            # KL/reference pressure vs the frozen v4 reference
NUM_EPOCHS: float = 1.0          # one short pass
WARMUP_RATIO: float = 0.1
LR_SCHEDULER: str = "cosine"
WEIGHT_DECAY: float = 0.0
OPTIMIZER: str = "adamw_8bit"
PER_DEVICE_BATCH: int = 1
GRAD_ACCUM: int = 8              # eff batch 8 (DPO forms chosen+rejected per example)
LOGGING_STEPS: int = 1
SAVE_STEPS: int = 25
SAVE_TOTAL_LIMIT: int = 8       # keep every checkpoint for per-ckpt selection
SEED: int = 3407

TARGET_PAIRS: int = 840         # ~280 per tier after tier/phase/rank balancing
SMOKE_MAX_PAIRS: int = 24
SMOKE_MAX_STEPS: int = 8

# eval decode (same greedy recipe as the v4 honest-eval generator). Selection only
# needs the recommended MOVE (the "I'd play <MOVE>." opener), so a short cap is
# sufficient for tier-policy match + soundness and keeps the eval cheap.
EVAL_MAX_NEW_TOKENS: int = 48
EVAL_BATCH: int = 32
TIERS: Tuple[str, ...] = ("beginner", "intermediate", "advanced")

ASSISTANT_MARKER: str = "<|im_start|>assistant\n"

# --------------------------------------------------------------------------- #
# Modal infra
# --------------------------------------------------------------------------- #
GPU: str = "A100-80GB"
TIMEOUT_S: int = 5 * 3600
CUDA_TAG: str = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION: str = "3.11"

PIP_PACKAGES: List[str] = [
    "unsloth", "trl", "peft", "bitsandbytes", "transformers", "datasets",
    "accelerate", "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf",
]


# --------------------------------------------------------------------------- #
# Local DPO-pair construction (runs at import under modal.is_local())
# --------------------------------------------------------------------------- #
def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _rank_bucket(rank: Optional[int]) -> str:
    if rank is None:
        return "na"
    if rank <= 0:
        return "best"
    if rank <= 2:
        return "near"
    if rank <= 5:
        return "mid"
    return "tail"


def _swap_move(text: str, canonical_san: str, rejected_san: str) -> str:
    """Replace every whole-token occurrence of ``canonical_san`` with
    ``rejected_san`` (SAN tokens use letters/digits/+#=xO- so we guard those on
    both sides), producing a style-matched rejected response."""
    pat = re.compile(
        r"(?<![A-Za-z0-9+#=x\-])" + re.escape(canonical_san) + r"(?![A-Za-z0-9+#=x\-])"
    )
    return pat.sub(rejected_san, text)


def _resolve_san(fen: str, uci: str, pool: List[dict]) -> Optional[str]:
    for p in pool:
        if p.get("uci") == uci and p.get("san"):
            return p["san"]
    try:
        import chess

        board = chess.Board(fen)
        return board.san(chess.Move.from_uci(uci))
    except Exception:  # noqa: BLE001
        return None


def build_dpo_pairs(train_path: Path) -> List[dict]:
    """Style-matched DPO pairs from the v6 training rows (chosen=canonical prose,
    rejected=same prose with the move swapped to ``dpo_rejected_uci``)."""
    pairs: List[dict] = []
    for row in _iter_jsonl(train_path):
        prov = row["provenance"]
        rej_uci = prov.get("dpo_rejected_uci")
        can_uci = prov.get("canonical_uci")
        if not rej_uci or rej_uci == can_uci:
            continue
        msgs = row["messages"]
        if len(msgs) < 3 or msgs[2]["role"] != "assistant":
            continue
        chosen = msgs[2]["content"]
        can_san = prov.get("canonical_san") or ""
        if not chosen.startswith(f"I'd play {can_san}."):
            continue
        rej_san = _resolve_san(prov["fen"], rej_uci, prov.get("sound_pool", []))
        if not rej_san or rej_san == can_san:
            continue
        rejected = _swap_move(chosen, can_san, rej_san)
        if rejected == chosen or not rejected.startswith(f"I'd play {rej_san}."):
            continue
        pairs.append({
            "system": msgs[0]["content"],
            "user": msgs[1]["content"],
            "chosen": chosen,
            "rejected": rejected,
            "meta": {
                "tier": prov.get("tier"),
                "phase": prov.get("phase"),
                "rank": prov.get("canonical_pool_rank"),
                "rank_bucket": _rank_bucket(prov.get("canonical_pool_rank")),
                "weight": float(prov.get("weight", 1.0)),
                "canonical_uci": can_uci,
                "rejected_uci": rej_uci,
            },
        })
    return pairs


def balance_pairs(pairs: List[dict], target: int, seed: int = SEED) -> List[dict]:
    """Round-robin weighted stratified sampling to ~``target``//3 per tier, and
    within each tier as evenly as availability allows across (phase, rank_bucket),
    preferring high provenance ``weight``. Balances the moat without letting the
    beginner tier (by far the largest) dominate."""
    import random
    from collections import defaultdict

    rng = random.Random(seed)
    per_tier = max(1, target // len(TIERS))
    by_tier: Dict[str, List[dict]] = defaultdict(list)
    for p in pairs:
        by_tier[p["meta"]["tier"]].append(p)

    out: List[dict] = []
    for tier in TIERS:
        plist = by_tier.get(tier, [])
        want = min(per_tier, len(plist))
        strata: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for p in plist:
            strata[(p["meta"]["phase"], p["meta"]["rank_bucket"])].append(p)
        keys = list(strata)
        for k in keys:
            strata[k].sort(key=lambda p: (-p["meta"]["weight"], rng.random()))
        idx = {k: 0 for k in keys}
        picked: List[dict] = []
        # round-robin across strata: take the next-best from each in turn
        while len(picked) < want and any(idx[k] < len(strata[k]) for k in keys):
            rng.shuffle(keys)
            for k in keys:
                if len(picked) >= want:
                    break
                if idx[k] < len(strata[k]):
                    picked.append(strata[k][idx[k]])
                    idx[k] += 1
        out.extend(picked)
    rng.shuffle(out)
    return out


def _distribution(pairs: List[dict]) -> Dict[str, Any]:
    from collections import Counter

    return {
        "n": len(pairs),
        "tier": dict(Counter(p["meta"]["tier"] for p in pairs)),
        "phase": dict(Counter(p["meta"]["phase"] for p in pairs)),
        "rank_bucket": dict(Counter(p["meta"]["rank_bucket"] for p in pairs)),
    }


def _write_pairs(pairs: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")


if modal.is_local():
    _THIS = Path(__file__).resolve().parent
    REPO_ROOT: Optional[Path] = _THIS.parent
    LOCAL_TRAIN: Optional[Path] = REPO_ROOT / "data" / "dataset" / "train_v6.jsonl"
    LOCAL_VALID: Optional[Path] = REPO_ROOT / "data" / "dataset" / "valid_v6.jsonl"
    LOCAL_V4: Optional[Path] = REPO_ROOT / "models" / "adapters" / "chess-coach-v4" / "adapter"
    LOCAL_PAIRS: Optional[Path] = REPO_ROOT / "data" / "dataset" / "_dpo_pairs_v6.jsonl"
    LOCAL_OUT_DIR: Optional[Path] = REPO_ROOT / "models" / "adapters" / RUN_NAME

    _missing = [p for p in (LOCAL_TRAIN, LOCAL_VALID) if not p.exists()]
    _missing += [p for p in [LOCAL_V4 / "adapter_model.safetensors"] if not p.exists()]
    if _missing:
        raise SystemExit(
            "BLOCKED: missing input(s):\n  " + "\n  ".join(str(p) for p in _missing)
            + "\n(need v6 train/valid shards and the local v4 adapter)"
        )
    _all_pairs = build_dpo_pairs(LOCAL_TRAIN)
    _bal = balance_pairs(_all_pairs, TARGET_PAIRS)
    _write_pairs(_bal, LOCAL_PAIRS)
    print(f"[pairs] usable={len(_all_pairs)} -> balanced={len(_bal)}")
    print(f"[pairs] balanced distribution: {json.dumps(_distribution(_bal))}")
else:
    REPO_ROOT = LOCAL_TRAIN = LOCAL_VALID = LOCAL_V4 = LOCAL_PAIRS = LOCAL_OUT_DIR = None


image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .apt_install("git")
    .pip_install(*PIP_PACKAGES)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONUNBUFFERED": "1"})   # real-time logs (block-buffered stdout hides hangs)
)
if modal.is_local():
    image = (
        image
        .add_local_file(LOCAL_PAIRS.as_posix(), REMOTE_PAIRS)
        .add_local_file(LOCAL_VALID.as_posix(), REMOTE_VALID)
        .add_local_dir(LOCAL_V4.as_posix(), V4_ADAPTER_REMOTE)
    )

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


# --------------------------------------------------------------------------- #
# Shared remote helpers
# --------------------------------------------------------------------------- #
def _gpu_banner() -> Optional[str]:
    import torch

    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(f"[gpu] torch={torch.__version__} cuda={torch.cuda.is_available()} device={name}")
    return name


def _load_base_4bit(max_seq_len: int):
    """Load the v4 base 4-bit with Unsloth (so the v4 adapter sits on the SAME
    quantized base it was trained on), returning (model, tokenizer)."""
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=max_seq_len, load_in_4bit=True, dtype=None,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _render_pair(tok, system: str, user: str, chosen: str, rejected: str
                 ) -> Optional[Tuple[str, str, str]]:
    """(prompt, chosen_completion, rejected_completion) rendered EXACTLY as the v6
    SFT target: full-conversation chat template, split at the assistant marker."""
    def full(assistant: str) -> str:
        return tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user},
             {"role": "assistant", "content": assistant}],
            tokenize=False, add_generation_prompt=False,
        )
    fc = full(chosen)
    fr = full(rejected)
    ic = fc.rfind(ASSISTANT_MARKER)
    ir = fr.rfind(ASSISTANT_MARKER)
    if ic < 0 or ir < 0:
        return None
    prompt = fc[: ic + len(ASSISTANT_MARKER)]
    return prompt, fc[ic + len(ASSISTANT_MARKER):], fr[ir + len(ASSISTANT_MARKER):]


def _filter_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    import inspect

    valid = set(inspect.signature(cls.__init__).parameters)
    return {k: v for k, v in kwargs.items() if k in valid}


def _resolve_adapter_dir(path: str) -> str:
    """A saved checkpoint may hold the adapter at the root or nested under the
    adapter name (``default``). Return the dir that actually has an adapter."""
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    for cand in ("default",):
        d = os.path.join(path, cand)
        if os.path.exists(os.path.join(d, "adapter_config.json")):
            return d
    return path


# --------------------------------------------------------------------------- #
# Remote: DPO train (two-adapter, reference = frozen v4)
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={VOL_MOUNT: volume})
def train(smoke: bool = False, beta: float = DPO_BETA, lr: float = LEARNING_RATE,
          epochs: float = NUM_EPOCHS, grad_accum: int = GRAD_ACCUM,
          save_steps: int = SAVE_STEPS) -> dict:
    import unsloth  # noqa: F401  (MUST precede trl/peft/transformers so its patches apply)
    from unsloth import is_bfloat16_supported

    import glob as _glob

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import TrainerCallback
    from trl import DPOConfig, DPOTrainer

    gpu_name = _gpu_banner()

    model, tok = _load_base_4bit(MAX_SEQ_LEN)
    print(f"[peft] policy(default)=v4 + reference(frozen)=v4 from {V4_ADAPTER_REMOTE}")
    model = PeftModel.from_pretrained(model, V4_ADAPTER_REMOTE, is_trainable=True,
                                      adapter_name="default")
    model.load_adapter(V4_ADAPTER_REMOTE, adapter_name="reference")
    model.set_adapter("default")
    try:
        model.enable_input_require_grads()
    except Exception as exc:  # noqa: BLE001
        print(f"[peft] enable_input_require_grads: {exc}")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[peft] trainable params={n_train:,}")

    rows = [json.loads(x) for x in open(REMOTE_PAIRS, encoding="utf-8") if x.strip()]
    if smoke:
        rows = rows[:SMOKE_MAX_PAIRS]
    recs: List[dict] = []
    for r in rows:
        rp = _render_pair(tok, r["system"], r["user"], r["chosen"], r["rejected"])
        if rp is None:
            continue
        prompt, chosen, rejected = rp
        recs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    if not recs:
        raise RuntimeError("no DPO pairs rendered")
    dataset = Dataset.from_list(recs)
    print(f"[data] dpo_pairs={len(recs)} (smoke={smoke})")
    print("[data] sample prompt tail:\n" + recs[0]["prompt"][-320:])
    print("[data] sample chosen:  " + recs[0]["chosen"][:120].replace("\n", " "))
    print("[data] sample rejected:" + recs[0]["rejected"][:120].replace("\n", " "))

    max_steps = SMOKE_MAX_STEPS if smoke else -1
    eff_save = 4 if smoke else save_steps          # smoke: force a checkpoint to validate layout
    eff_accum = 1 if smoke else grad_accum
    cfg_kwargs = dict(
        output_dir=TRAINER_DIR,
        beta=beta,
        model_adapter_name="default", ref_adapter_name="reference",
        max_length=MAX_SEQ_LEN, max_prompt_length=MAX_PROMPT_LEN,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=eff_accum,
        warmup_ratio=WARMUP_RATIO, num_train_epochs=(1.0 if smoke else epochs),
        max_steps=max_steps, learning_rate=lr, logging_steps=LOGGING_STEPS,
        optim=OPTIMIZER, weight_decay=WEIGHT_DECAY, lr_scheduler_type=LR_SCHEDULER,
        seed=SEED, save_strategy="steps", save_steps=eff_save,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0, dataset_num_proc=1,   # avoid fork deadlock (old kernel)
        remove_unused_columns=False, report_to="none",
    )
    dpo_config = DPOConfig(**_filter_kwargs(DPOConfig, cfg_kwargs))

    trainer_kwargs = dict(model=model, ref_model=None, args=dpo_config,
                          train_dataset=dataset, tokenizer=tok, processing_class=tok)
    trainer_kwargs = _filter_kwargs(DPOTrainer, trainer_kwargs)
    trainer = DPOTrainer(**trainer_kwargs)

    class _VolCommit(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
            try:
                volume.commit()
                print(f"[ckpt] committed at step {state.global_step}")
            except Exception as exc:  # noqa: BLE001
                print(f"[ckpt] commit failed: {exc}")

    trainer.add_callback(_VolCommit())

    resume = None
    if not smoke:
        volume.reload()
        ckpts = _glob.glob(f"{TRAINER_DIR}/checkpoint-*")
        if ckpts:
            resume = max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))
            print(f"[resume] {resume}")
    print(f"[train] beta={beta} lr={lr} epochs={epochs} eff_batch={PER_DEVICE_BATCH*eff_accum} "
          f"max_steps={max_steps} save_steps={eff_save}")
    try:
        out = trainer.train(resume_from_checkpoint=resume)
    except Exception as exc:  # noqa: BLE001
        if resume is None:
            raise
        print(f"[resume] failed ({exc}); fresh restart")
        out = trainer.train()

    hist = trainer.state.log_history
    losses = [d["loss"] for d in hist if "loss" in d]
    accs = [d["rewards/accuracies"] for d in hist if "rewards/accuracies" in d]
    margins = [d.get("rewards/margins") for d in hist if "rewards/margins" in d]
    print(f"[train] steps={len(losses)} first_loss={losses[0] if losses else None} "
          f"last_loss={losses[-1] if losses else None} last_acc={accs[-1] if accs else None}")

    print(f"[save] default adapter -> {ADAPTER_DIR}")
    model.save_pretrained(ADAPTER_DIR, selected_adapters=["default"])
    tok.save_pretrained(ADAPTER_DIR)
    volume.commit()

    volume.reload()
    ckpts = sorted(_glob.glob(f"{TRAINER_DIR}/checkpoint-*"),
                   key=lambda p: int(p.rsplit("-", 1)[-1]))
    return {
        "gpu": gpu_name, "smoke": smoke, "beta": beta, "lr": lr,
        "trainable_params": n_train, "n_pairs": len(recs),
        "steps": len(losses), "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "last_acc": accs[-1] if accs else None,
        "last_margin": margins[-1] if margins else None,
        "adapter_dir": ADAPTER_DIR, "checkpoints": ckpts, "run_name": RUN_NAME,
        "metrics": getattr(out, "metrics", None),
    }


# --------------------------------------------------------------------------- #
# Remote: generate valid_v6 completions for a set of adapters (base loaded once)
# --------------------------------------------------------------------------- #
def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.replace("<think>", "").replace("</think>", "").strip()


def _clean_lead(text: str) -> str:
    t = text.strip()
    if t.startswith("I'd play") or t.startswith("I\u2019d play"):
        return t
    idx = t.find("I'd play")
    if idx < 0:
        idx = t.find("I\u2019d play")
    return t[idx:].strip() if 0 < idx <= 160 else t


@app.function(image=image, gpu=GPU, timeout=3 * 3600, volumes={VOL_MOUNT: volume})
def eval_valid(specs: Dict[str, str], limit: int = 0) -> Dict[str, List[dict]]:
    """For each {name: adapter_dir}, greedy-generate over valid_v6 (same decode as
    the v4 honest eval) and return {name: [{"i", "output"}]} for LOCAL scoring."""
    import unsloth  # noqa: F401  (import before peft/transformers so its patches apply)
    from unsloth import FastLanguageModel

    import time

    import torch
    from peft import PeftModel

    volume.reload()
    rows = [json.loads(x) for x in open(REMOTE_VALID, encoding="utf-8") if x.strip()]
    if limit:
        rows = rows[:limit]
    prompts = [(r["messages"][0]["content"], r["messages"][1]["content"]) for r in rows]
    print(f"[eval] valid rows={len(rows)} adapters={list(specs)}")

    model, tok = _load_base_4bit(3072)
    names = list(specs)
    dirs = {nm: _resolve_adapter_dir(specs[nm]) for nm in names}
    model = PeftModel.from_pretrained(model, dirs[names[0]], adapter_name=names[0])
    for nm in names[1:]:
        model.load_adapter(dirs[nm], adapter_name=nm)
    FastLanguageModel.for_inference(model)   # once (avoid per-adapter resets)

    results: Dict[str, List[dict]] = {}
    for nm in names:
        model.set_adapter(nm)
        outs: List[dict] = []
        t0 = time.time()
        for i in range(0, len(prompts), EVAL_BATCH):
            batch = prompts[i:i + EVAL_BATCH]
            texts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": s}, {"role": "user", "content": u}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                for s, u in batch
            ]
            tok.padding_side = "left"        # decoder-only: MUST left-pad for correct batched gen
            tok.truncation_side = "left"
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=3072).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=EVAL_MAX_NEW_TOKENS,
                                     do_sample=False, repetition_penalty=1.15,
                                     no_repeat_ngram_size=4, pad_token_id=tok.pad_token_id)
            for j, (g, inp) in enumerate(zip(gen, enc["input_ids"])):
                raw = _strip_think(tok.decode(g[inp.shape[0]:], skip_special_tokens=True))
                outs.append({"i": i + j, "output": _clean_lead(raw)})
        print(f"[eval] {nm}: {len(outs)} gens in {time.time()-t0:.0f}s")
        results[nm] = outs
    return results


# --------------------------------------------------------------------------- #
# Remote: promote a chosen checkpoint's adapter to the run's adapter dir
# --------------------------------------------------------------------------- #
@app.function(image=image, timeout=1800, volumes={VOL_MOUNT: volume})
def promote(ckpt_dir: str) -> dict:
    volume.reload()
    src = ckpt_dir
    # a trainer checkpoint may nest the adapter one level down
    if not os.path.exists(os.path.join(src, "adapter_model.safetensors")):
        for cand in (os.path.join(ckpt_dir, "default"), ckpt_dir):
            if os.path.exists(os.path.join(cand, "adapter_model.safetensors")):
                src = cand
                break
    if not os.path.exists(os.path.join(src, "adapter_model.safetensors")):
        raise RuntimeError(f"no adapter_model.safetensors under {ckpt_dir}")
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    for fn in os.listdir(src):
        if fn.startswith("checkpoint-") or fn in ("optimizer.pt", "scheduler.pt"):
            continue
        s = os.path.join(src, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(ADAPTER_DIR, fn))
    volume.commit()
    files = sorted(os.listdir(ADAPTER_DIR))
    print(f"[promote] {src} -> {ADAPTER_DIR}: {files}")
    return {"src": src, "adapter_dir": ADAPTER_DIR, "files": files}


# --------------------------------------------------------------------------- #
# Local scoring (canonical extractor) + orchestration
# --------------------------------------------------------------------------- #
def _score(outputs: List[dict], valid_rows: List[dict]) -> Dict[str, Any]:
    """tier-policy exact match (mean over tiers) vs canonical_uci; tiebreak soundness."""
    from statistics import mean

    from src.eval.evaluate import extract_recommended_move

    by_tier: Dict[str, List[int]] = {t: [0, 0] for t in TIERS}
    sound = [0, 0]
    named = 0
    for o in outputs:
        row = valid_rows[o["i"]]
        prov = row["provenance"]
        tier = prov.get("tier")
        fen = prov["fen"]
        student_uci = (prov.get("student") or {}).get("uci") or ""
        _san, uci = extract_recommended_move(o["output"], fen, student_uci)
        if tier in by_tier:
            by_tier[tier][1] += 1
            if uci and uci == prov.get("canonical_uci"):
                by_tier[tier][0] += 1
        if uci:
            named += 1
        pool = {p.get("uci") for p in prov.get("sound_pool", [])}
        sound[1] += 1
        if uci and uci in pool:
            sound[0] += 1
    per_tier = {t: (by_tier[t][0] / by_tier[t][1]) for t in TIERS if by_tier[t][1]}
    return {
        "tier_policy_match": mean(per_tier.values()) if per_tier else 0.0,
        "per_tier": {t: round(v, 4) for t, v in per_tier.items()},
        "move_sound": sound[0] / sound[1] if sound[1] else 0.0,
        "named_rate": named / len(outputs) if outputs else 0.0,
        "n": len(outputs),
    }


def _volume_get(remote_path: str, local_parent: Path) -> None:
    local_parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force",
                    VOLUME_NAME, remote_path, str(local_parent)], check=True)


def _push_hf(local_dir: Path, private: bool = True) -> str:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        raise SystemExit("BLOCKED: no HF_TOKEN in .env")
    from huggingface_hub import HfApi

    api = HfApi(token=tok)
    api.create_repo(HF_ADAPTER_REPO, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(folder_path=str(local_dir), repo_id=HF_ADAPTER_REPO, repo_type="model")
    return f"https://huggingface.co/{HF_ADAPTER_REPO}"


@app.local_entrypoint()
def main(smoke: bool = False, beta: float = DPO_BETA, lr: float = LEARNING_RATE,
         epochs: float = NUM_EPOCHS, grad_accum: int = GRAD_ACCUM,
         save_steps: int = SAVE_STEPS, skip_train: bool = False,
         skip_eval: bool = False, skip_push: bool = False, eval_limit: int = 0) -> None:
    print(f"=== {APP_NAME} ({'SMOKE' if smoke else 'FULL'}) ===")
    print(f"base={BASE_MODEL}  pairs={LOCAL_PAIRS}  v4_adapter={LOCAL_V4}")

    if not skip_train:
        res = train.remote(smoke=smoke, beta=beta, lr=lr, epochs=epochs,
                           grad_accum=grad_accum, save_steps=save_steps)
        print("\n=== train() ===\n" + json.dumps(res, indent=2, default=str))
        if smoke:
            return

    if skip_eval:
        print("[main] --skip-eval: stopping after train (run --skip-train to eval+select+push).")
        return

    # Evaluate v4 (baseline) + every DPO checkpoint on valid_v6, then select.
    specs: Dict[str, str] = {"v4": V4_ADAPTER_REMOTE}
    # discover checkpoints from the volume via a tiny remote listing
    ckpts = _list_checkpoints.remote()
    for c in ckpts:
        specs[os.path.basename(c)] = c   # final adapter == last checkpoint, so no separate entry
    print(f"[main] evaluating adapters: {list(specs)}")

    outputs = eval_valid.remote(specs, limit=eval_limit)
    valid_rows = [json.loads(x) for x in open(LOCAL_VALID, encoding="utf-8") if x.strip()]
    if eval_limit:
        valid_rows = valid_rows[:eval_limit]

    scores = {nm: _score(outs, valid_rows) for nm, outs in outputs.items()}
    print("\n=== valid_v6 tier-policy match (dev-set selection) ===")
    for nm in scores:
        s = scores[nm]
        print(f"  {nm:26} match={s['tier_policy_match']:.4f} sound={s['move_sound']:.4f} "
              f"named={s['named_rate']:.3f} per_tier={s['per_tier']}")

    v4 = scores.get("v4", {})
    cand = {nm: s for nm, s in scores.items() if nm != "v4"}
    best = max(cand, key=lambda nm: (cand[nm]["tier_policy_match"], cand[nm]["move_sound"]))
    print(f"\n[select] best={best} match={cand[best]['tier_policy_match']:.4f} "
          f"(v4={v4.get('tier_policy_match', 0):.4f}, "
          f"delta={cand[best]['tier_policy_match']-v4.get('tier_policy_match',0):+.4f})")

    if not skip_push:
        best_dir = specs[best]
        if best_dir != ADAPTER_DIR:
            print(promote.remote(best_dir))
        LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp = LOCAL_OUT_DIR.parent / "_v6dpo_dl"
        shutil.rmtree(tmp, ignore_errors=True)
        _volume_get(f"/{RUN_NAME}/adapter", tmp)
        got = tmp / "adapter"
        src = got if got.exists() else tmp
        shutil.rmtree(LOCAL_OUT_DIR, ignore_errors=True)
        shutil.move(str(src), str(LOCAL_OUT_DIR))
        shutil.rmtree(tmp, ignore_errors=True)
        url = _push_hf(LOCAL_OUT_DIR)
        print(f"[push] adapter -> {url}  (volume: {VOLUME_NAME}:/{RUN_NAME}/adapter)")

    print("\n=== SUMMARY ===\n" + json.dumps(
        {"scores": scores, "selected": best,
         "delta_vs_v4": cand[best]["tier_policy_match"] - v4.get("tier_policy_match", 0)},
        indent=2))


@app.function(image=image, timeout=600, volumes={VOL_MOUNT: volume})
def _list_checkpoints() -> List[str]:
    import glob as _glob

    volume.reload()
    return sorted(_glob.glob(f"{TRAINER_DIR}/checkpoint-*"),
                  key=lambda p: int(p.rsplit("-", 1)[-1]))
