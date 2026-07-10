#!/usr/bin/env python3
"""Track B of the ADVERSARIAL eval: run the untuned Qwen3-32B **base** and the
**v4** QLoRA over the adversarial prompts, RAW (ungated), greedy — on Modal.

This is the base-vs-v4 contrast for the robustness scorecard. It deliberately
does NOT run the verify-and-regenerate gate: it measures what the *weights* do
when attacked (including prompt-level injection, which the deployed API has no
channel for). The prompts (``data/adversarial/prompts_modal.jsonl``, built by
``scripts/adversarial_eval.py build`` with real local Stockfish + Maia) already
contain the exact ``system`` + ``user`` the live endpoint would render, plus, for
the injection cases, a ``variant="inj"`` row whose user message has the injection
appended.

Mirrors the proven ``src/eval/eval_modal_v4.py`` (Unsloth 4-bit + greedy) so the
generation matches the showcase/eval. The only new axis is ``--which`` selecting
the model source:

* ``base`` -> ``unsloth/Qwen3-32B-unsloth-bnb-4bit`` (no adapter)
* ``v4``   -> the same base + the v4 LoRA (Volume copy preferred, HF fallback)

Commands
--------
    python -m scripts.adversarial_eval build            # writes the prompts file
    MODAL_PROFILE=chess-instructor-3 modal run src/eval/adversarial_modal.py --block
    #   (runs BOTH base and v4, downloads gen/raw_base.jsonl + gen/raw_v4.jsonl)
    MODAL_PROFILE=chess-instructor-3 modal run src/eval/adversarial_modal.py --which base --block
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import modal

APP_NAME = "chess-coach-adversarial"
VOLUME_NAME = "chess-coach-lora"
RUN_NAME = "chess-coach-v4"
VOL_MOUNT = "/vol"
ADAPTER_DIR = f"{VOL_MOUNT}/{RUN_NAME}/adapter"
REMOTE_PROMPTS = "/data/prompts_adversarial.jsonl"

BASE_MODEL = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
HF_ADAPTER_REPO = "khoilamalphaai/chess-coach-modal-backup"
HF_ADAPTER_SUBFOLDER = "v4-lora-qwen3-32b"

GPU = "A100-80GB"
TIMEOUT_S = 2 * 3600
CUDA_TAG = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION = "3.11"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 16

if modal.is_local():
    REPO_ROOT: Optional[Path] = Path(__file__).resolve().parents[2]
    LOCAL_PROMPTS: Optional[Path] = REPO_ROOT / "data" / "adversarial" / "prompts_modal.jsonl"
    LOCAL_GEN_DIR: Optional[Path] = REPO_ROOT / "data" / "adversarial" / "gen"
else:
    REPO_ROOT = LOCAL_PROMPTS = LOCAL_GEN_DIR = None

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .apt_install("git")
    .pip_install("unsloth", "trl", "peft", "bitsandbytes", "transformers", "datasets",
                 "accelerate", "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)

if modal.is_local():
    if not LOCAL_PROMPTS.exists():
        raise SystemExit(
            f"BLOCKED: {LOCAL_PROMPTS} missing. Build it first:\n"
            "  python scripts/adversarial_eval.py build"
        )
    image = image.add_local_file(LOCAL_PROMPTS.as_posix(), REMOTE_PROMPTS)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)

# HF token for the base pull + adapter fallback (Modal secret created before deploy).
try:
    hf_secret = modal.Secret.from_name("chess-hf")
    SECRETS = [hf_secret]
except Exception:  # noqa: BLE001 - the base is public; secret only needed for the private adapter
    SECRETS = []


def _strip_think(text: str) -> str:
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.replace("<think>", "").replace("</think>", "").strip()


def _resolve_v4_adapter() -> str:
    """Local path to the v4 adapter — Volume copy preferred, HF snapshot fallback."""
    import os

    cfg = os.path.join(ADAPTER_DIR, "adapter_config.json")
    wts = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
    if os.path.isfile(cfg) and os.path.isfile(wts):
        print(f"[adv] v4 adapter from Volume: {ADAPTER_DIR}")
        return ADAPTER_DIR
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    local = snapshot_download(HF_ADAPTER_REPO, allow_patterns=[f"{HF_ADAPTER_SUBFOLDER}/*"], token=token)
    resolved = os.path.join(local, HF_ADAPTER_SUBFOLDER)
    print(f"[adv] v4 adapter from HF cache: {resolved}")
    return resolved


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={VOL_MOUNT: volume},
              secrets=SECRETS,
              retries=modal.Retries(max_retries=6, initial_delay=5.0, backoff_coefficient=1.0))
def generate(which: str) -> dict:
    """Generate RAW greedy coaching for every adversarial prompt with model ``which``."""
    import time

    import torch
    from unsloth import FastLanguageModel

    volume.reload()
    if which == "base":
        source = BASE_MODEL
    elif which == "v4":
        source = _resolve_v4_adapter()
    else:
        raise ValueError(f"which must be 'base' or 'v4', got {which!r}")

    print(f"[adv:{which}] loading {source}")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=source, max_seq_length=3072, load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    rows = [json.loads(l) for l in open(REMOTE_PROMPTS, encoding="utf-8") if l.strip()]
    remote_out = f"{VOL_MOUNT}/{RUN_NAME}/adversarial_raw_{which}.jsonl"

    done: set = set()
    if Path(remote_out).exists():
        for l in open(remote_out, encoding="utf-8"):
            if l.strip():
                try:
                    done.add(json.loads(l)["id"])
                except Exception:  # noqa: BLE001
                    pass
    todo = [r for r in rows if r["id"] not in done]
    print(f"[adv:{which}] {len(todo)} pending of {len(rows)} ({len(done)} done)")

    t0 = time.time()
    written = 0
    with open(remote_out, "a", encoding="utf-8") as out:
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
                out.write(json.dumps({
                    "id": r["id"], "case_id": r.get("case_id"), "category": r.get("category"),
                    "tier": r.get("tier"), "variant": r.get("variant"), "model": which,
                    "output": raw,
                }, ensure_ascii=False) + "\n")
                written += 1
            out.flush()
            volume.commit()
            dt = time.time() - t0
            n = i + len(batch)
            print(f"  [{which}] {n}/{len(todo)} ({dt/max(1,n):.2f}s/it)")
    volume.commit()
    return {"which": which, "written": written, "total": len(rows),
            "secs": round(time.time() - t0, 1), "out": remote_out}


def _download(which: str) -> None:
    LOCAL_GEN_DIR.mkdir(parents=True, exist_ok=True)
    dest = LOCAL_GEN_DIR / f"raw_{which}.jsonl"
    if dest.exists():
        dest.unlink()
    # `modal volume get VOL remote_file DEST` treats a non-existent DEST as the
    # target FILENAME, so download straight to the final path.
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force",
                    VOLUME_NAME, f"/{RUN_NAME}/adversarial_raw_{which}.jsonl", str(dest)], check=True)
    if dest.exists():
        print(f"DONE {which} -> {dest} ({sum(1 for _ in open(dest))} rows)")
    else:
        print(f"WARN: {which} output not found on volume")


@app.local_entrypoint()
def main(which: str = "", block: bool = False) -> None:
    # Run SEQUENTIALLY (base, then v4) so only one A100 is live at a time — cheaper
    # and gentler on the workspace GPU quota than two parallel cold starts.
    targets = [which] if which else ["base", "v4"]
    for w in targets:
        call = generate.spawn(w)
        print(f"SPAWNED {w} call_id={call.object_id}")
        if block:
            res = call.get()
            print(json.dumps(res, indent=2, default=str))
            _download(w)
