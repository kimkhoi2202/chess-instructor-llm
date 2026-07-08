#!/usr/bin/env python3
"""Finish the v4 (Qwen3-32B QLoRA) VAL gens on a FRESH Modal workspace, adapter from HF.

Resumes the honest-eval VAL slice on ``chess-instructor-3`` (a fresh workspace). It
loads the FINAL v4 adapter from Hugging Face
(``khoilamalphaai/chess-coach-modal-backup/v4-lora-qwen3-32b`` — private, via the
``chess-hf`` Modal secret) on top of the same ``unsloth/Qwen3-32B-unsloth-bnb-4bit``
base, and generates ONLY the still-missing VAL prompts with **byte-identical
decoding** to the 100 positions already produced (greedy + repetition_penalty 1.15 +
no_repeat_ngram 4 + ``_clean_lead``) — so the completed 120-position set stays
homogeneous.

Runs DETACHED and writes per-scenario to a Modal Volume (created on CI3), so the job
survives a client disconnect and is resumable. Download the result afterwards.

    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    MODAL_PROFILE=chess-instructor-3 modal run --detach src/eval/eval_modal_v4_hf.py
    # then, when the Volume file is complete:
    MODAL_PROFILE=chess-instructor-3 modal volume get --force chess-coach-lora \
        /ours_v4_val_remaining.jsonl data/benchmark_v4/gen/ours_v4_val_remaining.jsonl
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import modal

APP_NAME = "chess-coach-eval-v4-hf"
VOLUME_NAME = "chess-coach-lora"
VOL_MOUNT = "/vol"
HF_ADAPTER_REPO = "khoilamalphaai/chess-coach-modal-backup"
HF_ADAPTER_SUBFOLDER = "v4-lora-qwen3-32b"
REMOTE_PROMPTS = "/data/prompts_v4_val_still_missing.jsonl"
REMOTE_OUT = f"{VOL_MOUNT}/ours_v4_val_remaining.jsonl"

GPU = "A100-80GB"
TIMEOUT_S = 2 * 3600
CUDA_TAG = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION = "3.11"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 32

if modal.is_local():
    REPO_ROOT: Optional[Path] = Path(__file__).resolve().parents[2]
    LOCAL_PROMPTS: Optional[Path] = REPO_ROOT / "data" / "benchmark_v4" / "prompts_v4_val_still_missing.jsonl"
    LOCAL_OUT: Optional[Path] = REPO_ROOT / "data" / "benchmark_v4" / "gen" / "ours_v4_val_remaining.jsonl"
else:
    REPO_ROOT = LOCAL_PROMPTS = LOCAL_OUT = None

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .apt_install("git")
    .pip_install("unsloth", "trl", "peft", "bitsandbytes", "transformers", "datasets",
                 "accelerate", "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)
if modal.is_local():
    if not LOCAL_PROMPTS.exists():
        raise SystemExit(f"BLOCKED: {LOCAL_PROMPTS} missing (build the still-missing prompt subset first).")
    image = image.add_local_file(LOCAL_PROMPTS.as_posix(), REMOTE_PROMPTS)

hf_secret = modal.Secret.from_name("chess-hf")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _strip_think(text: str) -> str:
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.replace("<think>", "").replace("</think>", "").strip()


def _clean_lead(text: str) -> str:
    t = text.strip()
    if t.startswith("I'd play") or t.startswith("I\u2019d play"):
        return t
    idx = t.find("I'd play")
    if idx < 0:
        idx = t.find("I\u2019d play")
    if 0 < idx <= 160:
        return t[idx:].strip()
    return t


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, secrets=[hf_secret],
              volumes={VOL_MOUNT: volume})
def generate(limit: int = 0) -> dict:
    import os
    import time
    from datetime import datetime, timezone

    import torch
    from huggingface_hub import snapshot_download
    from unsloth import FastLanguageModel

    volume.reload()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[load] pulling adapter {HF_ADAPTER_REPO}/{HF_ADAPTER_SUBFOLDER} from HF ...", flush=True)
    snap = snapshot_download(HF_ADAPTER_REPO, allow_patterns=[f"{HF_ADAPTER_SUBFOLDER}/*"], token=token)
    adapter_dir = os.path.join(snap, HF_ADAPTER_SUBFOLDER)
    print(f"[load] Unsloth base 4-bit + adapter={adapter_dir}", flush=True)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=adapter_dir, max_seq_length=3072, load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    rows = [json.loads(l) for l in open(REMOTE_PROMPTS, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    done: set = set()
    if Path(REMOTE_OUT).exists():
        for l in open(REMOTE_OUT, encoding="utf-8"):
            if l.strip():
                try:
                    done.add(json.loads(l)["scenario_id"])
                except Exception:  # noqa: BLE001
                    pass
    todo = [r for r in rows if r["id"] not in done]
    print(f"[gen] {len(todo)} pending of {len(rows)} ({len(done)} done)", flush=True)

    t0 = time.time()
    written = n_lead_cleaned = 0
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
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to("cuda")
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
                    "scenario_id": r["id"], "model": "ours_v4", "condition": "grounded",
                    "tier": r["tier"], "phase": r["phase"], "severity": r["severity"],
                    "pos_id": r["pos_id"], "output": cleaned, "output_raw": raw,
                    "lead_cleaned": cleaned != raw,
                    "prompt_tokens": int(inp.shape[0]), "completion_tokens": 0,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")
                written += 1
            out.flush()
            volume.commit()
            print(f"  {i + len(batch)}/{len(todo)} ({(time.time()-t0)/max(1,i+len(batch)):.2f}s/it, "
                  f"lead_cleaned={n_lead_cleaned})", flush=True)
    volume.commit()
    print(f"DONE {written} rows in {time.time()-t0:.0f}s", flush=True)
    return {"written": written, "total": len(rows), "out": REMOTE_OUT}


@app.local_entrypoint()
def main(limit: int = 0, block: bool = False) -> None:
    call = generate.spawn(limit=limit)
    print(f"SPAWNED generate call_id={call.object_id} — detached on Modal; "
          f"poll {REMOTE_OUT} on the Volume for completion.")
    if block:
        print(json.dumps(call.get(), indent=2, default=str))
        LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = LOCAL_OUT.parent / "_v4_rem_dl"
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force",
                        VOLUME_NAME, "/ours_v4_val_remaining.jsonl", str(tmp)], check=True)
        got = tmp / "ours_v4_val_remaining.jsonl"
        if got.exists():
            shutil.move(str(got), str(LOCAL_OUT))
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"DONE -> {LOCAL_OUT}")
