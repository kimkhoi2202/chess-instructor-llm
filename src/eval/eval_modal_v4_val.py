#!/usr/bin/env python3
"""Complete the v4 (Qwen3-32B QLoRA) coach gens on the HONEST-eval VAL slice.

The full-803 v4 eval-gen (``eval_modal_v4.py``) stopped early, so only 47 of the
120 honest-eval VAL positions have v4 coaching. This is a thin, byte-for-byte
copy of that generator (same base 4-bit + v4 adapter, same greedy + rep-penalty
decoding, same ``_clean_lead``) that runs ONLY over the missing VAL prompts and
writes them to a DISTINCT volume file (``ours_v4_val_gen.jsonl``) so v4's
canonical ``ours_v4_gen.jsonl`` is never touched.

Prompts come from ``data/benchmark_v4/prompts_v4_val_missing.jsonl`` (built by the
honest-v4 driver: the 219 val prompts with no v4 gen yet). Output rows use the
identical benchmark generation schema, downloaded to
``data/benchmark_v4/gen/ours_v4_val.jsonl`` and then merged with the 141 already
generated rows into the honest-eval ``ours_v4`` contender.

Commands (adapter lives on the ``chess-instructor`` volume; image is warm there)::

    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    modal run --detach src/eval/eval_modal_v4_val.py           # spawn on Modal
    modal run src/eval/eval_modal_v4_val.py --block            # generate + wait + download
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import modal

APP_NAME = "chess-coach-eval-v4-val"
VOLUME_NAME = "chess-coach-lora"
RUN_NAME = "chess-coach-v4"
VOL_MOUNT = "/vol"
ADAPTER_DIR = f"{VOL_MOUNT}/{RUN_NAME}/adapter"
REMOTE_PROMPTS = "/data/prompts_v4_val_missing.jsonl"
REMOTE_OUT = f"{VOL_MOUNT}/{RUN_NAME}/ours_v4_val_gen.jsonl"

BASE_MODEL = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
GPU = "A100-80GB"
TIMEOUT_S = 2 * 3600
CUDA_TAG = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION = "3.11"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 32

if modal.is_local():
    REPO_ROOT: Optional[Path] = Path(__file__).resolve().parents[2]
    LOCAL_PROMPTS: Optional[Path] = REPO_ROOT / "data" / "benchmark_v4" / "prompts_v4_val_missing.jsonl"
    LOCAL_OUT: Optional[Path] = REPO_ROOT / "data" / "benchmark_v4" / "gen" / "ours_v4_val.jsonl"
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
        raise SystemExit(f"BLOCKED: {LOCAL_PROMPTS} missing (build the val-missing prompt subset first).")
    image = image.add_local_file(LOCAL_PROMPTS.as_posix(), REMOTE_PROMPTS)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


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


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={VOL_MOUNT: volume})
def generate(limit: int = 0) -> dict:
    import time

    import torch
    from unsloth import FastLanguageModel

    volume.reload()
    print(f"[load] Unsloth base 4-bit + adapter={ADAPTER_DIR}")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR, max_seq_length=3072, load_in_4bit=True, dtype=None,
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
    print(f"[gen] {len(todo)} pending of {len(rows)} ({len(done)} done)")

    from datetime import datetime, timezone
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
                    "scenario_id": r["id"], "model": "ours_v4", "condition": "grounded",
                    "tier": r["tier"], "phase": r["phase"], "severity": r["severity"],
                    "pos_id": r["pos_id"], "output": cleaned, "output_raw": raw,
                    "lead_cleaned": cleaned != raw,
                    "prompt_tokens": int(inp.shape[0]), "completion_tokens": 0,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")
                written += 1
            out.flush()
            dt = time.time() - t0
            n = i + len(batch)
            print(f"  {n}/{len(todo)} ({dt/max(1,n):.2f}s/it, "
                  f"eta {dt/max(1,n)*(len(todo)-n)/60:.0f}m, lead_cleaned={n_lead_cleaned})")
            volume.commit()
    volume.commit()
    return {"written": written, "total": len(rows), "lead_cleaned": n_lead_cleaned,
            "secs": round(time.time() - t0, 1), "out": REMOTE_OUT}


@app.local_entrypoint()
def main(limit: int = 0, block: bool = False) -> None:
    call = generate.spawn(limit=limit)
    print(f"SPAWNED generate call_id={call.object_id} — running detached on Modal; "
          f"poll {VOL_MOUNT}/{RUN_NAME}/ours_v4_val_gen.jsonl on the Volume for completion.")
    if block:
        res = call.get()
        print(json.dumps(res, indent=2, default=str))
        LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = LOCAL_OUT.parent / "_ours_v4_val_dl"
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force",
                        VOLUME_NAME, f"/{RUN_NAME}/ours_v4_val_gen.jsonl", str(tmp)], check=True)
        got = tmp / "ours_v4_val_gen.jsonl"
        if got.exists():
            shutil.move(str(got), str(LOCAL_OUT))
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"DONE -> {LOCAL_OUT}")
