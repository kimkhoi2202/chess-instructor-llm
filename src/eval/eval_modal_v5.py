#!/usr/bin/env python3
"""Generate **v5** coaching for the 120-position VAL slice ON MODAL (base 4-bit + v5 LoRA).

Identical model/prompt/decoding to ``eval_modal_v4.py`` (so ``ours_v5`` is
apples-to-apples with the reused ``ours_v4`` / 4B / frontier val gens), but points
at the **v5** adapter (``chess-coach-v5/adapter`` on the shared Volume) and the
VAL-only prompts (``data/benchmark_v5/prompts_v5_val.jsonl``, 360 scenarios), so
the eval-gen is cheap (~$2-3). Writes ``ours_v5`` rows to the Volume; download to
``data/benchmark_v5/gen/ours_v5.jsonl`` and slice into the honest_v5 field.

Commands (scrub tokens + pin the workspace)::

    python -m scripts.build_v5_val_prompts
    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; export MODAL_PROFILE=chess-instructor-4
    /Users/khoilam/.venvs/mlx/bin/modal run --detach src/eval/eval_modal_v5.py         # spawn
    /Users/khoilam/.venvs/mlx/bin/modal run src/eval/eval_modal_v5.py --block          # gen + wait + download
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import modal

APP_NAME = "chess-coach-eval-v5"
VOLUME_NAME = "chess-coach-lora"
RUN_NAME = "chess-coach-v5"
VOL_MOUNT = "/vol"
ADAPTER_DIR = f"{VOL_MOUNT}/{RUN_NAME}/adapter"
REMOTE_PROMPTS = "/data/prompts_v5_val.jsonl"
REMOTE_OUT = f"{VOL_MOUNT}/{RUN_NAME}/ours_v5_val_gen.jsonl"

BASE_MODEL = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
GPU = "A100-80GB"
TIMEOUT_S = 3 * 3600
CUDA_TAG = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION = "3.11"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 32

if modal.is_local():
    REPO_ROOT: Optional[Path] = Path(__file__).resolve().parents[2]
    LOCAL_PROMPTS: Optional[Path] = REPO_ROOT / "data" / "benchmark_v5" / "prompts_v5_val.jsonl"
    LOCAL_OUT: Optional[Path] = REPO_ROOT / "data" / "benchmark_v5" / "gen" / "ours_v5.jsonl"
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
        raise SystemExit(
            f"BLOCKED: {LOCAL_PROMPTS} missing. Build it first:\n"
            "  /Users/khoilam/.venvs/mlx/bin/python -m scripts.build_v5_val_prompts"
        )
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


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={VOL_MOUNT: volume},
              retries=modal.Retries(max_retries=10, initial_delay=5.0, backoff_coefficient=1.0))
def generate(limit: int = 0) -> dict:
    import time
    from datetime import datetime, timezone

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
                    "scenario_id": r["id"], "model": "ours_v5", "condition": "grounded",
                    "tier": r["tier"], "phase": r["phase"], "severity": r["severity"],
                    "pos_id": r["pos_id"], "output": cleaned, "output_raw": raw,
                    "lead_cleaned": cleaned != raw,
                    "prompt_tokens": int(inp.shape[0]), "completion_tokens": 0,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")
                written += 1
            out.flush()
            volume.commit()
            if (i // BATCH_SIZE) % 5 == 0 or i + BATCH_SIZE >= len(todo):
                dt = time.time() - t0
                n = i + len(batch)
                print(f"  {n}/{len(todo)} ({dt/max(1,n):.2f}s/it, "
                      f"eta {dt/max(1,n)*(len(todo)-n)/60:.0f}m, lead_cleaned={n_lead_cleaned})")
    volume.commit()
    return {"written": written, "total": len(rows), "lead_cleaned": n_lead_cleaned,
            "secs": round(time.time() - t0, 1), "out": REMOTE_OUT}


@app.local_entrypoint()
def main(limit: int = 0, block: bool = False) -> None:
    call = generate.spawn(limit=limit)
    print(f"SPAWNED generate call_id={call.object_id} — running detached on Modal; "
          f"poll {VOL_MOUNT}/{RUN_NAME}/ours_v5_val_gen.jsonl on the Volume for completion.")
    if block:
        res = call.get()
        print(json.dumps(res, indent=2, default=str))
        LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = LOCAL_OUT.parent / "_ours_v5_dl"
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force",
                        VOLUME_NAME, f"/{RUN_NAME}/ours_v5_val_gen.jsonl", str(tmp)], check=True)
        got = tmp / "ours_v5_val_gen.jsonl"
        if got.exists():
            shutil.move(str(got), str(LOCAL_OUT))
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"DONE -> {LOCAL_OUT}")
