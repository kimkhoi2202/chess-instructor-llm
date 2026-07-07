#!/usr/bin/env python3
"""QLoRA fine-tune Qwen3-1.7B into the chess-coach specialist on a Modal GPU.

This is the "training is a button-press" downstream of the dataset. It takes the
filtered chat rows (``data/dataset/{train,valid}.jsonl`` — one
``{"messages": [system, user, assistant]}`` per line, produced by the filter +
``split_data.py``), fine-tunes a small Qwen3 base with Unsloth QLoRA on a single
NVIDIA GPU, and saves BOTH a LoRA adapter and a merged 16-bit model to a Modal
Volume, then downloads them to ``models/adapters/chess-coach-v1/`` locally.

Base model
----------
The eval harness scores the base model ``mlx-community/Qwen3-1.7B-4bit`` — a 4-bit
MLX quant of ``Qwen/Qwen3-1.7B`` (Qwen3's post-trained / instruct hybrid-thinking
model; there is **no** separate ``Qwen3-1.7B-Instruct`` repo on HF). To keep the
base-vs-tuned comparison honest we fine-tune the *same* weights via Unsloth's
ready 4-bit repo ``unsloth/Qwen3-1.7B`` (Unsloth auto-selects its dynamic
``unsloth/Qwen3-1.7B-unsloth-bnb-4bit`` quant for ``load_in_4bit=True``). The base
download is public — no Hugging Face token required.

Training details
----------------
- 4-bit QLoRA via Unsloth ``FastLanguageModel`` (fits an A10G 24GB easily).
- LoRA on the standard attention + MLP projections (r=16, alpha=16, dropout=0).
- Rows are rendered with the model's own chat template; TRL ``SFTTrainer`` trains
  with Unsloth's ``train_on_responses_only`` so the loss is computed on the
  ASSISTANT turn only (the system+user prompt is masked out).
- lr=2e-4, cosine schedule with warmup, ~2 epochs for the full run.

Commands
--------
Smoke test (<=20 rows, ~20 steps; proves the whole loop end-to-end, cheap)::

    /Users/khoilam/.venvs/mlx/bin/modal run src/train/train_modal.py --smoke

Full run (same command, no flag — "one command away")::

    /Users/khoilam/.venvs/mlx/bin/modal run src/train/train_modal.py

Prerequisite: ``data/dataset/{train,valid}.jsonl`` must exist locally (run the
filter then ``src/train/split_data.py``). They are attached to the image at build.

Security: no secrets are embedded; Modal auth comes from the ambient profile
(``~/.modal.toml`` / ``MODAL_TOKEN_*``) and the base model is public.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import modal

# --------------------------------------------------------------------------- #
# Names / paths
# --------------------------------------------------------------------------- #
APP_NAME: str = "chess-coach-qlora"
VOLUME_NAME: str = "chess-coach-lora"           # persistent outputs live here
RUN_NAME: str = "chess-coach-v1"                # this training run's artifact dir

VOL_MOUNT: str = "/vol"                          # Volume mount inside the container
REMOTE_TRAIN: str = "/data/train.jsonl"          # image-mounted training rows
REMOTE_VALID: str = "/data/valid.jsonl"          # image-mounted held-out rows
ADAPTER_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/adapter"
MERGED_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/merged_16bit"

# Repo-relative locals — computed ONLY on the client. This same module is
# re-imported INSIDE the container (as ``/root/train_modal.py``), where this repo
# layout does not exist and ``.parents[1]`` would be out of range, so every
# client-only path is guarded with ``modal.is_local()``. The container never
# needs these (it reads the image-mounted ``REMOTE_*`` paths instead).
if modal.is_local():
    _THIS_DIR = Path(__file__).resolve().parent
    REPO_ROOT: Optional[Path] = _THIS_DIR.parents[1]
    LOCAL_TRAIN: Optional[Path] = REPO_ROOT / "data" / "dataset" / "train.jsonl"
    LOCAL_VALID: Optional[Path] = REPO_ROOT / "data" / "dataset" / "valid.jsonl"
    LOCAL_OUT_DIR: Optional[Path] = REPO_ROOT / "models" / "adapters" / RUN_NAME
else:
    REPO_ROOT = LOCAL_TRAIN = LOCAL_VALID = LOCAL_OUT_DIR = None

# --------------------------------------------------------------------------- #
# Hyper-parameters (named constants; no magic numbers buried in the body)
# --------------------------------------------------------------------------- #
BASE_MODEL: str = "unsloth/Qwen3-1.7B"           # Unsloth loads its 4-bit quant
MAX_SEQ_LEN: int = 2048

LORA_R: int = 16
LORA_ALPHA: int = 16
LORA_DROPOUT: float = 0.0
TARGET_MODULES: list[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",      # attention
    "gate_proj", "up_proj", "down_proj",          # MLP
]

LEARNING_RATE: float = 2e-4
NUM_EPOCHS: float = 2.0
WARMUP_RATIO: float = 0.05
LR_SCHEDULER: str = "cosine"
WEIGHT_DECAY: float = 0.01
OPTIMIZER: str = "adamw_8bit"
PER_DEVICE_BATCH: int = 2
GRAD_ACCUM: int = 4                               # effective batch = 8
LOGGING_STEPS: int = 1
SEED: int = 3407

# Smoke-test caps (fast/cheap end-to-end check).
SMOKE_MAX_ROWS: int = 20
SMOKE_MAX_STEPS: int = 20

# Qwen3 chat-template markers for prompt masking (train on the response only).
QWEN_INSTRUCTION_PART: str = "<|im_start|>user\n"
QWEN_RESPONSE_PART: str = "<|im_start|>assistant\n"

# --------------------------------------------------------------------------- #
# Modal infra: GPU, image, volume, app
# --------------------------------------------------------------------------- #
GPU: str = "A10G"
TIMEOUT_S: int = 3600
CUDA_TAG: str = "12.4.1-cudnn-devel-ubuntu22.04"  # devel => nvcc present for kernels
PY_VERSION: str = "3.11"

# Fine-tuning stack. Unsloth is listed first so the resolver honours its version
# caps for the rest (trl / transformers / peft cannot float past what it supports).
PIP_PACKAGES: list[str] = [
    "unsloth",
    "trl",
    "peft",
    "bitsandbytes",
    "transformers",
    "datasets",
    "accelerate",
    "huggingface_hub",
    "hf_transfer",
    "sentencepiece",
    "protobuf",
]


def _require_local_data() -> None:
    """Fail fast (before image build) if the dataset shards are missing."""
    missing = [p for p in (LOCAL_TRAIN, LOCAL_VALID) if not p.exists()]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        raise SystemExit(
            "BLOCKED: missing dataset shard(s):\n  "
            f"{names}\n"
            "Produce them first, e.g.:\n"
            "  /Users/khoilam/.venvs/mlx/bin/python src/filter/filter.py "
            "--candidates data/generated/candidates_v1.jsonl\n"
            "  /Users/khoilam/.venvs/mlx/bin/python src/train/split_data.py"
        )


image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .apt_install("git")
    .pip_install(*PIP_PACKAGES)
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",   # fast base-model download
            "TOKENIZERS_PARALLELISM": "false",  # quiet fork warnings
        }
    )
)

# Ship the (small) dataset shards as a runtime mount — a tail layer, so editing
# the data does not invalidate the heavy pip layers. Only the client attaches
# them (the container's image is already resolved server-side).
if modal.is_local():
    _require_local_data()
    image = (
        image
        .add_local_file(LOCAL_TRAIN.as_posix(), REMOTE_TRAIN)
        .add_local_file(LOCAL_VALID.as_posix(), REMOTE_VALID)
    )

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


# --------------------------------------------------------------------------- #
# Helpers that run REMOTELY (heavy imports live inside so the local CLI is light)
# --------------------------------------------------------------------------- #
def _read_chat_rows(path: str, *, limit: Optional[int] = None) -> list[dict]:
    """Load ``{"messages": [...]}`` rows from a JSONL file, optionally truncated."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _build_text_dataset(rows: list[dict], tokenizer: Any):
    """Render each chat row to a single ``text`` field via the model's template.

    ``add_generation_prompt=False`` renders the full conversation (the assistant
    turn is part of the text); ``train_on_responses_only`` later masks everything
    up to and including the assistant marker so only the response is learned.
    """
    from datasets import Dataset

    texts = [
        tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        for row in rows
    ]
    return Dataset.from_list([{"text": t} for t in texts])


def _make_sft_config(**kwargs: Any):
    """Construct an ``SFTConfig`` tolerant of TRL field renames across versions.

    Newer TRL moved ``max_seq_length`` -> ``max_length`` and may drop
    ``dataset_text_field``; we inspect the accepted parameters and adapt so the
    same call works across the TRL versions Unsloth may pull.
    """
    import inspect

    from trl import SFTConfig

    valid = set(inspect.signature(SFTConfig.__init__).parameters)
    if "max_seq_length" in kwargs and "max_seq_length" not in valid:
        kwargs["max_length"] = kwargs.pop("max_seq_length")
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return SFTConfig(**filtered)


def _make_trainer(**kwargs: Any):
    """Construct an ``SFTTrainer`` tolerant of the ``tokenizer`` -> ``processing_class`` rename."""
    import inspect

    from trl import SFTTrainer

    valid = set(inspect.signature(SFTTrainer.__init__).parameters)
    if "tokenizer" in kwargs and "tokenizer" not in valid and "processing_class" in valid:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return SFTTrainer(**filtered)


def _print_gpu_banner() -> Optional[str]:
    """Print nvidia-smi + torch CUDA info; return the GPU name (or None)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        print(f"[gpu] nvidia-smi: {out}")
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        print(f"[gpu] nvidia-smi unavailable: {exc}")

    import torch

    avail = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if avail else None
    print(f"[gpu] torch={torch.__version__} cuda_available={avail} device={name}")
    return name


# --------------------------------------------------------------------------- #
# Remote training function
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={VOL_MOUNT: volume})
def train(smoke: bool = False, merge_16bit: bool = True) -> dict:
    """Fine-tune the coach with QLoRA and save adapter (+ merged) to the Volume.

    Parameters
    ----------
    smoke:
        If True, read at most ``SMOKE_MAX_ROWS`` rows and cap training at
        ``SMOKE_MAX_STEPS`` steps — a fast, cheap end-to-end check.
    merge_16bit:
        If True, also export a merged 16-bit model (adapter folded into the base)
        alongside the LoRA adapter.

    Returns
    -------
    A JSON-serialisable dict with the GPU name, row/step counts, the first/last
    training loss, and the Volume-relative artifact paths.
    """
    # Unsloth must be imported before transformers/trl so its patches apply.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    gpu_name = _print_gpu_banner()

    # 1) Load the base in 4-bit and attach LoRA -------------------------------
    print(f"[load] base={BASE_MODEL!r} 4-bit max_seq_len={MAX_SEQ_LEN}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,  # auto (bf16 on Ampere+, else fp16)
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # 2) Data -----------------------------------------------------------------
    row_limit = SMOKE_MAX_ROWS if smoke else None
    rows = _read_chat_rows(REMOTE_TRAIN, limit=row_limit)
    if not rows:
        raise RuntimeError(f"No training rows found at {REMOTE_TRAIN}")
    dataset = _build_text_dataset(rows, tokenizer)
    print(f"[data] train_rows={len(rows)} (smoke={smoke})")
    print("[data] sample rendered row (first 600 chars):")
    print(dataset[0]["text"][:600])

    # 3) Trainer --------------------------------------------------------------
    max_steps = SMOKE_MAX_STEPS if smoke else -1
    num_epochs = 1.0 if smoke else NUM_EPOCHS
    sft_config = _make_sft_config(
        output_dir=f"{VOL_MOUNT}/{RUN_NAME}/_trainer",
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        optim=OPTIMIZER,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        seed=SEED,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        report_to="none",
    )
    trainer = _make_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )
    # Mask the prompt: compute loss on the assistant response tokens only.
    trainer = train_on_responses_only(
        trainer,
        instruction_part=QWEN_INSTRUCTION_PART,
        response_part=QWEN_RESPONSE_PART,
    )

    print(f"[train] starting: max_steps={max_steps} epochs={num_epochs} "
          f"lr={LEARNING_RATE} eff_batch={PER_DEVICE_BATCH * GRAD_ACCUM}")
    train_output = trainer.train()

    losses = [
        {"step": d.get("step"), "loss": d["loss"]}
        for d in trainer.state.log_history
        if "loss" in d
    ]
    first_loss = losses[0]["loss"] if losses else None
    last_loss = losses[-1]["loss"] if losses else None
    print(f"[train] done. steps_logged={len(losses)} "
          f"first_loss={first_loss} last_loss={last_loss}")

    # 4) Save artifacts to the Volume ----------------------------------------
    print(f"[save] LoRA adapter -> {ADAPTER_DIR}")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    saved_merged = False
    if merge_16bit:
        print(f"[save] merged 16-bit model -> {MERGED_DIR}")
        model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
        saved_merged = True

    volume.commit()  # make writes visible to `modal volume get`
    print("[save] volume committed.")

    return {
        "gpu": gpu_name,
        "smoke": smoke,
        "base_model": BASE_MODEL,
        "train_rows": len(rows),
        "max_steps": max_steps,
        "num_epochs": num_epochs,
        "steps_logged": len(losses),
        "first_loss": first_loss,
        "last_loss": last_loss,
        "train_metrics": getattr(train_output, "metrics", None),
        "adapter_dir": ADAPTER_DIR,
        "merged_dir": MERGED_DIR if saved_merged else None,
        "run_name": RUN_NAME,
    }


# --------------------------------------------------------------------------- #
# Local download helper + entrypoint
# --------------------------------------------------------------------------- #
def _volume_get(remote_path: str, local_parent: Path) -> None:
    """Download a Volume folder ``remote_path`` into ``local_parent`` via the CLI.

    ``modal volume get`` recreates ``remote_path``'s basename as a subdirectory
    under the destination (like ``cp -r`` — ``get vol /a/adapter dest`` writes to
    ``dest/adapter/``), so pass the PARENT directory and let it create the named
    subfolder. ``--force`` overwrites; the command is shown so the download is
    visible in the run logs.
    """
    local_parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "modal", "volume", "get", "--force",
        VOLUME_NAME, remote_path, str(local_parent),
    ]
    print(f"[download] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main(smoke: bool = False, merge: bool = True, download: bool = True) -> None:
    """Run training on Modal, then pull the artifacts back to ``models/adapters/``.

    Flags (Modal maps these to ``--smoke`` / ``--no-merge`` / ``--no-download``):
      smoke     — <=20 rows, ~20 steps for a fast/cheap check (default False).
      merge     — also export the merged 16-bit model remotely (default True).
      download  — pull artifacts locally after training (default True). In smoke
                  mode only the small LoRA adapter is downloaded (the merged
                  model is left on the Volume to save egress).
    """
    print(f"=== {APP_NAME}: {'SMOKE' if smoke else 'FULL'} run ===")
    print(f"base model : {BASE_MODEL}")
    print(f"train data : {LOCAL_TRAIN}")
    print(f"gpu        : {GPU}")

    result = train.remote(smoke=smoke, merge_16bit=merge)

    print("\n=== remote train() result ===")
    print(json.dumps(result, indent=2, default=str))

    if not download:
        print("\n[download] skipped (--no-download).")
        return

    # Fresh local dir so a rerun doesn't merge stale files.
    shutil.rmtree(LOCAL_OUT_DIR, ignore_errors=True)

    # Always fetch the (small) adapter; fetch merged only for full runs. Each
    # call lands as ``LOCAL_OUT_DIR/<basename>/`` (adapter/ and merged_16bit/).
    _volume_get(f"/{RUN_NAME}/adapter", LOCAL_OUT_DIR)
    if merge and not smoke:
        _volume_get(f"/{RUN_NAME}/merged_16bit", LOCAL_OUT_DIR)
    elif merge and smoke:
        print(f"[download] smoke: merged 16-bit left on Volume at {MERGED_DIR}")

    print(f"\nDONE. Artifacts under: {LOCAL_OUT_DIR}")
