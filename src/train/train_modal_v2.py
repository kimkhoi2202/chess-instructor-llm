#!/usr/bin/env python3
"""QLoRA fine-tune Qwen3-1.7B into the chess-coach **v2** specialist on a Modal GPU.

This is a v2-suffixed copy of ``train_modal.py`` — same trainer, same base model,
same hyper-parameters — pointed at the v2 dataset
(``data/dataset/{train_v2,valid_v2}.jsonl``) and writing to a separate run dir
(``chess-coach-v2``) so **nothing v1 is overwritten**. Keeping the recipe
identical to v1 makes the v1->v2 comparison a clean DATA intervention.

Commands
--------
Smoke (<=20 rows, ~20 steps; proves the loop cheaply)::

    /Users/khoilam/.venvs/mlx/bin/modal run src/train/train_modal_v2.py --smoke

Full run::

    /Users/khoilam/.venvs/mlx/bin/modal run src/train/train_modal_v2.py

Prereq: ``data/dataset/{train_v2,valid_v2}.jsonl`` must exist locally (build via
the v2 filter + ``split_data.py`` with v2 paths). Merged 16-bit output is
downloaded to ``models/adapters/chess-coach-v2/`` and then converted to 4-bit MLX
at ``models/mlx/chess-coach-v2`` by ``scripts`` (mlx_lm.convert), outside Modal.
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
# Names / paths (v2)
# --------------------------------------------------------------------------- #
APP_NAME: str = "chess-coach-qlora-v2"
VOLUME_NAME: str = "chess-coach-lora"           # shared volume; v2 uses its own run dir
RUN_NAME: str = "chess-coach-v2"                # this run's artifact dir (v2)

VOL_MOUNT: str = "/vol"
REMOTE_TRAIN: str = "/data/train_v2.jsonl"
REMOTE_VALID: str = "/data/valid_v2.jsonl"
ADAPTER_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/adapter"
MERGED_DIR: str = f"{VOL_MOUNT}/{RUN_NAME}/merged_16bit"

if modal.is_local():
    _THIS_DIR = Path(__file__).resolve().parent
    REPO_ROOT: Optional[Path] = _THIS_DIR.parents[1]
    LOCAL_TRAIN: Optional[Path] = REPO_ROOT / "data" / "dataset" / "train_v2.jsonl"
    LOCAL_VALID: Optional[Path] = REPO_ROOT / "data" / "dataset" / "valid_v2.jsonl"
    LOCAL_OUT_DIR: Optional[Path] = REPO_ROOT / "models" / "adapters" / RUN_NAME
else:
    REPO_ROOT = LOCAL_TRAIN = LOCAL_VALID = LOCAL_OUT_DIR = None

# --------------------------------------------------------------------------- #
# Hyper-parameters (identical to v1 for a clean data-only comparison)
# --------------------------------------------------------------------------- #
BASE_MODEL: str = "unsloth/Qwen3-1.7B"
MAX_SEQ_LEN: int = 2048

LORA_R: int = 16
LORA_ALPHA: int = 16
LORA_DROPOUT: float = 0.0
TARGET_MODULES: list[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LEARNING_RATE: float = 2e-4
NUM_EPOCHS: float = 2.0
WARMUP_RATIO: float = 0.05
LR_SCHEDULER: str = "cosine"
WEIGHT_DECAY: float = 0.01
OPTIMIZER: str = "adamw_8bit"
PER_DEVICE_BATCH: int = 2
GRAD_ACCUM: int = 4
LOGGING_STEPS: int = 1
SEED: int = 3407

SMOKE_MAX_ROWS: int = 20
SMOKE_MAX_STEPS: int = 20

QWEN_INSTRUCTION_PART: str = "<|im_start|>user\n"
QWEN_RESPONSE_PART: str = "<|im_start|>assistant\n"

# --------------------------------------------------------------------------- #
# Modal infra
# --------------------------------------------------------------------------- #
GPU: str = "A10G"
TIMEOUT_S: int = 3600
CUDA_TAG: str = "12.4.1-cudnn-devel-ubuntu22.04"
PY_VERSION: str = "3.11"

PIP_PACKAGES: list[str] = [
    "unsloth", "trl", "peft", "bitsandbytes", "transformers", "datasets",
    "accelerate", "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf",
]


def _require_local_data() -> None:
    missing = [p for p in (LOCAL_TRAIN, LOCAL_VALID) if not p.exists()]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        raise SystemExit(
            "BLOCKED: missing v2 dataset shard(s):\n  "
            f"{names}\n"
            "Build them first, e.g.:\n"
            "  python src/filter/filter.py --candidates data/generated/candidates_v2.jsonl "
            "--faithfulness reject --dedup-key fen_tier_move --target-format v2 "
            "--train-out data/dataset/train_v2.jsonl --rejects-out data/generated/rejects_v2.jsonl\n"
            "  python src/train/split_data.py --input data/dataset/train_v2.jsonl "
            "--train-out data/dataset/train_v2.jsonl --valid-out data/dataset/valid_v2.jsonl"
        )


image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .apt_install("git")
    .pip_install(*PIP_PACKAGES)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)

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
# Remote helpers
# --------------------------------------------------------------------------- #
def _read_chat_rows(path: str, *, limit: Optional[int] = None) -> list[dict]:
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
    from datasets import Dataset

    texts = [
        tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        for row in rows
    ]
    return Dataset.from_list([{"text": t} for t in texts])


def _make_sft_config(**kwargs: Any):
    import inspect

    from trl import SFTConfig

    valid = set(inspect.signature(SFTConfig.__init__).parameters)
    if "max_seq_length" in kwargs and "max_seq_length" not in valid:
        kwargs["max_length"] = kwargs.pop("max_seq_length")
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return SFTConfig(**filtered)


def _make_trainer(**kwargs: Any):
    import inspect

    from trl import SFTTrainer

    valid = set(inspect.signature(SFTTrainer.__init__).parameters)
    if "tokenizer" in kwargs and "tokenizer" not in valid and "processing_class" in valid:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return SFTTrainer(**filtered)


def _print_gpu_banner() -> Optional[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        print(f"[gpu] nvidia-smi: {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"[gpu] nvidia-smi unavailable: {exc}")

    import torch

    avail = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if avail else None
    print(f"[gpu] torch={torch.__version__} cuda_available={avail} device={name}")
    return name


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={VOL_MOUNT: volume})
def train(smoke: bool = False, merge_16bit: bool = True) -> dict:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    gpu_name = _print_gpu_banner()

    print(f"[load] base={BASE_MODEL!r} 4-bit max_seq_len={MAX_SEQ_LEN}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, target_modules=TARGET_MODULES, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, bias="none",
        use_gradient_checkpointing="unsloth", random_state=SEED,
    )

    row_limit = SMOKE_MAX_ROWS if smoke else None
    rows = _read_chat_rows(REMOTE_TRAIN, limit=row_limit)
    if not rows:
        raise RuntimeError(f"No training rows found at {REMOTE_TRAIN}")
    dataset = _build_text_dataset(rows, tokenizer)
    print(f"[data] train_rows={len(rows)} (smoke={smoke})")
    print("[data] sample rendered row (first 700 chars):")
    print(dataset[0]["text"][:700])

    max_steps = SMOKE_MAX_STEPS if smoke else -1
    num_epochs = 1.0 if smoke else NUM_EPOCHS
    sft_config = _make_sft_config(
        output_dir=f"{VOL_MOUNT}/{RUN_NAME}/_trainer",
        dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=PER_DEVICE_BATCH, gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO, num_train_epochs=num_epochs, max_steps=max_steps,
        learning_rate=LEARNING_RATE, logging_steps=LOGGING_STEPS, optim=OPTIMIZER,
        weight_decay=WEIGHT_DECAY, lr_scheduler_type=LR_SCHEDULER, seed=SEED,
        bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(), report_to="none",
    )
    trainer = _make_trainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=sft_config)
    trainer = train_on_responses_only(
        trainer, instruction_part=QWEN_INSTRUCTION_PART, response_part=QWEN_RESPONSE_PART,
    )

    print(f"[train] starting: max_steps={max_steps} epochs={num_epochs} "
          f"lr={LEARNING_RATE} eff_batch={PER_DEVICE_BATCH * GRAD_ACCUM}")
    train_output = trainer.train()

    losses = [
        {"step": d.get("step"), "loss": d["loss"]}
        for d in trainer.state.log_history if "loss" in d
    ]
    first_loss = losses[0]["loss"] if losses else None
    last_loss = losses[-1]["loss"] if losses else None
    print(f"[train] done. steps_logged={len(losses)} first_loss={first_loss} last_loss={last_loss}")

    print(f"[save] LoRA adapter -> {ADAPTER_DIR}")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    saved_merged = False
    if merge_16bit:
        print(f"[save] merged 16-bit model -> {MERGED_DIR}")
        model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
        saved_merged = True

    volume.commit()
    print("[save] volume committed.")

    return {
        "gpu": gpu_name, "smoke": smoke, "base_model": BASE_MODEL, "train_rows": len(rows),
        "max_steps": max_steps, "num_epochs": num_epochs, "steps_logged": len(losses),
        "first_loss": first_loss, "last_loss": last_loss,
        "train_metrics": getattr(train_output, "metrics", None),
        "adapter_dir": ADAPTER_DIR, "merged_dir": MERGED_DIR if saved_merged else None,
        "run_name": RUN_NAME,
    }


def _volume_get(remote_path: str, local_parent: Path) -> None:
    local_parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "modal", "volume", "get", "--force",
           VOLUME_NAME, remote_path, str(local_parent)]
    print(f"[download] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main(smoke: bool = False, merge: bool = True, download: bool = True) -> None:
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

    shutil.rmtree(LOCAL_OUT_DIR, ignore_errors=True)
    _volume_get(f"/{RUN_NAME}/adapter", LOCAL_OUT_DIR)
    if merge and not smoke:
        _volume_get(f"/{RUN_NAME}/merged_16bit", LOCAL_OUT_DIR)
    elif merge and smoke:
        print(f"[download] smoke: merged 16-bit left on Volume at {MERGED_DIR}")

    print(f"\nDONE. Artifacts under: {LOCAL_OUT_DIR}")
