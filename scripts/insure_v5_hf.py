#!/usr/bin/env python3
"""Insure the trained v5 LoRA adapter to a private Hugging Face repo (post-train).

Decoupled from the Modal training run: pulls the adapter from the shared Volume
(``chess-coach-lora:/chess-coach-v5/adapter``, pinned to the chess-instructor-4
workspace) and uploads it to a private HF model repo using the token from the
gitignored ``.env`` (never argv, never committed). The volume adapter is the
PRIMARY insurance; this is the durable off-Modal copy.

Run AFTER the full train finishes::

    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    export MODAL_PROFILE=chess-instructor-4
    /Users/khoilam/.venvs/mlx/bin/python -m scripts.insure_v5_hf
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

MODAL_BIN = "/Users/khoilam/.venvs/mlx/bin/modal"
VOLUME = "chess-coach-lora"
RUN_NAME = "chess-coach-v5"
REMOTE_ADAPTER = f"/{RUN_NAME}/adapter"
HF_REPO = "khoilamalphaai/chess-coach-32b-v5-adapter"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="chess-instructor-4")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--local-dir", default=str(_ROOT / "models" / "adapters" / RUN_NAME))
    args = ap.parse_args(argv)

    load_dotenv(_ROOT / ".env")
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        raise SystemExit("BLOCKED: no HF_TOKEN in .env")

    local = Path(args.local_dir)
    # Pull the adapter from the volume (only if we don't already have it locally).
    if not (local / "adapter_model.safetensors").exists():
        tmp = local.parent / "_v5_adapter_dl"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.pop("MODAL_TOKEN_ID", None)
        env.pop("MODAL_TOKEN_SECRET", None)
        env["MODAL_PROFILE"] = args.profile
        print(f"[dl] modal volume get {VOLUME}:{REMOTE_ADAPTER} -> {tmp}")
        subprocess.run([MODAL_BIN, "volume", "get", "--force", VOLUME, REMOTE_ADAPTER, str(tmp)],
                       env=env, check=True)
        nested = tmp / "adapter"
        src = nested if nested.exists() else tmp
        shutil.rmtree(local, ignore_errors=True)
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(local))
        shutil.rmtree(tmp, ignore_errors=True)

    if not (local / "adapter_model.safetensors").exists():
        raise SystemExit(f"BLOCKED: adapter not found at {local} after download")

    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True, private=not args.public)
    print(f"[hf] uploading {local} -> {HF_REPO} (private={not args.public})")
    api.upload_folder(folder_path=str(local), repo_id=HF_REPO, repo_type="model")
    print(f"INSURED -> https://huggingface.co/{HF_REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
