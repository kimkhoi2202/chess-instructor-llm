#!/usr/bin/env python3
"""Insure the trained v5 LoRA adapter to a private Hugging Face repo (post-train).

Decoupled from the Modal training run: pulls the adapter from the shared Volume
(``chess-coach-lora:/chess-coach-v5/adapter``, pinned to the chess-instructor-4
workspace) and uploads it to a private HF model repo using the token from the
gitignored ``.env`` (never argv, never committed). The volume adapter is the
PRIMARY insurance; this is the durable off-Modal copy.

Run AFTER the full train finishes.

Mac (named modal profile in ``~/.modal.toml``)::

    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    export MODAL_PROFILE=chess-instructor-4
    /Users/khoilam/.venvs/mlx/bin/python -m scripts.insure_v5_hf

Linux cloud VM (no named profile — env-var modal auth)::

    export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...   # or MODAL_CI4_TOKEN_ID / _SECRET
    python3 -m scripts.insure_v5_hf

The modal CLI is auto-resolved (``$MODAL_BIN`` -> PATH -> ``<repo>/.venv/bin/modal``
-> ``~/.venvs/mlx/bin/modal`` -> ``python -m modal``); auth uses the named profile
when ``~/.modal.toml`` defines it, else falls back to the env-var tokens above.
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

VOLUME = "chess-coach-lora"
RUN_NAME = "chess-coach-v5"
REMOTE_ADAPTER = f"/{RUN_NAME}/adapter"
HF_REPO = "khoilamalphaai/chess-coach-32b-v5-adapter"


def _resolve_modal_cmd() -> list:
    """Resolve the modal CLI as an argv prefix, portable Mac<->cloud.

    Order: ``$MODAL_BIN`` override -> ``modal`` on PATH -> ``<repo>/.venv/bin/modal``
    -> the Mac MLX venv (``~/.venvs/mlx/bin/modal``) -> ``python -m modal`` (works
    wherever the ``modal`` package is installed). Returns argv only; no secrets.
    """
    override = os.environ.get("MODAL_BIN")
    if override:
        return [override]
    for cand in (
        shutil.which("modal"),
        str(_ROOT / ".venv" / "bin" / "modal"),
        os.path.expanduser("~/.venvs/mlx/bin/modal"),
    ):
        if cand and Path(cand).exists():
            return [cand]
    return [sys.executable, "-m", "modal"]


def _profile_available(profile: str) -> bool:
    """True iff ``~/.modal.toml`` defines the named profile (the Mac auth path)."""
    toml_path = Path(os.path.expanduser("~/.modal.toml"))
    if not toml_path.exists():
        return False
    try:
        import tomllib
        with toml_path.open("rb") as fh:
            return profile in tomllib.load(fh)
    except Exception:  # noqa: BLE001  (missing tomllib / malformed file)
        try:
            return f"[{profile}]" in toml_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return False


def _modal_auth_env(profile: str) -> dict:
    """Env for the modal subprocess, portable Mac<->cloud (never prints secrets).

    Mac: a named profile exists in ``~/.modal.toml`` -> use it (scrub any stray
    token env so the profile wins — current behavior). Cloud: no named profile ->
    env-var auth; pass through ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET`` if present,
    else promote ``MODAL_CI4_TOKEN_ID``/``MODAL_CI4_TOKEN_SECRET`` into them.
    """
    env = dict(os.environ)
    if _profile_available(profile):
        env.pop("MODAL_TOKEN_ID", None)
        env.pop("MODAL_TOKEN_SECRET", None)
        env["MODAL_PROFILE"] = profile
        print(f"[auth] modal named profile: {profile}")
        return env
    env.pop("MODAL_PROFILE", None)
    if not (env.get("MODAL_TOKEN_ID") and env.get("MODAL_TOKEN_SECRET")):
        tid = os.environ.get("MODAL_CI4_TOKEN_ID")
        tsec = os.environ.get("MODAL_CI4_TOKEN_SECRET")
        if tid and tsec:
            env["MODAL_TOKEN_ID"] = tid
            env["MODAL_TOKEN_SECRET"] = tsec
    if env.get("MODAL_TOKEN_ID") and env.get("MODAL_TOKEN_SECRET"):
        print("[auth] modal env-var tokens (MODAL_TOKEN_ID/MODAL_TOKEN_SECRET)")
    else:
        print("[auth] WARNING: no named modal profile and no MODAL_TOKEN_ID/SECRET "
              "(set MODAL_TOKEN_ID/SECRET or MODAL_CI4_TOKEN_ID/SECRET) — the modal "
              "CLI will fail to authenticate.")
    return env


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
        modal_cmd = _resolve_modal_cmd()
        env = _modal_auth_env(args.profile)
        print(f"[dl] {' '.join(modal_cmd)} volume get {VOLUME}:{REMOTE_ADAPTER} -> {tmp}")
        subprocess.run([*modal_cmd, "volume", "get", "--force", VOLUME, REMOTE_ADAPTER, str(tmp)],
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
