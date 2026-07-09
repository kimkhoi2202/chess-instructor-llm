#!/usr/bin/env python3
"""Publish the curated 32B training set to a Modal volume + a private HF repo.

CPU + a *small* volume write only — no Modal GPU, and it never touches the
kim-lam workspace or the running 32B eval (it only writes NEW files under a
``/curated/`` prefix). Uploads ``train_curated_32b.jsonl`` /
``valid_curated_32b.jsonl`` / ``manifest.json`` to:

* the Modal ``chess-data`` volume (``chess-instructor-2`` profile) under
  ``/curated/chess-coach-32b-v1/`` (override with ``--volume`` / ``--profile``),
* a private Hugging Face dataset repo (default
  ``khoilamalphaai/chess-coach-curated-32b``).

Run:
    python -m src.curate.publish            # both targets
    python -m src.curate.publish --skip-hf  # volume only
    python -m src.curate.publish --skip-modal
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

from config import settings  # noqa: E402
from src.curate.label import MANIFEST, TRAIN_OUT, VALID_OUT  # noqa: E402

log = logging.getLogger("curate.publish")

MODAL_BIN = "/Users/khoilam/.venvs/mlx/bin/modal"
DEFAULT_VOLUME = "chess-data"
DEFAULT_PROFILE = "chess-instructor-2"
DEFAULT_REMOTE_DIR = "/curated/chess-coach-32b-v1"
HF_REPO = "khoilamalphaai/chess-coach-curated-32b"


def _files() -> List[Path]:
    missing = [p for p in (TRAIN_OUT, VALID_OUT, MANIFEST) if not p.exists()]
    if missing:
        raise SystemExit(f"BLOCKED: build first — missing {', '.join(map(str, missing))}")
    return [TRAIN_OUT, VALID_OUT, MANIFEST]


def publish_modal(volume: str, profile: str, remote_dir: str) -> None:
    """Copy the curated files to the Modal volume (scrubbed env; never kim-lam)."""
    env = dict(os.environ)
    env.pop("MODAL_TOKEN_ID", None)
    env.pop("MODAL_TOKEN_SECRET", None)
    env["MODAL_PROFILE"] = profile
    for p in _files():
        remote = f"{remote_dir}/{p.name}"
        log.info("modal put %s -> %s:%s", p.name, volume, remote)
        subprocess.run([MODAL_BIN, "volume", "put", "--force", volume,
                        str(p), remote], env=env, check=True)
    log.info("Modal volume upload complete: %s:%s", volume, remote_dir)


def _readme(manifest: dict) -> str:
    c = manifest.get("counts", {})
    b = manifest.get("balance", {})
    return (
        "---\nlicense: cc0-1.0\ntags:\n- chess\n- coaching\n- sft\n- multi-tier\n"
        "pretty_name: Chess Coach Curated (32B, discriminating multi-tier)\n---\n\n"
        "# Chess Coach — Curated 32B training set\n\n"
        "Highest-quality curated SFT set of **discriminating multi-tier** chess-coaching "
        "positions: positions where the tier-appropriate move genuinely differs between a "
        "beginner (most human-findable sound move) and an advanced player (sharpest sound "
        "move). Mined on CPU (Stockfish sound pool + Maia human policy + a deterministic "
        "per-tier move selector), then labeled by three cross-family teacher models "
        "**independently** (gpt-5.5 / claude-opus-4-8 / gemini-3.1-pro), gate-verified "
        "(zero fabrication via a widened faithfulness checker, tier-appropriateness, a "
        "named transferable principle in the takeaway, no engine-speak), and **best-of-N** "
        "selected by an instructiveness score (+ a blinded cross-family judge tiebreak).\n\n"
        f"- Rows: **{c.get('total_rows')}** (train {c.get('train_rows')}, valid {c.get('valid_rows')})\n"
        f"- Distinct positions: **{c.get('distinct_positions')}**, full 3-tier triads: {c.get('full_triads_all_3_tiers')}\n"
        f"- By tier: {b.get('by_tier')}\n"
        f"- Best-of-N winner by family: {b.get('winner_by_family')}\n\n"
        "Positions derive from the CC0 Lichess puzzle database (used as a position source, "
        "not solutions); the coaching text is model-generated and engine/gate-verified. "
        "See `manifest.json` for full provenance, gate pass-rates, and cost.\n\n"
        "Format: chat JSONL (`messages`: system coach prompt, grounded user prompt, "
        "assistant coaching target). Ready to feed a 32B QLoRA SFT.\n"
    )


def publish_hf(repo: str, private: bool) -> None:
    from huggingface_hub import HfApi

    load_dotenv(settings.ROOT / ".env")
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        raise SystemExit("BLOCKED: no HF_TOKEN in .env")
    api = HfApi(token=tok)
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=private)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for p in _files():
        log.info("hf upload %s -> %s", p.name, repo)
        api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                        repo_id=repo, repo_type="dataset")
    readme = _ROOT / "data" / "curate" / "_hf_README.md"
    readme.write_text(_readme(manifest), encoding="utf-8")
    api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")
    log.info("HF upload complete: https://huggingface.co/datasets/%s (private=%s)",
             repo, private)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--volume", default=DEFAULT_VOLUME)
    p.add_argument("--profile", default=DEFAULT_PROFILE)
    p.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    p.add_argument("--hf-repo", default=HF_REPO)
    p.add_argument("--public", action="store_true", help="Make the HF repo public.")
    p.add_argument("--skip-modal", action="store_true")
    p.add_argument("--skip-hf", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _files()  # fail fast if not built
    if not args.skip_modal:
        publish_modal(args.volume, args.profile, args.remote_dir)
    if not args.skip_hf:
        publish_hf(args.hf_repo, private=not args.public)
    print("PUBLISH DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
