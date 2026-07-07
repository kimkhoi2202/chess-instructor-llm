#!/usr/bin/env python3
"""Shuffle the filtered SFT set and split it into train / valid shards.

The quality filter (``src/filter/filter.py``) writes every kept chat row to
``data/dataset/train.jsonl`` — one ``{"messages": [system, user, assistant]}``
object per line. This script shuffles those rows with a **fixed seed** (so the
split is reproducible across machines and reruns) and writes:

- ``data/dataset/train.jsonl`` — the ~95% training shard (overwritten in place), and
- ``data/dataset/valid.jsonl`` — the ~5% held-out shard.

The entire input is read into memory *before* anything is written, so reading and
writing the same ``train.jsonl`` path in one pass is safe. Because it overwrites
``train.jsonl``, run it once after each filter pass (or point ``--input`` at an
untouched full file if you want to re-split without shrinking the training shard).

Run from the project root with the pinned interpreter::

    /Users/khoilam/.venvs/mlx/bin/python src/train/split_data.py
    /Users/khoilam/.venvs/mlx/bin/python src/train/split_data.py --valid-frac 0.05 --seed 3407
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

# --- Paths (single source of truth relative to the repo root) ---------------
ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data" / "dataset"

# --- Split hyper-parameters (named constants; no magic numbers inline) -------
DEFAULT_SEED: int = 3407          # fixed => reproducible shuffle/split
DEFAULT_VALID_FRAC: float = 0.05  # ~95 / 5 train / valid split


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of objects, skipping blank lines.

    Raises ``SystemExit`` with a clear message if the file is missing or a line
    is not valid JSON (the upstream filter only ever writes valid rows, so a
    parse error means the wrong file was passed).
    """
    if not path.exists():
        raise SystemExit(
            f"BLOCKED: input not found: {path}\n"
            "Run the filter first, e.g.:\n"
            "  /Users/khoilam/.venvs/mlx/bin/python src/filter/filter.py "
            "--candidates data/generated/candidates_v1.jsonl"
        )
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"BLOCKED: {path}:{lineno} is not valid JSON: {exc}")
    return rows


def write_jsonl(rows: list[dict], out_path: Path) -> None:
    """Write ``rows`` to ``out_path`` atomically (temp file + os.replace)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)


def split_rows(
    rows: list[dict], *, valid_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Shuffle ``rows`` with ``seed`` and return ``(train_rows, valid_rows)``.

    Guarantees at least one row in each shard whenever ``len(rows) >= 2`` so a
    tiny (e.g. smoke) dataset still yields a usable held-out set. With a single
    row everything goes to train and valid is empty.
    """
    if not rows:
        raise SystemExit("BLOCKED: no rows to split (input was empty).")

    ordered = list(rows)
    random.Random(seed).shuffle(ordered)

    n = len(ordered)
    if n < 2:
        return ordered, []
    n_valid = max(1, round(n * valid_frac))
    n_valid = min(n_valid, n - 1)  # keep at least one training row
    valid_rows = ordered[:n_valid]
    train_rows = ordered[n_valid:]
    return train_rows, valid_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shuffle + split the filtered SFT set into train/valid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=DATASET_DIR / "train.jsonl",
        help="Filtered chat-rows JSONL to shuffle and split.",
    )
    parser.add_argument(
        "--train-out", type=Path, default=DATASET_DIR / "train.jsonl",
        help="Output path for the ~95%% training shard.",
    )
    parser.add_argument(
        "--valid-out", type=Path, default=DATASET_DIR / "valid.jsonl",
        help="Output path for the ~5%% held-out shard.",
    )
    parser.add_argument(
        "--valid-frac", type=float, default=DEFAULT_VALID_FRAC,
        help="Fraction of rows held out for validation.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Fixed RNG seed for a reproducible shuffle/split.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not (0.0 < args.valid_frac < 1.0):
        raise SystemExit(f"BLOCKED: --valid-frac must be in (0, 1), got {args.valid_frac}")

    rows = read_jsonl(args.input)
    train_rows, valid_rows = split_rows(
        rows, valid_frac=args.valid_frac, seed=args.seed
    )

    # Read fully above, so writing back to the input path is safe.
    write_jsonl(train_rows, args.train_out)
    write_jsonl(valid_rows, args.valid_out)

    total = len(rows)
    print("=== split_data ===")
    print(f"input:      {args.input}  ({total} rows)")
    print(f"seed:       {args.seed}   valid_frac: {args.valid_frac}")
    print(f"train ->    {args.train_out}  ({len(train_rows)} rows)")
    print(f"valid ->    {args.valid_out}  ({len(valid_rows)} rows)")
    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
