"""Tiny JSONL + checkpoint helpers shared by every benchmark phase.

The whole benchmark is *append-only + resumable*: each phase writes one JSON
object per unit of work and, on restart, skips units whose key is already on
disk. That means a mid-run crash (or a killed API call) never loses completed
work and never double-spends on a frontier call.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

_WRITE_LOCK = threading.Lock()


def ensure_dir(path: Path) -> None:
    """Create ``path`` (a directory) and its parents if missing."""
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Append one JSON object as a line and flush (thread-safe).

    A process-wide lock serialises writers so concurrent threads (frontier
    generation / judging) never interleave partial lines in the checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping blank/unparseable lines."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def done_keys(path: Path, fields: Sequence[str]) -> Set[Tuple[Any, ...]]:
    """Return the set of ``fields`` tuples already present in ``path``.

    Used for resume: e.g. ``done_keys(GENERATIONS_PATH, ["scenario_id",
    "model", "condition"])`` is every generation already completed.
    """
    keys: Set[Tuple[Any, ...]] = set()
    for row in read_jsonl(path):
        try:
            keys.add(tuple(row[f] for f in fields))
        except KeyError:
            continue
    return keys


def write_json(path: Path, obj: Any) -> None:
    """Pretty-write a JSON document (atomic-ish: temp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
