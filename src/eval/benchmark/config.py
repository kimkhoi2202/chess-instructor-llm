"""Static configuration for the 2x2x5 coaching benchmark.

One place for: the competitor / judge registry, the two conditions, output paths,
and (estimated) token prices used for the cost readout. Everything downstream
imports from here so the grid is defined exactly once.

Nothing here is a secret. Model *ids* are public strings; API keys live only in
``ROOT/.env`` and are read at call time (never stored in an artifact).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import settings

# --------------------------------------------------------------------------- #
# Paths (overridable via env so a smoke run can use an isolated directory)
# --------------------------------------------------------------------------- #

BENCH_DIR: Path = Path(os.environ.get("BENCH_DIR", str(settings.DATA / "benchmark")))

SCENARIOS_PATH: Path = BENCH_DIR / "scenarios.jsonl"
GENERATIONS_PATH: Path = BENCH_DIR / "generations.jsonl"
OBJECTIVE_PATH: Path = BENCH_DIR / "objective.jsonl"
COUNCIL_PATH: Path = BENCH_DIR / "council.jsonl"
RESULTS_JSON_PATH: Path = BENCH_DIR / "results.json"

BLIND_LABEL_JSONL: Path = BENCH_DIR / "blind_label.jsonl"
BLIND_LABEL_HTML: Path = BENCH_DIR / "blind_label.html"
BLIND_KEY_JSON: Path = BENCH_DIR / "blind_key.json"

REPORT_MD_PATH: Path = Path(
    os.environ.get("BENCH_REPORT", str(settings.ROOT / "RESULTS_BENCHMARK.md"))
)

# --------------------------------------------------------------------------- #
# The grid: competitors, conditions, judges
# --------------------------------------------------------------------------- #

#: The two experimental conditions. Order is meaningful for report tables.
CONDITIONS: Tuple[str, ...] = ("ungrounded", "grounded")

CONDITION_LABEL: Dict[str, str] = {
    "ungrounded": "WITHOUT grounding",
    "grounded": "WITH grounding",
}


@dataclass(frozen=True)
class Model:
    """One competitor in the benchmark.

    ``kind`` is ``"mlx"`` (local, run via mlx_lm; free) or ``"tfy"`` (frontier,
    run via the TrueFoundry OpenAI-compatible gateway). ``family`` groups a
    frontier model with the judge from the same lab, for the self-preference
    check. ``price_in`` / ``price_out`` are **estimated** USD per 1M tokens used
    only for the cost readout.

    ``reasoning_effort`` is model-specific and empirically tuned against the
    gateway: GPT-5.5 emits *nothing* at default effort (it spends the whole token
    budget reasoning), so it needs ``"low"``; Claude goes empty *with* the param
    and is fast/clean without it, so it uses ``None``; Gemini is cheaper at
    ``"low"``. The backend drops the param automatically if a model rejects it.
    """

    key: str
    display: str
    kind: str
    ident: str              # mlx repo/path, or gateway model id
    family: str             # "gpt" | "claude" | "gemini" | "local"
    price_in: float = 0.0
    price_out: float = 0.0
    reasoning_effort: Optional[str] = None


#: The five competitors. ``MODEL_ORDER`` fixes their column order in every table.
MODELS: Dict[str, Model] = {
    "ours": Model(
        key="ours",
        display="OURS (chess-coach-v1, 1.7B tuned)",
        kind="mlx",
        ident=str(settings.MODELS / "mlx" / "chess-coach-v1"),
        family="local",
    ),
    "base": Model(
        key="base",
        display="BASE (Qwen3-1.7B-4bit, untuned)",
        kind="mlx",
        ident="mlx-community/Qwen3-1.7B-4bit",
        family="local",
    ),
    "gpt": Model(
        key="gpt",
        display="GPT-5.5",
        kind="tfy",
        ident="openai-group/gpt-5.5",
        family="gpt",
        price_in=1.25,
        price_out=10.0,
        reasoning_effort="low",
    ),
    "claude": Model(
        key="claude",
        display="Claude Opus 4.8",
        kind="tfy",
        ident="claude-group/claude-opus-4-8",
        family="claude",
        price_in=15.0,
        price_out=75.0,
        reasoning_effort=None,
    ),
    "gemini": Model(
        key="gemini",
        display="Gemini 3.1 Pro",
        kind="tfy",
        ident="gemini-group/gemini-3.1-pro",
        family="gemini",
        price_in=1.25,
        price_out=10.0,
        reasoning_effort="low",
    ),
}

MODEL_ORDER: Tuple[str, ...] = ("ours", "base", "gpt", "claude", "gemini")

#: The blinded council: the three frontier models act as cross-family judges.
JUDGE_KEYS: Tuple[str, ...] = ("gpt", "claude", "gemini")

#: Anonymisation labels (one per competitor) shown to the council + human labeler.
ANON_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E")

# --------------------------------------------------------------------------- #
# Generation / judging knobs
# --------------------------------------------------------------------------- #

#: Max new tokens for a coaching generation (local + frontier). Frontier
#: reasoning models count reasoning toward output; with per-model reasoning
#: effort tuned above, actual usage is ~200-1700 tokens, so 4000 is safe
#: headroom that prevents truncation without inflating cost (billed on usage).
GEN_MAX_TOKENS_LOCAL: int = 400
GEN_MAX_TOKENS_TFY: int = 4000

#: Max new tokens for a council ranking (rank 5 + 5x3 rubric + note, with
#: reasoning headroom).
JUDGE_MAX_TOKENS: int = 4000

#: Deterministic RNG seed for scenario shuffling + per-item anonymisation.
SEED: int = 20260706


def price_for(model_key: str) -> Tuple[float, float]:
    """(price_in, price_out) per 1M tokens for ``model_key`` (0 for local)."""
    m = MODELS[model_key]
    return m.price_in, m.price_out
