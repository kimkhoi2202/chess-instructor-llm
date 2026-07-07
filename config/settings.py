"""Central configuration for the chess-instructor pipeline.

Single source of truth for tiers, engine tolerances, Maia mapping, paths, and
model ids. Imported by ingest / teacher / filter / eval so every stage agrees.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
POSITIONS = DATA / "positions"
TRANSCRIPTS = DATA / "transcripts"
GENERATED = DATA / "generated"
DATASET = DATA / "dataset"
MODELS = ROOT / "models"
PROMPTS = ROOT / "prompts"

# --- Tiers (rating bands) -------------------------------------------------
# Coaching granularity is coarse on purpose: fine (100-elo) buckets are false
# precision no judge can grade. Three tiers = gradeable + data-efficient.
TIERS = {
    "beginner":     {"low": 1000, "high": 1200, "maia": "maia-1100", "ply_cap": 2},
    "intermediate": {"low": 1300, "high": 1600, "maia": "maia-1500", "ply_cap": 4},
    "advanced":     {"low": 1700, "high": 2000, "maia": "maia-1900", "ply_cap": 6},
}

def tier_for_rating(rating: int) -> str | None:
    for name, t in TIERS.items():
        if t["low"] <= rating <= t["high"]:
            return name
    return None

# --- Engine (Stockfish) tolerances ---------------------------------------
STOCKFISH_BIN = "/opt/homebrew/bin/stockfish"
SOUND_TOLERANCE_CP = 150      # a move within this of best is "sound" (teachable)
BLUNDER_CP = 250              # cp_loss >= this is a blunder (never recommend)
MISTAKE_CP = 100
INACCURACY_CP = 50
DEFAULT_MOVETIME_MS = 300
MULTIPV = 8

# --- Maia -----------------------------------------------------------------
MAIA_DIR = MODELS / "maia"

# --- Teacher / judge models ----------------------------------------------
TEACHER_MODEL = "gpt-5.5"            # override via .env TEACHER_MODEL
TEACHER_REASONING_EFFORT = "high"    # "maximum reasoning"
TEACHER_MODEL_HARD = "gpt-5.5-pro"   # optional, for hard positions
# Judge must be a DIFFERENT family than the teacher (no grading own homework).
JUDGE_MODEL = "claude"               # resolve to a concrete Anthropic id when wired

# --- Behavior Spec (the gate; grades every output) -----------------------
BEHAVIOR_SPEC = """\
Given a position, the student's rating tier, the move the student played, and \
full-strength engine analysis (a sound-move pool with evals + short lines) plus \
the tier's human-move likelihoods, the coach recommends exactly ONE move drawn \
from the sound pool whose idea is explainable using only concepts appropriate to \
that tier, explains it in plain human terms tied to a concrete plan and to the \
student's actual mistake, and NEVER: states raw engine numbers/centipawns, cites \
lines deeper than the tier's ply cap, recommends a blunder, or fabricates a tactic \
absent from the analysis. Every response ends with one transferable takeaway, and \
the same position yields simpler ideas for Beginner than for Advanced."""
