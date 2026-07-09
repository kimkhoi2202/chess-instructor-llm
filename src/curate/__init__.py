"""Curation pipeline for the next 32B chess-coach training set.

Three stages, all CPU + TrueFoundry (no Modal GPU):

1. ``mine``  — stratified sample of the Lichess puzzle bank -> Stockfish sound
   pool + Maia human-likelihood + deterministic per-tier canonical move
   (``tier_select``); keep only the *discriminating* positions where the
   beginner (most human-findable sound) move != advanced (sharpest) move — the
   moat — deduped against the existing train/eval sets (no leakage).
2. ``label`` — for every mined (position, tier), generate coaching INDEPENDENTLY
   with the three cross-family teachers (gpt-5.5 / claude-opus-4-8 /
   gemini-3.1-pro), gate every candidate (faithfulness, tier-fit,
   principle-in-takeaway, no engine-speak, correctness), then best-of-N select
   the most instructive surviving label per example.
3. ``build`` — assemble the curated train/valid jsonl + manifest, ready for the
   next 32B QLoRA.
"""
