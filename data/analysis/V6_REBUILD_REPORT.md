# v6 Training-Label Rebuild — Report

A foundational, data-first rebuild of the chess-coach training **labels** (the move + provenance), deep-verified on Modal CPU with Stockfish 17 (two-depth root search + WDL bands) + Syzygy (Lichess tablebase API, ≤7 pieces) + Maia-2 human-likelihood. It feeds the downstream DPO + engine-distillation retrains. Existing v4 data + shipped docs are untouched.

## 1. What changed (the audit fixes, in the data)

| Fix | Mechanism | Result |
|---|---|---|
| #1 deep, robust sound pool | SF17 depth [14, 20] root-search, 2-depth agreement, WDL bands, Syzygy | **1567/5419** old benchmark sound-pool moves (28.9%) are rejected as not-actually-sound under deep search |
| #2 advanced = verified engine-best | rule: advanced := engine_best | old benchmark advanced!=engine_best: **43**; v6: **0** |
| #3 Maia as constraint | gate on human-likelihood, rank by robustness | min-max blend removed; beginner stays human-findable |
| #4 complete triads, no B=A!=I | atomic per-board selection + collapse fix | B=A!=I collapses: **0**; all-same capped + down-weighted |
| #5 coherent move-review | endorse unless worse by a margin | review actions: {'endorse': 3818, 'soft': 895, 'correct': 2418} |
| #6 quality over volume | sampling weights, fresh mining | discriminating boards prioritised; endgame/quiet mined |

## 2. Label-quality deltas

- **Training labels changed under the v6 rule + deeper search:** 3307 changed / 7269 comparable (45.5%).
- **Old recommended moves now rejected as unsound** (deep pool): 706 training positions had an old per-tier recommendation that is no longer in the deep sound pool.
- **Advanced-bug fixes:** 43 benchmark advanced labels diverged from the persisted engine_best in v4; v6 = 0.
- **Benchmark canonical labels re-derived:** 1089 of 2409 changed canonical move; 1074 changed engine_best under deep search.

## 3. v6 dataset stats

- train rows: **6768**, valid rows: **363** (game-disjoint holdout).
- unique boards: **2377**, discriminating boards: **2123**, all-same kept (capped): **254**.
- triad completeness: every board carries all 3 tiers (complete groups by construction).
- rows by tier: {'advanced': 2377, 'intermediate': 2377, 'beginner': 2377}
- rows by source: {'reused': 3210, 'mined_puzzle': 3921}
- prose provenance: {'reused': 4465, 'clean': 14063} (reused vetted teacher text where the move is unchanged; clean engine-grounded text otherwise).
- v6 phase share: {'endgame': 0.446, 'middlegame': 0.43, 'opening': 0.124}
- (context) old benchmark phase share: {'middlegame': 0.58, 'opening': 0.253, 'endgame': 0.167}; mined-position phase mix: {'endgame': 0.647, 'middlegame': 0.352, 'opening': 0.0}.

## 4. Provenance retained per row

Each row carries `provenance`: pos_id, fen, tier, phase, source, game_id, engine_best (uci/san/cp/wdl), canonical_uci/san + pool-rank + is_engine_best, maia_policy (pick + engine_best), severity + student move, review_action, discriminating / high_conf / pattern, the full sound_pool (uci/san/cp/wdl + per-tier maia policy), the deep-rejected moves + reasons, a `dpo_rejected_uci` contrast move, sampling `weight`, and the engine settings. This makes every label auditable and directly consumable by DPO (chosen/rejected) and engine-distillation (verified engine_best).

## 5. Engine configuration

```
{
  "stockfish": "17",
  "depths": [
    14,
    20
  ],
  "time_caps_s": [
    1.0,
    6.0
  ],
  "multipv": 10,
  "tol_cp": 120,
  "maia": "maia2-rapid @1100/1500/1900",
  "syzygy": "lichess tablebase API (<=7 pieces)",
  "endgame_tb_rate": 0.071
}
```

## 6. Ready for downstream

- **Engine-distillation:** `provenance.engine_best` is the deep-verified, WDL/tablebase-checked best move for every board.
- **DPO:** `canonical_uci` (chosen) + `dpo_rejected_uci` (an engine-rejected or over-levelled move) give ready contrast pairs; sampling `weight` favors high-confidence discriminating boards.
- Benchmark canonical labels refreshed in `data/benchmark_gap803/scenarios_v6.jsonl` with the **120 val ids stable** (360 val rows).