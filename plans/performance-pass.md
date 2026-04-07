# First Performance Pass (Bot-Agnostic Core Optimizations)

## Summary
Implement a focused speed pass that removes major overheads without coupling to one algorithm: repeated state clones, repeated board scans, and repeated move scoring setup. Keep reusable primitives in `core` and `search` so future bots can share them.

## Key Changes
- Core state operations:
  - `GameState.fast_clone()`
  - `GameState.apply_move_for(symbol, move)`
  - `GameState.occupied_moves()`
- Reusable search layer:
  - `SearchContext` for per-move occupied/candidate caching.
  - `BoundedCache` for lightweight generic memoization.
- Search/tactics/value:
  - Tactics use `fast_clone` and `apply_move_for`.
  - Value model exposes bulk `score_moves(...)`.
- MCTS integration:
  - Context-backed candidate generation.
  - Cached leaf evaluation.
  - Keep `Bot` protocol unchanged.
- Benchmarking:
  - `scripts/benchmark_bots.py` for local perf comparison.

## Test Plan
- Core correctness for clone/apply/occupied helpers.
- Search context and cache behavior.
- Value-model bulk scoring parity with single scoring.
- Existing MCTS functional tests remain green.
- Perf-sanity checks rely on deterministic cache behavior (not wall-clock thresholds).
