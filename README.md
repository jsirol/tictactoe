# tictactoe

Configurable Tic Tac Toe / five-in-a-row game in Python with a bot interface.

For full self-play + training usage, see [`SELFPLAY_README.md`](SELFPLAY_README.md).
For alternating headless self-play/train loop helper, use `scripts/run_selfplay_train_loop.py`.

## Implemented Features

- Python project managed with `uv`
- Configurable square board size
  - Minimum size: `10`
  - Default size in CLI: `15`
- Win condition: streak of **at least 5** in a row
  - Horizontal, vertical, and both diagonals
- Symbols: `X` and `O`
- Modes:
  - `play`: terminal UI (human vs bot)
  - `simulate`: headless bot vs bot simulation
  - `selfplay`: multi-process self-play data generation
  - `web`: browser UI (human vs bot, default: MCTS)
- Bot architecture:
  - Shared bot protocol/interface
  - Implemented bots: `RandomBot`, `MCTSBot`, and `AlphaBetaBot` (seedable RNG support)
- Reusable search modules for stronger bots:
    - frontier move policy
    - pluggable move generation modes (`full_legal`, `frontier`, `threat_frontier`)
    - tactical immediate win/block checks
    - forcing-line threat solver utility
    - heuristic value model (reusable across bots)
    - policy/value model abstraction for MCTS (heuristic or torch-backed)
- Browser UI:
  - Clickable grid board in the web page
  - `New Game` button to reset/start a match
  - Current game state shown (turn/winner/draw)
- Test suite with `pytest` for core game logic, bots, CLI/web behavior, search modules, and seeded bot-strength regression checks
- Local benchmark script for bot performance comparisons
 - Puzzle-pack loader and tactical solve-rate evaluation workflow

## Requirements

- Python `>= 3.11`
- `uv` installed: https://docs.astral.sh/uv/

## Running

Run commands from the repository root.

### Run tests

```bash
uv run --extra dev pytest
```

### Run benchmark

```bash
uv run python scripts/benchmark_bots.py --size 15 --games 10 --bot-x alphabeta --bot-o mcts --seed 42
```

For alpha-beta bots, benchmark output also includes search stats (`depth`, `nodes`, `tt_hits`, `cutoffs`).
Default alpha-beta search budget is tuned for stronger tactical play (`~800ms` per move).

### Evaluate tactical puzzle packs

```bash
uv run python scripts/evaluate_puzzles.py --pack data/puzzles/tuning_10x10.jsonl --bot alphabeta --seed 1
```

Optional:
- `--weights-file <PATH>` load heuristic weights from registry/report/json file

You can also include puzzle evaluation inside benchmark runs:

```bash
uv run python scripts/benchmark_bots.py --size 10 --games 2 --bot-x alphabeta --bot-o mcts --seed 1 --puzzle-pack data/puzzles/holdout_15x15.jsonl
```

Current packs:
- `data/puzzles/tuning_10x10.jsonl` for fast iteration
- `data/puzzles/holdout_15x15.jsonl` for transfer/holdout validation

### Tune heuristic weights

```bash
uv run python scripts/tune_weights.py --samples 20 --seed 1 --tuning-pack data/puzzles/tuning_10x10.jsonl --holdout-pack data/puzzles/holdout_15x15.jsonl
```

This writes:
- per-run reports to `data/weights/runs/<run_id>.json`
- canonical best profile to `data/weights/best.json` only when metrics improve

Promotion rule:
1. higher holdout solve-rate
2. tie-break with tuning solve-rate
3. tie-break with larger evaluated puzzle count

### Play in terminal UI (Human vs Bot)

```bash
uv run tictactoe play
```

Optional flags:

- `--size <N>` board size (must be `>= 10`)
- `--seed <INT>` deterministic bot randomness
- `--bot <random|mcts|alphabeta>` bot to play against

Example:

```bash
uv run tictactoe play --size 15 --seed 42
```

Move input format is `row col` using **0-based** coordinates (example: `7 8`).

### Run headless simulations (Bot vs Bot)

```bash
uv run tictactoe simulate --size 15 --games 100 --seed 42
```

Outputs total `X` wins, `O` wins, and draws.

Optional flags:

- `--bot-x <random|mcts|alphabeta>` bot playing as X (default: `random`)
- `--bot-o <random|mcts|alphabeta>` bot playing as O (default: `random`)

Example with different bots:

```bash
uv run tictactoe simulate --size 15 --games 50 --seed 42 --bot-x alphabeta --bot-o mcts
```

### Run multi-process self-play generation

```bash
uv run tictactoe selfplay --size 10 --games 40 --workers 12 --batch-size 32 --output-dir data/selfplay --seed 1
```

Optional:
- `--model-path <PATH>` load a torchscript policy/value model (falls back to heuristic when unavailable)
- `--determinism <balanced|strict|fast>` reproducibility/throughput mode

Self-play writes a JSONL shard plus `manifest.json` under `data/selfplay/`.  
Training samples (`kind=sample`) include: `obs`, `policy_target`, `value_target`, `action_mask`, `player_to_move`, `game_id`, `ply`, `model_version`.

### Run one headless policy/value training pass

Install RL dependency once:

```bash
uv sync --extra rl
```

Train from the latest self-play manifest:

```bash
uv run python scripts/train_policy_value.py --data-manifest data/selfplay/manifest.json --epochs 5 --batch-size 128 --out-dir data/models
```

This creates:
- `<run_name>.pt` (training checkpoint)
- `<run_name>.torchscript.pt` (inference artifact for `--model-path`)
- `<run_name>.meta.json` (run summary + metrics + source data)

### Run browser UI

```bash
uv run tictactoe web
```

Then open `http://127.0.0.1:8000` in your browser.

Optional flags:

- `--host <HOST>` server host (default: `127.0.0.1`)
- `--port <PORT>` server port (default: `8000`)
- `--size <N>` default new-game board size (must be `>= 10`)
- `--seed <INT>` optional deterministic seed for bot behavior
- `--bot <random|mcts|alphabeta>` bot used in web games (default: `mcts`)

Example:

```bash
uv run tictactoe web --host 127.0.0.1 --port 8000 --size 15 --seed 42 --bot mcts
```
