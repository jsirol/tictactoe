# tictactoe

Configurable Tic Tac Toe / five-in-a-row game in Python with a bot interface.

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
- Browser UI:
  - Clickable grid board in the web page
  - `New Game` button to reset/start a match
  - Current game state shown (turn/winner/draw)
- Test suite with `pytest` for core game logic, bots, CLI/web behavior, search modules, and seeded bot-strength regression checks
- Local benchmark script for bot performance comparisons

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
