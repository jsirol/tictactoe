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
- Bot architecture:
  - Shared bot protocol/interface
  - First bot implemented: `RandomBot` (seedable RNG support)
- Test suite with `pytest` for core game logic, bots, and CLI behavior

## Requirements

- Python `>= 3.11`
- `uv` installed: https://docs.astral.sh/uv/

## Running

Run commands from the repository root.

### Run tests

```bash
uv run --extra dev pytest
```

### Play in terminal UI (Human vs Bot)

```bash
uv run tictactoe play
```

Optional flags:

- `--size <N>` board size (must be `>= 10`)
- `--seed <INT>` deterministic bot randomness
- `--bot random` currently supported bot

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
