from __future__ import annotations

import argparse
import random
from typing import Sequence

import uvicorn

from .bots import AlphaBetaBot, Bot, MCTSBot, RandomBot
from .core import MIN_BOARD_SIZE, GameState, InvalidMove, Move, Symbol
from .model_loader import load_mcts_policy_value_model
from .selfplay import SelfPlayConfig, run_selfplay
from .search.value_model import HeuristicValueModel
from .weight_store import load_best_weights_if_exists, load_weights_file
from .web import create_app


def parse_move(raw: str) -> Move:
    parts = raw.strip().split()
    if len(parts) != 2:
        raise ValueError("Expected two integers: row col")
    try:
        row, col = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise ValueError("Row and col must be integers") from exc
    return Move(row=row, col=col)


def render_board(state: GameState) -> str:
    header = "   " + " ".join(f"{c:2d}" for c in range(state.board.size))
    rows: list[str] = [header]
    for row_idx, row in enumerate(state.board.cells):
        cells = " ".join(f"{(cell.value if cell else '.'):>2}" for cell in row)
        rows.append(f"{row_idx:2d} {cells}")
    return "\n".join(rows)


def get_bot(name: str, weights_file: str | None = None) -> Bot:
    if name == "random":
        return RandomBot()
    if name == "mcts":
        policy_model = load_mcts_policy_value_model()
        if policy_model is None:
            return MCTSBot()
        return MCTSBot(policy_value_model=policy_model)
    if name == "alphabeta":
        weights = None
        if weights_file:
            weights = load_weights_file(weights_file)
        else:
            weights = load_best_weights_if_exists()
        if weights is None:
            return AlphaBetaBot()
        return AlphaBetaBot(value_model=HeuristicValueModel(weights=weights))
    raise ValueError(f"Unsupported bot: {name}")


def play_bot_vs_bot(size: int, bot_x: Bot, bot_o: Bot, rng: random.Random) -> GameState:
    state = GameState.new(size=size)
    while not state.is_over:
        symbol = state.next_symbol
        bot = bot_x if symbol is Symbol.X else bot_o
        move = bot.choose_move(state=state, symbol=symbol, rng=rng)
        state.apply_move(move)
    return state


def run_simulation(
    size: int,
    games: int,
    seed: int | None = None,
    bot_x_name: str = "random",
    bot_o_name: str = "random",
    weights_file: str | None = None,
) -> dict[str, int]:
    if size < MIN_BOARD_SIZE:
        raise ValueError(f"--size must be >= {MIN_BOARD_SIZE}")
    if games < 1:
        raise ValueError("--games must be >= 1")

    rng = random.Random(seed)
    bot_x = get_bot(bot_x_name, weights_file=weights_file)
    bot_o = get_bot(bot_o_name, weights_file=weights_file)
    result = {"X": 0, "O": 0, "draw": 0}
    for _ in range(games):
        state = play_bot_vs_bot(size=size, bot_x=bot_x, bot_o=bot_o, rng=rng)
        if state.winner is Symbol.X:
            result["X"] += 1
        elif state.winner is Symbol.O:
            result["O"] += 1
        else:
            result["draw"] += 1
    return result


def run_web_server(
    host: str,
    port: int,
    size: int,
    seed: int | None = None,
    bot_name: str = "mcts",
    weights_file: str | None = None,
) -> int:
    if size < MIN_BOARD_SIZE:
        raise ValueError(f"--size must be >= {MIN_BOARD_SIZE}")
    app = create_app(
        default_size=size, default_seed=seed, default_bot=bot_name, default_weights_file=weights_file
    )
    uvicorn.run(app, host=host, port=port)
    return 0


def run_play(
    size: int, seed: int | None = None, bot_name: str = "random", weights_file: str | None = None
) -> int:
    if size < MIN_BOARD_SIZE:
        raise ValueError(f"--size must be >= {MIN_BOARD_SIZE}")

    rng = random.Random(seed)
    bot = get_bot(bot_name, weights_file=weights_file)
    state = GameState.new(size=size)

    while not state.is_over:
        print(render_board(state))
        if state.next_symbol is Symbol.X:
            raw = input("Your move (row col, 0-based): ")
            try:
                move = parse_move(raw)
                state.apply_move(move)
            except (ValueError, InvalidMove) as exc:
                print(f"Invalid move: {exc}")
                continue
        else:
            move = bot.choose_move(state=state, symbol=Symbol.O, rng=rng)
            state.apply_move(move)
            print(f"Bot plays: {move.row} {move.col}")

    print(render_board(state))
    if state.winner:
        print(f"Winner: {state.winner.value}")
    else:
        print("Draw")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tictactoe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    play = subparsers.add_parser("play", help="Play human vs bot in terminal UI")
    play.add_argument("--size", type=int, default=15)
    play.add_argument("--seed", type=int, default=None)
    play.add_argument("--bot", type=str, default="random", choices=["random", "mcts", "alphabeta"])
    play.add_argument("--weights-file", type=str, default=None)

    simulate = subparsers.add_parser("simulate", help="Run headless bot vs bot simulations")
    simulate.add_argument("--size", type=int, default=15)
    simulate.add_argument("--games", type=int, default=1)
    simulate.add_argument("--seed", type=int, default=None)
    simulate.add_argument("--bot-x", type=str, default="random", choices=["random", "mcts", "alphabeta"])
    simulate.add_argument("--bot-o", type=str, default="random", choices=["random", "mcts", "alphabeta"])
    simulate.add_argument("--weights-file", type=str, default=None)

    web = subparsers.add_parser("web", help="Run browser-based UI")
    web.add_argument("--host", type=str, default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    web.add_argument("--size", type=int, default=15)
    web.add_argument("--seed", type=int, default=None)
    web.add_argument("--bot", type=str, default="mcts", choices=["random", "mcts", "alphabeta"])
    web.add_argument("--weights-file", type=str, default=None)

    selfplay = subparsers.add_parser("selfplay", help="Run multi-process self-play data generation")
    selfplay.add_argument("--size", type=int, default=10)
    selfplay.add_argument("--games", type=int, default=20)
    selfplay.add_argument("--workers", type=int, default=12)
    selfplay.add_argument("--seed", type=int, default=None)
    selfplay.add_argument("--model-path", type=str, default=None)
    selfplay.add_argument("--output-dir", type=str, default="data/selfplay")
    selfplay.add_argument("--batch-size", type=int, default=32)
    selfplay.add_argument("--batch-wait-ms", type=int, default=3)
    selfplay.add_argument("--simulations", type=int, default=200)
    selfplay.add_argument("--time-budget-ms", type=int, default=120)
    selfplay.add_argument("--high-temperature", type=float, default=1.0)
    selfplay.add_argument("--low-temperature", type=float, default=0.1)
    selfplay.add_argument("--temperature-cutoff-ply", type=int, default=12)
    selfplay.add_argument(
        "--determinism", type=str, default="balanced", choices=["balanced", "strict", "fast"]
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "play":
            return run_play(size=args.size, seed=args.seed, bot_name=args.bot, weights_file=args.weights_file)
        if args.command == "simulate":
            result = run_simulation(
                size=args.size,
                games=args.games,
                seed=args.seed,
                bot_x_name=args.bot_x,
                bot_o_name=args.bot_o,
                weights_file=args.weights_file,
            )
            print(f"X wins: {result['X']}, O wins: {result['O']}, draws: {result['draw']}")
            return 0
        if args.command == "web":
            return run_web_server(
                host=args.host,
                port=args.port,
                size=args.size,
                seed=args.seed,
                bot_name=args.bot,
                weights_file=args.weights_file,
            )
        if args.command == "selfplay":
            summary = run_selfplay(
                SelfPlayConfig(
                    size=args.size,
                    games=args.games,
                    workers=args.workers,
                    seed=args.seed,
                    model_path=args.model_path,
                    output_dir=args.output_dir,
                    batch_size=args.batch_size,
                    batch_wait_ms=args.batch_wait_ms,
                    simulations=args.simulations,
                    time_budget_ms=args.time_budget_ms,
                    determinism=args.determinism,
                    high_temperature=args.high_temperature,
                    low_temperature=args.low_temperature,
                    temperature_cutoff_ply=args.temperature_cutoff_ply,
                )
            )
            print(
                f"Self-play complete: games={summary['games']}, samples={summary['samples']}, "
                f"elapsed={summary['elapsed_sec']:.2f}s, games_per_min={summary['games_per_min']:.2f}, "
                f"path={summary['path']}"
            )
            return 0
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
