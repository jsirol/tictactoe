from __future__ import annotations

import argparse
import random
import time

from tictactoe.cli import get_bot, play_bot_vs_bot


def run_benchmark(size: int, games: int, bot_x_name: str, bot_o_name: str, seed: int | None) -> None:
    rng = random.Random(seed)
    bot_x = get_bot(bot_x_name)
    bot_o = get_bot(bot_o_name)

    start = time.perf_counter()
    total_moves = 0
    for _ in range(games):
        state = play_bot_vs_bot(size=size, bot_x=bot_x, bot_o=bot_o, rng=rng)
        occupied = sum(1 for row in state.board.cells for cell in row if cell is not None)
        total_moves += occupied

    elapsed = time.perf_counter() - start
    avg_game_ms = (elapsed / games) * 1000
    avg_move_ms = (elapsed / max(1, total_moves)) * 1000
    print(
        "Benchmark: "
        f"games={games}, board={size}x{size}, X={bot_x_name}, O={bot_o_name}, "
        f"elapsed={elapsed:.2f}s, avg_game={avg_game_ms:.1f}ms, avg_move={avg_move_ms:.2f}ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=15)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--bot-x", type=str, default="mcts", choices=["random", "mcts", "alphabeta"])
    parser.add_argument("--bot-o", type=str, default="random", choices=["random", "mcts", "alphabeta"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_benchmark(
        size=args.size, games=args.games, bot_x_name=args.bot_x, bot_o_name=args.bot_o, seed=args.seed
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
