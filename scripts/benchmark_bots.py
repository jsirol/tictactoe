from __future__ import annotations

import argparse
import random
import time

from tictactoe.cli import get_bot, play_bot_vs_bot
from tictactoe.puzzle_eval import evaluate_pack


def run_benchmark(
    size: int,
    games: int,
    bot_x_name: str,
    bot_o_name: str,
    seed: int | None,
    weights_file: str | None = None,
) -> None:
    rng = random.Random(seed)
    bot_x = get_bot(bot_x_name, weights_file=weights_file)
    bot_o = get_bot(bot_o_name, weights_file=weights_file)
    weight_source = weights_file or "data/weights/best.json (auto if exists)"

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
        f"elapsed={elapsed:.2f}s, avg_game={avg_game_ms:.1f}ms, avg_move={avg_move_ms:.2f}ms, "
        f"weights={weight_source}"
    )
    if hasattr(bot_x, "last_stats"):
        stats = bot_x.last_stats
        print(
            f"X stats: depth={stats.depth_reached}, nodes={stats.nodes}, "
            f"tt_hits={stats.tt_hits}, cutoffs={stats.cutoffs}"
        )
    if hasattr(bot_o, "last_stats"):
        stats = bot_o.last_stats
        print(
            f"O stats: depth={stats.depth_reached}, nodes={stats.nodes}, "
            f"tt_hits={stats.tt_hits}, cutoffs={stats.cutoffs}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=15)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--bot-x", type=str, default="mcts", choices=["random", "mcts", "alphabeta"])
    parser.add_argument("--bot-o", type=str, default="random", choices=["random", "mcts", "alphabeta"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--weights-file", type=str, default=None)
    parser.add_argument("--puzzle-pack", type=str, default=None)
    parser.add_argument("--puzzle-bot", type=str, default=None, choices=["random", "mcts", "alphabeta"])
    args = parser.parse_args()

    run_benchmark(
        size=args.size,
        games=args.games,
        bot_x_name=args.bot_x,
        bot_o_name=args.bot_o,
        seed=args.seed,
        weights_file=args.weights_file,
    )
    if args.puzzle_pack:
        puzzle_bot = args.puzzle_bot or args.bot_x
        result = evaluate_pack(
            path=args.puzzle_pack, bot_name=puzzle_bot, seed=args.seed, weights_file=args.weights_file
        )
        print(
            f"Puzzle eval: pack={args.puzzle_pack}, bot={puzzle_bot}, "
            f"solved={result['solved']}/{result['total']} ({result['solve_rate']:.2%})"
        )
        print("Puzzle by size:")
        for size, stats in result["by_size"].items():
            solved, total, rate = stats
            print(f"  {size}x{size}: {solved}/{total} ({rate:.2%})")
        print("Puzzle by kind:")
        for kind, stats in result["by_kind"].items():
            solved, total, rate = stats
            print(f"  {kind}: {solved}/{total} ({rate:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
