from __future__ import annotations

import argparse

from tictactoe.puzzle_eval import evaluate_pack


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=str, required=True)
    parser.add_argument("--bot", type=str, default="alphabeta", choices=["random", "mcts", "alphabeta"])
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    result = evaluate_pack(path=args.pack, bot_name=args.bot, seed=args.seed)
    print(
        f"Puzzle eval: pack={args.pack}, bot={args.bot}, solved={result['solved']}/{result['total']} "
        f"({result['solve_rate']:.2%})"
    )
    print("By size:")
    for size, stats in result["by_size"].items():
        solved, total, rate = stats
        print(f"  {size}x{size}: {solved}/{total} ({rate:.2%})")
    print("By kind:")
    for kind, stats in result["by_kind"].items():
        solved, total, rate = stats
        print(f"  {kind}: {solved}/{total} ({rate:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
