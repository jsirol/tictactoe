from __future__ import annotations

import random
from collections import defaultdict

from tictactoe.cli import get_bot
from tictactoe.puzzles import load_puzzle_pack
from tictactoe.bots import Bot


def evaluate_pack(
    path: str,
    bot_name: str,
    seed: int | None,
    *,
    bot: Bot | None = None,
    weights_file: str | None = None,
) -> dict[str, object]:
    rng = random.Random(seed)
    if bot is None:
        bot = get_bot(bot_name, weights_file=weights_file)
    puzzles = load_puzzle_pack(path)
    solved = 0
    by_size: dict[int, list[bool]] = defaultdict(list)
    by_kind: dict[str, list[bool]] = defaultdict(list)

    for puzzle in puzzles:
        state = puzzle.to_state()
        move = bot.choose_move(state=state, symbol=state.next_symbol, rng=rng)
        ok = move in puzzle.expected_moves
        solved += 1 if ok else 0
        by_size[puzzle.size].append(ok)
        kind = "none" if puzzle.expected_kind is None else puzzle.expected_kind.value
        by_kind[kind].append(ok)

    return {
        "total": len(puzzles),
        "solved": solved,
        "solve_rate": (solved / len(puzzles)) if puzzles else 0.0,
        "by_size": {
            size: (sum(results), len(results), (sum(results) / len(results)) if results else 0.0)
            for size, results in sorted(by_size.items())
        },
        "by_kind": {
            kind: (sum(results), len(results), (sum(results) / len(results)) if results else 0.0)
            for kind, results in sorted(by_kind.items())
        },
    }
