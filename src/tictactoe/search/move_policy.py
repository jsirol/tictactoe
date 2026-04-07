from __future__ import annotations

from tictactoe.core import GameState, Move


def candidate_moves(state: GameState, radius: int = 1) -> list[Move]:
    legal = state.board.legal_moves()
    if not legal:
        return []

    occupied: list[Move] = []
    for row in range(state.board.size):
        for col in range(state.board.size):
            if state.board.cells[row][col] is not None:
                occupied.append(Move(row, col))

    if not occupied:
        center = state.board.size // 2
        return [Move(center, center)]

    frontier: set[Move] = set()
    for stone in occupied:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                move = Move(stone.row + dr, stone.col + dc)
                if state.board.in_bounds(move) and state.board.cells[move.row][move.col] is None:
                    frontier.add(move)

    if not frontier:
        return legal

    center = (state.board.size - 1) / 2.0
    return sorted(frontier, key=lambda m: (abs(m.row - center) + abs(m.col - center), m.row, m.col))
