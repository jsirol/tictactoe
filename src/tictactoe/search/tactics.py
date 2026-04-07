from __future__ import annotations

from tictactoe.core import GameState, Move, Symbol
from tictactoe.search.move_policy import candidate_moves


def find_immediate_winning_move(state: GameState, symbol: Symbol) -> Move | None:
    for move in candidate_moves(state, radius=1):
        trial = clone_state(state)
        if trial.next_symbol is not symbol:
            trial.next_symbol = symbol
        trial.apply_move(move)
        if trial.winner is symbol:
            return move
    return None


def clone_state(state: GameState) -> GameState:
    cloned = GameState.new(size=state.board.size)
    cloned.next_symbol = state.next_symbol
    cloned.winner = state.winner
    for row in range(state.board.size):
        for col in range(state.board.size):
            cloned.board.cells[row][col] = state.board.cells[row][col]
    return cloned
