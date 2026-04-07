from __future__ import annotations

from tictactoe.core import GameState, Move, Symbol
from tictactoe.search.context import SearchContext
from tictactoe.search.move_policy import candidate_moves


def find_immediate_winning_move(
    state: GameState,
    symbol: Symbol,
    candidates: list[Move] | None = None,
    context: SearchContext | None = None,
) -> Move | None:
    if candidates is None:
        if context is not None:
            candidates = context.candidate_moves(state, radius=1)
        else:
            candidates = candidate_moves(state, radius=1)

    for move in candidates:
        trial = clone_state(state)
        trial.apply_move_for(symbol, move)
        if trial.winner is symbol:
            return move
    return None


def clone_state(state: GameState) -> GameState:
    return state.fast_clone()
