from tictactoe.core import GameState, Move, Symbol
from tictactoe.search.context import SearchContext
from tictactoe.search.move_policy import candidate_moves


def test_search_context_reuses_cached_candidates():
    state = GameState.new(size=10)
    state.board.place(Symbol.X, Move(5, 5))
    state.board.place(Symbol.O, Move(4, 5))
    context = SearchContext()

    first = context.candidate_moves(state, radius=1)
    second = context.candidate_moves(state, radius=1)
    assert first is second


def test_context_matches_candidate_moves_under_progressing_state():
    state = GameState.new(size=10)
    context = SearchContext()
    for move in [Move(5, 5), Move(5, 6), Move(6, 6)]:
        state.apply_move(move)
        assert context.candidate_moves(state, radius=1) == candidate_moves(state, radius=1)
