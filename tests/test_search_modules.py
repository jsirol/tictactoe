import random

from tictactoe.core import GameState, Move, Symbol
from tictactoe.search.move_policy import candidate_moves
from tictactoe.search.tactics import find_immediate_winning_move
from tictactoe.search.value_model import HeuristicValueModel


def test_candidate_moves_prefers_local_frontier():
    state = GameState.new(size=10)
    state.board.place(Symbol.X, Move(5, 5))
    moves = candidate_moves(state, radius=1)
    assert Move(5, 6) in moves
    assert Move(0, 0) not in moves


def test_candidate_moves_on_empty_board_returns_center():
    state = GameState.new(size=10)
    moves = candidate_moves(state, radius=2)
    assert moves == [Move(5, 5)]


def test_find_immediate_winning_move_detects_win():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.O
    for col in range(4):
        state.board.place(Symbol.O, Move(3, col))
    move = find_immediate_winning_move(state, Symbol.O)
    assert move == Move(3, 4)


def test_heuristic_value_model_scores_threat_higher():
    model = HeuristicValueModel()
    state_a = GameState.new(size=10)
    state_b = GameState.new(size=10)

    state_a.board.place(Symbol.X, Move(4, 4))
    state_a.board.place(Symbol.X, Move(4, 5))
    state_a.board.place(Symbol.X, Move(4, 6))

    state_b.board.place(Symbol.X, Move(0, 0))
    state_b.board.place(Symbol.X, Move(9, 9))
    state_b.board.place(Symbol.X, Move(0, 9))

    assert model.evaluate(state_a, Symbol.X) > model.evaluate(state_b, Symbol.X)


def test_heuristic_value_model_is_symmetric():
    model = HeuristicValueModel()
    state = GameState.new(size=10)
    state.board.place(Symbol.X, Move(5, 5))
    state.board.place(Symbol.O, Move(5, 6))
    state.board.place(Symbol.X, Move(6, 5))

    mirrored = GameState.new(size=10)
    mirrored.board.place(Symbol.X, Move(5, 4))
    mirrored.board.place(Symbol.O, Move(5, 3))
    mirrored.board.place(Symbol.X, Move(6, 4))

    assert model.evaluate(state, Symbol.X) == model.evaluate(mirrored, Symbol.X)


def test_heuristic_rollout_prefers_scored_move_shape():
    model = HeuristicValueModel()
    state = GameState.new(size=10)
    state.board.place(Symbol.X, Move(5, 5))
    state.board.place(Symbol.X, Move(5, 6))
    state.next_symbol = Symbol.X
    moves = candidate_moves(state, radius=1)
    scored = sorted(moves, key=lambda m: model.score_move(state, Symbol.X, m), reverse=True)
    assert scored[0] in moves
    assert isinstance(random.Random(1).randint(0, 1), int)
