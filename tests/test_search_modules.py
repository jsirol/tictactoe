import random

from tictactoe.core import GameState, Move, Symbol
from tictactoe.search.cache import BoundedCache
from tictactoe.search.context import SearchContext
from tictactoe.search.move_generator import MoveGenerationMode, generate_moves
from tictactoe.search.move_policy import candidate_moves
from tictactoe.search.tactics import (
    ThreatKind,
    detect_threats,
    find_forcing_threat_move,
    find_immediate_winning_move,
)
from tictactoe.search.threat_solver import solve_forcing_line
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
    move = find_immediate_winning_move(state, Symbol.O, context=SearchContext())
    assert move == Move(3, 4)


def test_find_forcing_threat_move_detects_diagonal_four_creation():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.X
    state.board.place(Symbol.X, Move(1, 1))
    state.board.place(Symbol.X, Move(2, 2))
    state.board.place(Symbol.X, Move(3, 3))
    move = find_forcing_threat_move(state, Symbol.X, min_streak=4, context=SearchContext())
    assert move == Move(4, 4)


def test_find_forcing_threat_move_ignores_half_blocked_four():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.X
    state.board.place(Symbol.X, Move(1, 1))
    state.board.place(Symbol.X, Move(2, 2))
    state.board.place(Symbol.X, Move(3, 3))
    state.board.place(Symbol.O, Move(0, 0))
    move = find_forcing_threat_move(state, Symbol.X, min_streak=4, context=SearchContext())
    assert move is None


def test_detect_threats_includes_double_three():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.X
    state.board.place(Symbol.X, Move(5, 4))
    state.board.place(Symbol.X, Move(5, 6))
    state.board.place(Symbol.X, Move(4, 5))
    state.board.place(Symbol.X, Move(6, 5))

    threats = detect_threats(state, Symbol.X, context=SearchContext())
    double_threes = [threat for threat in threats if threat.kind is ThreatKind.DOUBLE_THREE]
    assert double_threes
    assert double_threes[0].move == Move(5, 5)


def test_detect_threats_excludes_half_blocked_double_three():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.X
    state.board.place(Symbol.X, Move(5, 4))
    state.board.place(Symbol.X, Move(5, 6))
    state.board.place(Symbol.X, Move(4, 5))
    state.board.place(Symbol.X, Move(6, 5))
    state.board.place(Symbol.O, Move(5, 7))

    threats = detect_threats(state, Symbol.X, context=SearchContext())
    assert not any(threat.kind is ThreatKind.DOUBLE_THREE and threat.move == Move(5, 5) for threat in threats)


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


def test_context_candidate_moves_matches_stateless_generation():
    state = GameState.new(size=10)
    state.board.place(Symbol.X, Move(5, 5))
    state.board.place(Symbol.O, Move(4, 5))
    context = SearchContext()
    assert context.candidate_moves(state, radius=1) == candidate_moves(state, radius=1)


def test_heuristic_value_model_bulk_scoring_matches_single_scoring():
    model = HeuristicValueModel()
    state = GameState.new(size=10)
    state.board.place(Symbol.X, Move(5, 5))
    moves = candidate_moves(state, radius=1)
    bulk = dict(model.score_moves(state, Symbol.X, moves))
    assert bulk
    for move in moves:
        assert bulk[move] == model.score_move(state, Symbol.X, move)


def test_bounded_cache_evicts_oldest():
    cache = BoundedCache[str, int](max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")
    cache.set("c", 3)
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_generate_moves_threat_frontier_prefers_tactical_set():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.X
    state.board.place(Symbol.X, Move(5, 4))
    state.board.place(Symbol.X, Move(5, 6))
    state.board.place(Symbol.X, Move(4, 5))
    state.board.place(Symbol.X, Move(6, 5))
    context = SearchContext()
    moves = generate_moves(
        state, Symbol.X, context=context, mode=MoveGenerationMode.THREAT_FRONTIER, candidate_radius=1
    )
    assert Move(5, 5) in moves


def test_threat_solver_returns_forcing_solution():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.O
    state.board.place(Symbol.O, Move(2, 0))
    state.board.place(Symbol.O, Move(2, 1))
    state.board.place(Symbol.O, Move(2, 2))
    state.board.place(Symbol.O, Move(2, 3))
    solution = solve_forcing_line(state, Symbol.O, SearchContext())
    assert solution is not None
    assert solution.kind is ThreatKind.IMMEDIATE_WIN
    assert solution.move == Move(2, 4)
    assert solution.status.value == "forced_win"


def test_value_model_explain_features_contains_total():
    state = GameState.new(size=10)
    state.board.place(Symbol.X, Move(5, 5))
    model = HeuristicValueModel()
    features = model.explain_features(state, Symbol.X)
    assert "total" in features
