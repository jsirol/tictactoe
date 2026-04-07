import random

from tictactoe.bots import AlphaBetaBot
from tictactoe.core import GameState, Move, Symbol
from tictactoe.search.context import SearchContext
from tictactoe.search.threat_solver import ThreatSolutionStatus, solve_forcing_line


def test_threat_solver_finds_forced_defense():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.O
    state.board.place(Symbol.X, Move(1, 1))
    state.board.place(Symbol.X, Move(1, 2))
    state.board.place(Symbol.X, Move(1, 3))
    state.board.place(Symbol.X, Move(1, 4))
    solution = solve_forcing_line(state, Symbol.O, SearchContext(), max_ply=4)
    assert solution is not None
    assert solution.status is ThreatSolutionStatus.FORCED_DEFENSE
    assert solution.move in {Move(1, 0), Move(1, 5)}


def test_threat_solver_finds_forced_win():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.O
    state.board.place(Symbol.O, Move(3, 0))
    state.board.place(Symbol.O, Move(3, 1))
    state.board.place(Symbol.O, Move(3, 2))
    state.board.place(Symbol.O, Move(3, 3))
    solution = solve_forcing_line(state, Symbol.O, SearchContext(), max_ply=4)
    assert solution is not None
    assert solution.status is ThreatSolutionStatus.FORCED_WIN
    assert solution.move == Move(3, 4)
    assert solution.line


def test_threat_solver_can_return_unresolved():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.X
    state.board.place(Symbol.X, Move(5, 4))
    state.board.place(Symbol.X, Move(5, 6))
    state.board.place(Symbol.X, Move(4, 5))
    state.board.place(Symbol.X, Move(6, 5))
    solution = solve_forcing_line(state, Symbol.X, SearchContext(), max_ply=1)
    assert solution is not None
    assert solution.status in {ThreatSolutionStatus.UNRESOLVED, ThreatSolutionStatus.FORCED_WIN}


def test_alphabeta_solves_tactical_defense_puzzle():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.O
    state.board.place(Symbol.X, Move(2, 2))
    state.board.place(Symbol.X, Move(2, 3))
    state.board.place(Symbol.X, Move(2, 4))
    state.board.place(Symbol.X, Move(2, 5))

    bot = AlphaBetaBot(time_budget_ms=60, max_depth=2)
    move = bot.choose_move(state=state, symbol=Symbol.O, rng=random.Random(42))
    assert move in {Move(2, 1), Move(2, 6)}


def test_alphabeta_solves_tactical_attack_puzzle():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.X
    state.board.place(Symbol.X, Move(5, 1))
    state.board.place(Symbol.X, Move(5, 2))
    state.board.place(Symbol.X, Move(5, 3))
    state.board.place(Symbol.X, Move(5, 4))

    bot = AlphaBetaBot(time_budget_ms=60, max_depth=2)
    move = bot.choose_move(state=state, symbol=Symbol.X, rng=random.Random(7))
    assert move in {Move(5, 0), Move(5, 5)}
