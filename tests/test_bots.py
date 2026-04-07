import random

import pytest

from tictactoe.bots import AlphaBetaBot, MCTSBot, RandomBot
from tictactoe.core import GameState, Move, Symbol


def test_random_bot_returns_legal_move():
    state = GameState.new(size=10)
    state.apply_move(Move(0, 0))
    bot = RandomBot()
    move = bot.choose_move(state=state, symbol=Symbol.O, rng=random.Random(1))
    assert move in state.board.legal_moves()


def test_random_bot_is_deterministic_with_same_seed():
    state = GameState.new(size=10)
    state.apply_move(Move(0, 0))
    bot = RandomBot()
    rng_a = random.Random(123)
    rng_b = random.Random(123)
    assert bot.choose_move(state, Symbol.O, rng_a) == bot.choose_move(state, Symbol.O, rng_b)


def test_random_bot_raises_when_no_legal_moves():
    state = GameState.new(size=10)
    for row in range(10):
        for col in range(10):
            state.board.place(Symbol.X, Move(row, col))
    bot = RandomBot()
    with pytest.raises(ValueError):
        bot.choose_move(state=state, symbol=Symbol.O, rng=random.Random(1))


def test_mcts_bot_returns_legal_move():
    state = GameState.new(size=10)
    state.apply_move(Move(0, 0))
    bot = MCTSBot(simulations=40)
    move = bot.choose_move(state=state, symbol=Symbol.O, rng=random.Random(1))
    assert move in state.board.legal_moves()


def test_mcts_bot_is_deterministic_with_same_seed():
    state = GameState.new(size=10)
    state.apply_move(Move(0, 0))
    bot = MCTSBot(simulations=50, time_budget_ms=10_000)
    rng_a = random.Random(123)
    rng_b = random.Random(123)
    assert bot.choose_move(state, Symbol.O, rng_a) == bot.choose_move(state, Symbol.O, rng_b)


def test_mcts_bot_prefers_immediate_winning_move():
    state = GameState.new(size=10)
    state.board.place(Symbol.O, Move(2, 0))
    state.board.place(Symbol.O, Move(2, 1))
    state.board.place(Symbol.O, Move(2, 2))
    state.board.place(Symbol.O, Move(2, 3))
    state.next_symbol = Symbol.O
    bot = MCTSBot(simulations=10)
    assert bot.choose_move(state=state, symbol=Symbol.O, rng=random.Random(5)) == Move(2, 4)


def test_mcts_bot_blocks_diagonal_forcing_four_when_no_immediate_win():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.O
    state.board.place(Symbol.X, Move(1, 1))
    state.board.place(Symbol.X, Move(2, 2))
    state.board.place(Symbol.X, Move(3, 3))
    state.board.place(Symbol.O, Move(0, 9))
    state.board.place(Symbol.O, Move(9, 0))

    bot = MCTSBot(simulations=20)
    assert bot.choose_move(state=state, symbol=Symbol.O, rng=random.Random(7)) == Move(4, 4)


def test_mcts_bot_plays_own_double_three_before_search():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.O
    state.board.place(Symbol.O, Move(5, 4))
    state.board.place(Symbol.O, Move(5, 6))
    state.board.place(Symbol.O, Move(4, 5))
    state.board.place(Symbol.O, Move(6, 5))

    bot = MCTSBot(simulations=20)
    assert bot.choose_move(state=state, symbol=Symbol.O, rng=random.Random(11)) == Move(5, 5)


def test_mcts_bot_blocks_opponent_double_three_when_no_higher_threat():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.O
    state.board.place(Symbol.X, Move(5, 4))
    state.board.place(Symbol.X, Move(5, 6))
    state.board.place(Symbol.X, Move(4, 5))
    state.board.place(Symbol.X, Move(6, 5))
    state.board.place(Symbol.O, Move(0, 0))

    bot = MCTSBot(simulations=20)
    assert bot.choose_move(state=state, symbol=Symbol.O, rng=random.Random(12)) == Move(5, 5)


def test_mcts_prioritizes_open_four_over_blocking_opponent_double_three():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.O
    state.board.place(Symbol.O, Move(2, 1))
    state.board.place(Symbol.O, Move(2, 2))
    state.board.place(Symbol.O, Move(2, 3))
    state.board.place(Symbol.X, Move(5, 4))
    state.board.place(Symbol.X, Move(5, 6))
    state.board.place(Symbol.X, Move(4, 5))
    state.board.place(Symbol.X, Move(6, 5))

    bot = MCTSBot(simulations=20)
    assert bot.choose_move(state=state, symbol=Symbol.O, rng=random.Random(13)) == Move(2, 4)


def test_alphabeta_bot_returns_legal_move():
    state = GameState.new(size=10)
    state.apply_move(Move(0, 0))
    bot = AlphaBetaBot(time_budget_ms=30, max_depth=2)
    move = bot.choose_move(state=state, symbol=Symbol.O, rng=random.Random(1))
    assert move in state.board.legal_moves()


def test_alphabeta_bot_is_deterministic_with_same_seed():
    state = GameState.new(size=10)
    state.apply_move(Move(0, 0))
    bot = AlphaBetaBot(time_budget_ms=10_000, max_depth=2)
    rng_a = random.Random(99)
    rng_b = random.Random(99)
    assert bot.choose_move(state, Symbol.O, rng_a) == bot.choose_move(state, Symbol.O, rng_b)


def test_alphabeta_bot_prefers_immediate_winning_move():
    state = GameState.new(size=10)
    state.board.place(Symbol.O, Move(2, 0))
    state.board.place(Symbol.O, Move(2, 1))
    state.board.place(Symbol.O, Move(2, 2))
    state.board.place(Symbol.O, Move(2, 3))
    state.next_symbol = Symbol.O
    bot = AlphaBetaBot(time_budget_ms=30, max_depth=2)
    assert bot.choose_move(state=state, symbol=Symbol.O, rng=random.Random(5)) == Move(2, 4)


def test_alphabeta_bot_blocks_immediate_win():
    state = GameState.new(size=10)
    state.next_symbol = Symbol.O
    state.board.place(Symbol.X, Move(1, 1))
    state.board.place(Symbol.X, Move(1, 2))
    state.board.place(Symbol.X, Move(1, 3))
    state.board.place(Symbol.X, Move(1, 4))
    bot = AlphaBetaBot(time_budget_ms=30, max_depth=2)
    assert bot.choose_move(state=state, symbol=Symbol.O, rng=random.Random(2)) in {Move(1, 0), Move(1, 5)}


def test_alphabeta_exposes_search_stats():
    state = GameState.new(size=10)
    state.apply_move(Move(5, 5))
    bot = AlphaBetaBot(time_budget_ms=30, max_depth=2)
    bot.choose_move(state=state, symbol=state.next_symbol, rng=random.Random(5))
    assert bot.last_stats.depth_reached >= 1
