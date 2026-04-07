import random

import pytest

from tictactoe.bots import RandomBot
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
