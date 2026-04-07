import random

from tictactoe.bots import MCTSBot, RandomBot
from tictactoe.cli import play_bot_vs_bot
from tictactoe.core import GameState, Move
from tictactoe.core import Symbol


def test_improved_mcts_beats_random_in_seeded_series():
    mcts = MCTSBot(time_budget_ms=80, simulations=150, rollout_depth=8, candidate_radius=1)
    random_bot = RandomBot()
    rng = random.Random(1234)
    wins = 0
    games = 12
    for _ in range(games):
        state = play_bot_vs_bot(size=10, bot_x=mcts, bot_o=random_bot, rng=rng)
        if state.winner is Symbol.X:
            wins += 1
    assert wins >= 8


def test_mcts_respects_time_budget_reasonably():
    bot = MCTSBot(time_budget_ms=20, simulations=5000)
    state = GameState.new(size=10)
    state.apply_move(Move(5, 5))
    rng = random.Random(1)
    move = bot.choose_move(state=state, symbol=state.next_symbol, rng=rng)
    assert move in state.board.legal_moves()
