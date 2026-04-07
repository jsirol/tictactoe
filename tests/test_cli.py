import pytest

from tictactoe.cli import get_bot, main, parse_move, run_simulation
from tictactoe.core import Move


def test_parse_move_accepts_two_zero_based_ints():
    assert parse_move("0 9") == Move(0, 9)


def test_parse_move_rejects_bad_input():
    for raw in ["", "1", "a b", "1,2", "1 2 3"]:
        try:
            parse_move(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for input: {raw!r}")


def test_run_simulation_returns_totals_for_all_games():
    result = run_simulation(size=10, games=5, seed=7)
    assert result["X"] + result["O"] + result["draw"] == 5


def test_run_simulation_supports_different_bots():
    result = run_simulation(size=10, games=3, seed=7, bot_x_name="mcts", bot_o_name="random")
    assert result["X"] + result["O"] + result["draw"] == 3


def test_main_simulate_subcommand_runs():
    code = main(["simulate", "--size", "10", "--games", "2", "--seed", "1"])
    assert code == 0


def test_main_simulate_subcommand_passes_bot_selection(monkeypatch):
    called = {}

    def fake_run(*, size, games, seed, bot_x_name, bot_o_name):
        called["args"] = (size, games, seed, bot_x_name, bot_o_name)
        return {"X": 1, "O": 0, "draw": 0}

    monkeypatch.setattr("tictactoe.cli.run_simulation", fake_run)
    code = main(
        ["simulate", "--size", "10", "--games", "1", "--seed", "4", "--bot-x", "mcts", "--bot-o", "random"]
    )
    assert code == 0
    assert called["args"] == (10, 1, 4, "mcts", "random")


def test_main_rejects_invalid_size():
    code = main(["simulate", "--size", "9"])
    assert code == 2


def test_get_bot_supports_mcts():
    bot = get_bot("mcts")
    assert bot.name == "mcts"


def test_get_bot_supports_alphabeta():
    bot = get_bot("alphabeta")
    assert bot.name == "alphabeta"


def test_get_bot_rejects_unknown():
    with pytest.raises(ValueError):
        get_bot("unknown")


def test_main_rejects_invalid_size_for_web():
    code = main(["web", "--size", "9"])
    assert code == 2


def test_main_web_subcommand_runs(monkeypatch):
    called = {}

    def fake_run(*, host, port, size, seed, bot_name):
        called["args"] = (host, port, size, seed, bot_name)
        return 0

    monkeypatch.setattr("tictactoe.cli.run_web_server", fake_run)
    code = main(
        ["web", "--host", "127.0.0.1", "--port", "9000", "--size", "10", "--seed", "2", "--bot", "random"]
    )
    assert code == 0
    assert called["args"] == ("127.0.0.1", 9000, 10, 2, "random")


def test_main_web_subcommand_defaults_to_mcts(monkeypatch):
    called = {}

    def fake_run(*, host, port, size, seed, bot_name):
        called["args"] = (host, port, size, seed, bot_name)
        return 0

    monkeypatch.setattr("tictactoe.cli.run_web_server", fake_run)
    code = main(["web"])
    assert code == 0
    assert called["args"] == ("127.0.0.1", 8000, 15, None, "mcts")
