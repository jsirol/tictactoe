from tictactoe.cli import main, parse_move, run_simulation
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


def test_main_simulate_subcommand_runs():
    code = main(["simulate", "--size", "10", "--games", "2", "--seed", "1"])
    assert code == 0


def test_main_rejects_invalid_size():
    code = main(["simulate", "--size", "9"])
    assert code == 2


def test_main_rejects_invalid_size_for_web():
    code = main(["web", "--size", "9"])
    assert code == 2


def test_main_web_subcommand_runs(monkeypatch):
    called = {}

    def fake_run(*, host, port, size, seed):
        called["args"] = (host, port, size, seed)
        return 0

    monkeypatch.setattr("tictactoe.cli.run_web_server", fake_run)
    code = main(["web", "--host", "127.0.0.1", "--port", "9000", "--size", "10", "--seed", "2"])
    assert code == 0
    assert called["args"] == ("127.0.0.1", 9000, 10, 2)
