from pathlib import Path

from tictactoe.puzzle_eval import evaluate_pack
from tictactoe.puzzles import load_puzzle_pack


def test_load_puzzle_pack_reads_entries():
    pack = load_puzzle_pack("data/puzzles/tuning_10x10.jsonl")
    assert len(pack) >= 3
    assert pack[0].size == 10
    assert pack[0].expected_moves


def test_load_puzzle_pack_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    try:
        load_puzzle_pack(path)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid JSON")


def test_evaluate_pack_is_deterministic_for_seed():
    a = evaluate_pack("data/puzzles/tuning_10x10.jsonl", bot_name="alphabeta", seed=7)
    b = evaluate_pack("data/puzzles/tuning_10x10.jsonl", bot_name="alphabeta", seed=7)
    assert a["solved"] == b["solved"]
    assert a["solve_rate"] == b["solve_rate"]
