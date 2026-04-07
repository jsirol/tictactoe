import pytest

from tictactoe.core import GameState, InvalidMove, Move, Symbol


def test_board_size_below_minimum_is_rejected():
    with pytest.raises(ValueError):
        GameState.new(size=9)


def test_turn_switches_after_valid_move():
    state = GameState.new(size=10)
    state.apply_move(Move(0, 0))
    assert state.next_symbol == Symbol.O


def test_rejects_out_of_bounds_move():
    state = GameState.new(size=10)
    with pytest.raises(InvalidMove):
        state.apply_move(Move(10, 0))


def test_rejects_occupied_cell():
    state = GameState.new(size=10)
    state.apply_move(Move(0, 0))
    with pytest.raises(InvalidMove):
        state.apply_move(Move(0, 0))


@pytest.mark.parametrize(
    ("moves", "last"),
    [
        ([Move(0, i) for i in range(5)], Move(0, 4)),
        ([Move(i, 0) for i in range(5)], Move(4, 0)),
        ([Move(i, i) for i in range(5)], Move(4, 4)),
        ([Move(i, 4 - i) for i in range(5)], Move(4, 0)),
    ],
)
def test_winning_streak_detected_in_all_directions(moves, last):
    state = GameState.new(size=10)
    for move in moves:
        state.board.place(Symbol.X, move)
    assert state.board.has_winning_streak(Symbol.X, last)


def test_fewer_than_five_is_not_a_win():
    state = GameState.new(size=10)
    for col in range(4):
        state.board.place(Symbol.X, Move(1, col))
    assert not state.board.has_winning_streak(Symbol.X, Move(1, 3))


def test_at_least_five_counts_as_win():
    state = GameState.new(size=10)
    for col in range(6):
        state.board.place(Symbol.X, Move(1, col))
    assert state.board.has_winning_streak(Symbol.X, Move(1, 5))


def test_game_state_sets_winner_on_winning_move():
    state = GameState.new(size=10)
    state.board.place(Symbol.X, Move(1, 0))
    state.board.place(Symbol.X, Move(1, 1))
    state.board.place(Symbol.X, Move(1, 2))
    state.board.place(Symbol.X, Move(1, 3))
    state.next_symbol = Symbol.X

    state.apply_move(Move(1, 4))
    assert state.winner == Symbol.X


def test_draw_when_board_full_and_no_winner():
    state = GameState.new(size=10)
    for row in range(10):
        for col in range(10):
            state.board.place(Symbol.X if (row + col) % 2 == 0 else Symbol.O, Move(row, col))
    assert state.board.is_full()
    assert state.winner is None
    assert state.is_draw


def test_fast_clone_copies_state_without_aliasing():
    state = GameState.new(size=10)
    state.apply_move(Move(1, 1))
    cloned = state.fast_clone()
    assert cloned.next_symbol == state.next_symbol
    assert cloned.board.cells == state.board.cells
    cloned.apply_move(Move(2, 2))
    assert state.board.cells[2][2] is None


def test_apply_move_for_plays_for_explicit_symbol():
    state = GameState.new(size=10)
    state.apply_move_for(Symbol.O, Move(0, 0))
    assert state.board.cells[0][0] is Symbol.O
    assert state.next_symbol is Symbol.X


def test_occupied_moves_returns_only_filled_cells():
    state = GameState.new(size=10)
    state.board.place(Symbol.X, Move(0, 0))
    state.board.place(Symbol.O, Move(2, 2))
    occupied = state.occupied_moves()
    assert set(occupied) == {Move(0, 0), Move(2, 2)}
