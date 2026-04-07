from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

MIN_BOARD_SIZE = 10
WIN_STREAK = 5


class Symbol(StrEnum):
    X = "X"
    O = "O"

    def other(self) -> "Symbol":
        return Symbol.O if self is Symbol.X else Symbol.X


@dataclass(frozen=True)
class Move:
    row: int
    col: int


class InvalidMove(ValueError):
    pass


@dataclass
class Board:
    size: int
    cells: list[list[Symbol | None]] = field(init=False)

    def __post_init__(self) -> None:
        if self.size < MIN_BOARD_SIZE:
            raise ValueError(f"Board size must be >= {MIN_BOARD_SIZE}")
        self.cells = [[None for _ in range(self.size)] for _ in range(self.size)]

    def in_bounds(self, move: Move) -> bool:
        return 0 <= move.row < self.size and 0 <= move.col < self.size

    def get(self, move: Move) -> Symbol | None:
        if not self.in_bounds(move):
            raise InvalidMove(f"Move out of bounds: {move}")
        return self.cells[move.row][move.col]

    def place(self, symbol: Symbol, move: Move) -> None:
        if not self.in_bounds(move):
            raise InvalidMove(f"Move out of bounds: {move}")
        if self.cells[move.row][move.col] is not None:
            raise InvalidMove(f"Cell already occupied: {move}")
        self.cells[move.row][move.col] = symbol

    def legal_moves(self) -> list[Move]:
        moves: list[Move] = []
        for row in range(self.size):
            for col in range(self.size):
                if self.cells[row][col] is None:
                    moves.append(Move(row, col))
        return moves

    def is_full(self) -> bool:
        return all(cell is not None for row in self.cells for cell in row)

    def has_winning_streak(self, symbol: Symbol, move: Move) -> bool:
        if not self.in_bounds(move):
            return False
        if self.cells[move.row][move.col] != symbol:
            return False

        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for dr, dc in directions:
            count = 1
            count += self._count_direction(symbol, move, dr, dc)
            count += self._count_direction(symbol, move, -dr, -dc)
            if count >= WIN_STREAK:
                return True
        return False

    def _count_direction(self, symbol: Symbol, start: Move, dr: int, dc: int) -> int:
        row = start.row + dr
        col = start.col + dc
        count = 0
        while 0 <= row < self.size and 0 <= col < self.size:
            if self.cells[row][col] != symbol:
                break
            count += 1
            row += dr
            col += dc
        return count


@dataclass
class GameState:
    board: Board
    next_symbol: Symbol = Symbol.X
    winner: Symbol | None = None

    @classmethod
    def new(cls, size: int) -> "GameState":
        return cls(board=Board(size=size))

    @property
    def is_draw(self) -> bool:
        return self.winner is None and self.board.is_full()

    @property
    def is_over(self) -> bool:
        return self.winner is not None or self.is_draw

    def apply_move(self, move: Move) -> None:
        if self.is_over:
            raise InvalidMove("Game is already over")

        symbol = self.next_symbol
        self.board.place(symbol, move)
        if self.board.has_winning_streak(symbol, move):
            self.winner = symbol
            return
        self.next_symbol = symbol.other()
