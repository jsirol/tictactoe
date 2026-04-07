from __future__ import annotations

import random
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


@dataclass(frozen=True)
class UndoToken:
    move: Move
    previous_next_symbol: Symbol
    previous_winner: Symbol | None


_ZOBRIST_CACHE: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {}
_ZOBRIST_TURN_X = 0x9E3779B185EBCA87


def _zobrist_table(size: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    cached = _ZOBRIST_CACHE.get(size)
    if cached is not None:
        return cached
    rng = random.Random(0xC0DE + size * 17)
    table = tuple(
        tuple((rng.getrandbits(64), rng.getrandbits(64)) for _ in range(size)) for _ in range(size)
    )
    _ZOBRIST_CACHE[size] = table
    return table


@dataclass
class Board:
    size: int
    cells: list[list[Symbol | None]] = field(init=False)
    occupied_count: int = field(init=False, default=0)
    _hash: int = field(init=False, default=0)
    _zobrist: tuple[tuple[tuple[int, int], ...], ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.size < MIN_BOARD_SIZE:
            raise ValueError(f"Board size must be >= {MIN_BOARD_SIZE}")
        self.cells = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.occupied_count = 0
        self._hash = 0
        self._zobrist = _zobrist_table(self.size)

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
        self.occupied_count += 1
        self._hash ^= self._cell_hash(symbol, move)

    def remove(self, move: Move) -> Symbol:
        if not self.in_bounds(move):
            raise InvalidMove(f"Move out of bounds: {move}")
        symbol = self.cells[move.row][move.col]
        if symbol is None:
            raise InvalidMove(f"Cannot remove empty cell: {move}")
        self.cells[move.row][move.col] = None
        self.occupied_count -= 1
        self._hash ^= self._cell_hash(symbol, move)
        return symbol

    def legal_moves(self) -> list[Move]:
        moves: list[Move] = []
        for row in range(self.size):
            for col in range(self.size):
                if self.cells[row][col] is None:
                    moves.append(Move(row, col))
        return moves

    def occupied_moves(self) -> list[Move]:
        moves: list[Move] = []
        for row in range(self.size):
            for col in range(self.size):
                if self.cells[row][col] is not None:
                    moves.append(Move(row, col))
        return moves

    def is_full(self) -> bool:
        return self.occupied_count == self.size * self.size

    def board_key(self) -> int:
        return self._hash

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

    def _cell_hash(self, symbol: Symbol, move: Move) -> int:
        x_hash, o_hash = self._zobrist[move.row][move.col]
        return x_hash if symbol is Symbol.X else o_hash


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
        self.make_move(move)

    def make_move(self, move: Move) -> UndoToken:
        if self.is_over:
            raise InvalidMove("Game is already over")

        symbol = self.next_symbol
        token = UndoToken(move=move, previous_next_symbol=self.next_symbol, previous_winner=self.winner)
        self.board.place(symbol, move)
        if self.board.has_winning_streak(symbol, move):
            self.winner = symbol
        else:
            self.next_symbol = symbol.other()
        return token

    def unmake_move(self, token: UndoToken) -> None:
        move = token.move
        if not self.board.in_bounds(move):
            raise InvalidMove(f"Move out of bounds: {move}")
        self.board.remove(move)
        self.next_symbol = token.previous_next_symbol
        self.winner = token.previous_winner

    def apply_move_for(self, symbol: Symbol, move: Move) -> None:
        self.next_symbol = symbol
        self.make_move(move)

    def occupied_moves(self) -> list[Move]:
        return self.board.occupied_moves()

    def fast_clone(self) -> "GameState":
        cloned = GameState.new(size=self.board.size)
        cloned.next_symbol = self.next_symbol
        cloned.winner = self.winner
        cloned.board.cells = [row[:] for row in self.board.cells]
        cloned.board.occupied_count = self.board.occupied_count
        cloned.board._hash = self.board._hash
        return cloned

    def state_key(self) -> int:
        turn_hash = _ZOBRIST_TURN_X if self.next_symbol is Symbol.X else 0
        return self.board.board_key() ^ turn_hash
