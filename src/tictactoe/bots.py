from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from .core import GameState, Move, Symbol


class Bot(Protocol):
    name: str

    def choose_move(self, state: GameState, symbol: Symbol, rng: random.Random) -> Move:
        """Return the next move for a symbol."""


@dataclass
class RandomBot:
    name: str = "random"

    def choose_move(self, state: GameState, symbol: Symbol, rng: random.Random) -> Move:
        legal = state.board.legal_moves()
        if not legal:
            raise ValueError("No legal moves available")
        return rng.choice(legal)
