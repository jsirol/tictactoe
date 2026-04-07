from __future__ import annotations

from dataclasses import dataclass, field

from tictactoe.core import GameState, Move
from tictactoe.search.move_policy import candidate_moves


@dataclass
class SearchContext:
    occupied_cache: dict[tuple[tuple[str, ...], ...], list[Move]] = field(default_factory=dict)
    candidate_cache: dict[tuple[tuple[tuple[str, ...], ...], int], list[Move]] = field(
        default_factory=dict
    )

    def occupied_moves(self, state: GameState) -> list[Move]:
        state_key = state.state_key()
        cached = self.occupied_cache.get(state_key)
        if cached is not None:
            return cached
        occupied = state.occupied_moves()
        self.occupied_cache[state_key] = occupied
        return occupied

    def candidate_moves(self, state: GameState, radius: int) -> list[Move]:
        state_key = state.state_key()
        cache_key = (state_key, radius)
        cached = self.candidate_cache.get(cache_key)
        if cached is not None:
            return cached

        occupied = self.occupied_moves(state)
        moves = candidate_moves(state, radius=radius, occupied=occupied)
        self.candidate_cache[cache_key] = moves
        return moves
