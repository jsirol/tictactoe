from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tictactoe.core import GameState, Move, Symbol
from tictactoe.search.value_model import ValueModel


class PolicyValueModel(Protocol):
    def predict(
        self, state: GameState, symbol: Symbol, candidate_moves: list[Move]
    ) -> tuple[dict[Move, float], float]:
        """Return move priors and value in [-1, 1] from `symbol` perspective."""

    def predict_batch(
        self, items: list[tuple[GameState, Symbol, list[Move]]]
    ) -> list[tuple[dict[Move, float], float]]:
        """Batch prediction variant for self-play workers/batchers."""


@dataclass(frozen=True)
class HeuristicPolicyValueModel:
    value_model: ValueModel

    def predict(
        self, state: GameState, symbol: Symbol, candidate_moves: list[Move]
    ) -> tuple[dict[Move, float], float]:
        if not candidate_moves:
            return {}, self.value_model.evaluate(state, symbol)
        scored = self.value_model.score_moves(state, symbol, candidate_moves)
        probs = _softmax_over_moves(scored)
        value = self.value_model.evaluate(state, symbol)
        return probs, value

    def predict_batch(
        self, items: list[tuple[GameState, Symbol, list[Move]]]
    ) -> list[tuple[dict[Move, float], float]]:
        return [self.predict(state, symbol, moves) for state, symbol, moves in items]


def _softmax_over_moves(scored: list[tuple[Move, float]]) -> dict[Move, float]:
    if not scored:
        return {}
    best = max(score for _, score in scored)
    exps: list[tuple[Move, float]] = []
    denom = 0.0
    for move, score in scored:
        # Fast, stable pseudo-softmax shape for move priors.
        value = 2.718281828459045 ** (score - best)
        exps.append((move, value))
        denom += value
    if denom <= 0.0:
        uniform = 1.0 / len(scored)
        return {move: uniform for move, _ in scored}
    return {move: value / denom for move, value in exps}
