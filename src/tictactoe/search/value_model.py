from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tictactoe.core import GameState, Move, Symbol
from tictactoe.search.tactics import ThreatKind, clone_state, detect_threats


class ValueModel(Protocol):
    def evaluate(self, state: GameState, for_symbol: Symbol) -> float:
        """Return a normalized value estimate in [-1, 1]."""

    def score_move(self, state: GameState, symbol: Symbol, move: Move) -> float:
        """Score a candidate move for move ordering and rollouts."""

    def score_moves(
        self, state: GameState, symbol: Symbol, moves: list[Move]
    ) -> list[tuple[Move, float]]:
        """Score multiple candidate moves for ordering."""

    def explain_features(self, state: GameState, symbol: Symbol) -> dict[str, float]:
        """Return feature contributions for debugging and tuning."""


@dataclass(frozen=True)
class HeuristicValueModel:
    line_weight: float = 0.35
    center_weight: float = 0.05
    stone_weight: float = 0.03
    open_four_bonus: float = 0.55
    double_three_bonus: float = 0.25
    open_three_bonus: float = 0.08
    allow_opp_open_four_penalty: float = 0.45
    allow_opp_immediate_penalty: float = 0.9
    open_two_weight: float = 0.04
    open_three_weight: float = 0.12
    open_four_weight: float = 0.3

    def evaluate(self, state: GameState, for_symbol: Symbol) -> float:
        if state.winner is for_symbol:
            return 1.0
        if state.winner is for_symbol.other():
            return -1.0
        if state.is_draw:
            return 0.0

        our = self._feature_score(state, for_symbol)
        their = self._feature_score(state, for_symbol.other())
        raw = our["total"] - their["total"]
        return max(-1.0, min(1.0, raw))

    def score_move(self, state: GameState, symbol: Symbol, move: Move) -> float:
        trial = clone_state(state)
        trial.apply_move_for(symbol, move)
        score = self.evaluate(trial, symbol)

        for threat in detect_threats(state, symbol, candidates=[move]):
            if threat.kind is ThreatKind.IMMEDIATE_WIN:
                return 1.0
            if threat.kind is ThreatKind.OPEN_FOUR:
                score += self.open_four_bonus
            elif threat.kind is ThreatKind.DOUBLE_THREE:
                score += self.double_three_bonus
            elif threat.kind is ThreatKind.OPEN_THREE:
                score += self.open_three_bonus

        opponent_threats = detect_threats(trial, symbol.other())
        if any(threat.kind is ThreatKind.IMMEDIATE_WIN for threat in opponent_threats):
            score -= self.allow_opp_immediate_penalty
        elif any(threat.kind is ThreatKind.OPEN_FOUR for threat in opponent_threats):
            score -= self.allow_opp_open_four_penalty
        return max(-1.0, min(1.0, score))

    def score_moves(
        self, state: GameState, symbol: Symbol, moves: list[Move]
    ) -> list[tuple[Move, float]]:
        return [(move, self.score_move(state, symbol, move)) for move in moves]

    def explain_features(self, state: GameState, symbol: Symbol) -> dict[str, float]:
        return self._feature_score(state, symbol)

    def _feature_score(self, state: GameState, symbol: Symbol) -> dict[str, float]:
        stones = 0
        center_bonus = 0.0
        max_line = 0
        center = (state.board.size - 1) / 2.0
        for row in range(state.board.size):
            for col in range(state.board.size):
                if state.board.cells[row][col] is symbol:
                    stones += 1
                    center_bonus += (state.board.size - (abs(row - center) + abs(col - center)))
                    max_line = max(max_line, _max_line_from(state, symbol, Move(row, col)))

        if stones == 0:
            return {
                "line": 0.0,
                "center": 0.0,
                "stones": 0.0,
                "open_two": 0.0,
                "open_three": 0.0,
                "open_four": 0.0,
                "total": 0.0,
            }
        normalized_center = center_bonus / (state.board.size * stones * 2)
        line_term = (max_line / 5.0) ** 2
        stone_term = stones / (state.board.size * state.board.size)
        open_two = 0.0
        open_three = 0.0
        open_four = 0.0
        for threat in detect_threats(state, symbol):
            if threat.kind is ThreatKind.OPEN_THREE:
                open_three += self.open_three_weight
            elif threat.kind is ThreatKind.DOUBLE_THREE:
                open_three += self.open_three_weight * 2
            elif threat.kind is ThreatKind.OPEN_FOUR:
                open_four += self.open_four_weight
            elif threat.kind is ThreatKind.IMMEDIATE_WIN:
                open_four += self.open_four_weight * 2
        # Lightweight proxy for open two pressure from local line potential.
        open_two = max(0.0, (line_term - 0.1)) * self.open_two_weight

        line_score = self.line_weight * line_term
        center_score = self.center_weight * normalized_center
        stone_score = self.stone_weight * stone_term
        total = line_score + center_score + stone_score + open_two + open_three + open_four
        return {
            "line": line_score,
            "center": center_score,
            "stones": stone_score,
            "open_two": open_two,
            "open_three": open_three,
            "open_four": open_four,
            "total": total,
        }


def _max_line_from(state: GameState, symbol: Symbol, move: Move) -> int:
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    best = 1
    for dr, dc in directions:
        count = 1
        count += _count_direction(state, symbol, move, dr, dc)
        count += _count_direction(state, symbol, move, -dr, -dc)
        best = max(best, count)
    return best


def _count_direction(state: GameState, symbol: Symbol, start: Move, dr: int, dc: int) -> int:
    row = start.row + dr
    col = start.col + dc
    count = 0
    while 0 <= row < state.board.size and 0 <= col < state.board.size:
        if state.board.cells[row][col] is not symbol:
            break
        count += 1
        row += dr
        col += dc
    return count
