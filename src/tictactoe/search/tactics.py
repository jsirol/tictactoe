from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tictactoe.core import GameState, Move, Symbol
from tictactoe.search.context import SearchContext
from tictactoe.search.move_policy import candidate_moves


class ThreatKind(StrEnum):
    IMMEDIATE_WIN = "immediate_win"
    OPEN_FOUR = "open_four"
    DOUBLE_THREE = "double_three"
    OPEN_THREE = "open_three"


@dataclass(frozen=True)
class Threat:
    kind: ThreatKind
    move: Move
    severity: int
    forcing: bool
    guaranteed: bool


_SEVERITY = {
    ThreatKind.IMMEDIATE_WIN: 100,
    ThreatKind.OPEN_FOUR: 80,
    ThreatKind.DOUBLE_THREE: 60,
    ThreatKind.OPEN_THREE: 40,
}


@dataclass(frozen=True)
class _LinePattern:
    run_len: int
    left_open: bool
    right_open: bool


def detect_threats(
    state: GameState,
    symbol: Symbol,
    candidates: list[Move] | None = None,
    context: SearchContext | None = None,
) -> list[Threat]:
    if candidates is None:
        if context is not None:
            candidates = context.candidate_moves(state, radius=1)
        else:
            candidates = candidate_moves(state, radius=1)

    found: list[Threat] = []
    for move in candidates:
        if state.board.cells[move.row][move.col] is not None:
            continue
        previous_next = state.next_symbol
        state.next_symbol = symbol
        token = state.make_move(move)

        if state.winner is symbol:
            found.append(_mk_threat(ThreatKind.IMMEDIATE_WIN, move, forcing=True, guaranteed=True))
            state.unmake_move(token)
            state.next_symbol = previous_next
            continue

        patterns = [_line_pattern(state, symbol, move, dr, dc) for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1))]
        if any(_is_open_four(pattern) for pattern in patterns):
            found.append(_mk_threat(ThreatKind.OPEN_FOUR, move, forcing=True, guaranteed=False))
            state.unmake_move(token)
            state.next_symbol = previous_next
            continue

        open_three_count = sum(1 for pattern in patterns if _is_open_three(pattern))
        if open_three_count >= 2:
            found.append(_mk_threat(ThreatKind.DOUBLE_THREE, move, forcing=True, guaranteed=False))
        elif open_three_count == 1:
            found.append(_mk_threat(ThreatKind.OPEN_THREE, move, forcing=False, guaranteed=False))
        state.unmake_move(token)
        state.next_symbol = previous_next

    return sorted(found, key=lambda t: (-t.severity, t.move.row, t.move.col))


def find_immediate_winning_move(
    state: GameState,
    symbol: Symbol,
    candidates: list[Move] | None = None,
    context: SearchContext | None = None,
) -> Move | None:
    for threat in detect_threats(state, symbol, candidates=candidates, context=context):
        if threat.kind is ThreatKind.IMMEDIATE_WIN:
            return threat.move
    return None


def find_forcing_threat_move(
    state: GameState,
    symbol: Symbol,
    min_streak: int = 4,
    candidates: list[Move] | None = None,
    context: SearchContext | None = None,
) -> Move | None:
    if min_streak == 4:
        for threat in detect_threats(state, symbol, candidates=candidates, context=context):
            if threat.kind is ThreatKind.OPEN_FOUR:
                return threat.move
        return None

    if candidates is None:
        if context is not None:
            candidates = context.candidate_moves(state, radius=1)
        else:
            candidates = candidate_moves(state, radius=1)

    best_move: Move | None = None
    best_streak = 0
    for move in candidates:
        if state.board.cells[move.row][move.col] is not None:
            continue
        previous_next = state.next_symbol
        state.next_symbol = symbol
        token = state.make_move(move)
        streak = _max_line_from_move(state, symbol, move)
        state.unmake_move(token)
        state.next_symbol = previous_next
        if streak >= min_streak and streak > best_streak:
            best_streak = streak
            best_move = move
    return best_move


def _max_line_from_move(state: GameState, symbol: Symbol, move: Move) -> int:
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    best = 1
    for dr, dc in directions:
        count = 1
        count += _count_direction(state, symbol, move, dr, dc)
        count += _count_direction(state, symbol, move, -dr, -dc)
        best = max(best, count)
    return best


def _line_pattern(state: GameState, symbol: Symbol, move: Move, dr: int, dc: int) -> _LinePattern:
    neg_len = _count_direction(state, symbol, move, -dr, -dc)
    pos_len = _count_direction(state, symbol, move, dr, dc)
    run_len = 1 + neg_len + pos_len
    left_end = Move(move.row - dr * (neg_len + 1), move.col - dc * (neg_len + 1))
    right_end = Move(move.row + dr * (pos_len + 1), move.col + dc * (pos_len + 1))
    left_open = state.board.in_bounds(left_end) and state.board.cells[left_end.row][left_end.col] is None
    right_open = state.board.in_bounds(right_end) and state.board.cells[right_end.row][right_end.col] is None
    return _LinePattern(run_len=run_len, left_open=left_open, right_open=right_open)


def _is_open_four(pattern: _LinePattern) -> bool:
    return pattern.run_len == 4 and pattern.left_open and pattern.right_open


def _is_open_three(pattern: _LinePattern) -> bool:
    return pattern.run_len == 3 and pattern.left_open and pattern.right_open


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


def _mk_threat(kind: ThreatKind, move: Move, forcing: bool, guaranteed: bool) -> Threat:
    return Threat(kind=kind, move=move, severity=_SEVERITY[kind], forcing=forcing, guaranteed=guaranteed)


def clone_state(state: GameState) -> GameState:
    return state.fast_clone()
