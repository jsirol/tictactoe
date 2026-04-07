from __future__ import annotations

from enum import StrEnum

from tictactoe.core import GameState, Move, Symbol
from tictactoe.search.context import SearchContext
from tictactoe.search.tactics import ThreatKind, detect_threats


class MoveGenerationMode(StrEnum):
    FULL_LEGAL = "full_legal"
    FRONTIER = "frontier"
    THREAT_FRONTIER = "threat_frontier"


def generate_moves(
    state: GameState,
    symbol: Symbol,
    context: SearchContext,
    mode: MoveGenerationMode,
    candidate_radius: int = 1,
) -> list[Move]:
    if mode is MoveGenerationMode.FULL_LEGAL:
        return state.board.legal_moves()
    if mode is MoveGenerationMode.FRONTIER:
        return context.candidate_moves(state, radius=candidate_radius)

    frontier = context.candidate_moves(state, radius=candidate_radius)
    if not frontier:
        return []

    threats = detect_threats(state, symbol, candidates=frontier, context=context)
    opp_threats = detect_threats(state, symbol.other(), candidates=frontier, context=context)
    tactical = {
        threat.move
        for threat in threats + opp_threats
        if threat.kind in (ThreatKind.IMMEDIATE_WIN, ThreatKind.OPEN_FOUR, ThreatKind.DOUBLE_THREE)
    }
    if tactical:
        return sorted(tactical, key=lambda m: (m.row, m.col))
    return frontier
