from __future__ import annotations

from dataclasses import dataclass

from tictactoe.core import GameState, Move, Symbol
from tictactoe.search.context import SearchContext
from tictactoe.search.tactics import ThreatKind, detect_threats


@dataclass(frozen=True)
class ThreatSolution:
    move: Move
    kind: ThreatKind
    forcing: bool
    guaranteed: bool


def solve_forcing_line(
    state: GameState,
    symbol: Symbol,
    context: SearchContext,
) -> ThreatSolution | None:
    own = detect_threats(state, symbol, context=context)
    opp = detect_threats(state, symbol.other(), context=context)
    for kind in (ThreatKind.IMMEDIATE_WIN, ThreatKind.OPEN_FOUR, ThreatKind.DOUBLE_THREE):
        own_move = _first(own, kind)
        if own_move is not None:
            return own_move
        opp_move = _first(opp, kind)
        if opp_move is not None:
            return opp_move
    return None


def _first(threats: list, kind: ThreatKind) -> ThreatSolution | None:
    for threat in threats:
        if threat.kind is kind:
            return ThreatSolution(
                move=threat.move,
                kind=threat.kind,
                forcing=threat.forcing,
                guaranteed=threat.guaranteed,
            )
    return None
