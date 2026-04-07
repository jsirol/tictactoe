from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tictactoe.core import GameState, Move, Symbol
from tictactoe.search.context import SearchContext
from tictactoe.search.move_generator import MoveGenerationMode, generate_moves
from tictactoe.search.tactics import ThreatKind, detect_threats


class ThreatSolutionStatus(StrEnum):
    FORCED_WIN = "forced_win"
    FORCED_DEFENSE = "forced_defense"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ThreatSolution:
    move: Move
    kind: ThreatKind
    forcing: bool
    guaranteed: bool
    status: ThreatSolutionStatus
    line: tuple[Move, ...] = ()


def solve_forcing_line(
    state: GameState,
    symbol: Symbol,
    context: SearchContext,
    max_ply: int = 4,
) -> ThreatSolution | None:
    own = detect_threats(state, symbol, context=context)
    opp = detect_threats(state, symbol.other(), context=context)
    unresolved_candidate: ThreatSolution | None = None
    for kind in (ThreatKind.IMMEDIATE_WIN, ThreatKind.OPEN_FOUR, ThreatKind.DOUBLE_THREE):
        own_move = _first(own, kind, status=ThreatSolutionStatus.FORCED_WIN)
        if own_move is not None:
            if unresolved_candidate is None:
                unresolved_candidate = ThreatSolution(
                    move=own_move.move,
                    kind=own_move.kind,
                    forcing=own_move.forcing,
                    guaranteed=False,
                    status=ThreatSolutionStatus.UNRESOLVED,
                    line=(own_move.move,),
                )
            result = _search_forcing(
                state=state.fast_clone(),
                attacker=symbol,
                to_move=symbol,
                first_move=own_move.move,
                path=(own_move.move,),
                ply_left=max_ply,
                context=context,
            )
            if result is not None:
                return result
        opp_move = _first(opp, kind, status=ThreatSolutionStatus.FORCED_DEFENSE)
        if opp_move is not None:
            return opp_move
    return unresolved_candidate


def _search_forcing(
    state: GameState,
    attacker: Symbol,
    to_move: Symbol,
    first_move: Move,
    path: tuple[Move, ...],
    ply_left: int,
    context: SearchContext,
) -> ThreatSolution | None:
    if ply_left <= 0:
        return None

    threats = detect_threats(state, to_move, context=context)
    forcing = [t for t in threats if t.kind in (ThreatKind.IMMEDIATE_WIN, ThreatKind.OPEN_FOUR, ThreatKind.DOUBLE_THREE)]
    if not forcing:
        return None

    if to_move is attacker:
        for threat in forcing:
            child = state.fast_clone()
            child.apply_move_for(to_move, threat.move)
            next_path = path + (threat.move,)
            if child.winner is attacker:
                return ThreatSolution(
                    move=first_move,
                    kind=ThreatKind.IMMEDIATE_WIN if threat.kind is ThreatKind.IMMEDIATE_WIN else threat.kind,
                    forcing=True,
                    guaranteed=threat.kind is ThreatKind.IMMEDIATE_WIN,
                    status=ThreatSolutionStatus.FORCED_WIN,
                    line=next_path,
                )
            continuation = _search_forcing(
                state=child,
                attacker=attacker,
                to_move=to_move.other(),
                first_move=first_move,
                path=next_path,
                ply_left=ply_left - 1,
                context=context,
            )
            if continuation is not None:
                return continuation
        return None

    # Defender turn: all tactical refutations must fail for attack to be forced.
    refutations = _defender_refutations(state, to_move, context=context)
    if not refutations:
        return ThreatSolution(
            move=first_move,
            kind=ThreatKind.DOUBLE_THREE,
            forcing=True,
            guaranteed=False,
            status=ThreatSolutionStatus.FORCED_WIN,
            line=path,
        )

    for defense in refutations:
        child = state.fast_clone()
        child.apply_move_for(to_move, defense)
        continuation = _search_forcing(
            state=child,
            attacker=attacker,
            to_move=to_move.other(),
            first_move=first_move,
            path=path + (defense,),
            ply_left=ply_left - 1,
            context=context,
        )
        if continuation is None:
            return None
    return ThreatSolution(
        move=first_move,
        kind=ThreatKind.DOUBLE_THREE,
        forcing=True,
        guaranteed=False,
        status=ThreatSolutionStatus.FORCED_WIN,
        line=path,
    )


def _defender_refutations(state: GameState, defender: Symbol, context: SearchContext) -> list[Move]:
    # Restrict defender search to tactical frontier to keep solver fast.
    moves = generate_moves(
        state,
        defender,
        context=context,
        mode=MoveGenerationMode.THREAT_FRONTIER,
        candidate_radius=1,
    )
    if moves:
        return moves
    return context.candidate_moves(state, radius=1)


def _first(threats: list, kind: ThreatKind, status: ThreatSolutionStatus) -> ThreatSolution | None:
    for threat in threats:
        if threat.kind is kind:
            return ThreatSolution(
                move=threat.move,
                kind=threat.kind,
                forcing=threat.forcing,
                guaranteed=threat.guaranteed,
                status=status,
                line=(threat.move,),
            )
    return None
