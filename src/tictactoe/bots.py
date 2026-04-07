from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from math import log, sqrt
from typing import Protocol

from .core import GameState, Move, Symbol
from .search.cache import BoundedCache
from .search.context import SearchContext
from .search.tactics import Threat, ThreatKind, clone_state, detect_threats
from .search.value_model import HeuristicValueModel, ValueModel


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


@dataclass
class _Node:
    state: GameState
    parent: "_Node | None"
    move: Move | None
    untried_moves: list[Move]
    children: list["_Node"]
    visits: int = 0
    total_value: float = 0.0

    @classmethod
    def from_state(
        cls,
        state: GameState,
        context: SearchContext,
        candidate_radius: int,
        parent: "_Node | None" = None,
        move: Move | None = None,
    ) -> "_Node":
        return cls(
            state=state,
            parent=parent,
            move=move,
            untried_moves=list(context.candidate_moves(state, radius=candidate_radius)),
            children=[],
        )


@dataclass
class MCTSBot:
    name: str = "mcts"
    simulations: int = 1200
    time_budget_ms: int = 300
    rollout_depth: int = 10
    candidate_radius: int = 1
    epsilon: float = 0.15
    exploration_constant: float = 1.41421356237
    value_model: ValueModel = field(default_factory=HeuristicValueModel)
    cache_size: int = 4096

    def choose_move(self, state: GameState, symbol: Symbol, rng: random.Random) -> Move:
        legal = state.board.legal_moves()
        if not legal:
            raise ValueError("No legal moves available")

        context = SearchContext()
        value_cache: BoundedCache[tuple[tuple[tuple[str, ...], ...], str], float] = BoundedCache(
            max_size=self.cache_size
        )

        tactical = self._pick_tactical_move(state, symbol, context=context)
        if tactical is not None:
            return tactical

        root = _Node.from_state(
            clone_state(state), context=context, candidate_radius=self.candidate_radius
        )
        deadline = time.perf_counter() + (self.time_budget_ms / 1000.0)
        iterations = 0

        while iterations < max(1, self.simulations) and time.perf_counter() < deadline:
            node = root

            while not node.untried_moves and node.children and not node.state.is_over:
                node = self._select_child(node, rng)

            if node.untried_moves and not node.state.is_over:
                widen_cap = max(1, int(sqrt(node.visits + 1)))
                if len(node.children) >= widen_cap:
                    node = self._select_child(node, rng)
                else:
                    move = rng.choice(node.untried_moves)
                    node.untried_moves.remove(move)
                    next_state = clone_state(node.state)
                    next_state.apply_move(move)
                    child = _Node.from_state(
                        next_state,
                        context=context,
                        candidate_radius=self.candidate_radius,
                        parent=node,
                        move=move,
                    )
                    node.children.append(child)
                    node = child

            value = self._rollout_value(node.state, symbol, rng, context, value_cache)
            while node is not None:
                node.visits += 1
                node.total_value += value
                node = node.parent
            iterations += 1

        if not root.children:
            return rng.choice(context.candidate_moves(state, radius=self.candidate_radius) or legal)

        best_visits = max(child.visits for child in root.children)
        candidates = [child for child in root.children if child.visits == best_visits]
        picked = rng.choice(candidates).move
        if picked is None:
            return rng.choice(legal)
        return picked

    def _rollout_value(
        self,
        state: GameState,
        root_symbol: Symbol,
        rng: random.Random,
        context: SearchContext,
        value_cache: BoundedCache[tuple[tuple[tuple[str, ...], ...], str], float],
    ) -> float:
        rollout_state = clone_state(state)
        depth = 0
        while not rollout_state.is_over and depth < self.rollout_depth:
            to_move = rollout_state.next_symbol
            tactical = self._pick_tactical_move(rollout_state, to_move, context=context)
            if tactical is not None:
                rollout_state.apply_move(tactical)
                depth += 1
                continue

            moves = context.candidate_moves(rollout_state, radius=self.candidate_radius)
            if not moves:
                break
            if rng.random() < self.epsilon:
                chosen = rng.choice(moves)
            else:
                scored = self.value_model.score_moves(rollout_state, to_move, moves)
                chosen = max(scored, key=lambda pair: pair[1])[0]
            rollout_state.apply_move(chosen)
            depth += 1

        if rollout_state.is_over:
            return _score_terminal_result(rollout_state.winner, root_symbol)
        cache_key = (rollout_state.state_key(), root_symbol.value)
        cached = value_cache.get(cache_key)
        if cached is not None:
            return cached
        value = self.value_model.evaluate(rollout_state, root_symbol)
        value_cache.set(cache_key, value)
        return value

    def _pick_tactical_move(self, state: GameState, symbol: Symbol, context: SearchContext) -> Move | None:
        own = detect_threats(state, symbol, context=context)
        opp = detect_threats(state, symbol.other(), context=context)
        for kind in (
            ThreatKind.IMMEDIATE_WIN,
            ThreatKind.OPEN_FOUR,
            ThreatKind.DOUBLE_THREE,
        ):
            own_move = _first_threat_move(own, kind)
            if own_move is not None:
                return own_move
            opp_move = _first_threat_move(opp, kind)
            if opp_move is not None:
                return opp_move
        return None

    def _select_child(self, node: _Node, rng: random.Random) -> _Node:
        if not node.children:
            raise ValueError("Cannot select child from empty node")
        scored: list[tuple[float, _Node]] = []
        parent_visits = max(1, node.visits)
        for child in node.children:
            if child.visits == 0:
                score = float("inf")
            else:
                exploit = child.total_value / child.visits
                explore = self.exploration_constant * sqrt(log(parent_visits) / child.visits)
                score = exploit + explore
            scored.append((score, child))
        best_score = max(score for score, _ in scored)
        candidates = [child for score, child in scored if score == best_score]
        return rng.choice(candidates)


@dataclass(frozen=True)
class _TTEntry:
    depth: int
    score: float
    flag: str
    best_move: Move | None


@dataclass
class AlphaBetaBot:
    name: str = "alphabeta"
    time_budget_ms: int = 500
    max_depth: int | None = None
    candidate_radius: int = 1
    threat_extension_depth: int = 1
    tt_max_size: int = 200_000
    value_model: ValueModel = field(default_factory=HeuristicValueModel)

    def choose_move(self, state: GameState, symbol: Symbol, rng: random.Random) -> Move:
        legal = state.board.legal_moves()
        if not legal:
            raise ValueError("No legal moves available")

        context = SearchContext()
        tactical = self._pick_tactical_move(state, symbol, context)
        if tactical is not None:
            return tactical

        deadline = time.perf_counter() + (self.time_budget_ms / 1000.0)
        tt: BoundedCache[tuple[tuple[tuple[str, ...], ...], str], _TTEntry] = BoundedCache(
            max_size=self.tt_max_size
        )
        max_depth = self.max_depth if self.max_depth is not None else 99

        best_move = legal[0]
        depth = 1
        while depth <= max_depth:
            if time.perf_counter() >= deadline:
                break
            alpha = -1_000_000_000.0
            beta = 1_000_000_000.0
            depth_best: Move | None = None
            depth_best_score = -1_000_000_000.0
            moves = self._ordered_moves(
                state=state, symbol=symbol, context=context, tt=tt, rng=rng
            )
            for move in moves:
                if time.perf_counter() >= deadline:
                    break
                child = state.fast_clone()
                child.apply_move(move)
                score = -self._negamax(
                    state=child,
                    depth=depth - 1,
                    alpha=-beta,
                    beta=-alpha,
                    root_symbol=symbol,
                    context=context,
                    tt=tt,
                    deadline=deadline,
                    rng=rng,
                )
                if score > depth_best_score:
                    depth_best_score = score
                    depth_best = move
                if score > alpha:
                    alpha = score
            if depth_best is not None:
                best_move = depth_best
            if time.perf_counter() >= deadline:
                break
            depth += 1

        return best_move

    def _negamax(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        root_symbol: Symbol,
        context: SearchContext,
        tt: BoundedCache[tuple[tuple[tuple[str, ...], ...], str], _TTEntry],
        deadline: float,
        rng: random.Random,
    ) -> float:
        if time.perf_counter() >= deadline:
            return self._evaluate(state, root_symbol)

        if state.is_over:
            return self._terminal_score(state, root_symbol)

        key = (state.state_key(), state.next_symbol.value)
        cached = tt.get(key)
        if cached is not None and cached.depth >= depth:
            if cached.flag == "exact":
                return cached.score
            if cached.flag == "lower":
                alpha = max(alpha, cached.score)
            elif cached.flag == "upper":
                beta = min(beta, cached.score)
            if alpha >= beta:
                return cached.score

        extension = 0
        if depth <= 0:
            threats = detect_threats(state, state.next_symbol, context=context)
            if any(t.kind in (ThreatKind.IMMEDIATE_WIN, ThreatKind.OPEN_FOUR, ThreatKind.DOUBLE_THREE) for t in threats):
                extension = self.threat_extension_depth
            else:
                return self._evaluate(state, root_symbol)

        effective_depth = max(0, depth + extension)
        if effective_depth == 0:
            return self._evaluate(state, root_symbol)

        original_alpha = alpha
        best_score = -1_000_000_000.0
        best_move: Move | None = None
        moves = self._ordered_moves(
            state=state,
            symbol=state.next_symbol,
            context=context,
            tt=tt,
            rng=rng,
        )
        if not moves:
            return self._evaluate(state, root_symbol)

        for move in moves:
            if time.perf_counter() >= deadline:
                break
            child = state.fast_clone()
            child.apply_move(move)
            score = -self._negamax(
                state=child,
                depth=effective_depth - 1,
                alpha=-beta,
                beta=-alpha,
                root_symbol=root_symbol,
                context=context,
                tt=tt,
                deadline=deadline,
                rng=rng,
            )
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        if best_score <= original_alpha:
            flag = "upper"
        elif best_score >= beta:
            flag = "lower"
        else:
            flag = "exact"
        tt.set(key, _TTEntry(depth=effective_depth, score=best_score, flag=flag, best_move=best_move))
        return best_score

    def _ordered_moves(
        self,
        state: GameState,
        symbol: Symbol,
        context: SearchContext,
        tt: BoundedCache[tuple[tuple[tuple[str, ...], ...], str], _TTEntry],
        rng: random.Random,
    ) -> list[Move]:
        candidates = context.candidate_moves(state, radius=self.candidate_radius)
        if not candidates:
            return []

        tactical_threats = detect_threats(state, symbol, candidates=candidates, context=context)
        tactical_map = {threat.move: threat.severity for threat in tactical_threats}
        scored = dict(self.value_model.score_moves(state, symbol, candidates))

        tt_move: Move | None = None
        tt_entry = tt.get((state.state_key(), state.next_symbol.value))
        if tt_entry is not None and tt_entry.best_move in candidates:
            tt_move = tt_entry.best_move

        # deterministic shuffle for equal-score move groups to keep seeded behavior stable.
        ranked = []
        for move in candidates:
            noise = rng.random() * 1e-6
            rank = (
                1 if tt_move == move else 0,
                tactical_map.get(move, 0),
                scored.get(move, -1.0),
                -noise,
            )
            ranked.append((rank, move))
        ranked.sort(reverse=True, key=lambda item: item[0])
        return [move for _, move in ranked]

    def _terminal_score(self, state: GameState, root_symbol: Symbol) -> float:
        if state.winner is None:
            return 0.0
        return 100000.0 if state.winner is root_symbol else -100000.0

    def _evaluate(self, state: GameState, root_symbol: Symbol) -> float:
        return self.value_model.evaluate(state, root_symbol) * 10_000.0

    def _pick_tactical_move(self, state: GameState, symbol: Symbol, context: SearchContext) -> Move | None:
        own = detect_threats(state, symbol, context=context)
        opp = detect_threats(state, symbol.other(), context=context)
        for kind in (ThreatKind.IMMEDIATE_WIN, ThreatKind.OPEN_FOUR, ThreatKind.DOUBLE_THREE):
            own_move = _first_threat_move(own, kind)
            if own_move is not None:
                return own_move
            opp_move = _first_threat_move(opp, kind)
            if opp_move is not None:
                return opp_move
        return None

def _score_terminal_result(winner: Symbol | None, root_symbol: Symbol) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner is root_symbol else -1.0


def _first_threat_move(threats: list[Threat], kind: ThreatKind) -> Move | None:
    for threat in threats:
        if threat.kind is kind:
            return threat.move
    return None
