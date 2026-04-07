from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from math import sqrt
from typing import Protocol

from .core import GameState, Move, Symbol
from .search.cache import BoundedCache
from .search.context import SearchContext
from .search.move_generator import MoveGenerationMode, generate_moves
from .search.policy_value import HeuristicPolicyValueModel, PolicyValueModel
from .search.tactics import Threat, ThreatKind, clone_state, detect_threats
from .search.threat_solver import ThreatSolutionStatus, solve_forcing_line
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
    prior: float
    unexpanded_moves: list[Move]
    children: dict[Move, "_Node"]
    policy_priors: dict[Move, float] = field(default_factory=dict)
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
            prior=1.0,
            unexpanded_moves=list(context.candidate_moves(state, radius=candidate_radius)),
            children={},
            policy_priors={},
        )


@dataclass
class MCTSBot:
    name: str = "mcts"
    simulations: int = 1200
    time_budget_ms: int = 300
    candidate_radius: int = 1
    rollout_depth: int = 0
    epsilon: float = 0.0
    exploration_constant: float = 1.41421356237
    temperature: float = 0.0
    dirichlet_alpha: float = 0.3
    root_noise_fraction: float = 0.0
    value_model: ValueModel = field(default_factory=HeuristicValueModel)
    policy_value_model: PolicyValueModel | None = None
    cache_size: int = 4096
    last_policy: dict[Move, float] = field(default_factory=dict, init=False)

    def choose_move(self, state: GameState, symbol: Symbol, rng: random.Random) -> Move:
        legal = state.board.legal_moves()
        if not legal:
            raise ValueError("No legal moves available")

        context = SearchContext()

        tactical = self._pick_tactical_move(state, symbol, context=context)
        if tactical is not None:
            return tactical

        root = _Node.from_state(
            clone_state(state), context=context, candidate_radius=self.candidate_radius
        )
        deadline = time.perf_counter() + (self.time_budget_ms / 1000.0)
        iterations = 0
        pv_model = self.policy_value_model or HeuristicPolicyValueModel(self.value_model)

        while iterations < max(1, self.simulations) and time.perf_counter() < deadline:
            node = root
            path: list[_Node] = [root]
            leaf_value: float | None = None

            while not node.state.is_over and not node.unexpanded_moves and node.children:
                node = self._select_child(node, rng)
                path.append(node)

            if node.state.is_over:
                leaf_value = _score_terminal_result(node.state.winner, symbol)
            else:
                if not node.policy_priors:
                    candidates = context.candidate_moves(node.state, radius=self.candidate_radius)
                    priors, value = pv_model.predict(node.state, symbol, candidates)
                    if node.parent is None and priors and self.root_noise_fraction > 0.0:
                        priors = self._apply_root_noise(priors, rng)
                    node.policy_priors = priors
                    if not node.unexpanded_moves:
                        node.unexpanded_moves = list(candidates)
                    leaf_value = value

                if node.unexpanded_moves:
                    move = rng.choice(node.unexpanded_moves)
                    node.unexpanded_moves.remove(move)
                    next_state = clone_state(node.state)
                    next_state.apply_move(move)
                    child = _Node.from_state(
                        next_state,
                        context=context,
                        candidate_radius=self.candidate_radius,
                        parent=node,
                        move=move,
                    )
                    child.prior = node.policy_priors.get(move, 0.0)
                    node.children[move] = child
                    node = child
                    path.append(child)
                    if node.state.is_over:
                        leaf_value = _score_terminal_result(node.state.winner, symbol)
                    else:
                        _, leaf_value = pv_model.predict(
                            node.state, symbol, context.candidate_moves(node.state, radius=self.candidate_radius)
                        )

            for seen in path:
                seen.visits += 1
                seen.total_value += leaf_value if leaf_value is not None else 0.0
            iterations += 1

        if not root.children:
            return rng.choice(context.candidate_moves(state, radius=self.candidate_radius) or legal)

        children = list(root.children.values())
        best_visits = max(child.visits for child in children)
        total_visits = sum(max(1, child.visits) for child in children)
        self.last_policy = {
            child.move: (max(1, child.visits) / total_visits) for child in children if child.move is not None
        }
        if self.temperature > 0.0:
            picked = self._sample_temperature(children, rng)
        else:
            candidates = [child for child in children if child.visits == best_visits]
            picked = rng.choice(candidates).move
        if picked is None:
            return rng.choice(legal)
        return picked

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
        for child in node.children.values():
            exploit = 0.0 if child.visits == 0 else child.total_value / child.visits
            explore = self.exploration_constant * child.prior * sqrt(parent_visits) / (1 + child.visits)
            score = exploit + explore
            scored.append((score, child))
        best_score = max(score for score, _ in scored)
        candidates = [child for score, child in scored if score == best_score]
        return rng.choice(candidates)

    def _apply_root_noise(self, priors: dict[Move, float], rng: random.Random) -> dict[Move, float]:
        moves = list(priors.keys())
        gamma = [rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in moves]
        total = sum(gamma)
        if total <= 0.0:
            return priors
        mixed: dict[Move, float] = {}
        for idx, move in enumerate(moves):
            noise = gamma[idx] / total
            mixed[move] = (1.0 - self.root_noise_fraction) * priors[move] + self.root_noise_fraction * noise
        return mixed

    def _sample_temperature(self, children: list[_Node], rng: random.Random) -> Move | None:
        weights: list[float] = []
        for child in children:
            visits = max(1, child.visits)
            if self.temperature == 1.0:
                weight = float(visits)
            else:
                weight = visits ** (1.0 / self.temperature)
            weights.append(weight)
        total = sum(weights)
        if total <= 0.0:
            return rng.choice(children).move
        pick = rng.random() * total
        seen = 0.0
        for idx, child in enumerate(children):
            seen += weights[idx]
            if seen >= pick:
                return child.move
        return children[-1].move


@dataclass(frozen=True)
class _TTEntry:
    depth: int
    score: float
    flag: str
    best_move: Move | None
    generation: int


@dataclass(frozen=True)
class AlphaBetaStats:
    depth_reached: int
    nodes: int
    tt_hits: int
    cutoffs: int


@dataclass
class AlphaBetaBot:
    name: str = "alphabeta"
    time_budget_ms: int = 800
    max_depth: int | None = None
    candidate_radius: int = 1
    threat_extension_depth: int = 1
    tt_max_size: int = 200_000
    threat_solver_max_ply: int = 6
    value_model: ValueModel = field(default_factory=HeuristicValueModel)
    aspiration_window: float = 200.0
    last_stats: AlphaBetaStats = field(default_factory=lambda: AlphaBetaStats(0, 0, 0, 0))

    def choose_move(self, state: GameState, symbol: Symbol, rng: random.Random) -> Move:
        legal = state.board.legal_moves()
        if not legal:
            raise ValueError("No legal moves available")

        context = SearchContext()
        tactical = solve_forcing_line(state, symbol, context, max_ply=self.threat_solver_max_ply)
        if tactical is not None:
            return tactical.move

        deadline = time.perf_counter() + (self.time_budget_ms / 1000.0)
        tt: BoundedCache[int, _TTEntry] = BoundedCache(max_size=self.tt_max_size)
        self._killer_table = {}
        self._history_table = {}
        max_depth = self.max_depth if self.max_depth is not None else 99

        best_move = legal[0]
        prev_score = 0.0
        nodes = 0
        tt_hits = 0
        cutoffs = 0
        depth_reached = 0
        depth = 1
        generation = 0
        while depth <= max_depth:
            if time.perf_counter() >= deadline:
                break
            generation += 1
            alpha = prev_score - self.aspiration_window
            beta = prev_score + self.aspiration_window
            depth_best: Move | None = None
            depth_best_score = -1_000_000_000.0
            moves = self._ordered_moves(
                state=state, symbol=symbol, context=context, tt=tt, rng=rng
            )
            if alpha >= beta:
                alpha, beta = -1_000_000_000.0, 1_000_000_000.0
            for move in moves:
                if time.perf_counter() >= deadline:
                    break
                child = state.fast_clone()
                child.apply_move(move)
                score, inc_nodes, inc_hits, inc_cutoffs = self._negamax(
                    state=child,
                    depth=depth - 1,
                    alpha=-beta,
                    beta=-alpha,
                    root_symbol=symbol,
                    context=context,
                    tt=tt,
                    deadline=deadline,
                    rng=rng,
                    generation=generation,
                )
                score = -score
                nodes += inc_nodes
                tt_hits += inc_hits
                cutoffs += inc_cutoffs
                if score > depth_best_score:
                    depth_best_score = score
                    depth_best = move
                if score > alpha:
                    alpha = score
                if score >= beta:
                    # aspiration fail-high: retry this depth with full window
                    alpha = -1_000_000_000.0
                    beta = 1_000_000_000.0
                    break
            if depth_best is not None:
                best_move = depth_best
                prev_score = depth_best_score
                depth_reached = depth
            if time.perf_counter() >= deadline:
                break
            depth += 1

        self.last_stats = AlphaBetaStats(
            depth_reached=depth_reached, nodes=nodes, tt_hits=tt_hits, cutoffs=cutoffs
        )
        return best_move

    def _negamax(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        root_symbol: Symbol,
        context: SearchContext,
        tt: BoundedCache[int, _TTEntry],
        deadline: float,
        rng: random.Random,
        generation: int,
    ) -> tuple[float, int, int, int]:
        nodes = 1
        tt_hits = 0
        cutoffs = 0
        if time.perf_counter() >= deadline:
            return self._evaluate(state, root_symbol), nodes, tt_hits, cutoffs

        if state.is_over:
            return self._terminal_score(state, root_symbol), nodes, tt_hits, cutoffs

        key = state.state_key()
        cached = tt.get(key)
        if cached is not None and cached.depth >= depth:
            tt_hits += 1
            if cached.flag == "exact":
                return cached.score, nodes, tt_hits, cutoffs
            if cached.flag == "lower":
                alpha = max(alpha, cached.score)
            elif cached.flag == "upper":
                beta = min(beta, cached.score)
            if alpha >= beta:
                cutoffs += 1
                return cached.score, nodes, tt_hits, cutoffs

        extension = 0
        if depth <= 0:
            tactical = solve_forcing_line(
                state, state.next_symbol, context, max_ply=min(3, self.threat_solver_max_ply)
            )
            if tactical is not None and tactical.status is not ThreatSolutionStatus.UNRESOLVED:
                extension = self.threat_extension_depth
            else:
                return self._evaluate(state, root_symbol), nodes, tt_hits, cutoffs

        effective_depth = max(0, depth + extension)
        if effective_depth == 0:
            return self._evaluate(state, root_symbol), nodes, tt_hits, cutoffs

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
            return self._evaluate(state, root_symbol), nodes, tt_hits, cutoffs

        first = True
        for move in moves:
            if time.perf_counter() >= deadline:
                break
            token = state.make_move(move)
            if first:
                child_score, n, h, c = self._negamax(
                    state=state,
                    depth=effective_depth - 1,
                    alpha=-beta,
                    beta=-alpha,
                    root_symbol=root_symbol,
                    context=context,
                    tt=tt,
                    deadline=deadline,
                    rng=rng,
                    generation=generation,
                )
                score = -child_score
                first = False
            else:
                # Principal variation search: null-window first.
                child_score, n, h, c = self._negamax(
                    state=state,
                    depth=effective_depth - 1,
                    alpha=-(alpha + 1),
                    beta=-alpha,
                    root_symbol=root_symbol,
                    context=context,
                    tt=tt,
                    deadline=deadline,
                    rng=rng,
                    generation=generation,
                )
                score = -child_score
                if alpha < score < beta:
                    child_score, n2, h2, c2 = self._negamax(
                        state=state,
                        depth=effective_depth - 1,
                        alpha=-beta,
                        beta=-alpha,
                        root_symbol=root_symbol,
                        context=context,
                        tt=tt,
                        deadline=deadline,
                        rng=rng,
                        generation=generation,
                    )
                    score = -child_score
                    n += n2
                    h += h2
                    c += c2
            state.unmake_move(token)
            nodes += n
            tt_hits += h
            cutoffs += c
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                self._record_cutoff(state, move, state.next_symbol)
                cutoffs += 1
                break

        if best_score <= original_alpha:
            flag = "upper"
        elif best_score >= beta:
            flag = "lower"
        else:
            flag = "exact"
        tt.set(
            key,
            _TTEntry(
                depth=effective_depth,
                score=best_score,
                flag=flag,
                best_move=best_move,
                generation=generation,
            ),
        )
        return best_score, nodes, tt_hits, cutoffs

    def _ordered_moves(
        self,
        state: GameState,
        symbol: Symbol,
        context: SearchContext,
        tt: BoundedCache[int, _TTEntry],
        rng: random.Random,
    ) -> list[Move]:
        own_threats = detect_threats(state, symbol, context=context)
        opp_threats = detect_threats(state, symbol.other(), context=context)
        high_tactical = any(
            t.kind in (ThreatKind.IMMEDIATE_WIN, ThreatKind.OPEN_FOUR, ThreatKind.DOUBLE_THREE)
            for t in (own_threats + opp_threats)
        )
        mode = MoveGenerationMode.THREAT_FRONTIER if high_tactical else MoveGenerationMode.FRONTIER
        candidates = generate_moves(
            state,
            symbol,
            context=context,
            mode=mode,
            candidate_radius=self.candidate_radius if high_tactical else max(1, self.candidate_radius + 1),
        )
        if not candidates:
            return []

        tactical_threats = detect_threats(state, symbol, candidates=candidates, context=context)
        tactical_map = {threat.move: threat.severity for threat in tactical_threats}
        scored = dict(self.value_model.score_moves(state, symbol, candidates))

        tt_move: Move | None = None
        tt_entry = tt.get(state.state_key())
        if tt_entry is not None and tt_entry.best_move in candidates:
            tt_move = tt_entry.best_move

        killer_primary = self._killer_move(state)
        ranked = []
        for move in candidates:
            history = self._history_score(symbol, move)
            noise = rng.random() * 1e-6
            rank = (
                1 if tt_move == move else 0,
                1 if killer_primary == move else 0,
                tactical_map.get(move, 0),
                scored.get(move, -1.0),
                history,
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
        solution = solve_forcing_line(state, symbol, context, max_ply=self.threat_solver_max_ply)
        return None if solution is None else solution.move

    def _killer_move(self, state: GameState) -> Move | None:
        return getattr(self, "_killer_table", {}).get((state.next_symbol.value, state.state_key()))

    def _history_score(self, symbol: Symbol, move: Move) -> float:
        return getattr(self, "_history_table", {}).get((symbol.value, move.row, move.col), 0.0)

    def _record_cutoff(self, state: GameState, move: Move, symbol: Symbol) -> None:
        if not hasattr(self, "_killer_table"):
            self._killer_table = {}
        if not hasattr(self, "_history_table"):
            self._history_table = {}
        self._killer_table[(symbol.value, state.state_key())] = move
        key = (symbol.value, move.row, move.col)
        self._history_table[key] = self._history_table.get(key, 0.0) + 1.0

def _score_terminal_result(winner: Symbol | None, root_symbol: Symbol) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner is root_symbol else -1.0


def _first_threat_move(threats: list[Threat], kind: ThreatKind) -> Move | None:
    for threat in threats:
        if threat.kind is kind:
            return threat.move
    return None
