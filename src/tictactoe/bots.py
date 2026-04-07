from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from math import log, sqrt
from typing import Protocol

from .core import GameState, Move, Symbol
from .search.move_policy import candidate_moves
from .search.tactics import clone_state, find_immediate_winning_move
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
        cls, state: GameState, candidate_radius: int, parent: "_Node | None" = None, move: Move | None = None
    ) -> "_Node":
        return cls(
            state=state,
            parent=parent,
            move=move,
            untried_moves=candidate_moves(state, radius=candidate_radius),
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

    def choose_move(self, state: GameState, symbol: Symbol, rng: random.Random) -> Move:
        legal = state.board.legal_moves()
        if not legal:
            raise ValueError("No legal moves available")

        win_now = find_immediate_winning_move(state, symbol)
        if win_now is not None:
            return win_now

        block_now = find_immediate_winning_move(state, symbol.other())
        if block_now is not None:
            return block_now

        root = _Node.from_state(clone_state(state), candidate_radius=self.candidate_radius)
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
                        next_state, candidate_radius=self.candidate_radius, parent=node, move=move
                    )
                    node.children.append(child)
                    node = child

            value = self._rollout_value(node.state, symbol, rng)
            while node is not None:
                node.visits += 1
                node.total_value += value
                node = node.parent
            iterations += 1

        if not root.children:
            return rng.choice(candidate_moves(state, radius=self.candidate_radius) or legal)

        best_visits = max(child.visits for child in root.children)
        candidates = [child for child in root.children if child.visits == best_visits]
        picked = rng.choice(candidates).move
        if picked is None:
            return rng.choice(legal)
        return picked

    def _rollout_value(self, state: GameState, root_symbol: Symbol, rng: random.Random) -> float:
        rollout_state = clone_state(state)
        depth = 0
        while not rollout_state.is_over and depth < self.rollout_depth:
            to_move = rollout_state.next_symbol
            win_now = find_immediate_winning_move(rollout_state, to_move)
            if win_now is not None:
                rollout_state.apply_move(win_now)
                depth += 1
                continue

            block_now = find_immediate_winning_move(rollout_state, to_move.other())
            if block_now is not None:
                rollout_state.apply_move(block_now)
                depth += 1
                continue

            moves = candidate_moves(rollout_state, radius=self.candidate_radius)
            if not moves:
                break
            if rng.random() < self.epsilon:
                chosen = rng.choice(moves)
            else:
                chosen = max(moves, key=lambda m: self.value_model.score_move(rollout_state, to_move, m))
            rollout_state.apply_move(chosen)
            depth += 1

        if rollout_state.is_over:
            return _score_terminal_result(rollout_state.winner, root_symbol)
        return self.value_model.evaluate(rollout_state, root_symbol)

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


def _score_terminal_result(winner: Symbol | None, root_symbol: Symbol) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner is root_symbol else -1.0
