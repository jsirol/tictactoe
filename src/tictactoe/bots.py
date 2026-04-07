from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from .core import GameState, Move, Symbol


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
    wins: float = 0.0

    @classmethod
    def from_state(
        cls, state: GameState, parent: "_Node | None" = None, move: Move | None = None
    ) -> "_Node":
        return cls(
            state=state,
            parent=parent,
            move=move,
            untried_moves=state.board.legal_moves(),
            children=[],
        )


@dataclass
class MCTSBot:
    name: str = "mcts"
    simulations: int = 300
    exploration_constant: float = 1.41421356237

    def choose_move(self, state: GameState, symbol: Symbol, rng: random.Random) -> Move:
        legal = state.board.legal_moves()
        if not legal:
            raise ValueError("No legal moves available")

        for move in legal:
            trial = _clone_state(state)
            trial.apply_move(move)
            if trial.winner is symbol:
                return move

        root = _Node.from_state(_clone_state(state))
        for _ in range(max(1, self.simulations)):
            node = root

            while not node.untried_moves and node.children and not node.state.is_over:
                node = self._select_child(node, rng)

            if node.untried_moves and not node.state.is_over:
                move = rng.choice(node.untried_moves)
                node.untried_moves.remove(move)
                next_state = _clone_state(node.state)
                next_state.apply_move(move)
                child = _Node.from_state(next_state, parent=node, move=move)
                node.children.append(child)
                node = child

            rollout_state = _clone_state(node.state)
            while not rollout_state.is_over:
                rollout_move = rng.choice(rollout_state.board.legal_moves())
                rollout_state.apply_move(rollout_move)

            result = _score_result(rollout_state.winner, root_symbol=symbol)
            while node is not None:
                node.visits += 1
                node.wins += result
                node = node.parent

        if not root.children:
            return rng.choice(legal)

        best_visits = max(child.visits for child in root.children)
        candidates = [child for child in root.children if child.visits == best_visits]
        return rng.choice(candidates).move  # type: ignore[return-value]

    def _select_child(self, node: _Node, rng: random.Random) -> _Node:
        scored = []
        for child in node.children:
            exploit = child.wins / child.visits
            explore = self.exploration_constant * ((node.visits**0.5) / (1 + child.visits))
            scored.append((exploit + explore, child))
        best_score = max(score for score, _ in scored)
        candidates = [child for score, child in scored if score == best_score]
        return rng.choice(candidates)


def _score_result(winner: Symbol | None, root_symbol: Symbol) -> float:
    if winner is None:
        return 0.5
    return 1.0 if winner is root_symbol else 0.0


def _clone_state(state: GameState) -> GameState:
    size = state.board.size
    cloned = GameState.new(size=size)
    cloned.next_symbol = state.next_symbol
    cloned.winner = state.winner
    for row in range(size):
        for col in range(size):
            cloned.board.cells[row][col] = state.board.cells[row][col]
    return cloned
