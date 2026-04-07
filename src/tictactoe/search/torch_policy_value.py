from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tictactoe.core import GameState, Move, Symbol

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


@dataclass
class TorchPolicyValueModel:
    model_path: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        if torch is None:  # pragma: no cover - optional dependency
            raise RuntimeError("PyTorch is not installed; install torch to use --model-path")
        self._model = torch.jit.load(self.model_path, map_location=self.device)
        self._model.eval()

    def predict(
        self, state: GameState, symbol: Symbol, candidate_moves: list[Move]
    ) -> tuple[dict[Move, float], float]:
        return self.predict_batch([(state, symbol, candidate_moves)])[0]

    def predict_batch(
        self, items: list[tuple[GameState, Symbol, list[Move]]]
    ) -> list[tuple[dict[Move, float], float]]:
        if not items:
            return []

        batch = [self._encode_state(state, symbol) for state, symbol, _ in items]
        with torch.inference_mode():
            inputs = torch.stack(batch, dim=0)
            policy_logits, values = self._model(inputs)
        out: list[tuple[dict[Move, float], float]] = []
        for idx, (_, _, moves) in enumerate(items):
            priors = self._project_policy(policy_logits[idx], moves)
            value = float(values[idx].item())
            out.append((priors, max(-1.0, min(1.0, value))))
        return out

    def _encode_state(self, state: GameState, symbol: Symbol) -> Any:
        size = state.board.size
        x_plane = torch.zeros((size, size), dtype=torch.float32, device=self.device)
        o_plane = torch.zeros((size, size), dtype=torch.float32, device=self.device)
        for row in range(size):
            for col in range(size):
                cell = state.board.cells[row][col]
                if cell is Symbol.X:
                    x_plane[row, col] = 1.0
                elif cell is Symbol.O:
                    o_plane[row, col] = 1.0
        stm = torch.full((size, size), 1.0 if symbol is Symbol.X else 0.0, dtype=torch.float32, device=self.device)
        return torch.stack((x_plane, o_plane, stm), dim=0)

    def _project_policy(self, logits: Any, moves: list[Move]) -> dict[Move, float]:
        if not moves:
            return {}
        weights = []
        for move in moves:
            weights.append(float(logits[move.row, move.col].item()))
        best = max(weights)
        exps = [2.718281828459045 ** (weight - best) for weight in weights]
        total = sum(exps)
        if total <= 0.0:
            uniform = 1.0 / len(moves)
            return {move: uniform for move in moves}
        return {move: exps[idx] / total for idx, move in enumerate(moves)}
