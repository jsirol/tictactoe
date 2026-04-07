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
        self._thread_configured = False

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

    def configure_threads(self, intraop_threads: int = 1, interop_threads: int = 1) -> None:
        if torch is None or self._thread_configured:  # pragma: no cover - optional dependency
            return
        torch.set_num_threads(max(1, intraop_threads))
        torch.set_num_interop_threads(max(1, interop_threads))
        self._thread_configured = True

    def predict_encoded_batch(
        self,
        encoded_planes: list[list[list[list[float]]]],
        candidate_indices_batch: list[list[int]],
    ) -> list[tuple[list[float], float]]:
        if not encoded_planes:
            return []
        tensor = torch.tensor(encoded_planes, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            policy_logits, values = self._model(tensor)
        flat_logits = policy_logits.view(policy_logits.shape[0], -1)
        outputs: list[tuple[list[float], float]] = []
        for idx, candidate_indices in enumerate(candidate_indices_batch):
            if not candidate_indices:
                outputs.append(([], max(-1.0, min(1.0, float(values[idx].item())))))
                continue
            indices = torch.tensor(candidate_indices, dtype=torch.long, device=self.device)
            selected = flat_logits[idx].index_select(0, indices)
            probs = torch.softmax(selected, dim=0).detach().cpu().tolist()
            outputs.append((probs, max(-1.0, min(1.0, float(values[idx].item())))))
        return outputs

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
