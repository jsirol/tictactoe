from __future__ import annotations

from pathlib import Path

from tictactoe.search.policy_value import PolicyValueModel
from tictactoe.search.torch_policy_value import TorchPolicyValueModel

DEFAULT_MCTS_MODEL_PATH = "data/models/latest.torchscript.pt"


def load_mcts_policy_value_model(model_path: str | None = None) -> PolicyValueModel | None:
    path = model_path or DEFAULT_MCTS_MODEL_PATH
    if not Path(path).exists():
        return None
    try:
        return TorchPolicyValueModel(model_path=path)
    except Exception:
        return None
