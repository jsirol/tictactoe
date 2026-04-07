import json
from pathlib import Path

from tictactoe.search.value_model import HeuristicWeights
from tictactoe.weight_store import WeightCandidate, WeightMetrics, load_weights_file, promote_best_if_improved


def test_load_weights_file_from_registry_shape(tmp_path: Path):
    path = tmp_path / "best.json"
    payload = {
        "profile_name": "p1",
        "weights": HeuristicWeights().to_dict(),
        "metrics": {
            "holdout_rate": 0.5,
            "tuning_rate": 0.6,
            "holdout_total": 10,
            "tuning_total": 20,
        },
        "source_run_id": "r1",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    weights = load_weights_file(path)
    assert isinstance(weights, HeuristicWeights)


def test_promote_best_if_improved_does_not_override_worse(tmp_path: Path):
    best_path = tmp_path / "best.json"
    better = WeightCandidate(
        profile_name="better",
        weights=HeuristicWeights(),
        metrics=WeightMetrics(holdout_rate=0.8, tuning_rate=0.8, holdout_total=10, tuning_total=10),
        source_run_id="r_better",
    )
    worse = WeightCandidate(
        profile_name="worse",
        weights=HeuristicWeights(line_weight=0.1),
        metrics=WeightMetrics(holdout_rate=0.7, tuning_rate=0.9, holdout_total=10, tuning_total=10),
        source_run_id="r_worse",
    )
    assert promote_best_if_improved(better, best_path=best_path) is True
    assert promote_best_if_improved(worse, best_path=best_path) is False
    loaded = load_weights_file(best_path)
    assert loaded == better.weights
