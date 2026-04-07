from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tictactoe.search.value_model import HeuristicWeights


@dataclass(frozen=True)
class WeightMetrics:
    holdout_rate: float
    tuning_rate: float
    holdout_total: int
    tuning_total: int


@dataclass(frozen=True)
class WeightCandidate:
    profile_name: str
    weights: HeuristicWeights
    metrics: WeightMetrics
    source_run_id: str


def load_weights_file(path: str | Path) -> HeuristicWeights:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Registry-like shape.
    if isinstance(raw, dict) and "weights" in raw:
        return HeuristicWeights.from_dict(raw["weights"])
    # Tune report shape.
    if isinstance(raw, dict) and "best_profile" in raw:
        best = raw["best_profile"]
        if isinstance(best, dict) and "weights" in best:
            return HeuristicWeights.from_dict(best["weights"])
    # Candidate list shape.
    if isinstance(raw, dict) and "candidates" in raw and raw["candidates"]:
        first = raw["candidates"][0]
        if isinstance(first, dict) and "weights" in first:
            return HeuristicWeights.from_dict(first["weights"])
    # Raw weights shape.
    if isinstance(raw, dict):
        return HeuristicWeights.from_dict(raw)

    raise ValueError(f"Unsupported weights file format: {p}")


def load_best_weights_if_exists(path: str | Path = "data/weights/best.json") -> HeuristicWeights | None:
    p = Path(path)
    if not p.exists():
        return None
    return load_weights_file(p)


def promote_best_if_improved(
    candidate: WeightCandidate,
    best_path: str | Path = "data/weights/best.json",
    force: bool = False,
) -> bool:
    p = Path(best_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not p.exists():
        _write_best(candidate, p)
        return True

    current = _load_candidate(p)
    if force or _is_better(candidate.metrics, current.metrics):
        _write_best(candidate, p)
        return True
    return False


def _is_better(candidate: WeightMetrics, current: WeightMetrics) -> bool:
    if candidate.holdout_rate != current.holdout_rate:
        return candidate.holdout_rate > current.holdout_rate
    if candidate.tuning_rate != current.tuning_rate:
        return candidate.tuning_rate > current.tuning_rate
    return (candidate.holdout_total + candidate.tuning_total) > (
        current.holdout_total + current.tuning_total
    )


def _load_candidate(path: Path) -> WeightCandidate:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    try:
        profile_name = str(raw["profile_name"])
        source_run_id = str(raw.get("source_run_id", "unknown"))
        weights = HeuristicWeights.from_dict(raw["weights"])
        metrics_raw = raw["metrics"]
        metrics = WeightMetrics(
            holdout_rate=float(metrics_raw["holdout_rate"]),
            tuning_rate=float(metrics_raw["tuning_rate"]),
            holdout_total=int(metrics_raw["holdout_total"]),
            tuning_total=int(metrics_raw["tuning_total"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid best weights registry: {path}") from exc
    return WeightCandidate(
        profile_name=profile_name, weights=weights, metrics=metrics, source_run_id=source_run_id
    )


def _write_best(candidate: WeightCandidate, path: Path) -> None:
    payload: dict[str, Any] = {
        "version": 1,
        "profile_name": candidate.profile_name,
        "weights": candidate.weights.to_dict(),
        "metrics": {
            "holdout_rate": candidate.metrics.holdout_rate,
            "tuning_rate": candidate.metrics.tuning_rate,
            "holdout_total": candidate.metrics.holdout_total,
            "tuning_total": candidate.metrics.tuning_total,
        },
        "source_run_id": candidate.source_run_id,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
