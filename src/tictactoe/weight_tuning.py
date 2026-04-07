from __future__ import annotations

from dataclasses import dataclass
from random import Random

from tictactoe.search.value_model import HeuristicWeights


@dataclass(frozen=True)
class ProfileResult:
    profile_name: str
    weights: HeuristicWeights
    tuning_rate: float
    holdout_rate: float
    tuning_total: int
    holdout_total: int


def build_weight_candidates(
    base: HeuristicWeights, samples: int, seed: int, perturbation: float = 0.2
) -> list[tuple[str, HeuristicWeights]]:
    rng = Random(seed)
    base_dict = base.to_dict()
    candidates: list[tuple[str, HeuristicWeights]] = [("base", base)]
    for i in range(samples):
        raw = {}
        for key, value in base_dict.items():
            factor = 1.0 + rng.uniform(-perturbation, perturbation)
            raw[key] = max(0.0, value * factor)
        candidates.append((f"candidate_{i+1:03d}", HeuristicWeights.from_dict(raw)))
    return candidates


def rank_profiles(results: list[ProfileResult]) -> list[ProfileResult]:
    return sorted(
        results,
        key=lambda r: (r.holdout_rate, r.tuning_rate, r.holdout_total + r.tuning_total),
        reverse=True,
    )
