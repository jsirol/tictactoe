from tictactoe.search.value_model import HeuristicWeights
from tictactoe.weight_tuning import ProfileResult, build_weight_candidates, rank_profiles


def test_build_weight_candidates_is_deterministic():
    base = HeuristicWeights()
    a = build_weight_candidates(base=base, samples=3, seed=7, perturbation=0.2)
    b = build_weight_candidates(base=base, samples=3, seed=7, perturbation=0.2)
    assert a == b


def test_rank_profiles_prefers_holdout_then_tuning():
    p1 = ProfileResult("a", HeuristicWeights(), tuning_rate=0.9, holdout_rate=0.6, tuning_total=10, holdout_total=10)
    p2 = ProfileResult("b", HeuristicWeights(), tuning_rate=0.7, holdout_rate=0.7, tuning_total=10, holdout_total=10)
    ranked = rank_profiles([p1, p2])
    assert ranked[0].profile_name == "b"
