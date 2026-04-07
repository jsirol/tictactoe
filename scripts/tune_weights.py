from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from tictactoe.bots import AlphaBetaBot
from tictactoe.puzzle_eval import evaluate_pack
from tictactoe.search.value_model import HeuristicValueModel, HeuristicWeights
from tictactoe.weight_store import (
    WeightCandidate,
    WeightMetrics,
    load_best_weights_if_exists,
    promote_best_if_improved,
)
from tictactoe.weight_tuning import ProfileResult, build_weight_candidates, rank_profiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-pack", type=str, default="data/puzzles/tuning_10x10.jsonl")
    parser.add_argument("--holdout-pack", type=str, default="data/puzzles/holdout_15x15.jsonl")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--perturbation", type=float, default=0.2)
    parser.add_argument("--time-budget-ms", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--best-path", type=str, default="data/weights/best.json")
    parser.add_argument("--runs-dir", type=str, default="data/weights/runs")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force-promote", action="store_true")
    args = parser.parse_args()

    base = load_best_weights_if_exists(args.best_path) or HeuristicWeights()
    candidates = build_weight_candidates(
        base=base, samples=args.samples, seed=args.seed, perturbation=args.perturbation
    )

    results: list[ProfileResult] = []
    for profile_name, weights in candidates:
        bot = AlphaBetaBot(
            time_budget_ms=args.time_budget_ms,
            max_depth=args.max_depth,
            value_model=HeuristicValueModel(weights=weights),
        )
        tuning = evaluate_pack(args.tuning_pack, bot_name="alphabeta", seed=args.seed, bot=bot)
        holdout = evaluate_pack(args.holdout_pack, bot_name="alphabeta", seed=args.seed, bot=bot)
        results.append(
            ProfileResult(
                profile_name=profile_name,
                weights=weights,
                tuning_rate=float(tuning["solve_rate"]),
                holdout_rate=float(holdout["solve_rate"]),
                tuning_total=int(tuning["total"]),
                holdout_total=int(holdout["total"]),
            )
        )

    ranked = rank_profiles(results)
    best = ranked[0]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = WeightCandidate(
        profile_name=best.profile_name,
        weights=best.weights,
        metrics=WeightMetrics(
            holdout_rate=best.holdout_rate,
            tuning_rate=best.tuning_rate,
            holdout_total=best.holdout_total,
            tuning_total=best.tuning_total,
        ),
        source_run_id=run_id,
    )
    promoted = promote_best_if_improved(candidate, best_path=args.best_path, force=args.force_promote)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    report_path = runs_dir / f"{run_id}.json"
    report = {
        "run_id": run_id,
        "seed": args.seed,
        "samples": args.samples,
        "perturbation": args.perturbation,
        "tuning_pack": args.tuning_pack,
        "holdout_pack": args.holdout_pack,
        "best_profile": {
            "profile_name": best.profile_name,
            "weights": best.weights.to_dict(),
            "tuning_rate": best.tuning_rate,
            "holdout_rate": best.holdout_rate,
            "tuning_total": best.tuning_total,
            "holdout_total": best.holdout_total,
        },
        "promoted_to_best": promoted,
        "candidates": [
            {
                "profile_name": r.profile_name,
                "weights": r.weights.to_dict(),
                "tuning_rate": r.tuning_rate,
                "holdout_rate": r.holdout_rate,
                "tuning_total": r.tuning_total,
                "holdout_total": r.holdout_total,
            }
            for r in ranked
        ],
    }
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"Run: {run_id}")
    print(f"Report: {report_path}")
    print(f"Promoted: {promoted}")
    print("Top profiles:")
    for idx, r in enumerate(ranked[: max(1, args.top_k)], start=1):
        print(
            f"  {idx}. {r.profile_name} holdout={r.holdout_rate:.2%} "
            f"tuning={r.tuning_rate:.2%} totals=({r.holdout_total},{r.tuning_total})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
