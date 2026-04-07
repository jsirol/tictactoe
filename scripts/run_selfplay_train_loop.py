from __future__ import annotations

import argparse
from datetime import UTC, datetime

from tictactoe.loop_runner import LoopConfig, run_loop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-run", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--initial-model-path", type=str, default=None)

    parser.add_argument("--selfplay-output-dir", type=str, default="data/selfplay")
    parser.add_argument("--models-dir", type=str, default="data/models/loops")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-wait-ms", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--time-budget-ms", type=int, default=120)
    parser.add_argument("--determinism", type=str, default="balanced", choices=["balanced", "strict", "fast"])
    parser.add_argument("--high-temperature", type=float, default=1.0)
    parser.add_argument("--low-temperature", type=float, default=0.1)
    parser.add_argument("--temperature-cutoff-ply", type=int, default=12)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-log-every-steps", type=int, default=20)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    run_name = args.run_name or f"loop_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_loop(
        LoopConfig(
            run_name=run_name,
            iterations=args.iterations,
            games_per_run=args.games_per_run,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed,
            initial_model_path=args.initial_model_path,
            selfplay_output_dir=args.selfplay_output_dir,
            models_dir=args.models_dir,
            board_size=args.size,
            workers=args.workers,
            batch_size=args.batch_size,
            batch_wait_ms=args.batch_wait_ms,
            simulations=args.simulations,
            time_budget_ms=args.time_budget_ms,
            determinism=args.determinism,
            high_temperature=args.high_temperature,
            low_temperature=args.low_temperature,
            temperature_cutoff_ply=args.temperature_cutoff_ply,
            train_epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            train_log_every_steps=args.train_log_every_steps,
            policy_loss_weight=args.policy_loss_weight,
            value_loss_weight=args.value_loss_weight,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
