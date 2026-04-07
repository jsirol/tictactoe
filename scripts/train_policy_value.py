from __future__ import annotations

import argparse
from datetime import UTC, datetime

from tictactoe.training import TrainingConfig, run_training


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=str, default="data/selfplay/manifest.json")
    parser.add_argument("--train-file", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="data/models")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    run_name = args.run_name or f"pv_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_training(
        TrainingConfig(
            data_manifest=args.data_manifest,
            train_file=args.train_file,
            out_dir=args.out_dir,
            run_name=run_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            log_every_steps=args.log_every_steps,
            policy_loss_weight=args.policy_loss_weight,
            value_loss_weight=args.value_loss_weight,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
