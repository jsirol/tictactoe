from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tictactoe.selfplay import SelfPlayConfig, run_selfplay_with_progress
from tictactoe.training import TrainingConfig, run_training


@dataclass(frozen=True)
class LoopConfig:
    run_name: str = "selfplay_train_loop"
    iterations: int = 10
    games_per_run: int = 10
    checkpoint_every: int = 5
    seed: int = 1
    initial_model_path: str | None = None

    selfplay_output_dir: str = "data/selfplay"
    models_dir: str = "data/models/loops"
    board_size: int = 10
    workers: int = 12
    batch_size: int = 32
    batch_wait_ms: int = 3
    simulations: int = 200
    time_budget_ms: int = 120
    determinism: str = "balanced"
    high_temperature: float = 1.0
    low_temperature: float = 0.1
    temperature_cutoff_ply: int = 12

    train_epochs: int = 5
    train_batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_log_every_steps: int = 20
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0


def run_loop(config: LoopConfig, logger: Callable[[str], None] = print) -> dict:
    if config.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if config.games_per_run < 1:
        raise ValueError("--games-per-run must be >= 1")
    if config.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be >= 1")

    base_dir = Path(config.models_dir) / config.run_name
    checkpoints_dir = base_dir / "checkpoints"
    base_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    history_path = base_dir / "loop_history.jsonl"
    latest_meta_path = base_dir / "latest_meta.json"
    best_meta_path = base_dir / "best_meta.json"
    latest_model_path = base_dir / "latest.torchscript.pt"
    best_model_path = base_dir / "best.torchscript.pt"
    final_model_path = base_dir / "final.torchscript.pt"

    best_loss = float("nan")
    best_iteration = 0
    current_model_path = config.initial_model_path
    loop_start = time.perf_counter()
    with history_path.open("w", encoding="utf-8") as history_file:
        for iteration in range(1, config.iterations + 1):
            logger(f"\n=== Iteration {iteration}/{config.iterations} ===")
            logger(f"[A] Self-play start (model={current_model_path or 'heuristic'})")
            selfplay_start = time.perf_counter()
            completed_games = 0
            produced_samples = 0

            def _on_progress(event: dict) -> None:
                nonlocal completed_games, produced_samples
                if event.get("kind") != "game_progress":
                    return
                completed_games += 1
                produced_samples += int(event.get("samples", 0))
                elapsed = max(1e-9, time.perf_counter() - selfplay_start)
                gpm = (completed_games / elapsed) * 60.0
                logger(
                    f"  self-play progress: {completed_games}/{config.games_per_run} games "
                    f"(samples={produced_samples}, gpm={gpm:.2f})"
                )

            selfplay_summary = run_selfplay_with_progress(
                config=SelfPlayConfig(
                    size=config.board_size,
                    games=config.games_per_run,
                    workers=config.workers,
                    seed=config.seed + iteration - 1,
                    model_path=current_model_path,
                    output_dir=config.selfplay_output_dir,
                    batch_size=config.batch_size,
                    batch_wait_ms=config.batch_wait_ms,
                    simulations=config.simulations,
                    time_budget_ms=config.time_budget_ms,
                    determinism=config.determinism,
                    high_temperature=config.high_temperature,
                    low_temperature=config.low_temperature,
                    temperature_cutoff_ply=config.temperature_cutoff_ply,
                ),
                on_progress=_on_progress,
            )
            logger(
                f"[A] Self-play done: games={selfplay_summary['games']}, samples={selfplay_summary['samples']}, "
                f"elapsed={selfplay_summary['elapsed_sec']:.2f}s, gpm={selfplay_summary['games_per_min']:.2f}"
            )

            logger("[B] Training start")
            train_run_name = f"iter_{iteration:03d}"
            train_out_dir = base_dir / train_run_name
            training_meta = run_training(
                TrainingConfig(
                    data_manifest=str(Path(config.selfplay_output_dir) / "manifest.json"),
                    out_dir=str(train_out_dir),
                    run_name=train_run_name,
                    epochs=config.train_epochs,
                    batch_size=config.train_batch_size,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    seed=config.seed + iteration - 1,
                    log_every_steps=config.train_log_every_steps,
                    policy_loss_weight=config.policy_loss_weight,
                    value_loss_weight=config.value_loss_weight,
                ),
                logger=lambda msg: logger(f"  train: {msg}"),
            )
            total_loss = float(training_meta["final_metrics"]["total_loss"])
            logger(f"[B] Training done: total_loss={total_loss:.6f}")

            iter_torchscript = Path(training_meta["artifacts"]["torchscript"])
            iter_checkpoint = Path(training_meta["artifacts"]["checkpoint"])
            iter_meta = train_out_dir / f"{train_run_name}.meta.json"

            shutil.copy2(iter_torchscript, latest_model_path)
            shutil.copy2(iter_meta, latest_meta_path)
            current_model_path = str(latest_model_path)

            # Promotion rule: latest model is always canonical "best" for online training loops.
            best_loss = total_loss
            best_iteration = iteration
            shutil.copy2(iter_torchscript, best_model_path)
            shutil.copy2(iter_meta, best_meta_path)
            logger(f"  promoted latest model at iteration {iteration} (loss={best_loss:.6f})")

            if iteration % config.checkpoint_every == 0:
                checkpoint_base = checkpoints_dir / f"iter_{iteration:03d}"
                shutil.copy2(iter_torchscript, checkpoint_base.with_suffix(".torchscript.pt"))
                shutil.copy2(iter_checkpoint, checkpoint_base.with_suffix(".pt"))
                shutil.copy2(iter_meta, checkpoint_base.with_suffix(".meta.json"))
                logger(f"  saved checkpoint snapshot at iteration {iteration}")

            history_entry = {
                "iteration": iteration,
                "selfplay": selfplay_summary,
                "training": {
                    "final_metrics": training_meta["final_metrics"],
                    "elapsed_sec": training_meta["elapsed_sec"],
                    "samples": training_meta["samples"],
                    "games": training_meta["games"],
                },
                "artifacts": {
                    "iter_torchscript": str(iter_torchscript),
                    "iter_checkpoint": str(iter_checkpoint),
                    "latest_torchscript": str(latest_model_path),
                    "best_torchscript": str(best_model_path) if best_model_path.exists() else None,
                },
            }
            history_file.write(json.dumps(history_entry, separators=(",", ":")) + "\n")
            history_file.flush()

    if latest_model_path.exists():
        shutil.copy2(latest_model_path, final_model_path)

    elapsed = time.perf_counter() - loop_start
    summary = {
        "run_name": config.run_name,
        "iterations": config.iterations,
        "games_per_run": config.games_per_run,
        "checkpoint_every": config.checkpoint_every,
        "elapsed_sec": elapsed,
        "best_iteration": best_iteration,
        "best_loss": best_loss,
        "latest_model": str(latest_model_path) if latest_model_path.exists() else None,
        "best_model": str(best_model_path) if best_model_path.exists() else None,
        "final_model": str(final_model_path) if final_model_path.exists() else None,
        "history": str(history_path),
    }
    logger(
        "\nLoop complete: "
        f"elapsed={summary['elapsed_sec']:.2f}s, best_iter={best_iteration}, best_loss={best_loss:.6f}\n"
        f"latest={summary['latest_model']}\n"
        f"best={summary['best_model']}\n"
        f"final={summary['final_model']}\n"
        f"history={summary['history']}"
    )
    return summary
