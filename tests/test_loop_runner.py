import json
from pathlib import Path

from tictactoe.loop_runner import LoopConfig, run_loop


def test_loop_runner_tracks_best_and_checkpoints(tmp_path, monkeypatch):
    losses = iter([1.2, 0.8, 1.1, 0.7, 0.9])
    init_paths: list[str | None] = []

    def fake_selfplay(config, on_progress=None):
        if on_progress is not None:
            for idx in range(config.games):
                on_progress({"kind": "game_progress", "samples": 2, "game_id": f"g{idx}"})
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        shard = out_dir / "selfplay_fake.jsonl"
        shard.write_text("", encoding="utf-8")
        (out_dir / "manifest.json").write_text(
            json.dumps({"games": config.games, "samples": config.games * 2, "path": str(shard)}),
            encoding="utf-8",
        )
        return {"games": config.games, "samples": config.games * 2, "elapsed_sec": 1.0, "games_per_min": 60.0}

    def fake_training(config, logger=print):
        init_paths.append(config.init_checkpoint_path)
        assert config.replay_shards == 3
        out_dir = Path(config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        loss = next(losses)
        checkpoint = out_dir / f"{config.run_name}.pt"
        torchscript = out_dir / f"{config.run_name}.torchscript.pt"
        meta = out_dir / f"{config.run_name}.meta.json"
        checkpoint.write_text("ckpt", encoding="utf-8")
        torchscript.write_text("ts", encoding="utf-8")
        payload = {
            "elapsed_sec": 0.5,
            "samples": 10,
            "games": 5,
            "final_metrics": {"total_loss": loss, "policy_loss": loss / 2, "value_loss": loss / 2},
            "artifacts": {"checkpoint": str(checkpoint), "torchscript": str(torchscript)},
        }
        meta.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("tictactoe.loop_runner.run_selfplay_with_progress", fake_selfplay)
    monkeypatch.setattr("tictactoe.loop_runner.run_training", fake_training)

    summary = run_loop(
        LoopConfig(
            run_name="testloop",
            iterations=5,
            games_per_run=3,
            checkpoint_every=5,
            models_dir=str(tmp_path / "models"),
            selfplay_output_dir=str(tmp_path / "selfplay"),
            web_mcts_model_path=str(tmp_path / "models" / "latest.torchscript.pt"),
        ),
        logger=lambda _: None,
    )
    base = tmp_path / "models" / "testloop"
    assert summary["best_iteration"] == 5
    assert Path(summary["best_model"]).exists()
    assert Path(summary["latest_model"]).exists()
    assert Path(summary["final_model"]).exists()
    assert Path(summary["web_mcts_model"]).exists()
    assert (base / "checkpoints" / "iter_005.torchscript.pt").exists()
    assert (base / "checkpoints" / "iter_005.pt").exists()
    assert (base / "checkpoints" / "iter_005.meta.json").exists()

    lines = (base / "loop_history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    assert init_paths[0] is None
    assert init_paths[1] is not None
