import json
from pathlib import Path

import pytest

from tictactoe.training import TrainingConfig, load_samples, resolve_data_source, run_training


def _sample_line(*, game_id: str = "g0", ply: int = 0, value: float = 1.0) -> dict:
    return {
        "kind": "sample",
        "game_id": game_id,
        "ply": ply,
        "seed": 1,
        "player_to_move": "X",
        "policy_target": {"5,5": 1.0},
        "action": {"row": 5, "col": 5},
        "action_mask": [[1 for _ in range(10)] for _ in range(10)],
        "obs": {
            "x": [[0 for _ in range(10)] for _ in range(10)],
            "o": [[0 for _ in range(10)] for _ in range(10)],
            "to_move": "X",
        },
        "model_version": "heuristic",
        "value_target": value,
    }


def test_load_samples_parses_valid_jsonl(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    lines = [_sample_line(game_id="g0", ply=0), _sample_line(game_id="g0", ply=1, value=-1.0)]
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")

    samples = load_samples(path)
    assert len(samples) == 2
    assert samples[0].game_id == "g0"
    assert samples[1].value_target == -1.0


def test_load_samples_rejects_missing_required_key(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    bad = _sample_line()
    del bad["policy_target"]
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_samples(path)


def test_resolve_data_source_uses_manifest_path(tmp_path: Path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps(_sample_line()) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"games": 1, "samples": 1, "path": str(train_file)}),
        encoding="utf-8",
    )
    source = resolve_data_source(TrainingConfig(data_manifest=str(manifest), run_name="x"))
    assert source.train_path == train_file
    assert source.train_paths == [train_file]
    assert source.manifest_games == 1
    assert source.manifest_samples == 1


def test_resolve_data_source_replay_mixing_picks_latest_shards(tmp_path: Path):
    paths: list[Path] = []
    for idx in range(5):
        path = tmp_path / f"selfplay_{1000+idx}.jsonl"
        path.write_text(json.dumps(_sample_line(game_id=f"g{idx}")) + "\n", encoding="utf-8")
        paths.append(path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"games": 1, "samples": 1, "path": str(paths[-1])}),
        encoding="utf-8",
    )
    source = resolve_data_source(
        TrainingConfig(data_manifest=str(manifest), replay_shards=3, run_name="mix")
    )
    assert source.train_paths == paths[-3:]


def test_training_smoke_exports_checkpoint_and_torchscript(tmp_path: Path):
    torch = pytest.importorskip("torch")
    train_file = tmp_path / "train.jsonl"
    with train_file.open("w", encoding="utf-8") as fh:
        for i in range(8):
            fh.write(json.dumps(_sample_line(game_id=f"g{i//2}", ply=i)) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"games": 4, "samples": 8, "path": str(train_file)}),
        encoding="utf-8",
    )

    out_dir = tmp_path / "models"
    meta = run_training(
        TrainingConfig(
            data_manifest=str(manifest),
            out_dir=str(out_dir),
            run_name="smoke",
            epochs=1,
            batch_size=4,
            log_every_steps=1,
        ),
        logger=lambda _: None,
    )
    assert (out_dir / "smoke.pt").exists()
    assert (out_dir / "smoke.torchscript.pt").exists()
    assert (out_dir / "smoke.meta.json").exists()
    loaded = torch.jit.load(str(out_dir / "smoke.torchscript.pt"), map_location="cpu")
    x = torch.zeros((1, 3, 10, 10), dtype=torch.float32)
    policy, value = loaded(x)
    assert tuple(policy.shape) == (1, 10, 10)
    assert tuple(value.shape) == (1, 1)
    assert meta["samples"] == 8
