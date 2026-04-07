from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class SampleRecord:
    game_id: str
    ply: int
    player_to_move: str
    obs_x: list[list[int]]
    obs_o: list[list[int]]
    obs_to_move: str
    action_mask: list[list[int]]
    policy_target: dict[str, float]
    value_target: float
    model_version: str


@dataclass(frozen=True)
class DataSource:
    train_paths: list[Path]
    manifest_path: Path | None
    manifest_games: int | None
    manifest_samples: int | None

    @property
    def train_path(self) -> Path:
        return self.train_paths[0]


@dataclass(frozen=True)
class TrainingConfig:
    data_manifest: str = "data/selfplay/manifest.json"
    train_file: str | None = None
    replay_shards: int = 3
    out_dir: str = "data/models"
    run_name: str = "pv_run"
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 1
    init_checkpoint_path: str | None = None
    log_every_steps: int = 20
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0


def run_training(config: TrainingConfig, logger: Callable[[str], None] = print) -> dict:
    torch, nn, F = _import_torch()
    logger("[1/5] Resolving data source")
    source = resolve_data_source(config)
    logger(f"  train_files={len(source.train_paths)}")
    for path in source.train_paths:
        logger(f"    - {path}")
    if source.manifest_path is not None:
        logger(
            f"  manifest={source.manifest_path}, games={source.manifest_games}, samples={source.manifest_samples}"
        )

    logger("[2/5] Loading and validating samples")
    samples = load_samples_from_paths(source.train_paths)
    if not samples:
        raise ValueError(f"No training samples found in {source.train_paths}")
    board_size = len(samples[0].obs_x)
    versions = Counter(sample.model_version for sample in samples)
    unique_games = len({sample.game_id for sample in samples})
    logger(
        f"  loaded_samples={len(samples)}, unique_games={unique_games}, board={board_size}x{board_size}, "
        f"versions={dict(versions)}"
    )

    logger("[3/5] Building tensors and dataloader")
    dataset = _build_dataset(torch=torch, samples=samples)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=max(1, config.batch_size),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    logger(f"  batches_per_epoch={len(loader)}")

    logger("[4/5] Training")
    torch.manual_seed(config.seed)
    model = SmallPolicyValueNet(board_size=board_size)
    if config.init_checkpoint_path:
        init_path = Path(config.init_checkpoint_path)
        if init_path.exists():
            raw = torch.load(str(init_path), map_location="cpu")
            state_dict = raw.get("model_state_dict", raw)
            model.load_state_dict(state_dict, strict=True)
            logger(f"  warm-started from checkpoint={init_path}")
        else:
            logger(f"  warm-start checkpoint not found, training from scratch: {init_path}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    model.train()
    global_step = 0
    start = time.perf_counter()
    processed = 0
    samples_per_game = (len(samples) / unique_games) if unique_games > 0 else 1.0
    last_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
    for epoch in range(1, config.epochs + 1):
        for x, policy, value, mask in loader:
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            policy_logits, value_pred = model(x)
            policy_loss = masked_policy_kl_loss(F=F, logits=policy_logits, target=policy, mask=mask)
            value_loss = F.mse_loss(value_pred.squeeze(-1), value)
            loss = config.policy_loss_weight * policy_loss + config.value_loss_weight * value_loss
            loss.backward()
            optimizer.step()

            processed += int(x.shape[0])
            last_metrics = {
                "policy_loss": float(policy_loss.detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "total_loss": float(loss.detach().item()),
            }
            if global_step % max(1, config.log_every_steps) == 0:
                elapsed = max(1e-9, time.perf_counter() - start)
                samples_per_sec = processed / elapsed
                games_per_min = ((processed / samples_per_game) / elapsed) * 60.0
                logger(
                    f"  epoch={epoch}/{config.epochs} step={global_step} "
                    f"loss={last_metrics['total_loss']:.4f} "
                    f"(policy={last_metrics['policy_loss']:.4f}, value={last_metrics['value_loss']:.4f}) "
                    f"samples/s={samples_per_sec:.1f} approx_games/min={games_per_min:.2f}"
                )

    logger("[5/5] Saving artifacts")
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{config.run_name}.pt"
    torchscript_path = out_dir / f"{config.run_name}.torchscript.pt"
    meta_path = out_dir / f"{config.run_name}.meta.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "board_size": board_size,
            "config": {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "seed": config.seed,
                "init_checkpoint_path": config.init_checkpoint_path,
            },
            "source": {
                "train_paths": [str(path) for path in source.train_paths],
                "manifest_path": None if source.manifest_path is None else str(source.manifest_path),
                "manifest_games": source.manifest_games,
                "manifest_samples": source.manifest_samples,
            },
        },
        checkpoint_path,
    )
    scripted = torch.jit.script(model.eval())
    torch.jit.save(scripted, str(torchscript_path))

    elapsed = max(1e-9, time.perf_counter() - start)
    meta = {
        "run_name": config.run_name,
        "board_size": board_size,
        "samples": len(samples),
        "games": unique_games,
        "model_versions": dict(versions),
        "elapsed_sec": elapsed,
        "final_metrics": last_metrics,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "torchscript": str(torchscript_path),
        },
        "source": {
            "train_paths": [str(path) for path in source.train_paths],
            "manifest_path": None if source.manifest_path is None else str(source.manifest_path),
            "manifest_games": source.manifest_games,
            "manifest_samples": source.manifest_samples,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger(
        f"  checkpoint={checkpoint_path}\n"
        f"  torchscript={torchscript_path}\n"
        f"  meta={meta_path}"
    )
    return meta


def resolve_data_source(config: TrainingConfig) -> DataSource:
    if config.train_file:
        path = Path(config.train_file)
        if not path.exists():
            raise FileNotFoundError(f"Train file does not exist: {path}")
        return DataSource(train_paths=[path], manifest_path=None, manifest_games=None, manifest_samples=None)

    manifest_path = Path(config.data_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest does not exist: {manifest_path}. "
            "Run `tictactoe selfplay` first or pass --train-file."
        )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = Path(raw["path"])
    if not train_path.exists():
        raise FileNotFoundError(f"Train file from manifest does not exist: {train_path}")
    train_paths = _resolve_replay_paths(train_path=train_path, replay_shards=config.replay_shards)
    return DataSource(
        train_paths=train_paths,
        manifest_path=manifest_path,
        manifest_games=int(raw.get("games", 0)),
        manifest_samples=int(raw.get("samples", 0)),
    )


def load_samples(path: Path) -> list[SampleRecord]:
    samples: list[SampleRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if raw.get("kind") != "sample":
                continue
            samples.append(_parse_sample(raw, line_no=line_no))
    return samples


def load_samples_from_paths(paths: list[Path]) -> list[SampleRecord]:
    samples: list[SampleRecord] = []
    for path in paths:
        samples.extend(load_samples(path))
    return samples


def _resolve_replay_paths(train_path: Path, replay_shards: int) -> list[Path]:
    if replay_shards <= 1:
        return [train_path]
    parent = train_path.parent
    candidates = sorted(parent.glob("selfplay_*.jsonl"))
    if not candidates:
        return [train_path]
    if train_path not in candidates:
        candidates.append(train_path)
        candidates = sorted(candidates)
    picked = candidates[-max(1, replay_shards) :]
    dedup: list[Path] = []
    for path in picked:
        if path not in dedup:
            dedup.append(path)
    return dedup


def _parse_sample(raw: dict, line_no: int) -> SampleRecord:
    required = [
        "game_id",
        "ply",
        "player_to_move",
        "obs",
        "action_mask",
        "policy_target",
        "value_target",
        "model_version",
    ]
    for key in required:
        if key not in raw:
            raise ValueError(f"Missing key `{key}` at line {line_no}")

    obs = raw["obs"]
    if not isinstance(obs, dict) or "x" not in obs or "o" not in obs or "to_move" not in obs:
        raise ValueError(f"Invalid obs at line {line_no}")
    obs_x = _parse_plane(obs["x"], "obs.x", line_no)
    obs_o = _parse_plane(obs["o"], "obs.o", line_no)
    mask = _parse_plane(raw["action_mask"], "action_mask", line_no)
    if len(obs_x) != len(obs_o) or len(obs_x) != len(mask):
        raise ValueError(f"Inconsistent board shapes at line {line_no}")
    size = len(obs_x)
    if size == 0 or any(len(row) != size for row in obs_x + obs_o + mask):
        raise ValueError(f"Non-square board at line {line_no}")

    policy = raw["policy_target"]
    if not isinstance(policy, dict):
        raise ValueError(f"Invalid policy_target at line {line_no}")
    parsed_policy: dict[str, float] = {}
    for key, value in policy.items():
        row_col = str(key).split(",")
        if len(row_col) != 2:
            raise ValueError(f"Invalid policy key `{key}` at line {line_no}")
        row = int(row_col[0])
        col = int(row_col[1])
        if not (0 <= row < size and 0 <= col < size):
            raise ValueError(f"Policy key out of bounds `{key}` at line {line_no}")
        parsed_policy[f"{row},{col}"] = float(value)

    return SampleRecord(
        game_id=str(raw["game_id"]),
        ply=int(raw["ply"]),
        player_to_move=str(raw["player_to_move"]),
        obs_x=obs_x,
        obs_o=obs_o,
        obs_to_move=str(obs["to_move"]),
        action_mask=mask,
        policy_target=parsed_policy,
        value_target=float(raw["value_target"]),
        model_version=str(raw["model_version"]),
    )


def _parse_plane(value: object, name: str, line_no: int) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list at line {line_no}")
    out: list[list[int]] = []
    for row in value:
        if not isinstance(row, list):
            raise ValueError(f"{name} row must be a list at line {line_no}")
        out.append([int(cell) for cell in row])
    return out


def _build_dataset(torch, samples: list[SampleRecord]):
    xs = []
    policies = []
    values = []
    masks = []
    size = len(samples[0].obs_x)
    for sample in samples:
        if len(sample.obs_x) != size:
            raise ValueError("Mixed board sizes in training file")
        x_plane = torch.tensor(sample.obs_x, dtype=torch.float32)
        o_plane = torch.tensor(sample.obs_o, dtype=torch.float32)
        stm_value = 1.0 if sample.obs_to_move == "X" else 0.0
        stm_plane = torch.full((size, size), stm_value, dtype=torch.float32)
        x = torch.stack((x_plane, o_plane, stm_plane), dim=0)

        policy = torch.zeros((size, size), dtype=torch.float32)
        total = 0.0
        for key, prob in sample.policy_target.items():
            row, col = key.split(",")
            r = int(row)
            c = int(col)
            policy[r, c] = float(prob)
            total += float(prob)
        if total > 0:
            policy /= total
        mask = torch.tensor(sample.action_mask, dtype=torch.float32)

        xs.append(x)
        policies.append(policy)
        values.append(torch.tensor(sample.value_target, dtype=torch.float32))
        masks.append(mask)

    x_tensor = torch.stack(xs, dim=0)
    p_tensor = torch.stack(policies, dim=0)
    v_tensor = torch.stack(values, dim=0)
    m_tensor = torch.stack(masks, dim=0)
    return torch.utils.data.TensorDataset(x_tensor, p_tensor, v_tensor, m_tensor)


def masked_policy_kl_loss(F, logits, target, mask):
    masked_logits = logits.masked_fill(mask <= 0, -1e9)
    log_probs = F.log_softmax(masked_logits.view(masked_logits.shape[0], -1), dim=-1)
    target_flat = target.view(target.shape[0], -1)
    mask_flat = mask.view(mask.shape[0], -1)
    # Defensive: keep policy targets only on legal actions.
    legal_target = target_flat * mask_flat
    legal_sum = legal_target.sum(dim=-1, keepdim=True)
    uniform_legal = mask_flat / mask_flat.sum(dim=-1, keepdim=True).clamp_min(1.0)
    target_norm = legal_target / legal_sum.clamp_min(1e-8)
    has_target = (legal_sum > 0).to(dtype=target_norm.dtype)
    target_norm = target_norm * has_target + uniform_legal * (1.0 - has_target)
    return -(target_norm * log_probs).sum(dim=-1).mean()


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for training. Install with: `uv sync --extra rl`"
        ) from exc
    return torch, nn, F


def _conv3x3(nn, channels: int):
    return nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)


class ResidualBlock:  # created dynamically with torch.nn base in factory
    pass


class SmallPolicyValueNet:  # created dynamically with torch.nn base in factory
    pass


# Dynamically define torch modules after optional import to keep module importable without torch.
try:
    _, _nn, _ = _import_torch()

    class ResidualBlock(_nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = _conv3x3(_nn, channels)
            self.bn1 = _nn.BatchNorm2d(channels)
            self.conv2 = _conv3x3(_nn, channels)
            self.bn2 = _nn.BatchNorm2d(channels)
            self.relu = _nn.ReLU(inplace=True)

        def forward(self, x):
            identity = x
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            x = self.relu(x + identity)
            return x


    class SmallPolicyValueNet(_nn.Module):
        def __init__(self, board_size: int, channels: int = 64, blocks: int = 3) -> None:
            super().__init__()
            self.board_size = board_size
            self.stem = _nn.Sequential(
                _nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
                _nn.BatchNorm2d(channels),
                _nn.ReLU(inplace=True),
            )
            self.trunk = _nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

            self.policy_head = _nn.Sequential(
                _nn.Conv2d(channels, 2, kernel_size=1, bias=False),
                _nn.BatchNorm2d(2),
                _nn.ReLU(inplace=True),
            )
            self.policy_out = _nn.Conv2d(2, 1, kernel_size=1)

            self.value_head = _nn.Sequential(
                _nn.Conv2d(channels, 1, kernel_size=1, bias=False),
                _nn.BatchNorm2d(1),
                _nn.ReLU(inplace=True),
            )
            self.value_fc1 = _nn.Linear(board_size * board_size, 64)
            self.value_fc2 = _nn.Linear(64, 1)
            self.tanh = _nn.Tanh()

        def forward(self, x):
            x = self.stem(x)
            x = self.trunk(x)

            policy = self.policy_head(x)
            policy = self.policy_out(policy).squeeze(1)  # [B, H, W]

            value = self.value_head(x).flatten(1)
            value = _nn.functional.relu(self.value_fc1(value))
            value = self.tanh(self.value_fc2(value))  # [B, 1]
            return policy, value

except RuntimeError:
    pass
