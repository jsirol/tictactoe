# Self-Play And Training Guide

This document covers how to:
- generate self-play data in headless mode,
- find and use the **latest** self-play manifest,
- run a single headless policy/value training run.

## 1) Prerequisites

From repo root:

```bash
uv sync
uv sync --extra rl
```

`--extra rl` installs PyTorch for training and TorchScript inference.

## 2) Generate Headless Self-Play Data

Run self-play:

```bash
uv run tictactoe selfplay \
  --size 10 \
  --games 200 \
  --workers 12 \
  --batch-size 32 \
  --simulations 200 \
  --time-budget-ms 120 \
  --output-dir data/selfplay \
  --seed 1
```

What gets written:
- `data/selfplay/selfplay_<timestamp>.jsonl`: sample shard for that run
- `data/selfplay/manifest.json`: pointer to the **latest completed run**

Important behavior:
- Every self-play run writes a new shard file.
- `manifest.json` is overwritten each run and always points to the latest shard.

## 3) Confirm The Latest Manifest

Inspect latest manifest:

```bash
cat data/selfplay/manifest.json
```

Expected shape:

```json
{
  "games": 200,
  "samples": 7000,
  "path": "data/selfplay/selfplay_1775580000.jsonl"
}
```

If the path exists, it is ready for training:

```bash
ls -lh "$(python - <<'PY'
import json
print(json.load(open('data/selfplay/manifest.json'))['path'])
PY
)"
```

## 4) Self-Play Data Schema Used For Training

Training consumes JSONL lines with `kind == "sample"` and fields:
- `obs` (planes: `x`, `o`, `to_move`)
- `policy_target` (sparse map `"row,col" -> probability`)
- `value_target` (float in `[-1, 1]`)
- `action_mask` (legal action mask)
- `player_to_move`
- `game_id`
- `ply`
- `model_version`

Non-sample lines (for example `kind == "game"`) are ignored by trainer.

## 5) Run One Headless Training Pass

Train from latest manifest:

```bash
uv run python scripts/train_policy_value.py \
  --data-manifest data/selfplay/manifest.json \
  --epochs 5 \
  --batch-size 128 \
  --lr 1e-3 \
  --out-dir data/models
```

Alternative: train from an explicit shard file:

```bash
uv run python scripts/train_policy_value.py \
  --train-file data/selfplay/selfplay_<timestamp>.jsonl \
  --epochs 5 \
  --batch-size 128 \
  --out-dir data/models
```

## 6) Training Observability

The trainer logs stages and progress:
- `[1/5]` resolve data source
- `[2/5]` validate/load samples
- `[3/5]` build tensors/dataloader
- `[4/5]` train loop
- `[5/5]` save artifacts

During stage 4 it prints:
- `epoch/step`
- total/policy/value loss
- `samples/s`
- `approx_games/min` (derived from samples-per-game in the loaded shard)

## 7) Output Artifacts

For a run named `pv_<timestamp>`:
- `data/models/pv_<timestamp>.pt` (training checkpoint)
- `data/models/pv_<timestamp>.torchscript.pt` (for `--model-path`)
- `data/models/pv_<timestamp>.meta.json` (metrics + source shard metadata)

Use exported model for next self-play:

```bash
uv run tictactoe selfplay \
  --size 10 \
  --games 200 \
  --workers 12 \
  --model-path data/models/pv_<timestamp>.torchscript.pt \
  --output-dir data/selfplay
```

To make web/CLI MCTS use your newest net by default, copy/link your chosen model to:

```bash
cp data/models/pv_<timestamp>.torchscript.pt data/models/latest.torchscript.pt
```

## 8) Typical Loop (Manual)

1. Run self-play to refresh `data/selfplay/manifest.json`.
2. Train one run from that manifest.
3. Use new TorchScript model in the next self-play run.

This repo currently keeps that loop manual by design (no full pipeline orchestrator yet).

## 9) Alternating Loop Helper (Headless)

You can run an alternating helper that performs:
1. self-play with current model,
2. one training update from latest manifest,
3. repeat.

Example:

```bash
uv run python scripts/run_selfplay_train_loop.py \
  --iterations 10 \
  --games-per-run 10 \
  --checkpoint-every 5 \
  --size 10 \
  --workers 12
```

By default, each loop iteration also publishes the newest model to:
`data/models/latest.torchscript.pt` (used automatically by web/CLI MCTS).
You can override with `--web-mcts-model-path`.

The loop warm-starts each training iteration from the previous iteration checkpoint by default.
Disable with `--no-warmstart-from-latest`.

Outputs under `data/models/loops/<run_name>/`:
- `latest.torchscript.pt`
- `best.torchscript.pt` (promoted latest model alias)
- `final.torchscript.pt`
- `checkpoints/iter_XXX.*` every `--checkpoint-every`
- `loop_history.jsonl` with per-iteration metrics

During self-play in each iteration, progress is printed per completed game with running games/min.
Progress lines include `worker=<id>` so parallel execution is visible.
