from __future__ import annotations

import json
import multiprocessing as mp
import queue
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tictactoe.bots import MCTSBot
from tictactoe.core import GameState, Move, Symbol
from tictactoe.search.policy_value import HeuristicPolicyValueModel, PolicyValueModel
from tictactoe.search.torch_policy_value import TorchPolicyValueModel
from tictactoe.search.value_model import HeuristicValueModel


@dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    worker_id: int
    encoded_planes: list[list[list[float]]]
    candidate_indices: list[int]
    board_size: int


@dataclass(frozen=True)
class InferenceResponse:
    request_id: int
    candidate_indices: list[int]
    probabilities: list[float]
    value: float


@dataclass(frozen=True)
class SelfPlayConfig:
    size: int
    games: int
    workers: int
    seed: int | None
    model_path: str | None
    output_dir: str
    batch_size: int
    batch_wait_ms: int
    simulations: int
    time_budget_ms: int
    determinism: str
    high_temperature: float
    low_temperature: float
    temperature_cutoff_ply: int
    max_plies: int | None = None


class QueuePolicyValueClient:
    def __init__(
        self,
        worker_id: int,
        request_queue: mp.Queue[InferenceRequest],
        response_queue: mp.Queue[InferenceResponse],
    ) -> None:
        self._worker_id = worker_id
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._next_request_id = 1

    def predict(
        self, state: GameState, symbol: Symbol, candidate_moves: list[Move]
    ) -> tuple[dict[Move, float], float]:
        board_size = state.board.size
        encoded_planes = _encode_planes(state, symbol)
        candidate_indices = [move.row * board_size + move.col for move in candidate_moves]
        request = InferenceRequest(
            request_id=self._next_request_id,
            worker_id=self._worker_id,
            encoded_planes=encoded_planes,
            candidate_indices=candidate_indices,
            board_size=board_size,
        )
        self._next_request_id += 1
        self._request_queue.put(request)
        while True:
            response = self._response_queue.get()
            if response.request_id == request.request_id:
                priors: dict[Move, float] = {}
                for idx, flat in enumerate(response.candidate_indices):
                    row = flat // board_size
                    col = flat % board_size
                    priors[Move(row, col)] = float(response.probabilities[idx])
                return priors, response.value

    def predict_batch(
        self, items: list[tuple[GameState, Symbol, list[Move]]]
    ) -> list[tuple[dict[Move, float], float]]:
        return [self.predict(state, symbol, moves) for state, symbol, moves in items]


def run_selfplay(config: SelfPlayConfig) -> dict[str, int | float]:
    return run_selfplay_with_progress(config=config, on_progress=None)


def run_selfplay_with_progress(
    config: SelfPlayConfig,
    on_progress: Callable[[dict], None] | None = None,
) -> dict[str, int | float]:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    request_queue: mp.Queue[InferenceRequest] = mp.Queue(maxsize=config.workers * 4)
    worker_response_queues = [mp.Queue(maxsize=32) for _ in range(config.workers)]
    writer_queue: mp.Queue[dict] = mp.Queue(maxsize=config.workers * 2)
    done_queue: mp.Queue[dict] = mp.Queue()
    progress_queue: mp.Queue[dict] = mp.Queue(maxsize=max(64, config.games * 2))

    batcher = mp.Process(
        target=_batcher_main,
        args=(config, request_queue, worker_response_queues),
        daemon=True,
    )
    writer = mp.Process(
        target=_writer_main,
        args=(config.output_dir, writer_queue, done_queue),
        daemon=True,
    )
    batcher.start()
    writer.start()

    base_seed = config.seed if config.seed is not None else int(time.time())
    game_queue: mp.Queue[int | None] = mp.Queue(maxsize=max(16, config.games))
    for game_idx in range(config.games):
        game_queue.put(game_idx)
    for _ in range(config.workers):
        game_queue.put(None)

    workers: list[mp.Process] = []
    for worker_id in range(config.workers):
        proc = mp.Process(
            target=_worker_main,
            args=(
                worker_id,
                config,
                base_seed + worker_id * 100_000,
                game_queue,
                request_queue,
                worker_response_queues[worker_id],
                writer_queue,
                progress_queue,
            ),
            daemon=True,
        )
        workers.append(proc)
        proc.start()

    alive = True
    while alive:
        alive = any(proc.is_alive() for proc in workers)
        try:
            event = progress_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if on_progress is not None:
            on_progress(event)

    for proc in workers:
        proc.join()

    while True:
        try:
            event = progress_queue.get_nowait()
        except queue.Empty:
            break
        if on_progress is not None:
            on_progress(event)

    writer_queue.put({"kind": "shutdown"})
    batcher.terminate()
    batcher.join(timeout=2)
    writer.join(timeout=5)
    elapsed = time.perf_counter() - start
    summary = done_queue.get(timeout=5)
    summary["elapsed_sec"] = elapsed
    summary["games_per_min"] = (config.games / elapsed) * 60.0 if elapsed > 0 else 0.0
    return summary


def _worker_main(
    worker_id: int,
    config: SelfPlayConfig,
    worker_seed: int,
    game_queue: mp.Queue[int | None],
    request_queue: mp.Queue[InferenceRequest],
    response_queue: mp.Queue[InferenceResponse],
    writer_queue: mp.Queue[dict],
    progress_queue: mp.Queue[dict],
) -> None:
    rng = random.Random(worker_seed)
    if config.determinism == "fast":
        rng = random.Random()
    policy_client = QueuePolicyValueClient(worker_id, request_queue, response_queue)
    bot_x = MCTSBot(
        simulations=config.simulations,
        time_budget_ms=config.time_budget_ms,
        policy_value_model=policy_client,
        root_noise_fraction=0.25,
        dirichlet_alpha=0.3,
    )
    bot_o = MCTSBot(
        simulations=config.simulations,
        time_budget_ms=config.time_budget_ms,
        policy_value_model=policy_client,
        root_noise_fraction=0.25,
        dirichlet_alpha=0.3,
    )

    max_plies = config.max_plies if config.max_plies is not None else config.size * config.size
    while True:
        game_idx = game_queue.get()
        if game_idx is None:
            break
        state = GameState.new(size=config.size)
        samples: list[dict] = []
        ply = 0
        game_id = f"{worker_id}:{game_idx}:{worker_seed}"
        while not state.is_over and ply < max_plies:
            symbol = state.next_symbol
            bot = bot_x if symbol is Symbol.X else bot_o
            bot.temperature = (
                config.high_temperature if ply < config.temperature_cutoff_ply else config.low_temperature
            )
            move = bot.choose_move(state=state, symbol=symbol, rng=rng)
            action_mask = [
                [1 if state.board.cells[row][col] is None else 0 for col in range(state.board.size)]
                for row in range(state.board.size)
            ]
            policy = {f"{mv.row},{mv.col}": prob for mv, prob in bot.last_policy.items()}
            samples.append(
                {
                    "kind": "sample",
                    "game_id": game_id,
                    "ply": ply,
                    "seed": worker_seed,
                    "player_to_move": symbol.value,
                    "policy_target": policy,
                    "action": {"row": move.row, "col": move.col},
                    "action_mask": action_mask,
                    "obs": _encode_obs(state, symbol),
                    "model_version": config.model_path or "heuristic",
                }
            )
            state.apply_move(move)
            ply += 1
        outcome = 0.0 if state.winner is None else (1.0 if state.winner is Symbol.X else -1.0)
        capped = (not state.is_over) and ply >= max_plies
        for sample in samples:
            sample["value_target"] = outcome if sample["player_to_move"] == "X" else -outcome
            writer_queue.put(sample)
        writer_queue.put(
            {"kind": "game", "game_id": game_id, "winner": None if state.winner is None else state.winner.value}
        )
        progress_queue.put(
            {
                "kind": "game_progress",
                "worker_id": worker_id,
                "game_id": game_id,
                "samples": len(samples),
                "plies": ply,
                "capped": capped,
            }
        )


def _batcher_main(
    config: SelfPlayConfig,
    request_queue: mp.Queue[InferenceRequest],
    worker_response_queues: list[mp.Queue[InferenceResponse]],
) -> None:
    model = _load_policy_value_model(config.model_path)
    if isinstance(model, TorchPolicyValueModel):
        model.configure_threads(intraop_threads=1, interop_threads=1)
    while True:
        batch: list[InferenceRequest] = []
        try:
            first = request_queue.get(timeout=0.5)
            batch.append(first)
        except queue.Empty:
            continue
        deadline = time.perf_counter() + (config.batch_wait_ms / 1000.0)
        while len(batch) < config.batch_size and time.perf_counter() < deadline:
            try:
                batch.append(request_queue.get_nowait())
            except queue.Empty:
                break
        if isinstance(model, TorchPolicyValueModel):
            encoded = [req.encoded_planes for req in batch]
            candidate_indices_batch = [req.candidate_indices for req in batch]
            outputs = model.predict_encoded_batch(encoded, candidate_indices_batch)
            for req, (probs, value) in zip(batch, outputs):
                worker_response_queues[req.worker_id].put(
                    InferenceResponse(
                        request_id=req.request_id,
                        candidate_indices=req.candidate_indices,
                        probabilities=probs,
                        value=value,
                    )
                )
        else:
            items = [
                (
                    _decode_state(req.encoded_planes, req.board_size),
                    Symbol.X if req.encoded_planes[2][0][0] >= 0.5 else Symbol.O,
                    [Move(flat // req.board_size, flat % req.board_size) for flat in req.candidate_indices],
                )
                for req in batch
            ]
            outputs = model.predict_batch(items)
            for req, (priors, value) in zip(batch, outputs):
                indices: list[int] = []
                probs: list[float] = []
                for move, prob in priors.items():
                    indices.append(move.row * req.board_size + move.col)
                    probs.append(float(prob))
                worker_response_queues[req.worker_id].put(
                    InferenceResponse(
                        request_id=req.request_id,
                        candidate_indices=indices,
                        probabilities=probs,
                        value=value,
                    )
                )


def _writer_main(output_dir: str, writer_queue: mp.Queue[dict], done_queue: mp.Queue[dict]) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"selfplay_{int(time.time())}.jsonl"
    games = 0
    samples = 0
    with file_path.open("w", encoding="utf-8") as f:
        while True:
            item = writer_queue.get()
            if item.get("kind") == "shutdown":
                break
            if item.get("kind") == "game":
                games += 1
            if item.get("kind") == "sample":
                samples += 1
            f.write(json.dumps(item, separators=(",", ":")) + "\n")
    manifest = {"games": games, "samples": samples, "path": str(file_path)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    done_queue.put(manifest)


def _load_policy_value_model(model_path: str | None) -> PolicyValueModel:
    if model_path:
        try:
            return TorchPolicyValueModel(model_path=model_path)
        except Exception:
            # Fall back to heuristic model when torch/runtime is unavailable.
            pass
    return HeuristicPolicyValueModel(HeuristicValueModel())


def _encode_obs(state: GameState, perspective: Symbol) -> dict:
    size = state.board.size
    x = [[0 for _ in range(size)] for _ in range(size)]
    o = [[0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for col in range(size):
            cell = state.board.cells[row][col]
            if cell is Symbol.X:
                x[row][col] = 1
            elif cell is Symbol.O:
                o[row][col] = 1
    return {"x": x, "o": o, "to_move": perspective.value}


def _encode_planes(state: GameState, perspective: Symbol) -> list[list[list[float]]]:
    size = state.board.size
    x = [[0.0 for _ in range(size)] for _ in range(size)]
    o = [[0.0 for _ in range(size)] for _ in range(size)]
    stm_value = 1.0 if perspective is Symbol.X else 0.0
    stm = [[stm_value for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for col in range(size):
            cell = state.board.cells[row][col]
            if cell is Symbol.X:
                x[row][col] = 1.0
            elif cell is Symbol.O:
                o[row][col] = 1.0
    return [x, o, stm]


def _decode_state(encoded_planes: list[list[list[float]]], board_size: int) -> GameState:
    state = GameState.new(size=board_size)
    x_plane = encoded_planes[0]
    o_plane = encoded_planes[1]
    for row in range(board_size):
        for col in range(board_size):
            if x_plane[row][col] >= 0.5:
                state.board.place(Symbol.X, Move(row, col))
            elif o_plane[row][col] >= 0.5:
                state.board.place(Symbol.O, Move(row, col))
    state.next_symbol = Symbol.X if encoded_planes[2][0][0] >= 0.5 else Symbol.O
    return state
