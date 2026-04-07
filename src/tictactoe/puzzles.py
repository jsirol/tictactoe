from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tictactoe.core import GameState, Move, Symbol
from tictactoe.search.tactics import ThreatKind


@dataclass(frozen=True)
class TacticalPuzzle:
    puzzle_id: str
    size: int
    to_move: Symbol
    stones: tuple[tuple[Symbol, Move], ...]
    expected_moves: tuple[Move, ...]
    expected_kind: ThreatKind | None
    max_ply: int | None

    def to_state(self) -> GameState:
        state = GameState.new(size=self.size)
        for symbol, move in self.stones:
            state.board.place(symbol, move)
        state.next_symbol = self.to_move
        return state


def load_puzzle_pack(path: str | Path) -> list[TacticalPuzzle]:
    puzzles: list[TacticalPuzzle] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {p}:{lineno}") from exc
            puzzles.append(_parse_puzzle(obj, source=f"{p}:{lineno}"))
    return puzzles


def _parse_puzzle(obj: dict, source: str) -> TacticalPuzzle:
    try:
        pid = str(obj["id"])
        size = int(obj["size"])
        to_move = Symbol(str(obj["to_move"]))
        stones_raw = obj["stones"]
        expected_moves_raw = obj["expected_moves"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid puzzle fields in {source}") from exc

    stones: list[tuple[Symbol, Move]] = []
    for item in stones_raw:
        try:
            symbol = Symbol(str(item["symbol"]))
            row = int(item["row"])
            col = int(item["col"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid stone in {source}") from exc
        stones.append((symbol, Move(row=row, col=col)))

    expected_moves: list[Move] = []
    for item in expected_moves_raw:
        try:
            row = int(item["row"])
            col = int(item["col"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid expected move in {source}") from exc
        expected_moves.append(Move(row=row, col=col))

    expected_kind: ThreatKind | None
    if obj.get("expected_kind") is None:
        expected_kind = None
    else:
        try:
            expected_kind = ThreatKind(str(obj["expected_kind"]))
        except ValueError as exc:
            raise ValueError(f"Invalid expected_kind in {source}") from exc

    max_ply = obj.get("max_ply")
    max_ply = None if max_ply is None else int(max_ply)

    return TacticalPuzzle(
        puzzle_id=pid,
        size=size,
        to_move=to_move,
        stones=tuple(stones),
        expected_moves=tuple(expected_moves),
        expected_kind=expected_kind,
        max_ply=max_ply,
    )
