from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .bots import Bot, MCTSBot, RandomBot
from .core import MIN_BOARD_SIZE, GameState, InvalidMove, Move, Symbol

SESSION_COOKIE = "ttt_session_id"


@dataclass
class SessionData:
    state: GameState
    rng: random.Random
    bot: Bot


class NewGameRequest(BaseModel):
    size: int | None = None
    seed: int | None = None
    bot: str | None = None


class MoveRequest(BaseModel):
    row: int
    col: int


def _serialize_state(state: GameState) -> dict[str, Any]:
    board = [[cell.value if cell is not None else None for cell in row] for row in state.board.cells]
    return {
        "size": state.board.size,
        "board": board,
        "next_symbol": state.next_symbol.value,
        "winner": state.winner.value if state.winner else None,
        "is_draw": state.is_draw,
        "is_over": state.is_over,
    }


def _get_bot(name: str) -> Bot:
    if name == "random":
        return RandomBot()
    if name == "mcts":
        return MCTSBot()
    raise ValueError(f"Unsupported bot: {name}")


def _new_session(size: int, seed: int | None, bot_name: str) -> SessionData:
    if size < MIN_BOARD_SIZE:
        raise ValueError(f"Board size must be >= {MIN_BOARD_SIZE}")
    return SessionData(state=GameState.new(size=size), rng=random.Random(seed), bot=_get_bot(bot_name))


def _ensure_session(request: Request) -> tuple[str, SessionData]:
    sessions: dict[str, SessionData] = request.app.state.sessions
    sid = request.cookies.get(SESSION_COOKIE)
    if sid and sid in sessions:
        return sid, sessions[sid]

    sid = uuid.uuid4().hex
    session = _new_session(
        request.app.state.default_size,
        request.app.state.default_seed,
        request.app.state.default_bot,
    )
    sessions[sid] = session
    return sid, session


def create_app(
    default_size: int = 15, default_seed: int | None = None, default_bot: str = "mcts"
) -> FastAPI:
    app = FastAPI(title="Tic Tac Toe Web")
    app.state.sessions = {}
    app.state.default_size = default_size
    app.state.default_seed = default_seed
    app.state.default_bot = default_bot

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tic Tac Toe</title>
  <style>
    :root { --bg:#f5f4ec; --ink:#1f2937; --panel:#ffffff; --accent:#0f766e; --grid:#cbd5e1; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: "Segoe UI", Tahoma, sans-serif; background: radial-gradient(circle at top, #fff, var(--bg)); color:var(--ink); }
    main { max-width: 980px; margin: 1.5rem auto; padding: 1rem; display:grid; gap:1rem; grid-template-columns: 280px 1fr; }
    .panel { background: var(--panel); border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; }
    h1 { margin: 0 0 .75rem; font-size: 1.5rem; }
    .status { min-height: 1.5rem; font-weight: 600; margin-bottom: .75rem; }
    button { background: var(--accent); color:#fff; border:0; border-radius: 8px; padding: .55rem .85rem; cursor:pointer; }
    input { width:100%; padding:.45rem; margin:.35rem 0 .75rem; border:1px solid #cbd5e1; border-radius:8px; }
    #board { display:grid; gap:2px; background: var(--grid); padding:2px; width: fit-content; max-width:100%; overflow:auto; }
    .cell { width:32px; height:32px; border:0; background:#fff; font-size:1rem; font-weight:700; color:#111827; }
    .cell:disabled { cursor: default; background:#f8fafc; }
    .cell.x { color:#be123c; }
    .cell.o { color:#0369a1; }
    @media (max-width: 780px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>Tic Tac Toe</h1>
      <label for="size">Board size</label>
      <input id="size" type="number" min="10" value="15" />
      <label for="seed">Seed (optional)</label>
      <input id="seed" type="number" placeholder="e.g. 42" />
      <button id="new-game">New Game</button>
      <p id="status" class="status"></p>
    </section>
    <section class="panel">
      <div id="board"></div>
    </section>
  </main>
  <script>
    const boardEl = document.getElementById("board");
    const statusEl = document.getElementById("status");
    const sizeEl = document.getElementById("size");
    const seedEl = document.getElementById("seed");
    const newBtn = document.getElementById("new-game");

    async function api(path, options = {}) {
      const response = await fetch(path, {
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({ detail: "Request failed" }));
        throw new Error(payload.detail || "Request failed");
      }
      return response.json();
    }

    function setStatus(state) {
      if (state.winner) {
        statusEl.textContent = "Winner: " + state.winner;
      } else if (state.is_draw) {
        statusEl.textContent = "Draw";
      } else {
        statusEl.textContent = "Turn: " + state.next_symbol;
      }
    }

    function renderBoard(state) {
      boardEl.style.gridTemplateColumns = `repeat(${state.size}, 32px)`;
      boardEl.innerHTML = "";
      setStatus(state);
      for (let row = 0; row < state.size; row++) {
        for (let col = 0; col < state.size; col++) {
          const cell = document.createElement("button");
          cell.className = "cell";
          const value = state.board[row][col];
          if (value === "X") cell.classList.add("x");
          if (value === "O") cell.classList.add("o");
          cell.textContent = value ?? "";
          const disabled = Boolean(value) || state.is_over || state.next_symbol !== "X";
          cell.disabled = disabled;
          cell.addEventListener("click", async () => {
            try {
              const next = await api("/api/game/move", {
                method: "POST",
                body: JSON.stringify({ row, col }),
              });
              renderBoard(next);
            } catch (error) {
              statusEl.textContent = error.message;
            }
          });
          boardEl.appendChild(cell);
        }
      }
    }

    async function loadCurrentGame() {
      try {
        const state = await api("/api/game");
        sizeEl.value = state.size;
        renderBoard(state);
      } catch (error) {
        statusEl.textContent = error.message;
      }
    }

    newBtn.addEventListener("click", async () => {
      try {
        const size = Number(sizeEl.value);
        const seedRaw = seedEl.value.trim();
        const seed = seedRaw === "" ? null : Number(seedRaw);
        const state = await api("/api/game/new", {
          method: "POST",
          body: JSON.stringify({ size, seed }),
        });
        renderBoard(state);
      } catch (error) {
        statusEl.textContent = error.message;
      }
    });

    loadCurrentGame();
  </script>
</body>
</html>"""

    @app.post("/api/game/new")
    def new_game(request: Request, payload: NewGameRequest) -> JSONResponse:
        size = payload.size if payload.size is not None else request.app.state.default_size
        seed = payload.seed if payload.seed is not None else request.app.state.default_seed
        bot_name = payload.bot if payload.bot is not None else request.app.state.default_bot
        try:
            session = _new_session(size=size, seed=seed, bot_name=bot_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        sid = uuid.uuid4().hex
        request.app.state.sessions[sid] = session
        response = JSONResponse(_serialize_state(session.state))
        response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
        return response

    @app.get("/api/game")
    def get_game(request: Request) -> JSONResponse:
        sid, session = _ensure_session(request)
        response = JSONResponse(_serialize_state(session.state))
        response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
        return response

    @app.post("/api/game/move")
    def play_move(request: Request, payload: MoveRequest) -> JSONResponse:
        sid, session = _ensure_session(request)
        move = Move(row=payload.row, col=payload.col)
        try:
            if session.state.next_symbol is not Symbol.X:
                raise InvalidMove("It is not the human player's turn")
            session.state.apply_move(move)
            if not session.state.is_over and session.state.next_symbol is Symbol.O:
                bot_move = session.bot.choose_move(session.state, Symbol.O, session.rng)
                session.state.apply_move(bot_move)
        except InvalidMove as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        response = JSONResponse(_serialize_state(session.state))
        response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
        return response

    return app
