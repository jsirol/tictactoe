from fastapi.testclient import TestClient

from tictactoe.web import create_app


def test_index_page_loads():
    client = TestClient(create_app(default_size=10, default_seed=1))
    response = client.get("/")
    assert response.status_code == 200
    assert "Tic Tac Toe" in response.text
    assert "New Game" in response.text


def test_new_game_creates_empty_board_and_sets_cookie():
    client = TestClient(create_app(default_size=10, default_seed=1))
    response = client.post("/api/game/new", json={"size": 10, "seed": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["size"] == 10
    assert payload["next_symbol"] == "X"
    assert payload["winner"] is None
    assert payload["is_over"] is False
    assert payload["board"][0][0] is None
    assert "ttt_session_id=" in response.headers["set-cookie"]


def test_get_game_returns_same_session_game():
    client = TestClient(create_app(default_size=10, default_seed=1))
    client.post("/api/game/new", json={"size": 10, "seed": 3})
    response = client.get("/api/game")
    assert response.status_code == 200
    assert response.json()["size"] == 10


def test_human_move_is_followed_by_bot_move_when_game_continues():
    client = TestClient(create_app(default_size=10, default_seed=0))
    client.post("/api/game/new", json={"size": 10, "seed": 0})
    response = client.post("/api/game/move", json={"row": 0, "col": 0})
    assert response.status_code == 200
    board = response.json()["board"]
    occupied = sum(1 for row in board for cell in row if cell is not None)
    assert board[0][0] == "X"
    assert occupied == 2
    assert response.json()["next_symbol"] == "X"


def test_invalid_human_move_returns_400():
    client = TestClient(create_app(default_size=10, default_seed=0))
    client.post("/api/game/new", json={"size": 10, "seed": 0})
    response = client.post("/api/game/move", json={"row": 0, "col": 100})
    assert response.status_code == 400
    assert "out of bounds" in response.json()["detail"].lower()


def test_new_game_rejects_small_size():
    client = TestClient(create_app(default_size=10, default_seed=0))
    response = client.post("/api/game/new", json={"size": 9})
    assert response.status_code == 400
