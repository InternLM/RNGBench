# RNG-Bench Playground

A single-page web app to **watch any model play RNG-Bench live**. Switch between
games, plug in one OpenAI-compatible **server URL** (+ model / params), and the
model plays turn by turn while you see its raw replies, the board, and a
scoreboard. The browser talks to *this* backend; the backend proxies the model
call — so there is **no CORS setup** and your API key never leaves your machine.

The per-game logic is reused from the benchmark code (`../1_matching_pairs_new`,
`../2_3d_maze`), so behaviour matches RNG-Bench.

## Run

```bash
pip install -r webapp/requirements.txt
uvicorn webapp.server:app --host 0.0.0.0 --port 8000   # run from the repo root
# open http://localhost:8000
```

## Use

1. Pick a game (top-right switch): **Matching Pairs** or **3D Maze**.
2. **Game setup** — board/maze size, theme, modality, etc. (fields vary per game).
3. **Model endpoint** — server URL (e.g. `https://api.openai.com/v1`), API key
   (optional, sent as Bearer), model name, `temperature`, `max_tokens`.
4. **New game** deals a board. **Auto-play** lets the model run turn after turn;
   **Step** advances one model call. 3D Maze also has manual Forward / Turn buttons.

## Add a new game

The backend is a small adapter registry — adding a game does **not** touch the
server or the frontend:

1. Create `webapp/games/<your_game>.py` with a class subclassing
   `games.base.GameAdapter` (implement `config_schema`, `new_session`, `view`,
   `step`, `done`, and optionally `manual` / `actions`).
2. Register it in `webapp/games/__init__.py` (`REGISTRY`).

The frontend reads `/api/games` and renders the game tab, its config form, board,
and log automatically.

## Notes

- Sessions are in-memory (capped); restarting the server clears them.
- This is a single-board live demo. For multi-seed / multi-config evaluation use
  the CLIs in `1_matching_pairs_new/` and `2_3d_maze/`.
