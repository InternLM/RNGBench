"""
RNG-Bench Playground — unified, game-agnostic backend.

One page, switchable games. Plug in any OpenAI-compatible endpoint and watch a
model play live; the per-game logic is reused from the benchmark code via small
adapters (see games/). The browser never calls the model directly — config is
POSTed here and the call is proxied server-side (no CORS, key stays on your box).

Run from the repo root:
    pip install -r webapp/requirements.txt
    uvicorn webapp.server:app --host 0.0.0.0 --port 8000
    # open http://localhost:8000
"""

import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for p in (str(_REPO), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from games import REGISTRY  # noqa: E402

_STATIC = _HERE / "static"
MAX_SESSIONS = 300
SESSIONS: Dict[str, Dict[str, Any]] = {}  # sid -> {"game": id, "state": {...}}

app = FastAPI(title="RNG-Bench Playground")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── model proxy ───────────────────────────────────────────────────────────────
class ModelCfg(BaseModel):
    api_base: str
    api_key: str = ""
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048


def _make_call_model(cfg: ModelCfg):
    url = cfg.api_base.strip().rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg.api_key.strip():
        headers["Authorization"] = f"Bearer {cfg.api_key.strip()}"

    def call(messages: List[Dict[str, Any]]) -> str:
        body = {"model": cfg.model, "messages": messages,
                "temperature": cfg.temperature, "max_tokens": cfg.max_tokens, "stream": False}
        try:
            with httpx.Client(timeout=180.0) as c:
                r = c.post(url, json=body, headers=headers)
        except httpx.RequestError as e:
            raise HTTPException(502, f"Could not reach endpoint: {e}")
        if r.status_code >= 400:
            raise HTTPException(502, f"Upstream {r.status_code}: {r.text[:400]}")
        try:
            msg = r.json()["choices"][0]["message"]
        except Exception:
            raise HTTPException(502, f"Unexpected response: {r.text[:400]}")
        content = msg.get("content") or ""
        if not content.strip() and msg.get("reasoning_content"):
            content = msg["reasoning_content"]
        return content
    return call


def _sess(sid: str):
    s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "Session not found — start a new game.")
    return s, REGISTRY[s["game"]]


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/games")
def list_games():
    return [{"id": a.id, "name": a.name, "description": a.description,
             "config": a.config_schema()} for a in REGISTRY.values()]


class NewReq(BaseModel):
    game: str
    config: Dict[str, Any] = {}


@app.post("/api/new")
def new_game(req: NewReq):
    adapter = REGISTRY.get(req.game)
    if adapter is None:
        raise HTTPException(400, f"Unknown game '{req.game}'.")
    try:
        state = adapter.new_session(req.config)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if len(SESSIONS) >= MAX_SESSIONS:
        SESSIONS.pop(next(iter(SESSIONS)))
    sid = uuid.uuid4().hex
    SESSIONS[sid] = {"game": adapter.id, "state": state}
    return {"session": sid, "game": adapter.id, "name": adapter.name,
            "view": adapter.view(state), "info": adapter.info(state),
            "stats": adapter.stats(state), "done": adapter.done(state),
            "actions": adapter.actions(state)}


class StepReq(ModelCfg):
    session: str


@app.post("/api/step")
def step(req: StepReq):
    s, adapter = _sess(req.session)
    if adapter.done(s["state"]):
        return {"done": True, "view": adapter.view(s["state"]), "stats": adapter.stats(s["state"]), "log": []}
    return adapter.step(s["state"], _make_call_model(req))


class ManualReq(BaseModel):
    session: str
    action: str


@app.post("/api/manual")
def manual(req: ManualReq):
    s, adapter = _sess(req.session)
    if adapter.done(s["state"]):
        return {"done": True, "view": adapter.view(s["state"]), "stats": adapter.stats(s["state"]), "log": []}
    try:
        return adapter.manual(s["state"], req.action)
    except NotImplementedError as e:
        raise HTTPException(400, str(e))


@app.get("/")
def index():
    return FileResponse(str(_STATIC / "index.html"))


app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
