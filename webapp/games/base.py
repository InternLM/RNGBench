"""
Game-adapter interface for the RNG-Bench playground.

Adding a new game = drop a new module under `games/` that subclasses
`GameAdapter` and register it in `games/__init__.py`. The server stays generic:
it only knows about sessions and forwards model calls.

An adapter owns its own per-session state (a plain dict). The server passes a
`call_model(messages) -> str` closure into `step`, so adapters never touch HTTP,
CORS, or API keys.
"""

import base64
import io
from typing import Any, Callable, Dict, List

CallModel = Callable[[List[Dict[str, Any]]], str]


def img_data_url(pil_img) -> str:
    """PIL.Image -> base64 PNG data URL (for the browser and for vision models)."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class GameAdapter:
    """Base class. Subclasses set id/name/description and implement the methods."""

    id: str = "game"
    name: str = "Game"
    description: str = ""

    # ── metadata for the frontend ────────────────────────────────────────────
    def config_schema(self) -> List[Dict[str, Any]]:
        """Form fields the UI renders. Each: {key,label,type(select|bool|int),...}."""
        return []

    # ── lifecycle ────────────────────────────────────────────────────────────
    def new_session(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Create and return a fresh per-session state dict."""
        raise NotImplementedError

    def view(self, s: Dict[str, Any]) -> Dict[str, Any]:
        """Current visualization: {board: <data-url>, caption?: str}."""
        raise NotImplementedError

    def info(self, s: Dict[str, Any]) -> str:
        """Short one-line config summary (size/seed/...)."""
        return ""

    def stats(self, s: Dict[str, Any]) -> Dict[str, Any]:
        """Scoreboard chips, e.g. {Score: 3, ...}."""
        return {}

    def done(self, s: Dict[str, Any]) -> bool:
        raise NotImplementedError

    # ── one model turn ───────────────────────────────────────────────────────
    def step(self, s: Dict[str, Any], call_model: CallModel) -> Dict[str, Any]:
        """Run ONE model call (with the game's own retry policy) and apply it.

        Returns: {log: [entries], view: {...}, stats: {...}, done: bool,
                  phase?: str, note?: str}
        """
        raise NotImplementedError

    # ── optional human action (click / button) ───────────────────────────────
    def manual(self, s: Dict[str, Any], action: str) -> Dict[str, Any]:
        raise NotImplementedError("This game does not support manual play.")

    def actions(self, s: Dict[str, Any]) -> List[Dict[str, str]]:
        """Manual-control buttons the UI shows: [{action, label}]. Empty = none."""
        return []
