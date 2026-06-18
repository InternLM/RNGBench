"""
Split a game.json into viewer-friendly states / meta.

Each turn -> one state (turn_id / call_type / round / player / phase / applied / verdict /
messages_sent / response / parse / env_result). The HTML template consumes the list of states directly.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional


def load_game(path: str) -> Dict[str, Any]:
    import json
    with open(path) as f:
        return json.load(f)


def _infer_phase_before(call_type: str) -> str:
    """call_type tells us what the model was asked to do this time."""
    if call_type in ("flip_first", "flip_second"):
        return call_type
    return "retry"


def _short_verdict(env_result: Dict[str, Any], parse: Dict[str, Any], call_type: str) -> str:
    if not parse.get("ok"):
        return "parse_fail"
    if not env_result.get("applied"):
        return f"invalid: {env_result.get('error', '?')}"
    if call_type == "flip_first":
        face = env_result.get("revealed_face")
        return f"flipped first → {face}"
    if call_type == "flip_second":
        face = env_result.get("revealed_face")
        phase_after = env_result.get("phase_after", "")
        # phase_after == "flip_first" means resolve happened
        return f"flipped second → {face}"
    return "applied"


def build_states(game: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One state per turn."""
    states: List[Dict[str, Any]] = []
    for t in game.get("turns", []):
        parse = t.get("parse") or {}
        env_result = t.get("env_result") or {}
        states.append({
            "turn_id": t["turn_id"],
            "call_type": t.get("call_type", ""),
            "retry_of": t.get("retry_of"),
            "round": t.get("round"),
            "player": t.get("player"),
            "phase_before": _infer_phase_before(t.get("call_type", "")),
            "parse": parse,
            "env_result": env_result,
            "verdict": _short_verdict(env_result, parse, t.get("call_type", "")),
            "messages_sent": t.get("messages_sent", []),
            "response": t.get("response", {}),
        })
    return states


def extract_meta(game: Dict[str, Any]) -> Dict[str, Any]:
    cfg = game.get("config", {})
    res = game.get("result", {})
    mode = cfg.get("mode", "")
    is_dual = mode.startswith("dual")

    if is_dual:
        title = f"{cfg.get('label_a', cfg.get('model_a', 'A'))}  vs  {cfg.get('label_b', cfg.get('model_b', 'B'))}"
        subtitle_scores = f"A: {res.get('score_a', 0)}  ·  B: {res.get('score_b', 0)}  /  {res.get('total_pairs', '?')}"
        render_desc = f"render_a={cfg.get('render_a')}, render_b={cfg.get('render_b')}"
    else:
        title = cfg.get("label", cfg.get("model", "model"))
        subtitle_scores = f"score: {res.get('score', 0)} / {res.get('total_pairs', '?')}"
        render_desc = f"render={cfg.get('render', '?')}"

    return {
        "mode": mode,
        "is_dual": is_dual,
        "title": title,
        "subtitle_scores": subtitle_scores,
        "render_desc": render_desc,
        "theme": cfg.get("theme"),
        "rows": cfg.get("rows"),
        "cols": cfg.get("cols"),
        "seed": cfg.get("seed"),
        "total_pairs": res.get("total_pairs"),
        "rounds_played": res.get("rounds_played"),
        "response_count": res.get("response_count"),
        "done": res.get("done"),
        "error": res.get("error"),
        "ground_truth": cfg.get("ground_truth"),
    }
