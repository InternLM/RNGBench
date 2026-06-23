"""Matching Pairs adapter — reuses 1_matching_pairs_new (env, rendering, parser, prompts)."""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

_REPO = Path(__file__).resolve().parent.parent.parent
_MATCH = _REPO / "1_matching_pairs_new"
for p in (str(_MATCH), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from env.matching_env import InvalidActionError, MatchingEnv  # noqa: E402
from env.board import coord_to_str  # noqa: E402
from common.parsing import parse_coord  # noqa: E402
from common.prompts import (  # noqa: E402
    FLIP_FIRST_INSTRUCTION, FLIP_SECOND_INSTRUCTION,
    action_format_hint, build_single_system, retry_invalid_coord, retry_parse_fail,
)

from .base import GameAdapter, img_data_url

_ASSETS = str(_MATCH / "assets")
_THEMES = ["ascii", "noise", "poker", "textures", "perlin", "voronoi"]
MAX_ATTEMPTS = 3  # 1 try + 2 retries (benchmark default)


class MatchingAdapter(GameAdapter):
    id = "matching"
    name = "Matching Pairs"
    description = ("Flip two face-down cards each turn; matched pairs are removed. "
                   "Recall identities seen earlier to find pairs — static, categorical hidden state.")

    def config_schema(self):
        return [
            {"key": "grid", "label": "Board size", "type": "select",
             "options": ["4x4", "6x6", "6x8", "8x8"], "default": "6x6"},
            {"key": "theme", "label": "Card theme", "type": "select",
             "options": _THEMES, "default": "ascii"},
            {"key": "modality", "label": "Sent to model", "type": "select",
             "options": ["image", "text"], "default": "image"},
            {"key": "cot", "label": "Allow chain-of-thought", "type": "bool", "default": True},
        ]

    # ── lifecycle ────────────────────────────────────────────────────────────
    def new_session(self, cfg):
        import random
        rows, cols = (int(x) for x in str(cfg.get("grid", "6x6")).lower().split("x"))
        if rows * cols % 2 or rows * cols > 100:
            raise ValueError("rows×cols must be even and ≤ 100.")
        theme = cfg.get("theme", "ascii")
        if theme not in _THEMES:
            raise ValueError("unknown theme")
        modality = cfg.get("modality", "image")
        cot = bool(cfg.get("cot", True))
        seed = int(cfg.get("seed", random.randint(0, 1_000_000)))
        env = MatchingEnv(rows=rows, cols=cols, seed=seed,
                          theme=(None if theme == "ascii" else theme), assets_dir=_ASSETS)
        env.reset()
        return {
            "env": env, "modality": modality, "cot": cot, "round": 1, "seed": seed,
            "theme": theme,
            "messages": [{"role": "system", "content": build_single_system(cot_enabled=cot)}],
            "stats": {"Score": 0, "Pairs": env.total_pairs, "Model calls": 0,
                      "Parse fail": 0, "Invalid": 0},
        }

    def view(self, s, both=None):
        env = s["env"]
        if both is not None:
            img = env.render_both_flips(both[0], both[1], mode="image")
        else:
            img = env.render_board(mode="image")
        return {"board": img_data_url(img)}

    def info(self, s):
        env = s["env"]
        return f"{env.rows}×{env.cols} · {s['theme']} · {s['modality']} · seed {s['seed']}"

    def stats(self, s):
        return s["stats"]

    def done(self, s):
        return s["env"].get_observation().done

    # ── message construction (mirrors modes/single_normal.py) ─────────────────
    def _msg_flip_first(self, s):
        env, modality, cot = s["env"], s["modality"], s["cot"]
        obs = env.get_observation()
        if obs.last_result is not None:
            v = "Match!" if obs.last_result["matched"] else "No match — cards flipped back."
            pre = f"Previous round: {v} Score: {obs.score}. Remaining pairs: {obs.remaining_pairs}."
        else:
            pre = f"Remaining pairs: {obs.remaining_pairs}. Score: {obs.score}."
        head = f"Round {s['round']} start — current board:"
        instr = FLIP_FIRST_INSTRUCTION + "\n" + action_format_hint(cot)
        if modality == "text":
            return {"role": "user", "content": f"{pre}\n{head}\n{env.render_board(mode='text')}\n{instr}"}
        return {"role": "user", "content": [
            {"type": "text", "text": f"{pre}\n{head}"},
            {"type": "image_url", "image_url": {"url": img_data_url(env.render_board(mode='image'))}},
            {"type": "text", "text": instr}]}

    def _msg_flip_second(self, s):
        env, modality, cot = s["env"], s["modality"], s["cot"]
        instr = FLIP_SECOND_INSTRUCTION + "\n" + action_format_hint(cot)
        if modality == "text":
            return {"role": "user", "content": f"Board after your first flip:\n{env.render_board(mode='text')}\n{instr}"}
        return {"role": "user", "content": [
            {"type": "text", "text": "Board after your first flip:"},
            {"type": "image_url", "image_url": {"url": img_data_url(env.render_board(mode='image'))}},
            {"type": "text", "text": instr}]}

    def _msg_outcome(self, s, c1, c2):
        env, modality = s["env"], s["modality"]
        r = env.get_observation().last_result or {}
        v = "Match!" if r.get("matched") else "No match — cards flipped back."
        tail = f"{v} Score: {env.get_observation().score}. Remaining: {env.get_observation().remaining_pairs}."
        if modality == "text":
            return {"role": "user", "content": f"Both cards revealed:\n{env.render_both_flips(c1, c2, mode='text')}\n{tail}"}
        return {"role": "user", "content": [
            {"type": "text", "text": "Both cards revealed:"},
            {"type": "image_url", "image_url": {"url": img_data_url(env.render_both_flips(c1, c2, mode='image'))}},
            {"type": "text", "text": tail}]}

    # ── one flip (first or second) with retries ───────────────────────────────
    def _do_flip(self, s, responder: Callable[[int], str], attempts: int):
        env = s["env"]
        phase = env.get_observation().phase
        c1_before = env.first_flip_coord
        s["messages"].append(self._msg_flip_first(s) if phase == "flip_first" else self._msg_flip_second(s))
        log, applied, chosen = [], False, None
        for attempt in range(attempts):
            content = responder(attempt)
            s["messages"].append({"role": "assistant", "content": content})
            s["stats"]["Model calls"] += 1
            coord = parse_coord(content, env.rows, env.cols)
            e = {"phase": phase, "attempt": attempt, "raw": content,
                 "coord": coord_to_str(coord) if coord else None}
            if coord is None:
                s["stats"]["Parse fail"] += 1
                e["result"] = "parse_fail"; log.append(e)
                if attempt < attempts - 1:
                    s["messages"].append({"role": "user", "content": retry_parse_fail(s["cot"])})
                continue
            try:
                env.step(coord); applied = True; chosen = coord
                e["result"] = "applied"; log.append(e); break
            except InvalidActionError as ex:
                s["stats"]["Invalid"] += 1
                e["result"] = f"invalid:{ex.code}"; log.append(e)
                if attempt < attempts - 1:
                    s["messages"].append({"role": "user", "content": retry_invalid_coord(ex.reason, s["cot"])})
        both, verdict, round_done = None, None, False
        if not applied:
            if env.phase == "flip_second":
                env.abort_round()
                s["messages"].append({"role": "user", "content":
                    "Could not complete a valid second flip after retries. First card flipped back. Round forfeited."})
            verdict, round_done = "forfeit", True
            s["round"] += 1
        elif phase == "flip_second":
            s["messages"].append(self._msg_outcome(s, c1_before, chosen))
            both = (c1_before, chosen)
            verdict = "match" if (env.get_observation().last_result or {}).get("matched") else "no_match"
            round_done = True
            s["round"] += 1
        s["stats"]["Score"] = env.get_observation().score
        out = {"log": log, "view": self.view(s, both=both), "stats": s["stats"],
               "done": self.done(s), "phase": env.get_observation().phase,
               "verdict": verdict, "round_done": round_done}
        return out

    def step(self, s, call_model):
        return self._do_flip(s, lambda attempt: call_model(s["messages"]), MAX_ATTEMPTS)

    # ── manual: a flip via clicking is driven from the frontend by coord ──────
    def manual(self, s, action):
        # action is a coordinate string like "cD"
        return self._do_flip(s, lambda attempt: f"Action: {action}", 1)
