"""3D Maze adapter — reuses 2_3d_maze (MazeGame3D, parser, prompts, step-message builder)."""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_MAZE = _REPO / "2_3d_maze"
for p in (str(_MAZE), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from game import MazeGame3D, parse_action  # noqa: E402
from runner import SYSTEM_PROMPT_V5, SYSTEM_PROMPT_V5_MINIMAP, _build_step_message  # noqa: E402

from .base import GameAdapter, img_data_url

MAX_ATTEMPTS = 3
_SIZES = ["7", "9", "11", "13"]


class MazeAdapter(GameAdapter):
    id = "maze"
    name = "3D Maze"
    description = ("Navigate a first-person maze to the goal. Assemble egocentric views into a "
                  "mental map — dynamic, spatial hidden state. Toggle the minimap for the Memory-Gap oracle.")

    def config_schema(self):
        return [
            {"key": "maze_size", "label": "Maze size", "type": "select",
             "options": _SIZES, "default": "9"},
            {"key": "minimap", "label": "Show minimap (Memory-Gap oracle)", "type": "bool", "default": False},
        ]

    def new_session(self, cfg):
        import random
        size = int(cfg.get("maze_size", 9))
        if size % 2 == 0 or not (5 <= size <= 15):
            raise ValueError("maze_size must be odd, 5–15.")
        minimap = bool(cfg.get("minimap", False))
        seed = int(cfg.get("seed", random.randint(0, 1_000_000)))
        env = MazeGame3D(maze_size=size, seed=seed, vision_range=4,
                         action_space="v5", maze_type="v6")
        optimal = env.compute_optimal_path_length()
        max_steps = max(80, 4 * optimal)
        tmpl = SYSTEM_PROMPT_V5_MINIMAP if minimap else SYSTEM_PROMPT_V5
        system = tmpl.format(vision_range=env.vision_range, maze_size=size, max_steps=max_steps)
        return {
            "env": env, "minimap": minimap, "seed": seed, "size": size,
            "optimal": optimal, "max_steps": max_steps, "feedback": "",
            "messages": [{"role": "system", "content": system}],
            "stats": {"Steps": 0, "Explored": f"0/{env.total_cells()}",
                      "Optimal": optimal, "Reached": "no"},
        }

    def view(self, s):
        env = s["env"]
        img = env.render_frame(show_minimap=s["minimap"], show_hud=True)
        return {"board": img_data_url(img)}

    def info(self, s):
        mm = " · minimap" if s["minimap"] else ""
        return f"{s['size']}×{s['size']} · v5 actions · optimal {s['optimal']} · max {s['max_steps']}{mm} · seed {s['seed']}"

    def stats(self, s):
        return s["stats"]

    def done(self, s):
        return s["env"].is_game_over() or s["env"].step_count >= s["max_steps"]

    def _refresh_stats(self, s):
        env = s["env"]
        s["stats"]["Steps"] = env.step_count
        s["stats"]["Explored"] = f"{env.explored_cells()}/{env.total_cells()}"
        s["stats"]["Reached"] = "yes" if env.is_game_over() else "no"

    def _apply(self, s, responder, attempts):
        env = s["env"]
        step_num = env.step_count + 1
        msg = _build_step_message(env, step_num, prefix=s.get("feedback", ""),
                                  show_minimap=s["minimap"], show_hud=True, obs_mode="scene")
        s["messages"].append(msg)
        log, applied, action = [], False, None
        for attempt in range(attempts):
            content = responder(attempt)
            s["messages"].append({"role": "assistant", "content": content})
            s["stats"].setdefault("Model calls", 0)
            s["stats"]["Model calls"] += 1
            action = parse_action(content, "v5")
            e = {"attempt": attempt, "raw": content, "action": action}
            if action is None:
                e["result"] = "parse_fail"; log.append(e)
                if attempt < attempts - 1:
                    s["messages"].append({"role": "user", "content":
                        "Could not parse an action. Reply with a final line: "
                        "Action: <move_forward|turn_left|turn_right>"})
                continue
            ok, info = env.move(action)
            e["result"] = "moved" if ok else "blocked"
            e["info"] = info
            log.append(e)
            s["feedback"] = info + "\n" if info else ""
            applied = True
            break
        self._refresh_stats(s)
        reached = env.is_game_over()
        out_of_steps = env.step_count >= s["max_steps"]
        verdict = "reached" if reached else ("failed" if out_of_steps else None)
        return {"log": log, "view": self.view(s), "stats": s["stats"],
                "done": reached or out_of_steps, "action": action,
                "verdict": verdict, "round_done": bool(verdict)}

    def step(self, s, call_model):
        return self._apply(s, lambda attempt: call_model(s["messages"]), MAX_ATTEMPTS)

    def actions(self, s):
        return [{"action": "move_forward", "label": "Forward"},
                {"action": "turn_left", "label": "Turn L"},
                {"action": "turn_right", "label": "Turn R"}]

    def manual(self, s, action):
        return self._apply(s, lambda attempt: f"Action: {action}", 1)
