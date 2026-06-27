"""
3D first-person maze LLM Runner.

Evaluates a multimodal LLM's navigation ability in a 3D fog-of-war maze.
Each step the model is given a screenshot of the current view (base64 JPEG)
and outputs an action command.

Action space: forward / backward / turn_left / turn_right

Fairness guarantees:
  - same seed -> identical maze structure and start point
  - same vision_range -> identical fog-of-war visibility
  - same max_steps -> identical step budget
  - the RNG seed is separate from fallback_rng_seed, so the model cannot
    influence maze generation
"""

import json
import logging
import os
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

load_dotenv(_REPO / ".env")

from framework.llm_client import LLMClient  # shared client (was embedded here)
from game import ACTIONS, ACTIONS_V5, MazeGame3D, parse_action

logger = logging.getLogger(__name__)

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are navigating a 3D first-person maze. Your goal is to reach the Goal (orange marker) from the Start in as few actions as possible.

=== Image Layout ===
The image you receive each turn has two parts:

1. **Main 3D View** (large area): Your first-person perspective inside the maze.
   - You see corridors and walls rendered in perspective.
   - Fog of war: only ~{vision_range} cells ahead are visible; beyond that is complete darkness.
   - Walls darken with distance (fog effect).

2. **HUD bar** (bottom strip):
   - Your current cell coordinates (Pos), facing direction (Facing), goal coordinates (Goal).
   - Steps taken, cells explored, and available actions.

=== Actions ===
Each turn, choose exactly ONE action:
  - `forward`     — Move one cell in your current facing direction (if not blocked by wall)
  - `backward`    — Move one cell in the opposite direction (if not blocked)
  - `turn_left`   — Rotate 90° counterclockwise (N→W, W→S, S→E, E→N). No position change.
  - `turn_right`  — Rotate 90° clockwise (N→E, E→S, S→W, W→N). No position change.

Turn actions cost 1 step but do NOT move you. Movement actions also cost 1 step and physically move you.

=== Navigation Strategy ===
- The Goal is at the BOTTOM-RIGHT corner of the maze (shown in HUD).
- Build a mental map from your observations over time. You will NOT receive a movement history summary — rely on your memory of this conversation.
- Plan ahead: turning wastes steps, so orient yourself correctly before moving.
- Use systematic strategies (e.g., right-hand wall following, DFS) to ensure coverage.

=== Output Format (STRICT) ===
Thought: <brief reasoning: current position, what you see, plan>
Action: <forward|backward|turn_left|turn_right>

Maze Config: {maze_size}×{maze_size} cells | Vision: {vision_range} cells | Max Steps: {max_steps}
"""

SYSTEM_PROMPT_MAP = """\
You are navigating a 3D first-person maze. Your goal is to reach the Goal (orange marker) from the Start in as few actions as possible.

=== Image Layout ===
The image you receive each turn has two parts:

1. **Main 3D View** (large area): Your first-person perspective inside the maze.
   - You see corridors and walls rendered in perspective.
   - Fog of war: only ~{vision_range} cells ahead are visible; beyond that is complete darkness.
   - Walls darken with distance (fog effect).

2. **HUD bar** (bottom strip):
   - Your current cell coordinates (Pos), facing direction (Facing), goal coordinates (Goal).
   - Steps taken, cells explored, and available actions.

=== Actions ===
Each turn, choose exactly ONE action:
  - `forward`     — Move one cell in your current facing direction (if not blocked by wall)
  - `backward`    — Move one cell in the opposite direction (if not blocked)
  - `turn_left`   — Rotate 90° counterclockwise (N→W, W→S, S→E, E→N). No position change.
  - `turn_right`  — Rotate 90° clockwise (N→E, E→S, S→W, W→N). No position change.

Turn actions cost 1 step but do NOT move you. Movement actions also cost 1 step and physically move you.

=== Navigation Strategy ===
- The Goal is at the BOTTOM-RIGHT corner of the maze (shown in HUD).
- Build and maintain an ASCII map from your observations. Update it every step.
- Plan ahead: turning wastes steps, so orient yourself correctly before moving.
- Use systematic strategies (e.g., right-hand wall following, DFS) to ensure coverage.

=== Output Format (STRICT) ===
Thought: <brief reasoning: current position, what you see, plan>
Map:
<your {wall_size}×{wall_size} wall map — cells at odd rows/cols, walls at even rows/cols>
Legend: #=wall  ' '=open passage  S=start  G=goal  @=you  .=visited  ?=unknown cell
Rows increase downward (row 0 = top), columns increase rightward (col 0 = left).
Cell (r,c) is at grid position (2r, 2c). Passage between (r,c) and (r,c+1) is at grid (2r, 2c+1).
Action: <forward|backward|turn_left|turn_right>

Maze Config: {maze_size}×{maze_size} cells | Vision: {vision_range} cells | Max Steps: {max_steps}
"""

SYSTEM_PROMPT_V5 = """\
You are navigating a 3D first-person maze. Your goal is to reach the Goal (orange marker) from the Start in as few actions as possible.

=== Image Layout ===
The image you receive each turn has two parts:

1. **Main 3D View** (large area): Your first-person perspective inside the maze.
   - You see corridors and walls rendered in perspective.
   - Fog of war: only ~{vision_range} cells ahead are visible; beyond that is complete darkness.
   - Walls darken with distance (fog effect).

2. **HUD bar** (bottom strip):
   - Your current cell coordinates (Pos), facing direction (Facing), goal coordinates (Goal).
   - Steps taken, cells explored, and available actions.

=== Actions ===
Each turn, choose exactly ONE action:
  - `move_forward`  — Move forward in your current facing direction, stopping automatically when you hit a wall OR reach a junction (any cell where you can turn or branch). You will pass through all intermediate corridor cells. Costs 1 step regardless of distance traveled.
  - `turn_left`     — Rotate 90° counterclockwise (N→W, W→S, S→E, E→N). No position change.
  - `turn_right`    — Rotate 90° clockwise (N→E, E→S, S→W, W→N). No position change.

Each move_forward carries you to the next decision point (wall or junction) in one step. Plan your turns carefully before moving.

=== Navigation Strategy ===
- The Goal is at the BOTTOM-RIGHT corner of the maze (shown in HUD).
- Build a mental map from your observations over time. You will NOT receive a movement history summary — rely on your memory of this conversation.
- Plan ahead: each move_forward takes you to the next wall or junction, so orient yourself correctly before moving.
- Use systematic strategies (e.g., right-hand wall following, DFS) to ensure coverage.

=== Output Format (STRICT) ===
Thought: <brief reasoning: current position, what you see, plan>
Action: <move_forward|turn_left|turn_right>

Maze Config: {maze_size}×{maze_size} cells | Vision: {vision_range} cells | Max Steps: {max_steps}
"""

SYSTEM_PROMPT_V5_MINIMAP = """\
You are navigating a 3D first-person maze. Your goal is to reach the Goal (orange marker) from the Start in as few actions as possible.

=== Image Layout ===
The image you receive each turn has THREE parts:

1. **Main 3D View** (large area): Your first-person perspective inside the maze.
   - You see corridors and walls rendered in perspective.
   - Fog of war: only ~{vision_range} cells ahead are visible; beyond that is complete darkness.
   - Walls darken with distance (fog effect).

2. **Mini-map** (top-right overlay, ~140×140 px): A top-down bird's-eye map of the maze.
   - Only cells you have already visited or seen are revealed (fog of war applies here too).
   - Light grey/white cells: explored open passages.
   - Dark grid lines between cells: walls.
   - Orange cell: Goal location.
   - Blue dot + arrow: your current position and facing direction (arrow points the way you face).
   - Black areas: unexplored regions you haven't visited yet.
   - Use this map to track where you've been and plan your route to the Goal.

3. **HUD bar** (bottom strip):
   - Your current cell coordinates (Pos), facing direction (Facing), goal coordinates (Goal).
   - Steps taken, cells explored, and available actions.

=== Actions ===
Each turn, choose exactly ONE action:
  - `move_forward`  — Move forward in your current facing direction, stopping automatically when you hit a wall OR reach a junction (any cell where you can turn or branch). You will pass through all intermediate corridor cells. Costs 1 step regardless of distance traveled.
  - `turn_left`     — Rotate 90° counterclockwise (N→W, W→S, S→E, E→N). No position change.
  - `turn_right`    — Rotate 90° clockwise (N→E, E→S, S→W, W→N). No position change.

Each move_forward carries you to the next decision point (wall or junction) in one step. Plan your turns carefully before moving.

=== Navigation Strategy ===
- The Goal is at the BOTTOM-RIGHT corner of the maze (shown in HUD and on the mini-map as an orange cell).
- Use the mini-map to avoid revisiting explored dead-ends and to track unexplored regions.
- Plan ahead: each move_forward takes you to the next wall or junction, so orient yourself correctly before moving.
- Use systematic strategies (e.g., right-hand wall following, DFS) to ensure coverage.

=== Output Format (STRICT) ===
Thought: <brief reasoning: current position, what you see on the mini-map, plan>
Action: <move_forward|turn_left|turn_right>

Maze Config: {maze_size}×{maze_size} cells | Vision: {vision_range} cells | Max Steps: {max_steps}
"""

SYSTEM_PROMPT_V5_TRAJECTORY = """\
You are navigating a 3D first-person maze. Your goal is to reach the Goal (orange marker) from the Start in as few actions as possible.

=== Image Layout ===
The image you receive each turn has two parts:

1. **Main 3D View** (large area): Your first-person perspective inside the maze.
   - You see corridors and walls rendered in perspective.
   - Fog of war: only ~{vision_range} cells ahead are visible; beyond that is complete darkness.
   - Walls darken with distance (fog effect).

2. **HUD bar** (bottom strip):
   - Your current cell coordinates (Pos), facing direction (Facing), goal coordinates (Goal).
   - Steps taken, cells explored, and available actions.

=== Actions ===
Each turn, choose exactly ONE action:
  - `move_forward`  — Move forward in your current facing direction, stopping automatically when you hit a wall OR reach a junction (any cell where you can turn or branch). You will pass through all intermediate corridor cells. Costs 1 step regardless of distance traveled.
  - `turn_left`     — Rotate 90° counterclockwise (N→W, W→S, S→E, E→N). No position change.
  - `turn_right`    — Rotate 90° clockwise (N→E, E→S, S→W, W→N). No position change.

Each move_forward carries you to the next decision point (wall or junction) in one step. Plan your turns carefully before moving.

=== Navigation Strategy ===
- The Goal is at the BOTTOM-RIGHT corner of the maze (shown in HUD).
- Build and maintain a simple visited-cells map from your observations every step.
- Only mark cells you have visited or can infer directly from your own movement; keep unknown cells as `?`.
- Plan ahead: each move_forward takes you to the next wall or junction, so orient yourself correctly before moving.
- Use systematic strategies (e.g., right-hand wall following, DFS) to ensure coverage.

=== Output Format (STRICT) ===
Thought: <brief reasoning: current position, what you see, plan>
Map:
<your {maze_size}×{maze_size} ASCII map — one row per line, columns space-separated>
Legend: S=start  G=goal  @=you  .=visited empty  ?=unknown
Rows increase downward (row 0 = top), columns increase rightward (col 0 = left).
Action: <move_forward|turn_left|turn_right>

Maze Config: {maze_size}×{maze_size} cells | Vision: {vision_range} cells | Max Steps: {max_steps}
"""

SYSTEM_PROMPT_V5_MAP = """\
You are navigating a 3D first-person maze. Your goal is to reach the Goal (orange marker) from the Start in as few actions as possible.

=== Image Layout ===
The image you receive each turn has two parts:

1. **Main 3D View** (large area): Your first-person perspective inside the maze.
   - You see corridors and walls rendered in perspective.
   - Fog of war: only ~{vision_range} cells ahead are visible; beyond that is complete darkness.
   - Walls darken with distance (fog effect).

2. **HUD bar** (bottom strip):
   - Your current cell coordinates (Pos), facing direction (Facing), goal coordinates (Goal).
   - Steps taken, cells explored, and available actions.

=== Actions ===
Each turn, choose exactly ONE action:
  - `move_forward`  — Move forward in your current facing direction, stopping automatically when you hit a wall OR reach a junction (any cell where you can turn or branch). You will pass through all intermediate corridor cells. Costs 1 step regardless of distance traveled.
  - `turn_left`     — Rotate 90° counterclockwise (N→W, W→S, S→E, E→N). No position change.
  - `turn_right`    — Rotate 90° clockwise (N→E, E→S, S→W, W→N). No position change.

Each move_forward carries you to the next decision point (wall or junction) in one step. Plan your turns carefully before moving.

=== Navigation Strategy ===
- The Goal is at the BOTTOM-RIGHT corner of the maze (shown in HUD).
- Build and maintain a wall map from your observations every step. Use it to plan routes.
- Plan ahead: each move_forward takes you to the next wall or junction, so orient yourself correctly before moving.
- Use systematic strategies (e.g., right-hand wall following, DFS) to ensure coverage.

=== Wall Map Format ===
The map is a {wall_size}×{wall_size} grid (no spaces between characters):
  - Outer border: always `#`
  - Cell (r,c): at grid row 2r+1, col 2c+1 — use S/G/@/./?
  - Passage between adjacent cells: at the grid position between them — use ` ` if open, `#` if wall, `?` if unknown
  - `?` = unknown (unvisited cell or unseen passage); `#` = confirmed wall; ` ` = confirmed open

Example: 3×3 maze (→ 7×7 grid), you are at cell (1,1), visited (0,0) and (0,1):
#######
#. .?##
## # ##
#?#@.?#
###?###
#?#?.G#
#######
(Cell (0,0)=S shown as `.` visited; (0,1)=`.`; `@`=you at (1,1); passage between (0,0)↔(0,1) shown as ` `; `#` between (0,1)↔(0,2) confirmed wall; `?` = unknown cells/passages)

=== Output Format (STRICT) ===
Thought: <brief reasoning: current position, what you see, plan>
Map:
<your {wall_size}×{wall_size} wall map, no spaces between characters>
Action: <move_forward|turn_left|turn_right>

Maze Config: {maze_size}×{maze_size} cells | Vision: {vision_range} cells | Max Steps: {max_steps}
"""


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """Single-step record."""
    step: int
    action: str
    success: bool
    message: str
    raw_output: str
    attempts: int
    used_fallback: bool
    facing_before: str
    facing_after: str
    position_before: Tuple[int, int]   # cell coords
    position_after: Tuple[int, int]
    cells_explored: int
    available_actions: List[str]
    cells_moved: int = 1               # cells traversed (v5 move_forward can be >1)
    map_output: str = ""               # ASCII map output from model (if ask_map/ask_trajectory=True)


@dataclass
class GameResult:
    """Full game evaluation result."""
    model: str
    maze_size: int
    seed: int
    vision_range: int
    total_steps: int
    optimal_steps: int
    reached_goal: bool
    cells_explored: int
    total_cells: int
    exploration_rate: float
    path_efficiency: float   # optimal / actual (1.0 = perfect), 0 if not reached
    random_fallback_count: int
    wall_bump_count: int
    turn_count: int
    action_space: str = "v4"
    obs_mode: str = "scene"
    loop_rate: float = 0.0
    n_clearings: int = -1
    clearing_radius: int = 1
    maze_type: str = "v5"
    wall_style: str = "plain"
    full_maze_text: str = ""
    conversation: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[StepRecord] = field(default_factory=list)


# ─── Message Builders ─────────────────────────────────────────────────────────

def _build_step_message(
    env: MazeGame3D,
    step_num: int,
    prefix: str = "",
    show_minimap: bool = False,
    show_hud: bool = True,
    obs_mode: str = "scene",
) -> Dict[str, Any]:
    """
    Build the per-step user message.
    Always includes the current-view screenshot (image_url content block).
    `prefix` is the previous step's movement-result feedback (text).

    obs_mode:
      - "scene":         3D first-person render (optional minimap overlay)
      - "mm2d-local":    3x3 local 2D patch (no history, low visual load)
      - "mm2d-cone":     facing-cone 2D patch + LOS (equivalent to 3D single-frame visibility)
      - "text-symbolic": pure-text description of the 4 adjacent directions (no image, zero visual load)
    """
    if obs_mode == "mm2d-local":
        b64 = env.render_local_patch_b64()
        view_label = (
            "Local 2D map (3×3 cells around you, world-aligned: top=North; no exploration history is shown). "
            "Color legend: cyan dot + white line = your position and facing arrow; "
            "light blue = open passable cell; dark slate = wall-blocked or out-of-bounds neighbor; "
            "orange line = wall between two cells (you cannot cross); "
            "orange-filled cell = the goal (only highlighted if it falls inside this 3×3 window):"
        )
    elif obs_mode == "mm2d-cone":
        b64 = env.render_local_patch_cone_b64()
        view_label = (
            f"Local 2D forward-cone map (visibility cone aligned with your facing, depth = {env.vision_range} cells, "
            f"FOV ≈ 66°; the canvas is rotated so YOUR FACING ALWAYS POINTS UP). "
            "This view shows exactly the per-frame visibility a 3D first-person camera would have. "
            "Color legend: cyan dot + white line = your position and facing arrow (always pointing up); "
            "light-blue cell = an open passable cell that is either in your line-of-sight, "
            "or a one-cell side-passage opening branching off the visible path (so you can see junctions ahead); "
            "dark slate = a cell that is OUTSIDE your FOV cone, OR is occluded by an intervening wall, "
            "OR is out of the maze bounds (you cannot tell which from this image alone — turning will reveal more); "
            "RED line = a wall (you cannot cross through it); "
            "ORANGE-filled cell = the GOAL (only highlighted if it is currently visible). "
            "A bottom HUD bar repeats your position, facing, goal, step count, explored cells, and available actions:"
        )
    elif obs_mode == "text-symbolic":
        b64 = None
        view_label = None
    else:
        b64 = env.render_image_b64(show_minimap=show_minimap, show_hud=show_hud)
        view_label = "Current 3D View:"

    cell_r, cell_c = env.agent_cell_pos()
    goal_r, goal_c = env.goal_cell_pos()
    avail = env.get_available_actions()

    if env.action_space == "v5":
        action_hint = "Action: <move_forward|turn_left|turn_right>"
    else:
        action_hint = "Action: <forward|backward|turn_left|turn_right>"

    info_text = (
        f"Step {step_num}.\n"
        f"Your cell position: ({cell_r}, {cell_c}) | Facing: {env.facing}\n"
        f"Goal cell position: ({goal_r}, {goal_c})\n"
        f"Steps taken: {env.step_count} | Cells explored: {env.explored_cells()}/{env.total_cells()}\n"
        f"Available actions: {', '.join(avail)}\n"
        f"Choose your next action.\nThought: ...\n{action_hint}"
    )

    parts = []
    if prefix:
        parts.append({"type": "text", "text": prefix})

    if obs_mode == "text-symbolic":
        # Pure text: no image block at all
        local_text = env.render_local_text()
        parts.append({"type": "text", "text": local_text})
    else:
        parts.append({"type": "text", "text": view_label})
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })
    parts.append({"type": "text", "text": info_text})

    return {"role": "user", "content": parts}


def _parse_map_output(raw: str) -> str:
    """Extract the ASCII map text between Map: ... and Action: in the model output."""
    import re
    m = re.search(r'Map:\s*\n(.*?)(?=\nAction:)', raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def parse_map_position(map_output: str, maze_size: int) -> Optional[Tuple[int, int]]:
    """Extract the position of @ from the model-produced map.
    Supports two formats:
      - (2N+1)x(2N+1) with-outer-walls format: cell (r,c) is at grid (2r+1, 2c+1)
      - N x N legacy format (space-separated): cell (r,c) is at row r, column c
    Returns the cell coords (row, col), or None if not found.
    """
    if not map_output:
        return None
    lines = map_output.strip().split("\n")
    # keep only lines containing map characters (excluding Legend / caption lines)
    grid_lines = [l for l in lines if "#" in l or any(c in l for c in "@.?SG")]

    wall_size = 2 * maze_size + 1   # (2N+1)×(2N+1) wall map

    # try the (2N+1)x(2N+1) format: line length close to wall_size (no spaces)
    wall_lines = [l for l in grid_lines if len(l.replace(" ", "")) >= wall_size - 2][:wall_size]
    if len(wall_lines) >= wall_size:
        for grid_r, line in enumerate(wall_lines):
            chars = line.replace(" ", "")
            for grid_c, ch in enumerate(chars):
                if ch == "@":
                    # cell (r,c) is at grid (2r+1, 2c+1), i.e. grid_r and grid_c are both odd
                    if grid_r % 2 == 1 and grid_c % 2 == 1:
                        return (grid_r // 2, grid_c // 2)
        return None

    # fallback: N x N format (space-separated tokens)
    cell_lines = [l for l in grid_lines if len(l.strip().split()) >= maze_size][:maze_size]
    for r, line in enumerate(cell_lines):
        cells = line.strip().split()
        for c, ch in enumerate(cells[:maze_size]):
            if ch == "@":
                return (r, c)
    return None


def _make_feedback(action: str, success: bool, message: str, pos_after: Tuple[int, int]) -> str:
    """Build the movement-result feedback text (used as the prefix of the next user message)."""
    if success:
        return f"Result: {message}\n\n"
    else:
        return f"Result: Action '{action}' FAILED — {message} Position unchanged: ({pos_after[0]}, {pos_after[1]}).\n\n"


# ─── Runner ───────────────────────────────────────────────────────────────────

class MazeRunner3D:
    """3D maze LLM evaluation runner (multi-turn dialogue, no history summary)."""

    def __init__(
        self,
        client: LLMClient,
        maze_size: int = 11,
        seed: int = 0,
        vision_range: int = 4,
        max_steps: int = 300,
        max_retries: int = 2,
        fallback_rng_seed: int = 0,
        screenshot_dir: Optional[str] = None,
        show_minimap: bool = False,
        show_hud: bool = True,
        action_space: str = "v4",
        checkpoint_dir: Optional[str] = None,
        history_window: int = 0,
        loop_rate: float = 0.15,
        n_clearings: int = -1,
        clearing_radius: int = 1,
        ask_map: bool = False,
        ask_trajectory: bool = False,
        maze_type: str = "v5",
        obs_mode: str = "scene",
        wall_style: str = "plain",
    ):
        self.client = client
        self.ask_map = ask_map
        self.ask_trajectory = ask_trajectory
        self.maze_size = maze_size
        self.seed = seed
        self.vision_range = vision_range
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.show_minimap = show_minimap
        self.show_hud = show_hud
        self.obs_mode = obs_mode
        self.loop_rate_param = loop_rate
        self.n_clearings_param = n_clearings
        self.clearing_radius_param = clearing_radius
        self.action_space = action_space
        self.maze_type = maze_type
        self.wall_style = wall_style
        self.history_window = history_window  # 0 = unlimited
        self.fallback_rng = random.Random(fallback_rng_seed)
        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else None
        if self.screenshot_dir:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint support
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.env = MazeGame3D(
            maze_size=maze_size,
            seed=seed,
            vision_range=vision_range,
            action_space=action_space,
            loop_rate=loop_rate,
            n_clearings=n_clearings,
            clearing_radius=clearing_radius,
            maze_type=maze_type,
            wall_style=wall_style,
        )

        if action_space == "v5":
            if ask_map:
                prompt_template = SYSTEM_PROMPT_V5_MAP
            elif ask_trajectory:
                prompt_template = SYSTEM_PROMPT_V5_TRAJECTORY
            elif show_minimap:
                prompt_template = SYSTEM_PROMPT_V5_MINIMAP
            else:
                prompt_template = SYSTEM_PROMPT_V5
        else:
            prompt_template = SYSTEM_PROMPT_MAP if ask_map else SYSTEM_PROMPT
        sys_prompt = prompt_template.format(
            maze_size=maze_size,
            wall_size=2 * maze_size + 1,
            vision_range=vision_range,
            max_steps=max_steps,
            minimap_size=140,
        )
        # ask_map / ask_trajectory need more tokens for the ASCII map output
        # (2N+1)^2 wall map is larger: 9x9→361 chars, 11x11→529 chars.
        # framework LLMClient reads max_tokens from extra_params (no .max_tokens attr)
        self.client.extra_params["max_tokens"] = 2048 if (ask_map or ask_trajectory) else 512
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": sys_prompt}
        ]

        # Runtime stats
        self.step_records: List[StepRecord] = []
        self.wall_bump_count = 0
        self.random_fallback_total = 0
        self.turn_count = 0

    def _checkpoint_path(self) -> Optional[Path]:
        if not self.checkpoint_dir:
            return None
        name = f"{self.client.model}__{self.maze_size}x{self.maze_size}__v{self.vision_range}__seed{self.seed}.ckpt.json"
        return self.checkpoint_dir / name

    def _checkpoint_run_config(self) -> Dict[str, Any]:
        """Used to decide whether a checkpoint is compatible with the current experiment config."""
        return {
            "action_space": self.action_space,
            "maze_type": self.maze_type,
            "wall_style": self.wall_style,
            "ask_map": self.ask_map,
            "ask_trajectory": self.ask_trajectory,
        }

    def _save_checkpoint(self, action_count: int, feedback_prefix: str):
        """Save a checkpoint after each step: action list + LLM raw outputs + stats."""
        ckpt_path = self._checkpoint_path()
        if not ckpt_path:
            return
        data = {
            "action_count": action_count,
            "feedback_prefix": feedback_prefix,
            "run_config": self._checkpoint_run_config(),
            "wall_bump_count": self.wall_bump_count,
            "random_fallback_total": self.random_fallback_total,
            "turn_count": self.turn_count,
            "steps": [
                {"action": sr.action, "raw_output": sr.raw_output,
                 "attempts": sr.attempts, "used_fallback": sr.used_fallback}
                for sr in self.step_records
            ],
        }
        tmp = ckpt_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, ensure_ascii=False)
        try:
            tmp.rename(ckpt_path)
        except OSError:
            import shutil
            try:
                shutil.move(str(tmp), str(ckpt_path))
            except OSError:
                pass

    def _load_checkpoint(self) -> Optional[dict]:
        ckpt_path = self._checkpoint_path()
        if not ckpt_path or not ckpt_path.exists():
            return None
        with open(ckpt_path) as f:
            data = json.load(f)
        if data.get("run_config") != self._checkpoint_run_config():
            logger.warning(f"Ignoring incompatible checkpoint: {ckpt_path}")
            return None
        return data

    def _delete_checkpoint(self):
        ckpt_path = self._checkpoint_path()
        if ckpt_path and ckpt_path.exists():
            ckpt_path.unlink()

    def _resume_from_checkpoint(self, ckpt: dict) -> Tuple[int, str]:
        """
        Resume from a checkpoint: replay actions to rebuild game state and dialogue history.
        Returns (action_count, feedback_prefix) to continue the run() loop.
        """
        logger.info(f"RESUMING from checkpoint: {ckpt['action_count']} steps completed")
        feedback_prefix = ""

        for i, step_data in enumerate(ckpt["steps"]):
            action = step_data["action"]
            raw_output = step_data["raw_output"]
            action_num = i + 1

            # Build step message (re-renders image from deterministic game state)
            step_msg = _build_step_message(self.env, action_num, prefix=feedback_prefix, show_minimap=self.show_minimap, show_hud=self.show_hud, obs_mode=self.obs_mode)
            self.messages.append(step_msg)

            # Append saved assistant output
            self.messages.append({"role": "assistant", "content": raw_output if raw_output.strip() else "..."})

            # Replay action on game
            pos_before = self.env.agent_cell_pos()
            facing_before = self.env.facing
            avail = self.env.get_available_actions()
            success, message = self.env.move(action)
            pos_after = self.env.agent_cell_pos()

            if action in ("turn_left", "turn_right"):
                self.turn_count += 1
            if not success:
                self.wall_bump_count += 1
            if step_data["used_fallback"]:
                self.random_fallback_total += 1

            feedback_prefix = _make_feedback(action, success, message, pos_after)

            self.step_records.append(StepRecord(
                step=action_num,
                action=action,
                success=success,
                message=message,
                raw_output=raw_output,
                attempts=step_data["attempts"],
                used_fallback=step_data["used_fallback"],
                facing_before=facing_before,
                facing_after=self.env.facing,
                position_before=pos_before,
                position_after=pos_after,
                cells_explored=self.env.explored_cells(),
                available_actions=avail,
                cells_moved=self.env.last_cells_moved,
                map_output=_parse_map_output(raw_output) if (self.ask_map or self.ask_trajectory) else "",
            ))

        logger.info(f"Resumed: pos={self.env.agent_cell_pos()}, facing={self.env.facing}, "
                     f"explored={self.env.explored_cells()}/{self.env.total_cells()}")
        return ckpt["action_count"], ckpt["feedback_prefix"]

    def _trim_messages_for_api(self) -> list:
        """Return messages; if history_window is set, keep only the last N dialogue turns."""
        if not self.history_window:
            return self.messages
        # messages[0] is the system prompt; each step after is [user, assistant]
        # keep system + the most recent history_window*2 messages
        system = self.messages[:1]
        history = self.messages[1:]
        trimmed = history[-(self.history_window * 2):]
        return system + trimmed

    def _choose_action(self) -> Tuple[str, str, int, bool]:
        """
        Have the LLM choose an action. The current step's user message is already appended to self.messages.
        Returns (action, raw_output, attempts, used_fallback).
        Transient network errors are retried inside framework's LLMClient.chat();
        a 400 content-filter is recovered here by trimming the oldest message pair.
        Any other error propagates so the game finalizes with an error.
        """
        valid_actions = ACTIONS_V5 if self.action_space == "v5" else ACTIONS
        action_hint = "|".join(valid_actions)

        MAX_FILTER_TRIMS = 20          # bound the 400 content-filter recovery
        raw = ""
        parse_attempts = 0
        filter_trims = 0

        while parse_attempts <= self.max_retries:
            try:
                raw = self.client.chat(self._trim_messages_for_api())["content"]
            except Exception as e:
                # Framework already retried transient network errors; anything that
                # reaches here is a real error. Recover only from a 400 content-filter
                # by trimming the oldest message pair — everything else propagates.
                err_str = str(e)
                is_filter = ("400" in err_str or "invalid_prompt" in err_str or "BadRequest" in err_str)
                if is_filter and filter_trims < MAX_FILTER_TRIMS and len(self.messages) > 3:
                    filter_trims += 1
                    self.messages = self.messages[:1] + self.messages[3:]
                    logger.warning(f"Trimmed oldest message pair due to API content filter "
                                   f"({filter_trims}/{MAX_FILTER_TRIMS}).")
                    continue
                raise

            action = parse_action(raw, action_space=self.action_space)

            if action is None:
                parse_attempts += 1
                logger.warning(f"Parse attempt {parse_attempts}/{self.max_retries + 1}: failed to parse action from: {raw}")
                if parse_attempts <= self.max_retries:
                    self.messages.append({"role": "assistant", "content": raw if raw.strip() else "..."})
                    self.messages.append({
                        "role": "user",
                        "content": (
                            f"Invalid action. Please choose exactly one of: "
                            f"{', '.join(valid_actions)}.\n"
                            f"Thought: ...\nAction: <{action_hint}>"
                        ),
                    })
                continue

            self.messages.append({"role": "assistant", "content": raw if raw.strip() else "..."})
            return action, raw, parse_attempts + 1, False

        # Fallback: random valid action
        avail = self.env.get_available_actions()
        fallback = self.fallback_rng.choice(avail if avail else valid_actions)
        logger.warning(f"All parse attempts exhausted (parse={parse_attempts}). Fallback: {fallback}")
        self.messages.append({"role": "assistant", "content": raw if raw.strip() else "..."})
        return fallback, raw, self.max_retries + 1, True

    def run(self) -> GameResult:
        """Run the full game evaluation and return a GameResult."""
        optimal = self.env.compute_optimal_path_length()
        logger.info(
            f"3D Maze start: {self.maze_size}×{self.maze_size}, "
            f"seed={self.seed}, vision={self.vision_range}"
        )
        logger.info(f"Optimal actions: {optimal}")
        logger.info(f"Start: {self.env.agent_cell_pos()}, Goal: {self.env.goal_cell_pos()}")

        feedback_prefix = ""
        action_count = 0

        # Try to resume from checkpoint
        ckpt = self._load_checkpoint()
        if ckpt:
            action_count, feedback_prefix = self._resume_from_checkpoint(ckpt)

        while not self.env.is_game_over() and action_count < self.max_steps:
            action_count += 1
            pos_before = self.env.agent_cell_pos()
            facing_before = self.env.facing
            avail = self.env.get_available_actions()

            logger.info(
                f"--- Step {action_count} | pos={pos_before} | facing={facing_before} | "
                f"explored={self.env.explored_cells()}/{self.env.total_cells()} ---"
            )

            # Save screenshots of current state before API call
            if self.screenshot_dir:
                frame_img = (self.env._build_local_patch() if self.obs_mode == "mm2d-local" else self.env._build_local_patch_cone(show_hud=self.show_hud) if self.obs_mode == "mm2d-cone" else self.env.render_frame(show_minimap=self.show_minimap, show_hud=self.show_hud))
                topdown_img = self.env.render_topdown_global(step=action_count)
                frame_img.save(self.screenshot_dir / f"step{action_count:04d}_3d.jpg", quality=88)
                topdown_img.save(self.screenshot_dir / f"step{action_count:04d}_topdown.jpg", quality=92)

            # Build and append user message (with screenshot)
            step_msg = _build_step_message(self.env, action_count, prefix=feedback_prefix, show_minimap=self.show_minimap, show_hud=self.show_hud, obs_mode=self.obs_mode)
            self.messages.append(step_msg)

            # LLM chooses action
            action, raw, attempts, used_fallback = self._choose_action()
            if used_fallback:
                self.random_fallback_total += 1
            if action in ("turn_left", "turn_right"):
                self.turn_count += 1

            # Execute action
            success, message = self.env.move(action)
            pos_after = self.env.agent_cell_pos()

            if not success:
                self.wall_bump_count += 1
                logger.info(f"BLOCKED: {action} ({message})")
            else:
                logger.info(f"OK: {action} → pos={pos_after}, facing={self.env.facing}")

            feedback_prefix = _make_feedback(action, success, message, pos_after)

            self.step_records.append(StepRecord(
                step=action_count,
                action=action,
                success=success,
                message=message,
                raw_output=raw,
                attempts=attempts,
                used_fallback=used_fallback,
                facing_before=facing_before,
                facing_after=self.env.facing,
                position_before=pos_before,
                position_after=pos_after,
                cells_explored=self.env.explored_cells(),
                available_actions=avail,
                cells_moved=self.env.last_cells_moved,
                map_output=_parse_map_output(raw) if (self.ask_map or self.ask_trajectory) else "",
            ))

            # Save checkpoint after each step
            self._save_checkpoint(action_count, feedback_prefix)

        reached_goal = self.env.is_game_over()
        total_steps = action_count

        # Save final state screenshot
        if self.screenshot_dir:
            frame_img = (self.env._build_local_patch() if self.obs_mode == "mm2d-local" else self.env._build_local_patch_cone(show_hud=self.show_hud) if self.obs_mode == "mm2d-cone" else self.env.render_frame(show_minimap=self.show_minimap, show_hud=self.show_hud))
            topdown_img = self.env.render_topdown_global(step=total_steps)
            frame_img.save(self.screenshot_dir / f"step{total_steps + 1:04d}_final_3d.jpg", quality=88)
            topdown_img.save(self.screenshot_dir / f"step{total_steps + 1:04d}_final_topdown.jpg", quality=92)

        # Final message
        if reached_goal:
            self.messages.append({
                "role": "user",
                "content": (
                    f"{feedback_prefix}Game over! You reached the goal in {total_steps} steps! "
                    f"Optimal was {optimal} actions."
                ),
            })
            logger.info(f"GOAL REACHED in {total_steps} steps (optimal: {optimal})")
        else:
            self.messages.append({
                "role": "user",
                "content": (
                    f"{feedback_prefix}Game terminated: max steps ({self.max_steps}) reached "
                    f"without finding the goal."
                ),
            })
            logger.warning(f"Max steps reached ({self.max_steps})")

        # Delete checkpoint on successful completion
        self._delete_checkpoint()

        path_efficiency = optimal / max(total_steps, 1) if reached_goal else 0.0
        conversation = _serialize_messages(self.messages)

        return GameResult(
            model=self.client.model,
            maze_size=self.maze_size,
            seed=self.seed,
            vision_range=self.vision_range,
            total_steps=total_steps,
            optimal_steps=optimal,
            reached_goal=reached_goal,
            cells_explored=self.env.explored_cells(),
            total_cells=self.env.total_cells(),
            exploration_rate=self.env.exploration_rate(),
            path_efficiency=path_efficiency,
            random_fallback_count=self.random_fallback_total,
            wall_bump_count=self.wall_bump_count,
            turn_count=self.turn_count,
            action_space=self.action_space,
            obs_mode=self.obs_mode,
            loop_rate=getattr(self, "loop_rate_param", 0.0),
            n_clearings=getattr(self, "n_clearings_param", -1),
            clearing_radius=getattr(self, "clearing_radius_param", 1),
            maze_type=self.maze_type,
            wall_style=self.wall_style,
            full_maze_text=self.env.render_top_down_text(full=True),
            conversation=conversation,
            steps=self.step_records,
        )


# ─── Serialization & Save ─────────────────────────────────────────────────────

def _serialize_messages(messages: list) -> list:
    """Serialize messages for JSON saving (image base64 replaced with a placeholder)."""
    serialized = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url":
                        parts.append("[IMAGE: 3D maze screenshot]")
                    elif part.get("type") == "text":
                        parts.append(part["text"])
                else:
                    parts.append(str(part))
            serialized.append({"role": msg["role"], "content": "\n".join(parts)})
        else:
            serialized.append({"role": msg["role"], "content": content})
    return serialized


def save_result(result: GameResult, output_dir: str = "results",
                rollout: int | None = None) -> Path:
    """Save the game result JSON + full dialogue log."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base = (
        f"{result.model}__{result.maze_size}x{result.maze_size}"
        f"__v{result.vision_range}__seed{result.seed}"
    )
    if rollout is not None:
        base += f"__r{rollout}"

    # JSON
    json_path = out_path / f"{base}.json"
    data = asdict(result)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Result saved to {json_path}")

    # Human-readable log
    log_dir = out_path / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{base}.log"
    with open(log_path, "w") as f:
        sep = "=" * 70
        f.write(f"{sep}\n3D First-Person Maze — Full Conversation Log\n{sep}\n")
        f.write(f"Model:         {result.model}\n")
        f.write(f"Maze Size:     {result.maze_size}×{result.maze_size}\n")
        f.write(f"Vision Range:  {result.vision_range} cells\n")
        f.write(f"Seed:          {result.seed}\n")
        f.write(f"Optimal Steps: {result.optimal_steps}\n")
        f.write(f"Total Steps:   {result.total_steps}\n")
        f.write(f"Result:        {'REACHED' if result.reached_goal else 'NOT REACHED'}\n")
        f.write(f"Efficiency:    {result.path_efficiency:.4f}\n")
        f.write(f"Exploration:   {result.cells_explored}/{result.total_cells} "
                f"({result.exploration_rate:.1%})\n")
        f.write(f"Wall Bumps:    {result.wall_bump_count}\n")
        f.write(f"Turns:         {result.turn_count}\n")
        f.write(f"Fallbacks:     {result.random_fallback_count}\n")
        f.write(f"{sep}\n\nFull Maze (top-down):\n{result.full_maze_text}\n\n{sep}\n")
        f.write("CONVERSATION TRACE\n")
        f.write(f"{sep}\n\n")
        for msg in result.conversation:
            f.write(f"[{msg['role'].upper()}]\n{msg['content']}\n\n")

    logger.info(f"Log saved to {log_path}")
    return json_path
