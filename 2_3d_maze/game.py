"""
3D first-person maze game environment.

Renders a first-person 3D view via raycasting (DDA), with distance fog.
Supports a seed to control randomness, ensuring fair multi-model evaluation.

Coordinate systems:
  - raw grid: (2*maze_size+1) x (2*maze_size+1), '#'=wall, '.'=path
  - player world coords: (px=col, py=row) float; px=col+0.5 means the center of raw_col
  - facing angle: 0deg=East, 90deg=South, 180deg=West, 270deg=North (clockwise)
"""

import base64
import colorsys
import hashlib
import io
import math
import random
import re
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── Action / Direction Constants ────────────────────────────────────────────

ACTIONS = ["forward", "backward", "turn_left", "turn_right"]       # v4
ACTIONS_V5 = ["move_forward", "turn_left", "turn_right"]           # v5 (slide-forward)

# Facing angle in degrees (0=E, 90=S, 180=W, 270=N)
FACING_ANGLE: Dict[str, float] = {"E": 0.0, "S": 90.0, "W": 180.0, "N": 270.0}

# (dcol, drow) per action in raw grid coords (each move = 2 raw cells)
FACE_DELTA: Dict[str, Tuple[int, int]] = {
    "E": (2, 0),
    "S": (0, 2),
    "W": (-2, 0),
    "N": (0, -2),
}

# (dr, dc) per facing direction in cell coords (1 cell)
FACE_DELTA_CELL: Dict[str, Tuple[int, int]] = {
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
    "N": (-1, 0),
}

TURN_LEFT: Dict[str, str] = {"E": "N", "N": "W", "W": "S", "S": "E"}
TURN_RIGHT: Dict[str, str] = {"E": "S", "S": "W", "W": "N", "N": "E"}
OPPOSITE: Dict[str, str] = {"E": "W", "W": "E", "N": "S", "S": "N"}

# ─── Render Constants ─────────────────────────────────────────────────────────

RENDER_W = 640        # 3D view width
RENDER_H = 360        # 3D view height
MINIMAP_SIZE = 140    # mini-map square size (pixels)
HUD_H = 60            # HUD bar height (2 lines: pos/facing/goal/steps/explored, actions)
IMG_W = RENDER_W
IMG_H = RENDER_H + HUD_H  # 640 × 420

FOV_DEG = 66          # horizontal field of view in degrees

# Color tuples (R, G, B) as numpy uint8 arrays
_C = lambda r, g, b: np.array([r, g, b], dtype=np.uint8)

WALL_COLOR_NS = _C(160, 165, 200)   # North/South wall faces (side=0)
WALL_COLOR_EW = _C(110, 115, 150)   # East/West wall faces (side=1), darker
GOAL_WALL_COLOR = _C(200, 120, 10)  # Walls immediately adjacent to goal cell


# ── Visual-pattern variant helpers ────────────────────────────────────────────
# These are used by `_build_3d_scene` when wall_style != "plain".
def _face_hash(map_y: int, map_x: int, side: int) -> int:
    h = hashlib.md5(f"{map_y},{map_x},{side}".encode()).digest()
    return int.from_bytes(h[:4], "big")


_GOLDEN_RATIO_CONJ = 0.61803398875


def _face_hue_color(map_y: int, map_x: int, side: int) -> np.ndarray:
    """Per-face deterministic hue (golden-ratio sampled for max separation)."""
    idx = _face_hash(map_y, map_x, side) & 0xFFF
    hue = (idx * _GOLDEN_RATIO_CONJ) % 1.0
    sat = 0.78 if side == 0 else 0.62
    val = 0.92 if side == 0 else 0.78
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return np.array([r * 255, g * 255, b * 255], dtype=np.float32)


def _repetitive_stripe_mod(y: int) -> float:
    """Same horizontal banding on every wall — visual complexity but no info."""
    return 0.55 + 0.45 * (math.sin(y * 0.18) * 0.5 + 0.5)


def _unique_face_stripe(map_y: int, map_x: int, side: int,
                        y: int, wall_y_start: int, wall_y_end: int) -> float:
    """Per-face vertical band pattern, seeded by (y,x,side)."""
    rng = np.random.default_rng(_face_hash(map_y, map_x, side))
    n_bands = 6
    bands = rng.uniform(0.4, 1.0, size=n_bands)
    height = max(1, wall_y_end - wall_y_start + 1)
    norm = (y - wall_y_start) / height
    idx = min(n_bands - 1, max(0, int(norm * n_bands)))
    return float(bands[idx])
CEILING_TOP = _C(8, 10, 22)
CEILING_BOT = _C(22, 28, 50)
FLOOR_TOP = _C(18, 15, 12)
FLOOR_BOT = _C(55, 48, 38)
FOG_RGB = _C(4, 4, 14)

MM_UNSEEN = (20, 20, 28)
MM_WALL_SEEN = (75, 78, 100)
MM_PATH_SEEN = (190, 195, 215)
MM_PATH_VISIBLE = (225, 230, 255)
MM_PLAYER = (0, 180, 255)
MM_GOAL = (255, 150, 0)
MM_START = (0, 210, 90)
MM_BORDER = (80, 80, 100)

HUD_BG = (8, 8, 20)
HUD_TEXT = (190, 195, 210)
HUD_ACCENT = (100, 160, 255)


# ─── MazeGame3D ───────────────────────────────────────────────────────────────

class MazeGame3D:
    """3D first-person maze game environment (raycasting render, fog-of-war)."""

    WALL = "#"
    PATH = "."
    START = "S"
    GOAL = "G"

    _VALID_WALL_STYLES = ("plain", "repetitive", "color_tag", "unique_poster")

    def __init__(
        self,
        maze_size: int = 11,
        seed: int = 0,
        vision_range: int = 4,
        action_space: str = "v4",
        loop_rate: float = 0.15,
        n_clearings: int = -1,
        clearing_radius: int = 1,
        maze_type: str = "v5",
        wall_style: str = "plain",
    ):
        if maze_size < 3:
            raise ValueError("maze_size must be >= 3")
        if maze_size % 2 == 0:
            raise ValueError("maze_size must be odd for DFS maze generation")
        if action_space not in ("v4", "v5"):
            raise ValueError("action_space must be 'v4' or 'v5'")
        if maze_type not in ("v5", "v6"):
            raise ValueError("maze_type must be 'v5' or 'v6'")
        if wall_style not in self._VALID_WALL_STYLES:
            raise ValueError(f"wall_style must be one of {self._VALID_WALL_STYLES}")

        self.loop_rate = loop_rate
        self.n_clearings = n_clearings  # -1 = auto
        self.clearing_radius = clearing_radius
        self.maze_type = maze_type
        self.wall_style = wall_style

        self.maze_size = maze_size
        self.seed = seed
        self.vision_range = vision_range  # visibility in cells
        self.action_space = action_space

        self.grid_h = 2 * maze_size + 1
        self.grid_w = 2 * maze_size + 1
        self.grid: List[List[str]] = []

        # Raw grid positions (row, col)
        self.start_pos: Tuple[int, int] = (1, 1)
        self.goal_pos: Tuple[int, int] = (0, 0)

        # Player world position (raw grid coordinates, continuous)
        # px = col + 0.5,  py = row + 0.5  when centered in a raw grid cell
        self.px: float = 1.5
        self.py: float = 1.5
        self.facing: str = "E"   # N / S / E / W
        self.angle: float = 0.0  # degrees for rendering

        # Evaluation tracking
        self.step_count: int = 0
        self.visited_cells: Set[Tuple[int, int]] = set()    # cell (row,col), 0-indexed
        self.seen_raw: Set[Tuple[int, int]] = set()          # raw (row,col) ever in view
        self.path_history: List[Tuple[int, int]] = []        # ordered cell positions (with repeats)
        self.last_cells_moved: int = 0                       # cells moved in last action (v5)

        self._generate_maze()
        self._update_visibility()

    # ── Maze Generation ──────────────────────────────────────────────────────

    def _generate_maze(self):
        """Generate the maze according to maze_type."""
        if self.maze_type == "v6":
            self._generate_maze_v6()
        else:
            self._generate_maze_v5()

    def _generate_maze_v5(self):
        """Generate the maze via DFS recursive backtracking, then add loops and clearings."""
        rng = random.Random(self.seed)
        h = w = self.maze_size

        # All walls to start
        self.grid = [[self.WALL] * self.grid_w for _ in range(self.grid_h)]

        # Open all cell positions (odd, odd)
        for r in range(h):
            for c in range(w):
                self.grid[2 * r + 1][2 * c + 1] = self.PATH

        # DFS carving (perfect maze)
        cell_visited = [[False] * w for _ in range(h)]
        stack = [(0, 0)]
        cell_visited[0][0] = True

        while stack:
            cr, cc = stack[-1]
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not cell_visited[nr][nc]:
                    neighbors.append((nr, nc, dr, dc))

            if neighbors:
                nr, nc, dr, dc = rng.choice(neighbors)
                self.grid[2 * cr + 1 + dr][2 * cc + 1 + dc] = self.PATH
                cell_visited[nr][nc] = True
                stack.append((nr, nc))
            else:
                stack.pop()

        # Post-processing: add loops and clearings
        if self.loop_rate > 0:
            self._add_loops(rng)
        self._add_clearings(rng)

        # Place start (top-left cell) and goal (bottom-right cell)
        self.start_pos = (1, 1)
        self.goal_pos = (2 * h - 1, 2 * w - 1)
        self.grid[self.start_pos[0]][self.start_pos[1]] = self.START
        self.grid[self.goal_pos[0]][self.goal_pos[1]] = self.GOAL

        # Initialize player at start
        self.px = float(self.start_pos[1]) + 0.5   # col + 0.5
        self.py = float(self.start_pos[0]) + 0.5   # row + 0.5
        self.facing = "E"
        self.angle = FACING_ANGLE["E"]

        start_cell = self._raw_to_cell(self.start_pos[0], self.start_pos[1])
        self.visited_cells = {start_cell}
        self.path_history = [start_cell]

    def _add_loops(self, rng: random.Random) -> None:
        """Randomly knock down some walls to introduce loops (multiple paths) in the perfect maze."""
        h = w = self.maze_size
        for r in range(h):
            for c in range(w):
                # horizontal: wall between (r,c) and (r,c+1)
                if c + 1 < w:
                    wall_row = 2 * r + 1
                    wall_col = 2 * c + 2
                    if self.grid[wall_row][wall_col] == self.WALL:
                        if rng.random() < self.loop_rate:
                            self.grid[wall_row][wall_col] = self.PATH
                # vertical: wall between (r,c) and (r+1,c)
                if r + 1 < h:
                    wall_row = 2 * r + 2
                    wall_col = 2 * c + 1
                    if self.grid[wall_row][wall_col] == self.WALL:
                        if rng.random() < self.loop_rate:
                            self.grid[wall_row][wall_col] = self.PATH

    def _add_clearings(self, rng: random.Random) -> None:
        """Randomly create open clearings in the maze, removing all interior walls within a rectangle."""
        h = w = self.maze_size
        r = self.clearing_radius
        num = self.n_clearings if self.n_clearings >= 0 else max(1, (h * w) // 25)

        for _ in range(num):
            cr = rng.randint(r, h - r - 1)
            cc = rng.randint(r, w - r - 1)
            r_min = cr - r
            r_max = cr + r
            c_min = cc - r
            c_max = cc + r

            for row in range(r_min, r_max + 1):
                for col in range(c_min, c_max):
                    self.grid[2 * row + 1][2 * col + 2] = self.PATH

            for row in range(r_min, r_max):
                for col in range(c_min, c_max + 1):
                    self.grid[2 * row + 2][2 * col + 1] = self.PATH

    def _generate_maze_v6(self):
        """v6 maze: DFS + very few loops + anti-2x2-block constraint.

        v6 characteristics:
        - DFS recursive backtracking (long corridors + long dead ends, fewer branches than Prim)
        - only 1-2 loops (and avoids forming a 2x2 open block)
        - no clearings
        """
        rng = random.Random(self.seed)
        h = w = self.maze_size

        self.grid = [[self.WALL] * self.grid_w for _ in range(self.grid_h)]
        for r in range(h):
            for c in range(w):
                self.grid[2 * r + 1][2 * c + 1] = self.PATH

        # DFS carving (perfect maze)
        cell_visited = [[False] * w for _ in range(h)]
        stack = [(0, 0)]
        cell_visited[0][0] = True
        while stack:
            cr, cc = stack[-1]
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not cell_visited[nr][nc]:
                    neighbors.append((nr, nc, dr, dc))
            if neighbors:
                nr, nc, dr, dc = rng.choice(neighbors)
                self.grid[2 * cr + 1 + dr][2 * cc + 1 + dc] = self.PATH
                cell_visited[nr][nc] = True
                stack.append((nr, nc))
            else:
                stack.pop()

        def would_form_square(wall_row: int, wall_col: int) -> bool:
            """Check whether opening this wall would fully open a 2x2 cell block's interior walls (a 2x2 open block)."""
            # determine the two adjacent cells this wall affects
            if wall_row % 2 == 1:  # horizontal wall (between cell r,c and r,c+1)
                cr = (wall_row - 1) // 2
                cc_left = (wall_col - 2) // 2
                cc_right = cc_left + 1
                # check the upper and lower 2x2 blocks
                for cr0 in [cr - 1, cr]:
                    if cr0 < 0 or cr0 + 1 >= h:
                        continue
                    cc0 = cc_left
                    if cc0 < 0 or cc0 + 1 >= w:
                        continue
                    # the 2x2 block's 4 interior walls (up-h, down-h, left-v, right-v)
                    inner_walls = [
                        (2 * cr0 + 1, 2 * cc0 + 2),       # top row horizontal
                        (2 * (cr0 + 1) + 1, 2 * cc0 + 2), # bottom row horizontal
                        (2 * cr0 + 2, 2 * cc0 + 1),       # left col vertical
                        (2 * cr0 + 2, 2 * (cc0 + 1) + 1), # right col vertical
                    ]
                    opened = sum(1 for r_, c_ in inner_walls if self.grid[r_][c_] != self.WALL)
                    if opened >= 3:  # opening this one too means all 4 open = 2x2 block
                        return True
            else:  # vertical wall
                cr_top = (wall_row - 2) // 2
                cr_bot = cr_top + 1
                cc = (wall_col - 1) // 2
                for cc0 in [cc - 1, cc]:
                    if cc0 < 0 or cc0 + 1 >= w:
                        continue
                    cr0 = cr_top
                    if cr0 < 0 or cr0 + 1 >= h:
                        continue
                    inner_walls = [
                        (2 * cr0 + 1, 2 * cc0 + 2),
                        (2 * (cr0 + 1) + 1, 2 * cc0 + 2),
                        (2 * cr0 + 2, 2 * cc0 + 1),
                        (2 * cr0 + 2, 2 * (cc0 + 1) + 1),
                    ]
                    opened = sum(1 for r_, c_ in inner_walls if self.grid[r_][c_] != self.WALL)
                    if opened >= 3:
                        return True
            return False

        # 1-2 loops: chosen from central candidates, avoiding 2x2 open blocks
        margin = max(1, h // 4)
        loop_candidates = []
        for r in range(h):
            for c in range(w):
                if c + 1 < w:
                    wall_row, wall_col = 2 * r + 1, 2 * c + 2
                    if self.grid[wall_row][wall_col] == self.WALL:
                        if margin <= r < h - margin and margin <= c < w - margin:
                            loop_candidates.append((wall_row, wall_col))
                if r + 1 < h:
                    wall_row, wall_col = 2 * r + 2, 2 * c + 1
                    if self.grid[wall_row][wall_col] == self.WALL:
                        if margin <= r < h - margin and margin <= c < w - margin:
                            loop_candidates.append((wall_row, wall_col))

        rng.shuffle(loop_candidates)
        target_loops = rng.randint(1, 2)
        loops_added = 0
        for wr, wc in loop_candidates:
            if loops_added >= target_loops:
                break
            if not would_form_square(wr, wc):
                self.grid[wr][wc] = self.PATH
                loops_added += 1

        # Place start and goal
        self.start_pos = (1, 1)
        self.goal_pos = (2 * h - 1, 2 * w - 1)
        self.grid[self.start_pos[0]][self.start_pos[1]] = self.START
        self.grid[self.goal_pos[0]][self.goal_pos[1]] = self.GOAL

        self.px = float(self.start_pos[1]) + 0.5
        self.py = float(self.start_pos[0]) + 0.5
        self.facing = "E"
        self.angle = FACING_ANGLE["E"]

        start_cell = self._raw_to_cell(self.start_pos[0], self.start_pos[1])
        self.visited_cells = {start_cell}
        self.path_history = [start_cell]

    # ── Coordinate Helpers ────────────────────────────────────────────────────

    def _raw_to_cell(self, raw_row: int, raw_col: int) -> Tuple[int, int]:
        """Raw grid coords -> cell coords (0-indexed)."""
        return ((raw_row - 1) // 2, (raw_col - 1) // 2)

    def _is_wall_raw(self, raw_col: int, raw_row: int) -> bool:
        """Check whether raw grid position (col, row) is a wall (out-of-bounds counts as wall)."""
        if 0 <= raw_row < self.grid_h and 0 <= raw_col < self.grid_w:
            return self.grid[raw_row][raw_col] == self.WALL
        return True

    def _current_raw_cell(self) -> Tuple[int, int]:
        """The raw grid cell (row, col) the player is currently in."""
        return (int(self.py), int(self.px))

    def agent_cell_pos(self) -> Tuple[int, int]:
        """The agent's position in cell coords (row, col), 0-indexed."""
        raw_row, raw_col = self._current_raw_cell()
        return self._raw_to_cell(raw_row, raw_col)

    def goal_cell_pos(self) -> Tuple[int, int]:
        return self._raw_to_cell(self.goal_pos[0], self.goal_pos[1])

    # ── Visibility & Fog of War ───────────────────────────────────────────────

    def _update_visibility(self):
        """Update seen_raw (minimap fog-of-war).

        Reveal policy: only unlock (a) the current cell itself, and (b) all of the
        current cell's "junction" branches (i.e. adjacent cells with no wall in
        between). That is: "reveal one cell per step + one extra cell for each
        junction branch".
        """
        cell_r, cell_c = self.agent_cell_pos()

        # (a) Current cell — add the cell center and the agent's own raw position.
        raw_row, raw_col = self._current_raw_cell()
        self.seen_raw.add((raw_row, raw_col))
        self.seen_raw.add((2 * cell_r + 1, 2 * cell_c + 1))

        # (b) Junction reveal: each open neighbor (no wall between current cell and it)
        # gets its center + the connecting wall cell added to seen_raw, so the minimap
        # renders the side passage and the open boundary between the two cells.
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cell_r + dr, cell_c + dc
            if not (0 <= nr < self.maze_size and 0 <= nc < self.maze_size):
                continue
            wall_r = cell_r + nr + 1
            wall_c = cell_c + nc + 1
            if self.grid[wall_r][wall_c] == self.WALL:
                continue
            self.seen_raw.add((2 * nr + 1, 2 * nc + 1))
            self.seen_raw.add((wall_r, wall_c))

    def _get_current_visible(self) -> Set[Tuple[int, int]]:
        """Get the set of raw grid cells visible in the current frame."""
        raw_row, raw_col = self._current_raw_cell()
        vr = self.vision_range * 2
        visible = set()
        for r in range(max(0, raw_row - vr), min(self.grid_h, raw_row + vr + 1)):
            for c in range(max(0, raw_col - vr), min(self.grid_w, raw_col + vr + 1)):
                if abs(r - raw_row) + abs(c - raw_col) <= vr:
                    visible.add((r, c))
        return visible

    # ── Game Logic ────────────────────────────────────────────────────────────

    def is_game_over(self) -> bool:
        raw_row, raw_col = self._current_raw_cell()
        return (raw_row, raw_col) == self.goal_pos

    def _can_move_face(self, face: str) -> bool:
        """Check whether one cell can be advanced in a given facing (checks the wall cell in between)."""
        dcol, drow = FACE_DELTA[face]
        cur_col = int(self.px)
        cur_row = int(self.py)
        wall_col = cur_col + dcol // 2
        wall_row = cur_row + drow // 2
        target_col = cur_col + dcol
        target_row = cur_row + drow
        if not (0 <= wall_row < self.grid_h and 0 <= wall_col < self.grid_w):
            return False
        if not (0 <= target_row < self.grid_h and 0 <= target_col < self.grid_w):
            return False
        return self.grid[wall_row][wall_col] != self.WALL

    def can_act(self, action: str) -> bool:
        """Check whether an action is possible (turns are always possible)."""
        if action in ("turn_left", "turn_right"):
            return True
        if action in ("forward", "move_forward"):
            return self._can_move_face(self.facing)
        if action == "backward":
            return self._can_move_face(OPPOSITE[self.facing])
        return False

    def _move_forward_until_wall(self) -> int:
        """v5: keep advancing in the current facing until a wall, the goal, or a junction (open left/right); returns cells moved."""
        cells_moved = 0
        left_face = TURN_LEFT[self.facing]
        right_face = TURN_RIGHT[self.facing]
        while self._can_move_face(self.facing):
            dcol, drow = FACE_DELTA[self.facing]
            self.px += dcol
            self.py += drow
            cells_moved += 1
            cell = self.agent_cell_pos()
            self.visited_cells.add(cell)
            self.path_history.append(cell)
            self._update_visibility()
            if self.is_game_over():
                break
            # stop at a junction (open passage on the left or right) as a decision point
            if self._can_move_face(left_face) or self._can_move_face(right_face):
                break
        return cells_moved

    def move(self, action: str) -> Tuple[bool, str]:
        """
        Execute an action. Returns (success, message).
        Every action (including wall bumps) counts toward step_count, for fair evaluation.
        """
        valid_actions = ACTIONS_V5 if self.action_space == "v5" else ACTIONS
        if action not in valid_actions:
            return False, f"Invalid action '{action}'. Choose from: {valid_actions}"

        self.step_count += 1
        self.last_cells_moved = 0

        if action == "turn_left":
            old = self.facing
            self.facing = TURN_LEFT[self.facing]
            self.angle = FACING_ANGLE[self.facing]
            self._update_visibility()
            return True, f"Turned left: {old} → {self.facing}."

        if action == "turn_right":
            old = self.facing
            self.facing = TURN_RIGHT[self.facing]
            self.angle = FACING_ANGLE[self.facing]
            self._update_visibility()
            return True, f"Turned right: {old} → {self.facing}."

        # v5: move_forward (slide until wall)
        if action == "move_forward":
            direction = self.facing
            n = self._move_forward_until_wall()
            self.last_cells_moved = n
            if n == 0:
                return False, "Cannot move forward: wall is directly ahead."
            cell = self.agent_cell_pos()
            if self.is_game_over():
                return True, f"Moved {n} cell(s) heading {direction}. You reached the GOAL!"
            return True, f"Moved {n} cell(s) heading {direction}. Now at cell ({cell[0]}, {cell[1]})."

        # v4: forward / backward (move 1 cell)
        face = self.facing if action == "forward" else OPPOSITE[self.facing]

        if not self._can_move_face(face):
            return False, f"Cannot move {action}: wall is blocking."

        dcol, drow = FACE_DELTA[face]
        self.px += dcol
        self.py += drow
        self.last_cells_moved = 1

        cell = self.agent_cell_pos()
        self.visited_cells.add(cell)
        self.path_history.append(cell)
        self._update_visibility()

        if self.is_game_over():
            return True, f"Moved 1 cell heading {face}. You reached the GOAL!"
        return True, f"Moved 1 cell heading {face}. Now at cell ({cell[0]}, {cell[1]})."

    def get_available_actions(self) -> List[str]:
        """Return all currently possible actions."""
        actions = ACTIONS_V5 if self.action_space == "v5" else ACTIONS
        return [a for a in actions if self.can_act(a)]

    def get_movable_directions(self) -> List[str]:
        """Return actions that actually move (excluding turns)."""
        if self.action_space == "v5":
            return ["move_forward"] if self.can_act("move_forward") else []
        return [a for a in ("forward", "backward") if self.can_act(a)]

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render_frame(self, show_minimap: bool = True, show_hud: bool = True) -> Image.Image:
        """
        Render the full evaluation frame:
          - main: 3D first-person raycasting view (640x360)
          - top-right: explored-maze top-down minimap (140x140), with fog (when show_minimap=True)
          - bottom: HUD info bar (640x60, two rows; when show_hud=True)
        Returns a PIL Image (640x420, or 640x360 when show_hud=False).
        """
        # 1. Build 3D scene as numpy array
        scene = self._build_3d_scene()

        # 2. Draw minimap and paste onto scene (only for visualization)
        if show_minimap:
            mm_img = self._build_minimap()
            mm_arr = np.array(mm_img)
            mm_x = RENDER_W - MINIMAP_SIZE - 8
            mm_y = 8
            scene[mm_y: mm_y + MINIMAP_SIZE, mm_x: mm_x + MINIMAP_SIZE] = mm_arr

        if not show_hud:
            return Image.fromarray(scene)

        # 3. Create full image (with HUD strip)
        full = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        full[:RENDER_H] = scene

        # 4. Draw HUD into bottom strip
        img = Image.fromarray(full)
        self._draw_hud(img)

        return img

    def _build_3d_scene(self) -> np.ndarray:
        """
        Main raycasting loop. Returns a RENDER_H x RENDER_W x 3 numpy uint8 array.
        Uses the DDA algorithm; fog_dist = vision_range * 2 raw grid units.
        """
        W, H = RENDER_W, RENDER_H
        pixels = np.zeros((H, W, 3), dtype=np.uint8)

        # ── Ceiling gradient ──
        ys = np.arange(H // 2, dtype=np.float32)
        t = ys / max(H // 2 - 1, 1)        # 0=top center, 1=top edge
        t_rev = 1.0 - t                      # 1=top center (brighter), 0=edge
        for i in range(3):
            col = (CEILING_BOT[i] * t_rev + CEILING_TOP[i] * t).astype(np.uint8)
            pixels[:H // 2, :, i] = col[:, np.newaxis]

        # ── Floor gradient ──
        ys_f = np.arange(H // 2, H, dtype=np.float32)
        t_f = (ys_f - H // 2) / max(H // 2 - 1, 1)   # 0=center, 1=bottom
        for i in range(3):
            col = (FLOOR_TOP[i] * (1 - t_f) + FLOOR_BOT[i] * t_f).astype(np.uint8)
            pixels[H // 2:, :, i] = col[:, np.newaxis]

        # ── Raycasting ──
        angle_rad = math.radians(self.angle)
        dir_x = math.cos(angle_rad)   # col direction
        dir_y = math.sin(angle_rad)   # row direction

        half_fov_tan = math.tan(math.radians(FOV_DEG / 2))
        plane_x = -dir_y * half_fov_tan
        plane_y = dir_x * half_fov_tan

        fog_dist = float(self.vision_range * 2)

        # Goal position for special coloring
        goal_r, goal_c = self.goal_pos

        for x in range(W):
            cam = 2.0 * x / W - 1.0     # -1 (left) to +1 (right)
            ray_dx = dir_x + plane_x * cam
            ray_dy = dir_y + plane_y * cam

            map_x = int(self.px)   # current raw grid col
            map_y = int(self.py)   # current raw grid row

            ddx = abs(1.0 / ray_dx) if abs(ray_dx) > 1e-12 else 1e30
            ddy = abs(1.0 / ray_dy) if abs(ray_dy) > 1e-12 else 1e30

            if ray_dx < 0:
                step_x = -1
                side_x = (self.px - map_x) * ddx
            else:
                step_x = 1
                side_x = (map_x + 1.0 - self.px) * ddx

            if ray_dy < 0:
                step_y = -1
                side_y = (self.py - map_y) * ddy
            else:
                step_y = 1
                side_y = (map_y + 1.0 - self.py) * ddy

            hit = False
            side = 0  # 0 = vertical (N/S) wall, 1 = horizontal (E/W) wall

            while True:
                if side_x < side_y:
                    side_x += ddx
                    map_x += step_x
                    side = 0
                else:
                    side_y += ddy
                    map_y += step_y
                    side = 1

                # Perpendicular distance for early fog cutoff
                perp = (side_x - ddx) if side == 0 else (side_y - ddy)
                if perp > fog_dist + 0.5:
                    break

                if not (0 <= map_y < self.grid_h and 0 <= map_x < self.grid_w):
                    break

                cell_type = self.grid[map_y][map_x]
                if cell_type == self.WALL:
                    hit = True
                    break

            if not hit:
                continue

            # Perpendicular wall distance (corrects fisheye)
            perp_dist = (side_x - ddx) if side == 0 else (side_y - ddy)
            perp_dist = max(0.08, perp_dist)

            # Wall strip height
            line_h = int(H / perp_dist)
            y_start = max(0, H // 2 - line_h // 2)
            y_end = min(H - 1, H // 2 + line_h // 2)

            # Fog factor: 1.0 = fully visible, 0.0 = black
            fog_f = max(0.0, 1.0 - perp_dist / fog_dist)

            # Base wall color (goal adjacency check)
            is_goal_adj = (map_y, map_x) == (goal_r, goal_c)
            if is_goal_adj:
                base = GOAL_WALL_COLOR.astype(np.float32)
            elif side == 0:
                base = WALL_COLOR_NS.astype(np.float32)
            else:
                base = WALL_COLOR_EW.astype(np.float32)

            style = self.wall_style
            if style == "plain" or is_goal_adj:
                # Plain (and always plain on the goal wall to keep the marker recognizable)
                color = (base * fog_f).astype(np.uint8)
                pixels[y_start: y_end + 1, x] = color
            elif style == "repetitive":
                for y in range(y_start, y_end + 1):
                    m = _repetitive_stripe_mod(y)
                    pixels[y, x] = (base * m * fog_f).astype(np.uint8)
            elif style == "color_tag":
                tint = _face_hue_color(map_y, map_x, side)
                pixels[y_start: y_end + 1, x] = (tint * fog_f).astype(np.uint8)
            elif style == "unique_poster":
                tint = _face_hue_color(map_y, map_x, side)
                for y in range(y_start, y_end + 1):
                    m = _unique_face_stripe(map_y, map_x, side, y, y_start, y_end)
                    pixels[y, x] = (tint * m * fog_f).astype(np.uint8)
            else:
                raise ValueError(f"unknown wall_style: {style}")

        return pixels

    def _build_minimap(self) -> Image.Image:
        """
        Render the top-down minimap (MINIMAP_SIZE x MINIMAP_SIZE).
        Cell-level rendering: each passable cell occupies an equal-size pixel block,
        with walls drawn as overlaid grid lines. No longer uses the raw grid
        (2n+1)x(2n+1) ratio, to avoid distorting clearing/corridor proportions.
        - black background: unexplored area (fog)
        - light gray / bright white: seen / currently visible cells
        - orange / green: goal / start
        - dark grid lines: walls
        - blue dot + arrow: player position and facing
        """
        n = self.maze_size
        cell_px = MINIMAP_SIZE / n          # e.g. 140/7 = 20.0 px per cell
        wall_w = max(2, round(cell_px * 0.15))   # wall line width

        WALL_C = (30, 32, 55)
        mm = Image.new("RGB", (MINIMAP_SIZE, MINIMAP_SIZE), (10, 10, 20))
        draw = ImageDraw.Draw(mm)
        current_visible = self._get_current_visible()

        def crect(r, c):
            """Pixel bounding box for cell (r, c)."""
            x1 = round(c * cell_px)
            y1 = round(r * cell_px)
            x2 = round((c + 1) * cell_px) - 1
            y2 = round((r + 1) * cell_px) - 1
            return x1, y1, x2, y2

        def cell_seen(r, c):
            return (2 * r + 1, 2 * c + 1) in self.seen_raw

        def is_wall(r1, c1, r2, c2):
            """Wall between adjacent cells — formula: raw midpoint = (r1+r2+1, c1+c2+1)."""
            return self.grid[r1 + r2 + 1][c1 + c2 + 1] == self.WALL

        # ── 1. Cell backgrounds ──
        for r in range(n):
            for c in range(n):
                if not cell_seen(r, c):
                    continue
                raw_r, raw_c = 2 * r + 1, 2 * c + 1
                if (raw_r, raw_c) == self.goal_pos:
                    color = MM_GOAL
                elif (raw_r, raw_c) == self.start_pos:
                    color = MM_START
                elif (raw_r, raw_c) in current_visible:
                    color = MM_PATH_VISIBLE
                else:
                    color = MM_PATH_SEEN
                draw.rectangle(list(crect(r, c)), fill=color)

        # ── 2. Wall lines (only where at least one adjacent cell has been seen) ──
        # Vertical walls between (r, c) and (r, c+1)
        for r in range(n):
            for c in range(n - 1):
                if not (cell_seen(r, c) or cell_seen(r, c + 1)):
                    continue
                if is_wall(r, c, r, c + 1):
                    x = round((c + 1) * cell_px)
                    y1, y2 = round(r * cell_px), round((r + 1) * cell_px)
                    draw.line([(x, y1), (x, y2)], fill=WALL_C, width=wall_w)

        # Horizontal walls between (r, c) and (r+1, c)
        for r in range(n - 1):
            for c in range(n):
                if not (cell_seen(r, c) or cell_seen(r + 1, c)):
                    continue
                if is_wall(r, c, r + 1, c):
                    y = round((r + 1) * cell_px)
                    x1, x2 = round(c * cell_px), round((c + 1) * cell_px)
                    draw.line([(x1, y), (x2, y)], fill=WALL_C, width=wall_w)

        # ── 3. Player dot + facing arrow ──
        cell_r, cell_c = self.agent_cell_pos()
        px_mm = (cell_c + 0.5) * cell_px
        py_mm = (cell_r + 0.5) * cell_px
        r_dot = max(2, int(cell_px * 0.28))
        draw.ellipse(
            [px_mm - r_dot, py_mm - r_dot, px_mm + r_dot, py_mm + r_dot],
            fill=MM_PLAYER,
        )
        ang = math.radians(self.angle)
        arrow_len = max(3, int(cell_px * 0.42))
        ex = px_mm + math.cos(ang) * arrow_len
        ey = py_mm + math.sin(ang) * arrow_len
        draw.line([(px_mm, py_mm), (ex, ey)], fill=(255, 255, 255), width=2)

        # ── 4. Outer border ──
        draw.rectangle(
            [0, 0, MINIMAP_SIZE - 1, MINIMAP_SIZE - 1],
            outline=WALL_C,
            width=wall_w,
        )

        return mm

    def _build_local_patch(self, cell_px: int = 60, pad: int = 4) -> Image.Image:
        """
        3×3 world-aligned local patch (mm2d-local obs mode).

        Strips visual perception load while preserving non-Markov memory burden:
          - center cell = current position with player dot + facing arrow
          - 4 cardinal neighbors: open (light), wall-blocked (dark), or out-of-bounds (black)
          - 4 corner cells: black (not relevant)
          - wall lines drawn between center and each cardinal neighbor
          - goal cell highlighted only if it falls inside this 3×3 window
          - NO exploration history is shown — same info-content discipline as 3D raycast view
        """
        size = 3 * cell_px + 2 * pad
        BG_C      = (15, 15, 25)       # canvas / corner cells
        CUR_C     = (75, 82, 125)      # current cell — medium slate so dot+arrow pop
        OPEN_C    = (195, 205, 230)    # open passable neighbor
        BLOCKED_C = (32, 34, 55)       # wall-blocked neighbor
        OOB_C     = (25, 26, 40)       # out-of-bounds — distinct from BG and BLOCKED
        GOAL_C    = MM_GOAL            # orange goal highlight
        WALL_C    = (255, 190, 50)     # amber wall line — high contrast on all cells
        ARROW_C   = (255, 255, 255)    # white facing arrow — visible on CUR_C
        DOT_C     = (0, 210, 255)      # cyan player dot
        BORDER_C  = (80, 85, 120)

        img = Image.new("RGB", (size, size), BG_C)
        draw = ImageDraw.Draw(img)

        cur_r, cur_c = self.agent_cell_pos()
        goal_r, goal_c = self.goal_cell_pos()
        wall_w = max(2, round(cell_px * 0.10))

        def patch_to_world(pr, pc):
            return cur_r + pr - 1, cur_c + pc - 1

        def in_bounds(r, c):
            return 0 <= r < self.maze_size and 0 <= c < self.maze_size

        def has_wall(r1, c1, r2, c2):
            if not (in_bounds(r1, c1) and in_bounds(r2, c2)):
                return True
            return self.grid[r1 + r2 + 1][c1 + c2 + 1] == self.WALL

        def crect(pr, pc):
            x1 = pad + pc * cell_px
            y1 = pad + pr * cell_px
            return [x1, y1, x1 + cell_px - 1, y1 + cell_px - 1]

        cardinal = {(0, 1), (1, 0), (1, 2), (2, 1)}

        # 1. Cell backgrounds
        for pr in range(3):
            for pc in range(3):
                wr, wc = patch_to_world(pr, pc)
                if (pr, pc) == (1, 1):
                    color = CUR_C
                elif (pr, pc) in cardinal:
                    if not in_bounds(wr, wc):
                        color = OOB_C
                    elif has_wall(cur_r, cur_c, wr, wc):
                        color = BLOCKED_C
                    elif (wr, wc) == (goal_r, goal_c):
                        color = GOAL_C
                    else:
                        color = OPEN_C
                else:
                    color = OOB_C
                draw.rectangle(crect(pr, pc), fill=color)

        # 2. Wall lines between center and each cardinal neighbor
        center_box = crect(1, 1)
        edges = [
            ((0, 1), [(center_box[0], center_box[1]), (center_box[2], center_box[1])]),  # north
            ((2, 1), [(center_box[0], center_box[3]), (center_box[2], center_box[3])]),  # south
            ((1, 0), [(center_box[0], center_box[1]), (center_box[0], center_box[3])]),  # west
            ((1, 2), [(center_box[2], center_box[1]), (center_box[2], center_box[3])]),  # east
        ]
        for (pr, pc), line_pts in edges:
            wr, wc = patch_to_world(pr, pc)
            if has_wall(cur_r, cur_c, wr, wc):
                draw.line(line_pts, fill=WALL_C, width=wall_w)

        # 3. Player dot + facing arrow at center
        cx = pad + cell_px + cell_px // 2
        cy = pad + cell_px + cell_px // 2
        r_dot = max(3, int(cell_px * 0.22))
        draw.ellipse([cx - r_dot, cy - r_dot, cx + r_dot, cy + r_dot], fill=DOT_C)
        ang = math.radians(self.angle)
        arrow_len = int(cell_px * 0.40)
        ex = cx + int(math.cos(ang) * arrow_len)
        ey = cy + int(math.sin(ang) * arrow_len)
        draw.line([(cx, cy), (ex, ey)], fill=ARROW_C, width=3)

        # 4. Outer border
        draw.rectangle([0, 0, size - 1, size - 1], outline=BORDER_C, width=2)

        return img

    def _build_local_patch_cone(self, cell_px: int = 50, pad: int = 4, show_hud: bool = True) -> Image.Image:
        """
        Forward-facing cone 2D patch (mm2d-cone obs mode), aligned with 3D scene visibility.

        Renders a (2*vision_range+1) x (2*vision_range+1) world-aligned grid centered on
        the agent. A cell is "visible" iff:
          - it lies within the agent's FOV cone (FOV_DEG, facing direction), AND
          - it is reachable by a ray from the agent that is not blocked by an intervening
            wall (line-of-sight, DDA across the raw grid), AND
          - its (Chebyshev / radial) distance is within vision_range.

        Visible cells display open / wall-blocked / goal coloring; non-visible cells are
        rendered as fog (uniform dark, no wall info) — matching the 3D scene's behavior
        where occluded geometry is invisible.

        - center cell = agent with player dot + facing arrow
        - wall lines drawn only between visible cells
        - the canvas is rotated so the agent's facing direction always points UP
        """
        vr = self.vision_range
        grid_n = 2 * vr + 1
        size = grid_n * cell_px + 2 * pad

        BG_C       = (15, 15, 25)
        FOG_C      = (20, 22, 35)        # outside FOV / occluded — dark fog
        CUR_C      = (75, 82, 125)
        OPEN_C     = (195, 205, 230)     # light blue = passable / corridor (the "path")
        OPEN_BORDER_C = (110, 130, 170)  # darker blue outline to make path crisp on fog
        BLOCKED_C  = (32, 34, 55)
        OOB_C      = (25, 26, 40)
        GOAL_C     = MM_GOAL             # orange — RESERVED for goal cell only
        WALL_C     = (255, 40, 40)       # vivid red wall lines (clearly NOT the orange goal)
        WALL_OUTLINE_C = (90, 0, 0)      # darker outline behind wall to boost contrast on fog
        ARROW_C    = (255, 255, 255)
        DOT_C      = (0, 210, 255)
        BORDER_C   = (80, 85, 120)

        cur_r, cur_c = self.agent_cell_pos()

        # Determine FOV cone in WORLD coordinates first; we'll rotate canvas at the end so
        # facing points up.
        face_angle = math.radians(self.angle)   # world angle (col=x, row=y)
        half_fov = math.radians(FOV_DEG / 2)

        def in_bounds_cell(r, c):
            return 0 <= r < self.maze_size and 0 <= c < self.maze_size

        def has_wall_between(r1, c1, r2, c2):
            """True iff the cell-edge between two adjacent cells is a wall."""
            if not (in_bounds_cell(r1, c1) and in_bounds_cell(r2, c2)):
                return True
            return self.grid[r1 + r2 + 1][c1 + c2 + 1] == self.WALL

        def los_clear(tr, tc):
            """True iff a straight ray from (cur_r,cur_c) center to (tr,tc) center is not
            blocked by any wall in between. Walks cell-by-cell along the line."""
            if (tr, tc) == (cur_r, cur_c):
                return True
            steps = max(abs(tr - cur_r), abs(tc - cur_c)) * 4 + 1
            for k in range(1, steps + 1):
                a = k / steps
                rr = cur_r + (tr - cur_r) * a
                cc = cur_c + (tc - cur_c) * a
                # Check the wall cell in raw grid that this point lies in.
                # raw row of cell border between (r,c) and (r+1,c) is 2r+2; between cells
                # in same row is 2c+2. We sample the raw grid cell at the current point.
                raw_r = int(round(rr * 2 + 1))   # cell center is at 2r+1
                raw_c = int(round(cc * 2 + 1))
                # If this raw point falls on a wall coordinate (even index in either axis),
                # check it.
                # Simpler: sample slightly between integer cell centers and detect wall cells
                rrr = rr * 2 + 1; ccc = cc * 2 + 1
                ri = int(round(rrr)); ci = int(round(ccc))
                if 0 <= ri < self.grid_h and 0 <= ci < self.grid_w:
                    # Treat any sampled raw cell that is a wall as blocking, except the
                    # endpoints (which are cell centers, never walls).
                    is_endpoint = (ri == 2 * cur_r + 1 and ci == 2 * cur_c + 1) or \
                                  (ri == 2 * tr + 1 and ci == 2 * tc + 1)
                    if not is_endpoint and self.grid[ri][ci] == self.WALL:
                        return False
            return True

        def in_fov(tr, tc):
            if (tr, tc) == (cur_r, cur_c):
                return True
            dy = tr - cur_r
            dx = tc - cur_c
            ang = math.atan2(dy, dx)
            d = ang - face_angle
            # normalize to [-pi, pi]
            while d > math.pi:
                d -= 2 * math.pi
            while d < -math.pi:
                d += 2 * math.pi
            return abs(d) <= half_fov

        # 1. Build canvas in WORLD orientation (north = up = -row), agent at center
        img = Image.new("RGB", (size, size), BG_C)
        draw = ImageDraw.Draw(img)
        wall_w = max(5, round(cell_px * 0.20))         # thick enough to read on fog
        wall_outline_w = wall_w + 4                     # dark outline behind wall

        def crect(pr, pc):
            x1 = pad + pc * cell_px
            y1 = pad + pr * cell_px
            return [x1, y1, x1 + cell_px - 1, y1 + cell_px - 1]

        # Determine visibility for each cell in the bounding window
        visible = {}
        for pr in range(grid_n):
            for pc in range(grid_n):
                wr = cur_r + (pr - vr)
                wc = cur_c + (pc - vr)
                if (pr, pc) == (vr, vr):
                    visible[(pr, pc)] = ("cur", wr, wc); continue
                if not in_bounds_cell(wr, wc):
                    visible[(pr, pc)] = ("oob", wr, wc); continue
                # Radial cap at vision_range cells (Chebyshev — matches a square of radius vr)
                if max(abs(wr - cur_r), abs(wc - cur_c)) > vr:
                    visible[(pr, pc)] = ("fog", wr, wc); continue
                if not in_fov(wr, wc):
                    visible[(pr, pc)] = ("fog", wr, wc); continue
                if not los_clear(wr, wc):
                    visible[(pr, pc)] = ("fog", wr, wc); continue
                visible[(pr, pc)] = ("vis", wr, wc)

        # Side-junction reveal: if a visible cell has a passable side opening (no wall),
        # spill 1 cell into the branch even if that neighbor was outside the FOV cone.
        # This mirrors what the 3D camera "sees" as a side-passage opening on the wall.
        for (pr, pc), (kind, wr, wc) in list(visible.items()):
            if kind != "vis" and kind != "cur":
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = wr + dr, wc + dc
                npr, npc = pr + dr, pc + dc
                if not (0 <= npr < grid_n and 0 <= npc < grid_n):
                    continue
                if not in_bounds_cell(nr, nc):
                    continue
                if has_wall_between(wr, wc, nr, nc):
                    continue
                if visible[(npr, npc)][0] in ("fog",):
                    visible[(npr, npc)] = ("vis", nr, nc)

        goal_r, goal_c = self.goal_cell_pos()

        # 2. Fill cell backgrounds
        for (pr, pc), (kind, wr, wc) in visible.items():
            if kind == "cur":
                color = CUR_C
            elif kind == "oob":
                color = OOB_C
            elif kind == "fog":
                color = FOG_C
            else:
                if (wr, wc) == (goal_r, goal_c):
                    color = GOAL_C
                else:
                    color = OPEN_C
            box = crect(pr, pc)
            draw.rectangle(box, fill=color)
            # Outline open / goal cells with a darker border so the passable path is
            # visually distinct from the dark fog cells around it.
            if kind == "vis" and (wr, wc) != (goal_r, goal_c):
                draw.rectangle(box, outline=OPEN_BORDER_C, width=2)

        # 3. Wall lines: only between two cells where at least one is "vis" or "cur"
        def is_visible_kind(k):
            return k in ("vis", "cur")

        for pr in range(grid_n):
            for pc in range(grid_n):
                if not is_visible_kind(visible[(pr, pc)][0]):
                    continue
                wr, wc = visible[(pr, pc)][1], visible[(pr, pc)][2]
                box = crect(pr, pc)
                # check 4 edges (north/south/west/east neighbors)
                neighbors = [
                    (-1, 0, [(box[0], box[1]), (box[2], box[1])]),    # north edge top
                    (1, 0,  [(box[0], box[3]), (box[2], box[3])]),    # south edge bottom
                    (0, -1, [(box[0], box[1]), (box[0], box[3])]),    # west edge left
                    (0, 1,  [(box[2], box[1]), (box[2], box[3])]),    # east edge right
                ]
                for dr, dc, line_pts in neighbors:
                    nwr, nwc = wr + dr, wc + dc
                    if has_wall_between(wr, wc, nwr, nwc):
                        # Draw dark outline first, then bright wall on top, so the wall
                        # is readable even when one side is dark fog.
                        draw.line(line_pts, fill=WALL_OUTLINE_C, width=wall_outline_w)
                        draw.line(line_pts, fill=WALL_C, width=wall_w)

        # 4. Player dot + facing arrow at center cell (drawn in WORLD orientation, i.e.,
        # arrow points in world direction)
        cx = pad + vr * cell_px + cell_px // 2
        cy = pad + vr * cell_px + cell_px // 2
        r_dot = max(3, int(cell_px * 0.22))
        draw.ellipse([cx - r_dot, cy - r_dot, cx + r_dot, cy + r_dot], fill=DOT_C)
        ang = face_angle
        arrow_len = int(cell_px * 0.40)
        ex = cx + int(math.cos(ang) * arrow_len)
        ey = cy + int(math.sin(ang) * arrow_len)
        draw.line([(cx, cy), (ex, ey)], fill=ARROW_C, width=3)

        # 5. Outer border
        draw.rectangle([0, 0, size - 1, size - 1], outline=BORDER_C, width=2)

        # 6. Rotate so facing points up: world facing angle -> rotate canvas by (-90 - angle_deg)
        # In world coords, angle=0 is east (right), angle=90 is south (down). To make facing
        # point up (image -90deg), rotate canvas by (angle_deg + 90) counter-clockwise.
        rot_deg = self.angle + 90
        img = img.rotate(rot_deg, resample=Image.BILINEAR, fillcolor=BG_C)

        if not show_hud:
            return img

        # 7. Append HUD strip (same content as 3D scene HUD)
        hud_h = HUD_H
        full = Image.new("RGB", (img.width, img.height + hud_h), HUD_BG)
        full.paste(img, (0, 0))
        d = ImageDraw.Draw(full)
        d.rectangle([0, img.height, img.width - 1, img.height + hud_h - 1], fill=HUD_BG)
        # Reuse the same HUD drawing code via a temporary translation: build a fake image
        # with just the HUD strip and copy. Easiest: replicate the text layout here.
        FONT_SIZE = 13
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", FONT_SIZE)
        except Exception:
            font = ImageFont.load_default()
        cell_r, cell_c = self.agent_cell_pos()
        goal_r, goal_c = self.goal_cell_pos()
        avail = self.get_available_actions()
        y0 = img.height
        # line 1
        line1_parts = [
            (f"Pos:({cell_r},{cell_c})", HUD_ACCENT),
            ("  ", HUD_TEXT),
            (f"Facing:{self.facing}", HUD_ACCENT),
            ("  ", HUD_TEXT),
            (f"Goal:({goal_r},{goal_c})", HUD_ACCENT),
            ("  ", HUD_TEXT),
            (f"Steps:{self.step_count}", HUD_TEXT),
            ("  ", HUD_TEXT),
            (f"Explored:{len(self.visited_cells)}/{self.maze_size**2}", HUD_TEXT),
        ]
        x = 8
        for txt, col in line1_parts:
            d.text((x, y0 + 6), txt, fill=col, font=font)
            try:
                bbox = d.textbbox((0, 0), txt, font=font)
                x += bbox[2] - bbox[0]
            except AttributeError:
                x += d.textlength(txt, font=font)
        # line 2
        line2_parts = [
            (f"Actions:[{', '.join(avail)}]", HUD_TEXT),
        ]
        x = 8
        for txt, col in line2_parts:
            d.text((x, y0 + 6 + FONT_SIZE + 4), txt, fill=col, font=font)
        return full

    def render_local_patch_cone_b64(self) -> str:
        """Render facing-cone 2D patch and return JPEG base64 for VLM API."""
        img = self._build_local_patch_cone()
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def render_local_text(self) -> str:
        """
        Text-symbolic local view (text-symbolic obs mode).

        Describes the 4 cardinal neighbours as OPEN / WALL / OUT-OF-BOUNDS,
        with GOAL tag if the goal cell is adjacent. No image is produced.
        Zero visual perception burden; full non-Markov memory burden preserved.
        """
        cur_r, cur_c = self.agent_cell_pos()
        goal_r, goal_c = self.goal_cell_pos()

        def in_bounds(r, c):
            return 0 <= r < self.maze_size and 0 <= c < self.maze_size

        def passage(dr, dc):
            nr, nc = cur_r + dr, cur_c + dc
            if not in_bounds(nr, nc):
                return "OUT-OF-BOUNDS"
            wall_r = cur_r + nr + 1
            wall_c = cur_c + nc + 1
            if self.grid[wall_r][wall_c] == self.WALL:
                return "WALL"
            if (nr, nc) == (goal_r, goal_c):
                return "OPEN (GOAL)"
            return "OPEN"

        dirs = [("North", -1, 0), ("East", 0, 1), ("South", 1, 0), ("West", 0, -1)]
        lines = ["Local surroundings (world-aligned, no history shown):"]
        for name, dr, dc in dirs:
            status = passage(dr, dc)
            marker = " ◀ FACING" if FACE_DELTA_CELL[self.facing] == (dr, dc) else ""
            lines.append(f"  {name:5s}: {status}{marker}")
        return "\n".join(lines)

    def render_local_patch_b64(self) -> str:
        """Render mm2d-local patch and return JPEG base64 for VLM API."""
        img = self._build_local_patch()
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=92, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _draw_hud(self, img: Image.Image):
        """Draw the HUD info bar at the bottom of the image (two rows)."""
        draw = ImageDraw.Draw(img)
        y0 = RENDER_H

        # Background
        draw.rectangle([0, y0, IMG_W - 1, IMG_H - 1], fill=HUD_BG)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

        cell_r, cell_c = self.agent_cell_pos()
        goal_r, goal_c = self.goal_cell_pos()
        avail = self.get_available_actions()

        # Line 1: state
        line1 = [
            (f"Pos:({cell_r},{cell_c})", HUD_ACCENT),
            ("  ", HUD_TEXT),
            (f"Facing:{self.facing}", HUD_ACCENT),
            ("  ", HUD_TEXT),
            (f"Goal:({goal_r},{goal_c})", HUD_ACCENT),
            ("  ", HUD_TEXT),
            (f"Steps:{self.step_count}", HUD_TEXT),
            ("  ", HUD_TEXT),
            (f"Explored:{len(self.visited_cells)}/{self.maze_size**2}", HUD_TEXT),
        ]
        # Line 2: available actions (full list, never truncated)
        line2 = [
            (f"Actions:[{', '.join(avail)}]", HUD_TEXT),
        ]

        for line_idx, parts in enumerate((line1, line2)):
            x = 10
            y = y0 + 6 + line_idx * 20
            for text, color in parts:
                draw.text((x, y), text, fill=color, font=font)
                bbox = draw.textbbox((x, y), text, font=font)
                x = bbox[2]

    # ── Image Export ──────────────────────────────────────────────────────────

    def render_topdown_global(self, step: int = 0, cell_px: int = 24) -> Image.Image:
        """
        Render the global top-down view (no fog):
          - full maze structure visible
          - the traversed path marked with a gradient trail (light blue -> dark blue)
          - current position: bright-blue dot + facing arrow
          - start green, goal orange
          - step count labeled in the top-right corner
        cell_px: pixel size of each raw grid cell
        """
        pad = 4
        w = self.grid_w * cell_px + pad * 2
        h = self.grid_h * cell_px + pad * 2 + 28  # 28px title bar

        img = Image.new("RGB", (w, h), (15, 15, 25))
        draw = ImageDraw.Draw(img)

        # ── Title bar ──
        try:
            font_title = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 13
            )
            font_small = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11
            )
        except (OSError, IOError):
            font_title = font_small = ImageFont.load_default()

        goal_r, goal_c = self.goal_cell_pos()
        cur_r, cur_c = self.agent_cell_pos()
        draw.rectangle([0, 0, w - 1, 27], fill=(25, 28, 50))
        draw.text(
            (6, 6),
            f"Step {step:3d}  |  Pos:({cur_r},{cur_c})  Facing:{self.facing}  "
            f"Goal:({goal_r},{goal_c})  Explored:{len(self.visited_cells)}/{self.maze_size**2}",
            fill=(180, 190, 220),
            font=font_small,
        )

        # ── Draw raw grid cells ──
        def cell_rect(raw_row, raw_col):
            x1 = pad + raw_col * cell_px
            y1 = 28 + pad + raw_row * cell_px
            return x1, y1, x1 + cell_px - 1, y1 + cell_px - 1

        WALL_C = (45, 48, 75)
        PATH_C = (210, 215, 230)
        START_C = (30, 200, 80)
        GOAL_C = (240, 150, 20)

        for row in range(self.grid_h):
            for col in range(self.grid_w):
                x1, y1, x2, y2 = cell_rect(row, col)
                ct = self.grid[row][col]
                if ct == self.WALL:
                    color = WALL_C
                elif (row, col) == self.goal_pos:
                    color = GOAL_C
                elif (row, col) == self.start_pos:
                    color = START_C
                else:
                    color = PATH_C
                draw.rectangle([x1, y1, x2, y2], fill=color)

        # ── Draw path trail (gradient blue: light=old → dark=new) ──
        n = len(self.path_history)
        for i, (cr, cc) in enumerate(self.path_history):
            t = i / max(n - 1, 1)  # 0=oldest, 1=newest
            raw_r = 2 * cr + 1
            raw_c = 2 * cc + 1
            x1, y1, x2, y2 = cell_rect(raw_r, raw_c)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            r_dot = max(2, cell_px // 3)
            # light blue → vivid blue
            blue = int(180 + 75 * t)
            green = int(200 - 120 * t)
            red = int(100 - 80 * t)
            draw.ellipse([cx - r_dot, cy - r_dot, cx + r_dot, cy + r_dot],
                         fill=(red, green, blue))

        # ── Draw player: circle + direction arrow ──
        raw_row, raw_col = self._current_raw_cell()
        x1, y1, x2, y2 = cell_rect(raw_row, raw_col)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        r_agent = max(3, cell_px // 2 - 2)
        draw.ellipse([cx - r_agent, cy - r_agent, cx + r_agent, cy + r_agent],
                     fill=(0, 200, 255), outline=(255, 255, 255), width=1)

        # Direction arrow
        ang = math.radians(self.angle)
        arrow_len = max(4, int(cell_px * 0.9))
        ex = int(cx + math.cos(ang) * arrow_len)
        ey = int(cy + math.sin(ang) * arrow_len)
        draw.line([(cx, cy), (ex, ey)], fill=(255, 255, 255), width=2)

        # ── Draw grid lines (subtle) ──
        for row in range(self.grid_h + 1):
            y = 28 + pad + row * cell_px
            draw.line([(pad, y), (pad + self.grid_w * cell_px, y)], fill=(30, 32, 55), width=1)
        for col in range(self.grid_w + 1):
            x = pad + col * cell_px
            draw.line([(x, 28 + pad), (x, 28 + pad + self.grid_h * cell_px)], fill=(30, 32, 55), width=1)

        return img

    def render_image_b64(self, show_minimap: bool = False, show_hud: bool = True) -> str:
        """Render a frame and return a JPEG base64 string (for VLM API calls)."""
        img = self.render_frame(show_minimap=show_minimap, show_hud=show_hud)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=88, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def render_full_state_image(self) -> Image.Image:
        """Debug: render the full-maze top-down view (no fog)."""
        img = self.render_frame()
        # Overlay unseen areas with seen_raw = all cells
        _saved = self.seen_raw.copy()
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                self.seen_raw.add((r, c))
        full_mm = self._build_minimap()
        self.seen_raw = _saved
        mm_x = RENDER_W - MINIMAP_SIZE - 8
        mm_y = 8
        img.paste(full_mm, (mm_x, mm_y))
        return img

    # ── Statistics & Evaluation ───────────────────────────────────────────────

    def _can_move_cell(self, r: int, c: int, face: str) -> bool:
        """Check whether one can move from cell (r,c) in direction face (used by BFS)."""
        dr, dc = FACE_DELTA_CELL[face]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.maze_size and 0 <= nc < self.maze_size):
            return False
        wall_row = 2 * r + 1 + dr
        wall_col = 2 * c + 1 + dc
        return self.grid[wall_row][wall_col] != self.WALL

    def compute_optimal_path_length(self) -> int:
        """
        BFS for the optimal action-sequence length (including turn cost).
        Initial state: start_cell, facing=E
        Terminal: goal_cell (any facing)
        Uses v4 or v5 rules according to self.action_space.
        """
        start_cell = self._raw_to_cell(self.start_pos[0], self.start_pos[1])
        goal_cell = self.goal_cell_pos()

        initial = (start_cell[0], start_cell[1], "E")
        queue = deque([(initial, 0)])
        seen = {initial}

        while queue:
            (r, c, face), dist = queue.popleft()
            if (r, c) == goal_cell:
                return dist

            # Turn actions (same for v4 and v5)
            for new_face in (TURN_LEFT[face], TURN_RIGHT[face]):
                ns = (r, c, new_face)
                if ns not in seen:
                    seen.add(ns)
                    queue.append((ns, dist + 1))

            if self.action_space == "v5":
                # v5: move_forward slides until wall OR junction (left/right passage)
                dr, dc = FACE_DELTA_CELL[face]
                cr, cc = r, c
                left_f = TURN_LEFT[face]
                right_f = TURN_RIGHT[face]
                while self._can_move_cell(cr, cc, face):
                    cr += dr
                    cc += dc
                    # Each reachable cell in one move_forward costs 1 step
                    ns = (cr, cc, face)
                    if ns not in seen:
                        seen.add(ns)
                        queue.append((ns, dist + 1))
                    if (cr, cc) == goal_cell:
                        break
                    # Junction-stop: mirror actual game behavior
                    if self._can_move_cell(cr, cc, left_f) or self._can_move_cell(cr, cc, right_f):
                        break
            else:
                # v4: forward / backward (move 1 cell)
                for move_face in (face, OPPOSITE[face]):
                    dr, dc = FACE_DELTA_CELL[move_face]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.maze_size and 0 <= nc < self.maze_size:
                        wall_row = 2 * r + 1 + dr
                        wall_col = 2 * c + 1 + dc
                        if self.grid[wall_row][wall_col] != self.WALL:
                            ns = (nr, nc, move_face)
                            if ns not in seen:
                                seen.add(ns)
                                queue.append((ns, dist + 1))

        return -1  # unreachable

    def total_cells(self) -> int:
        return self.maze_size * self.maze_size

    def explored_cells(self) -> int:
        return len(self.visited_cells)

    def exploration_rate(self) -> float:
        return self.explored_cells() / self.total_cells()

    def render_top_down_text(self, full: bool = False) -> str:
        """Text version of the top-down maze (debug)."""
        lines = []
        raw_row, raw_col = self._current_raw_cell()
        vr = self.vision_range * 2
        for r in range(self.grid_h):
            row_str = ""
            for c in range(self.grid_w):
                if (r, c) == (raw_row, raw_col):
                    row_str += "@"
                elif not full and abs(r - raw_row) + abs(c - raw_col) > vr:
                    row_str += "?"
                elif (r, c) == self.goal_pos:
                    row_str += "G"
                else:
                    row_str += self.grid[r][c]
            lines.append(row_str)
        return "\n".join(lines)


# ─── Action Parser ────────────────────────────────────────────────────────────

def parse_action(text: str, action_space: str = "v4") -> Optional[str]:
    """
    Parse an action from the model output.
    action_space="v4": forward, backward, turn_left, turn_right
    action_space="v5": move_forward, turn_left, turn_right
    """
    # Strip reasoning tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Focus on text after "Action:"
    if "action" in text.lower():
        idx = text.lower().rfind("action")
        text = text[idx:]

    text_lower = text.lower()

    if action_space == "v5":
        # v5: move_forward, turn_left, turn_right
        for action in ACTIONS_V5:
            pattern = action.replace("_", r"[_\s]")
            if re.search(r"\b" + pattern + r"\b", text_lower):
                return action
        # "forward" alone → move_forward in v5
        aliases_v5 = [
            (r"\bforward\b|\bgo\s+forward\b|\badvance\b|\bwalk\s+forward\b", "move_forward"),
            (r"\bturn\s+left\b|\brotate\s+left\b|\bleft\s+turn\b", "turn_left"),
            (r"\bturn\s+right\b|\brotate\s+right\b|\bright\s+turn\b", "turn_right"),
        ]
        for pattern, action in aliases_v5:
            if re.search(pattern, text_lower):
                return action
        return None

    # v4: forward, backward, turn_left, turn_right
    for action in ACTIONS:
        pattern = action.replace("_", r"[_\s]")
        if re.search(r"\b" + pattern + r"\b", text_lower):
            return action

    aliases = [
        (r"\bgo\s+forward\b|\bmove\s+forward\b|\badvance\b|\bwalk\s+forward\b", "forward"),
        (r"\bgo\s+back(ward)?\b|\bmove\s+back(ward)?\b|\bretreat\b", "backward"),
        (r"\bturn\s+left\b|\brotate\s+left\b|\bleft\s+turn\b", "turn_left"),
        (r"\bturn\s+right\b|\brotate\s+right\b|\bright\s+turn\b", "turn_right"),
    ]
    for pattern, action in aliases:
        if re.search(pattern, text_lower):
            return action

    return None
