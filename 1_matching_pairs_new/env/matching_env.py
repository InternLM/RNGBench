"""
MatchingEnv — ENV-style interface wrapper.

Action = the coord (r, c) of a single card.
Internal state machine: flip_first -> flip_second -> resolve -> next round's flip_first.

The env only handles game logic + rendering; it does not deal with image persistence
or message construction (the runner does that).
Rendering has two modes: "image" (returns PIL) and "text" (returns an ASCII string).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from env.board import Board, Coord, coord_to_str
from env.renderer import ImageCardRenderer, ImageRenderer, TextRenderer


class InvalidActionError(Exception):
    """Raised when step() rejects an action.

    code is a machine-readable category for prompt generation:
      - "out_of_range": coord outside the grid
      - "already_removed": coord points to a matched/removed card
      - "same_as_first": flip_second chose the same coord as the first flip
    reason is a human-readable string (may contain the coord).
    """
    def __init__(self, reason: str, code: str = "invalid", coord: Optional[Coord] = None):
        super().__init__(reason)
        self.reason = reason
        self.code = code
        self.coord = coord


@dataclass
class Observation:
    """Read-only state the env exposes to the runner (no render data)."""
    phase: str
    rows: int
    cols: int
    remaining_pairs: int
    score: int
    done: bool
    first_flip_coord: Optional[Coord] = None
    first_flip_face: Optional[str] = None
    last_result: Optional[Dict[str, Any]] = None


class MatchingEnv:
    """Memory matching-pairs environment.

    Args:
        rows, cols, num_cards, seed: board params
        theme: asset theme used in image mode; None -> render ASCII text into an image
        assets_dir, cell_size: asset config (only used in image mode + theme)
    """

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        num_cards: Optional[int] = None,
        seed: int = 0,
        theme: Optional[str] = None,
        assets_dir: str = "assets",
        cell_size: Tuple[int, int] = (64, 64),
    ):
        self.rows = rows
        self.cols = cols
        self.num_cards = num_cards
        self.seed = seed
        self.theme = theme
        self._assets_dir = assets_dir
        self._cell_size = cell_size

        self.board: Optional[Board] = None
        self._phase: str = "flip_first"
        self._first_flip_coord: Optional[Coord] = None
        self._first_flip_face: Optional[str] = None
        self._last_result: Optional[Dict[str, Any]] = None

        self._text_renderer = TextRenderer()
        self._image_renderer: Optional[Union[ImageCardRenderer, ImageRenderer]] = None

    # ── Lifecycle ───────────────────────────────────────────────────────

    def reset(self) -> Observation:
        self.board = Board(
            rows=self.rows, cols=self.cols,
            num_cards=self.num_cards, seed=self.seed,
        )
        self._phase = "flip_first"
        self._first_flip_coord = None
        self._first_flip_face = None
        self._last_result = None

        if self.theme:
            from env.card_assets import CardAssets
            assets = CardAssets(theme=self.theme, assets_dir=self._assets_dir, cell_size=self._cell_size)
            assets.select(self.board.symbols, self.seed)
            self._image_renderer = ImageCardRenderer(card_assets=assets)
        else:
            self._image_renderer = ImageRenderer()

        return self.get_observation()

    # ── Main interface ──────────────────────────────────────────────────

    def step(self, action: Coord) -> Observation:
        if self.board is None:
            raise RuntimeError("Env not reset")
        if self.board.is_game_over():
            raise RuntimeError("Game already over")

        self._validate(action)

        if self._phase == "flip_first":
            self._first_flip_coord = action
            self._first_flip_face = self.board.get_face(action)
            self._phase = "flip_second"
            self._last_result = None
        else:
            face1 = self._first_flip_face
            coord1 = self._first_flip_coord
            face2 = self.board.get_face(action)
            _, _, matched = self.board.flip(coord1, action)
            self._last_result = {
                "coord1": coord_to_str(coord1), "face1": face1,
                "coord2": coord_to_str(action), "face2": face2,
                "matched": matched,
            }
            self._first_flip_coord = None
            self._first_flip_face = None
            self._phase = "flip_first"

        return self.get_observation()

    def abort_round(self) -> Observation:
        if self._phase != "flip_second":
            raise RuntimeError(f"abort_round called in phase {self._phase}")
        self._first_flip_coord = None
        self._first_flip_face = None
        self._phase = "flip_first"
        self._last_result = None
        return self.get_observation()

    # ── Rendering ───────────────────────────────────────────────────────

    def render_board(
        self,
        mode: str = "image",
        current_flips: Optional[List[Coord]] = None,
    ):
        """Render the current board. mode="image" -> PIL.Image; mode="text" -> str.

        current_flips default: in the flip_second phase the first_flip is shown
        automatically; in the flip_first phase, none.
        """
        if self.board is None:
            raise RuntimeError("Env not reset")
        if current_flips is None:
            current_flips = [self._first_flip_coord] if self._first_flip_coord else []

        if mode == "text":
            return self._text_renderer.render(self.board, current_flips)
        elif mode == "image":
            return self._image_renderer.render(self.board, current_flips)
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def render_both_flips(self, coord1: Coord, coord2: Coord, mode: str = "image"):
        """Render the transient "both cards revealed" view (does not change env state)."""
        if mode == "text":
            return self._text_renderer.render(self.board, [coord1, coord2])
        elif mode == "image":
            return self._image_renderer.render(self.board, [coord1, coord2])
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def render_ground_truth(self, mode: str = "image"):
        """Ground-truth view with all card faces visible."""
        all_coords = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        if mode == "text":
            return self._text_renderer.render_full(self.board)
        elif mode == "image":
            return self._image_renderer.render(self.board, all_coords)
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    # ── State queries ───────────────────────────────────────────────────

    def _validate(self, action: Coord) -> None:
        r, c = action
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise InvalidActionError(
                f"coord {coord_to_str(action)} out of range", code="out_of_range", coord=action)
        if self.board.is_removed(action):
            raise InvalidActionError(
                f"{coord_to_str(action)} is already matched/removed", code="already_removed", coord=action)
        if self._phase == "flip_second" and action == self._first_flip_coord:
            raise InvalidActionError(
                f"second flip must differ from first ({coord_to_str(action)})",
                code="same_as_first", coord=action)

    def get_observation(self) -> Observation:
        return Observation(
            phase=self._phase,
            rows=self.rows, cols=self.cols,
            remaining_pairs=self.board.remaining_pairs() if self.board else 0,
            score=self._score(),
            done=self.board.is_game_over() if self.board else False,
            first_flip_coord=self._first_flip_coord,
            first_flip_face=self._first_flip_face,
            last_result=self._last_result,
        )

    def _score(self) -> int:
        if self.board is None:
            return 0
        return self.board.total_pairs - self.board.remaining_pairs()

    @property
    def total_pairs(self) -> int:
        return self.board.total_pairs

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def first_flip_coord(self) -> Optional[Coord]:
        return self._first_flip_coord

    @property
    def first_flip_face(self) -> Optional[str]:
        return self._first_flip_face

    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        return self._last_result
