"""
Board state management.

Responsibility: card layout, coordinate system, flip/match logic. Pure game
state — no rendering or LLM involvement.
"""

import random
import string
from typing import List, Optional, Tuple

Coord = Tuple[int, int]

# Card symbol pool: digits + uppercase + lowercase; numbered indices beyond that
_BASE_SYMBOLS = string.digits + string.ascii_uppercase + string.ascii_lowercase


def _get_symbols(n: int) -> List[str]:
    """Get n distinct card symbols."""
    if n <= len(_BASE_SYMBOLS):
        return list(_BASE_SYMBOLS[:n])
    # beyond 62, use numeric indices
    return [str(i) for i in range(n)]


def coord_to_str(coord: Coord) -> str:
    """(row, col) -> 'aA'-format coordinate."""
    r, c = coord
    return f"{string.ascii_lowercase[r]}{string.ascii_uppercase[c]}"


class Board:
    """Memory matching-pairs board.

    Manages card layout and match state. Supports a seed for reproducible randomness.

    Attributes:
        rows: number of rows
        cols: number of columns
        num_cards: number of distinct card faces
        seed: random seed
        cards: 2D array of card faces (cards[r][c] = card symbol)
        matched: 2D match state (matched[r][c] = whether removed)
    """

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        num_cards: Optional[int] = None,
        seed: int = 0,
    ):
        total = rows * cols
        if total % 2 != 0:
            raise ValueError("Grid must have even number of cells (rows * cols).")
        if num_cards is None:
            num_cards = total // 2
        if num_cards > total // 2:
            raise ValueError(f"num_cards ({num_cards}) must be <= {total // 2}.")
        if (total // 2) % num_cards != 0:
            raise ValueError("total_pairs must be divisible by num_cards.")

        self.rows = rows
        self.cols = cols
        self.num_cards = num_cards
        self.seed = seed
        self.total_pairs = total // 2
        self.repeat_count = (self.total_pairs // num_cards) * 2

        # generate the symbol pool
        self.symbols = _get_symbols(num_cards)

        # initialize the board
        self.cards: List[List[str]] = []
        self.matched: List[List[bool]] = [[False] * cols for _ in range(rows)]
        self._init_cards()

    def _init_cards(self) -> None:
        """Initialize the card layout from the seed."""
        rng = random.Random(self.seed)
        flat = []
        for i in range(self.total_pairs):
            card_id = i % self.num_cards
            flat.extend([self.symbols[card_id], self.symbols[card_id]])
        rng.shuffle(flat)
        self.cards = [flat[i * self.cols : (i + 1) * self.cols] for i in range(self.rows)]

    def flip(self, coord1: Coord, coord2: Coord) -> Tuple[str, str, bool]:
        """Flip two cards; mark them if matched. Returns (face1, face2, is_match)."""
        r1, c1 = coord1
        r2, c2 = coord2
        face1 = self.cards[r1][c1]
        face2 = self.cards[r2][c2]
        is_match = face1 == face2
        if is_match:
            self.matched[r1][c1] = True
            self.matched[r2][c2] = True
        return face1, face2, is_match

    def get_face(self, coord: Coord) -> str:
        """Get the card face at a coordinate."""
        return self.cards[coord[0]][coord[1]]

    def is_removed(self, coord: Coord) -> bool:
        """Whether this position has been removed."""
        return self.matched[coord[0]][coord[1]]

    def is_game_over(self) -> bool:
        return all(all(row) for row in self.matched)

    def remaining_pairs(self) -> int:
        count = sum(1 for r in range(self.rows) for c in range(self.cols) if not self.matched[r][c])
        return count // 2

    def available_coords(self, exclude: Optional[Coord] = None) -> List[Coord]:
        """All flippable coords (not removed and not in exclude)."""
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if not self.matched[r][c] and (r, c) != exclude
        ]
