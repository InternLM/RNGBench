"""
Card-face image asset management.

Loads images from an asset directory, supporting different themes (poker, icons,
abstract, etc.). Provides a unified interface for card faces, the card back, and
empty slots.
"""

import random
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw


class CardAssets:
    """Manage card-face images under one theme.

    Asset directory structure:
      assets/{theme}/
        |- back.png       # card back (optional; auto-generated if missing)
        |- face_001.png   # card-face images (sorted by filename)
        |- face_002.png
        \- ...

    Args:
        theme: theme name (a subdir under assets/)
        assets_dir: asset root directory
        cell_size: card size after resize (width, height)
    """

    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(
        self,
        theme: str,
        assets_dir: str = "assets",
        cell_size: tuple = (64, 64),
    ):
        self.theme = theme
        self.cell_size = cell_size
        self.theme_dir = Path(assets_dir) / theme

        if not self.theme_dir.is_dir():
            raise FileNotFoundError(f"Theme directory not found: {self.theme_dir}")

        # load all card-face image paths (sorted for seed reproducibility)
        self._all_face_paths = sorted(
            p for p in self.theme_dir.iterdir()
            if p.suffix.lower() in self.SUPPORTED_EXTS and p.name != "back.png"
        )
        if not self._all_face_paths:
            raise ValueError(f"No card face images found in {self.theme_dir}")

        # card back
        back_path = self.theme_dir / "back.png"
        if back_path.exists():
            self._back_img = Image.open(back_path).convert("RGBA").resize(cell_size)
        else:
            self._back_img = self._generate_back()

        # empty slot
        self._empty_img = self._generate_empty()

        # card faces chosen for the current game (filled after select)
        self.faces: List[Image.Image] = []
        # symbol ID -> image mapping
        self._id_to_img: dict = {}

    def select(self, symbols: List[str], seed: int) -> None:
        """Pick len(symbols) distinct faces from assets via seed and build the symbol -> image map."""
        n = len(symbols)
        if n > len(self._all_face_paths):
            raise ValueError(
                f"Need {n} card faces but theme '{self.theme}' only has {len(self._all_face_paths)}. "
                f"Add more images to {self.theme_dir}/"
            )
        rng = random.Random(seed)
        chosen = rng.sample(self._all_face_paths, n)
        self.faces = [Image.open(p).convert("RGBA").resize(self.cell_size) for p in chosen]
        self._id_to_img = {sym: img for sym, img in zip(symbols, self.faces)}

    def get_face(self, symbol: str) -> Image.Image:
        """Get the card-face image for a symbol ID."""
        return self._id_to_img[symbol]

    def get_back(self) -> Image.Image:
        return self._back_img

    def get_empty(self) -> Image.Image:
        return self._empty_img

    @property
    def num_available(self) -> int:
        return len(self._all_face_paths)

    def _generate_back(self) -> Image.Image:
        """Generate the default card back (blue checkerboard pattern)."""
        w, h = self.cell_size
        img = Image.new("RGBA", (w, h), (30, 80, 160))
        draw = ImageDraw.Draw(img)
        # checkerboard texture
        grid = max(w // 8, 4)
        for i in range(0, w, grid * 2):
            for j in range(0, h, grid * 2):
                draw.rectangle([i, j, i + grid, j + grid], fill=(40, 100, 180))
                draw.rectangle([i + grid, j + grid, i + grid * 2, j + grid * 2], fill=(40, 100, 180))
        draw.rectangle([1, 1, w - 2, h - 2], outline=(200, 200, 255), width=2)
        return img

    def _generate_empty(self) -> Image.Image:
        """Generate the empty-slot image (light-gray dashed border)."""
        w, h = self.cell_size
        img = Image.new("RGBA", (w, h), (240, 240, 240, 128))
        draw = ImageDraw.Draw(img)
        # dashed border
        dash = max(w // 10, 3)
        for x in range(0, w, dash * 2):
            draw.line([(x, 0), (min(x + dash, w), 0)], fill=(180, 180, 180), width=1)
            draw.line([(x, h - 1), (min(x + dash, w), h - 1)], fill=(180, 180, 180), width=1)
        for y in range(0, h, dash * 2):
            draw.line([(0, y), (0, min(y + dash, h))], fill=(180, 180, 180), width=1)
            draw.line([(w - 1, y), (w - 1, min(y + dash, h))], fill=(180, 180, 180), width=1)
        return img
