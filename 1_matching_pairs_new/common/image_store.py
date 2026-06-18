"""
ImageStore — semantic-name-based saving (no content dedup; identical frames may be
rewritten to the same filename, which is idempotent).

The filename is given by the caller, e.g. "round_03_flip1"; saved as
"<root>/<name>.jpg", returning a path relative to base_dir (default "images/<name>.jpg").

Before an LLM call, use load_as_data_url(path) to read it back as base64.
"""

import base64
from pathlib import Path
from typing import Optional


class ImageStore:
    def __init__(
        self,
        root: Path,
        base_dir: Optional[Path] = None,
        quality: int = 85,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_dir = Path(base_dir) if base_dir else self.root.parent
        self.quality = quality

    def save_pil(self, img, name: str) -> str:
        """Save a PIL Image as <root>/<name>.jpg by name; returns a path relative to base_dir.

        The same name overwrites (re-rendering an unchanged board on retry produces no extra files).
        """
        dest = self.root / f"{name}.jpg"
        img.convert("RGB").save(dest, format="JPEG", quality=self.quality, optimize=True)
        return str(dest.relative_to(self.base_dir))

    def load_as_data_url(self, rel_path: str) -> str:
        full = self.base_dir / rel_path
        data = full.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
