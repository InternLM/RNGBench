"""
Board renderers.

Responsibility: turn board state into text or image representations.
- TextRenderer: ASCII text
- ImageRenderer: render text to an image (text-symbol card faces)
- ImageCardRenderer: composite image assets (for different-pattern tests)
"""

import base64
import io
import string
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from env.board import Board, Coord


class TextRenderer:
    """Render the board as ASCII text.

    Format:
      [A][B][C][D]
    a [*][*][*][*]    [*] = face-down, [X] = revealed, [ ] = matched/removed
    b [*][*][*][*]
    """

    def render(self, board: Board, current_flips: Optional[List[Coord]] = None) -> str:
        current_flips = current_flips or []
        w = max((len(s) for s in board.symbols), default=1)

        def cell(text: str) -> str:
            return f"[{text:>{w}}]"

        lines = []
        header = " " * 2 + "".join(cell(string.ascii_uppercase[c]) for c in range(board.cols))
        lines.append(header)
        for r in range(board.rows):
            row_str = f"{string.ascii_lowercase[r]} "
            for c in range(board.cols):
                if (r, c) in current_flips:
                    row_str += cell(board.cards[r][c])
                elif board.matched[r][c]:
                    row_str += cell(" ")
                else:
                    row_str += cell("*")
            lines.append(row_str)
        return "\n".join(lines)

    def render_full(self, board: Board) -> str:
        """Show all card faces (debug)."""
        w = max((len(s) for s in board.symbols), default=1)

        def cell(text: str) -> str:
            return f"[{text:>{w}}]"

        lines = []
        header = " " * 2 + "".join(cell(string.ascii_uppercase[c]) for c in range(board.cols))
        lines.append(header)
        for r in range(board.rows):
            row_str = f"{string.ascii_lowercase[r]} "
            for c in range(board.cols):
                if board.matched[r][c]:
                    row_str += cell(" ")
                else:
                    row_str += cell(board.cards[r][c])
            lines.append(row_str)
        return "\n".join(lines)


class ImageRenderer:
    """Render the board as a PIL Image (for VLM evaluation).

    Generates an image from TextRenderer's text output using a monospace font.
    Future extensions:
    - different card patterns/colors
    - custom card assets
    """

    def __init__(self, font_size: int = 36, padding: int = 40):
        self.font_size = font_size
        self.padding = padding
        self._text_renderer = TextRenderer()

    def render(self, board: Board, current_flips: Optional[List[Coord]] = None):
        """Render to a PIL Image."""
        from PIL import Image, ImageDraw, ImageFont

        text = self._text_renderer.render(board, current_flips)
        lines = text.split("\n")

        font = self._load_font()

        # compute size
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)
        line_height = self.font_size + 8
        max_width = max(draw.textlength(line, font=font) for line in lines)
        img_w = int(max_width + self.padding * 2)
        img_h = int(len(lines) * line_height + self.padding * 2)

        # draw
        img = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(img)
        y = self.padding
        for line in lines:
            draw.text((self.padding, y), line, fill="black", font=font)
            y += line_height
        return img

    def render_to_base64(self, board: Board, current_flips: Optional[List[Coord]] = None) -> str:
        """Render to a JPEG base64 data URL."""
        img = self.render(board, current_flips)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _load_font(self):
        for path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]:
            try:
                return ImageFont.truetype(path, self.font_size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()


class ImageCardRenderer:
    """Render the board as a grid of composited image assets.

    Uses images from CardAssets as card faces, tiled into a board image.
    Used to test a VLM's memory for different visual patterns.

    Args:
        card_assets: a CardAssets instance (with select already called)
        gap: spacing between cards
        label_size: font size for the row/column labels
        bg_color: background color
    """

    def __init__(self, card_assets, gap: int = 4, label_size: int = 16, bg_color=(50, 120, 50)):
        self.assets = card_assets
        self.gap = gap
        self.label_size = label_size
        self.bg_color = bg_color
        self._font = self._load_label_font()

    def render(self, board: Board, current_flips: Optional[List[Coord]] = None) -> Image.Image:
        current_flips = current_flips or []
        cw, ch = self.assets.cell_size
        gap = self.gap

        # label margin width/height
        label_margin = self.label_size + 8

        img_w = label_margin + board.cols * (cw + gap) + gap
        img_h = label_margin + board.rows * (ch + gap) + gap

        img = Image.new("RGB", (img_w, img_h), self.bg_color)
        draw = ImageDraw.Draw(img)

        # column labels
        for c in range(board.cols):
            x = label_margin + gap + c * (cw + gap) + cw // 2
            draw.text((x, 2), string.ascii_uppercase[c], fill="white", font=self._font, anchor="mt")

        # row labels + cards
        for r in range(board.rows):
            y = label_margin + gap + r * (ch + gap)
            draw.text((label_margin // 2, y + ch // 2), string.ascii_lowercase[r], fill="white", font=self._font, anchor="mm")

            for c in range(board.cols):
                x = label_margin + gap + c * (cw + gap)
                coord = (r, c)

                if coord in current_flips:
                    card_img = self.assets.get_face(board.cards[r][c])
                elif board.matched[r][c]:
                    card_img = self.assets.get_empty()
                else:
                    card_img = self.assets.get_back()

                # handle RGBA alpha: composite onto a white background first
                if card_img.mode == "RGBA":
                    bg = Image.new("RGB", card_img.size, (255, 255, 255))
                    bg.paste(card_img, mask=card_img.split()[3])
                    img.paste(bg, (x, y))
                else:
                    img.paste(card_img.convert("RGB"), (x, y))

        return img

    def render_to_base64(self, board: Board, current_flips: Optional[List[Coord]] = None) -> str:
        img = self.render(board, current_flips)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _load_label_font(self):
        for path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]:
            try:
                return ImageFont.truetype(path, self.label_size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()
