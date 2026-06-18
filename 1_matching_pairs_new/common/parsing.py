"""
LLM output parsing.

Responsibility: extract a coordinate from the model output. Supports stripping
reasoning tags and extracting after the Action keyword.
"""

import re
from typing import Optional

from env.board import Coord


def parse_coord(text: str, rows: int, cols: int) -> Optional[Coord]:
    """Parse a coordinate from the LLM output.

    Requirement: an 'Action:' keyword must appear (case-insensitive, markdown **
    wrapping allowed). Takes the coordinate after the **last** Action:. 'aA' or
    'a A' (lowercase row + uppercase column).
    No whole-text fallback — avoids grabbing a coordinate mentioned offhand in the thought.
    Automatically strips <think>...</think> reasoning tags.

    Returns:
        (row, col) tuple, or None (parse failure)
    """
    # strip thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # strict: an Action: keyword must exist (markdown ** etc. allowed); take the last Action segment.
    # no whole-text [a-z][A-Z] fallback — it would mis-grab coords mentioned in the thought.
    markers = list(re.finditer(r"(?:^|[^A-Za-z])\**action\**\s*:", text, flags=re.IGNORECASE))
    if not markers:
        return None
    tail = text[markers[-1].end():]
    m = re.match(r"\s*\**\s*([a-z])\s*([A-Z])(?![a-zA-Z])", tail)
    if not m:
        return None
    r = ord(m.group(1)) - ord("a")
    c = ord(m.group(2)) - ord("A")

    if 0 <= r < rows and 0 <= c < cols:
        return (r, c)
    return None
