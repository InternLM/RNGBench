"""
CLI: render a single game.json into an HTML replay.

Usage:
    python -m visualize.replay <path/to/game.json>
    python -m visualize.replay <path/to/game.json> -o custom.html

Default output: <game.json's dir>/replay/replay.html; image relative paths are handled automatically.
"""

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from visualize.builder import load_game, build_states, extract_meta    # noqa: E402
from visualize.template import render_html                              # noqa: E402


def generate_replay(game_json: str, output: str = None) -> str:
    game = load_game(game_json)
    meta = extract_meta(game)
    states = build_states(game)

    game_path = Path(game_json).resolve()
    if output is None:
        replay_dir = game_path.parent / "replay"
        replay_dir.mkdir(exist_ok=True)
        output = str(replay_dir / "replay.html")
    else:
        output = str(Path(output).resolve())

    # image paths are relative to game.json's dir; HTML is under <game_dir>/replay/ -> prefix "../"
    out_dir = Path(output).parent
    try:
        rel = out_dir.relative_to(game_path.parent)
        # each level deeper needs one more "../"
        depth = len(rel.parts)
        prefix = ("../" * depth) if depth > 0 else ""
    except ValueError:
        # output is outside game_dir -> use an absolute-path prefix
        prefix = str(game_path.parent) + "/"

    html = render_html(meta, states, img_prefix=prefix)
    Path(output).write_text(html, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate HTML replay from game.json")
    parser.add_argument("game_json", help="Path to game.json")
    parser.add_argument("-o", "--output", default=None, help="Output HTML path")
    args = parser.parse_args()
    out = generate_replay(args.game_json, args.output)
    print(f"Replay written: {out}")


if __name__ == "__main__":
    main()
