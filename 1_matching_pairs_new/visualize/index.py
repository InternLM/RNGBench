"""
CLI: scan a results directory, generate a replay for each game.json, and build an index.html.

Usage:
    python -m visualize.index <results_root> [-o <index.html>]

Example:
    python -m visualize.index results_test/seed-2.0-lite-nothink
    # -> a seed_N/replay/replay.html for each + <results_root>/index.html
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from visualize.builder import load_game, build_states, extract_meta  # noqa: E402
from visualize.template import render_html, render_index_html        # noqa: E402
from visualize.replay import generate_replay                          # noqa: E402


def scan_games(root: Path) -> List[Path]:
    return sorted(root.rglob("game.json"))


def build_index(root: str, out: str = None) -> str:
    root_path = Path(root).resolve()
    games = scan_games(root_path)
    if not games:
        raise FileNotFoundError(f"No game.json under {root_path}")

    entries: List[Dict[str, Any]] = []
    for g in games:
        try:
            replay_path = Path(generate_replay(str(g)))
        except Exception as e:
            print(f"[warn] failed to render {g}: {e}", file=sys.stderr)
            continue
        game = load_game(str(g))
        meta = extract_meta(game)
        entries.append({
            "title": meta.get("title", ""),
            "subtitle_scores": meta.get("subtitle_scores", ""),
            "mode": meta.get("mode", ""),
            "render_desc": meta.get("render_desc", ""),
            "grid": f"{meta.get('rows')}x{meta.get('cols')}",
            "seed": meta.get("seed", ""),
            "done": meta.get("done", False),
            "href": str(replay_path.relative_to(root_path)) if replay_path.is_relative_to(root_path) else str(replay_path),
        })

    if out is None:
        out = str(root_path / "index.html")
    Path(out).write_text(
        render_index_html(entries, title=f"Replays · {root_path.name}"),
        encoding="utf-8",
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate HTML replays + index from a results directory")
    parser.add_argument("root", help="Results root directory (recursively scans for game.json)")
    parser.add_argument("-o", "--output", default=None, help="Output index.html path")
    args = parser.parse_args()
    out = build_index(args.root, args.output)
    print(f"Index written: {out}")


if __name__ == "__main__":
    main()
