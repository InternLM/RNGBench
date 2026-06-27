"""
Oracle-memory observation for Matching Pairs (the Memory-Gap upper bound).

Idea (opt-in, does NOT change the default eval): alongside the live board, show a
second "memory aid" board where every card the player has revealed so far and not
yet matched is rendered face-up. This hands the model a perfect record of its own
past observations, so any remaining failure is attributable to perception /
decision rather than forgetting. No prompt change is needed — only the observation
image gains a side-by-side memory panel.

Used only when `single_normal --oracle-memory` is passed.
"""

from typing import Any, Dict, List, Set, Tuple

Coord = Tuple[int, int]

_NOTE = (
    "The image has TWO boards. LEFT = the live board you act on. RIGHT = a MEMORY "
    "AID showing every card you have revealed so far that is not yet matched, all "
    "face-up. The memory aid contains no information beyond what you have already "
    "seen; use it instead of relying on recall."
)


def _active_seen(env, seen: Set[Coord]) -> List[Coord]:
    """Revealed-so-far cards that are still on the board (matched ones drop out)."""
    return [c for c in seen if not env.board.is_removed(c)]


def _compose_side_by_side(left, right, left_label: str, right_label: str):
    from PIL import Image, ImageDraw, ImageFont
    pad, gap, bar = 10, 30, 26
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
    body_h = max(left.height, right.height)
    W = pad + left.width + gap + right.width + pad
    H = bar + pad + body_h + pad
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 6), left_label, fill=(40, 40, 40), font=font)
    d.text((pad + left.width + gap, 6), right_label, fill=(40, 60, 120), font=font)
    divx = pad + left.width + gap // 2
    d.line([(divx, bar), (divx, H - pad)], fill=(205, 205, 205), width=2)
    canvas.paste(left, (pad, bar + pad))
    canvas.paste(right, (pad + left.width + gap, bar + pad))
    return canvas


def oracle_note_part() -> Dict[str, Any]:
    return {"type": "text", "text": _NOTE}


def oracle_observation_part(br, round_idx: int, seen: Set[Coord], phase_name: str) -> Dict[str, Any]:
    """One content part: [live board | memory board].

    Image mode → a single composed image (saved via br.image_store).
    Text mode  → the live board plus a labelled memory board, in one text part.
    """
    env = br.env
    active = _active_seen(env, seen)
    if br.render == "text":
        live = env.render_board(mode="text")
        mem = env.render_board(mode="text", current_flips=active)
        txt = (f"Live board:\n```\n{live}\n```\n\n"
               f"Memory aid — every card you've revealed so far, face-up:\n```\n{mem}\n```")
        return {"type": "text", "text": txt}
    live = env.render_board(mode="image")
    mem = env.render_board(mode="image", current_flips=active)
    combo = _compose_side_by_side(live, mem, "Live board", "Memory aid (revealed cards)")
    ref = br.image_store.save_pil(combo, name=f"round_{round_idx:03d}_{phase_name}_oracle")
    return br.ref_to_part(ref)
