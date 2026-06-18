"""
BoardRenderer — unifies text / image rendering for the mode runners.

Core idea: a render result is first "materialized" into an opaque ref:
  - image mode -> relative image path
  - text mode -> the rendered ASCII string (wrapped in a ```...``` code block as a text part)

Then the ref is wrapped into an OpenAI content part and put into messages.

Storing the ref in history is enough; any later turn reusing history calls
ref_to_part(ref) to reconstruct the part.
"""

from typing import Any, Dict, List, Optional


class BoardRenderer:
    def __init__(self, env, render: str, image_store=None):
        self.env = env
        self.render = render
        self.image_store = image_store
        if render == "image" and image_store is None:
            raise ValueError("image_store required for image mode")

    # ── Store (returns an opaque ref) ──────────────────────────────────

    def store_board(self, name: str, current_flips: Optional[List] = None) -> str:
        if self.render == "text":
            return self.env.render_board(mode="text", current_flips=current_flips)
        img = self.env.render_board(mode="image", current_flips=current_flips)
        return self.image_store.save_pil(img, name=name)

    def store_both_flips(self, name: str, coord1, coord2) -> str:
        if self.render == "text":
            return self.env.render_both_flips(coord1, coord2, mode="text")
        img = self.env.render_both_flips(coord1, coord2, mode="image")
        return self.image_store.save_pil(img, name=name)

    def store_ground_truth(self, name: str = "ground_truth") -> str:
        if self.render == "text":
            return self.env.render_ground_truth(mode="text")
        img = self.env.render_ground_truth(mode="image")
        return self.image_store.save_pil(img, name=name)

    # ── ref → content part ─────────────────────────────────────────

    def ref_to_part(self, ref: str) -> Dict[str, Any]:
        if self.render == "text":
            return {"type": "text", "text": f"```\n{ref}\n```"}
        # the storage layer keeps the standard OpenAI image_url shape with url = relative path;
        # materialize() swaps non-data-URL urls for real base64 data URLs before calling the LLM.
        return {"type": "image_url", "image_url": {"url": ref}}

    # ── One-shot convenience (store then immediately ref_to_part) ──────

    def board_part(self, name: str, current_flips: Optional[List] = None) -> Dict[str, Any]:
        return self.ref_to_part(self.store_board(name, current_flips))

    def both_flips_part(self, name: str, coord1, coord2) -> Dict[str, Any]:
        return self.ref_to_part(self.store_both_flips(name, coord1, coord2))

    def ground_truth_part(self, name: str = "ground_truth") -> Dict[str, Any]:
        return self.ref_to_part(self.store_ground_truth(name))
