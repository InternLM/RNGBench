"""
Output directory conventions:

single: <out_root>/<mode>/<label>/<render_leaf>/<RxC>/seed_<N>/
dual:   <out_root>/<mode>/<label_a>/vs_<label_b>/<render_leaf>/<RxC>/seed_<N>/

render_leaf:
  - single: "text" or "image-<theme>" / "image-ascii"
  - dual, same rendering: same as single
  - dual, mixed: e.g. "image-poker_vs_text"
"""

from pathlib import Path
from typing import Optional


def render_leaf(render: str, theme: Optional[str]) -> str:
    if render == "text":
        return "text"
    if render == "image":
        return f"image-{theme}" if theme else "image-ascii"
    raise ValueError(f"Unknown render: {render}")


def dual_render_leaf(render_a: str, theme_a: Optional[str],
                     render_b: str, theme_b: Optional[str]) -> str:
    la = render_leaf(render_a, theme_a)
    lb = render_leaf(render_b, theme_b)
    return la if la == lb else f"{la}_vs_{lb}"


def resolve_run_dir(
    out_root: Path,
    mode: str,
    leaf: str,
    seed: int,
    label: Optional[str] = None,
    label_b: Optional[str] = None,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    cot_enabled: bool = True,
) -> Path:
    """Build this run's directory. Does not create it.

    single: out_root / mode / label / leaf[_nocot] / RxC / seed_N
    dual:   out_root / mode / label / vs_label_b / leaf[_nocot] / RxC / seed_N

    The `_nocot` suffix is appended only when cot_enabled=False, to avoid
    overwriting the default CoT runs.
    """
    p = Path(out_root) / mode
    if label:
        p = p / label
    if label_b is not None:
        p = p / f"vs_{label_b}"
    leaf_dir = leaf if cot_enabled else f"{leaf}_nocot"
    p = p / leaf_dir
    if rows is not None and cols is not None:
        p = p / f"{rows}x{cols}"
    p = p / f"seed_{seed}"
    return p
