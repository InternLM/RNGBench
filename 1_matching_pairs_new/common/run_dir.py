"""
Handle the case where the run directory already exists.

Three strategies:
  - "overwrite": rm -rf the whole run_dir then recreate it (default)
  - "skip":      if run_dir/game.json exists and is complete (result field non-null),
                 skip and return None; if incomplete -> treated as overwrite (rm -rf, rerun)
  - "resume":    complete -> return None (skip);
                 incomplete (has partial turns) -> leave run_dir untouched and let
                 run_one_game read the partial game.json + replay env, continuing from
                 the last complete round
                 (currently only single_normal implements replay; other modes on resume
                  keep the old partial game.json but run_one_game overwrites from the start,
                  equivalent to overwrite-without-rm);
                 corrupted json -> rm -rf, rerun

If the caller gets None, do not call run_one_game; otherwise continue normally.
"""

import json
import shutil
from pathlib import Path
from typing import Optional


def _is_finalized(game_json: Path) -> Optional[bool]:
    """True: game.json exists and the result field is non-null (complete).
    False: game.json exists but is not finalized (partial).
    None: game.json does not exist or is corrupted."""
    if not game_json.exists():
        return None
    try:
        data = json.loads(game_json.read_text())
    except Exception:
        return None
    return data.get("result") is not None


def prepare_run_dir(run_dir: Path, on_exists: str) -> Optional[Path]:
    run_dir = Path(run_dir)
    game_json = run_dir / "game.json"

    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    state = _is_finalized(game_json)

    if on_exists == "skip":
        if state is True:
            print(f"[skip] {run_dir} already has finalized game.json; skipping.")
            return None
        # no usable result / corrupted -> treat as overwrite
        shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    if on_exists == "overwrite":
        shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    if on_exists == "resume":
        if state is True:
            print(f"[resume] {run_dir} already finalized; skipping.")
            return None
        if state is None and game_json.exists():
            # corrupted json -> treat as invalid, rerun
            shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        # state is False (partial) or game.json missing -> keep run_dir.
        # run_one_game will detect the partial game.json and replay/continue.
        return run_dir

    raise ValueError(f"Unknown on_exists: {on_exists!r}")


ON_EXISTS_CHOICES = ("overwrite", "skip", "resume")
