"""
Mode 1: single-player + show action (accumulating multi-turn dialogue).

Supports text / image board rendering. Every model reply (including retries) records a full turn.
"""

import argparse
import json
import logging
import string
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
_REPO = _ROOT.parent
for p in (str(_REPO), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from env.matching_env import InvalidActionError, MatchingEnv, Observation  # noqa: E402
from env.board import coord_to_str                                           # noqa: E402
from common.image_store import ImageStore                                    # noqa: E402
from common.board_renderer import BoardRenderer                              # noqa: E402
from common.materialize import materialize                                   # noqa: E402
from common.turn_logger import TurnLogger                                    # noqa: E402
from common.parsing import parse_coord                                       # noqa: E402
from common.output_layout import resolve_run_dir, render_leaf                # noqa: E402
from common.run_dir import prepare_run_dir, ON_EXISTS_CHOICES                # noqa: E402
from common.prompts import (                                                 # noqa: E402
    build_single_system,
    FLIP_FIRST_INSTRUCTION,
    FLIP_SECOND_INSTRUCTION,
    action_format_hint,
    retry_parse_fail,
    retry_invalid_coord,
)
from common.optimal import compute_optimal_resp_times                        # noqa: E402
from common.oracle_memory import oracle_observation_part, oracle_note_part   # noqa: E402
from model_presets import make_client, parse_grid_size                       # noqa: E402

logger = logging.getLogger(__name__)

MODE_NAME = "single_normal"


def _text(t: str) -> Dict[str, Any]:
    return {"type": "text", "text": t}


# ── Message construction ──────────────────────────────────────────────────

def _flip_first_user_msg(obs: Observation, br: BoardRenderer, round_idx: int,
                         cot_enabled: bool, oracle_seen=None) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = []
    if obs.last_result is not None:
        r = obs.last_result
        verdict = "Match!" if r["matched"] else "No match — cards flipped back."
        parts.append(_text(
            f"Previous round: {verdict} "
            f"Score: {obs.score}. Remaining pairs: {obs.remaining_pairs}."
        ))
    else:
        parts.append(_text(f"Remaining pairs: {obs.remaining_pairs}. Score: {obs.score}."))
    parts.append(_text(f"Round {round_idx} start — current board:"))
    if oracle_seen is not None:
        parts.append(oracle_note_part())
        parts.append(oracle_observation_part(br, round_idx, oracle_seen, "start"))
    else:
        parts.append(br.board_part(name=f"round_{round_idx:03d}_start"))
    parts.append(_text(FLIP_FIRST_INSTRUCTION + "\n" + action_format_hint(cot_enabled)))
    return {"role": "user", "content": parts}


def _flip_second_user_msg(obs: Observation, br: BoardRenderer, round_idx: int,
                          cot_enabled: bool, oracle_seen=None) -> Dict[str, Any]:
    parts = [_text("Board after your first flip:")]
    if oracle_seen is not None:
        parts.append(oracle_note_part())
        parts.append(oracle_observation_part(br, round_idx, oracle_seen, "flip1"))
    else:
        parts.append(br.board_part(name=f"round_{round_idx:03d}_flip1"))
    parts.append(_text(FLIP_SECOND_INSTRUCTION + "\n" + action_format_hint(cot_enabled)))
    return {"role": "user", "content": parts}


def _retry_msg(reason: str) -> Dict[str, Any]:
    return {"role": "user", "content": [_text(reason)]}


# ── Resume from partial game.json ──────────────────────────────────────

def _str_to_coord(s: str) -> tuple:
    """Inverse of env.board.coord_to_str: 'aA' → (0, 0)."""
    return (string.ascii_lowercase.index(s[0]),
            string.ascii_uppercase.index(s[1]))


def _try_resume_from_partial(
    game_json_path: Path,
    env: MatchingEnv,
    br: BoardRenderer,
    image_store: Optional[ImageStore],
) -> Optional[Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]]:
    """If game.json exists with un-finalized turns, replay env to the last
    fully-completed round and return (turns_to_keep, round_idx, messages).

    Returns None if no resumable state (file missing / already finalized /
    corrupt / no completed round).

    Discards any orphan turns from a partially-played round (since env is
    deterministic, those will be re-rolled cleanly). The caller must then
    construct a TurnLogger with `resume_turns=turns_to_keep` so log_call
    appends after them with monotonic turn_ids.
    """
    if not game_json_path.exists():
        return None
    try:
        data = json.loads(game_json_path.read_text())
    except Exception:
        return None
    if data.get("result") is not None:
        return None
    turns = data.get("turns") or []
    if not turns:
        return None

    # Per-round: which flips applied? Use phase_after to disambiguate
    # (call_type "retry" doesn't tell us which phase the apply was for).
    round_state: Dict[int, Dict[str, Any]] = {}
    for t in turns:
        env_result = t.get("env_result") or {}
        if not env_result.get("applied"):
            continue
        r = t.get("round")
        st = round_state.setdefault(r, {})
        phase_after = env_result.get("phase_after")
        coord_str = (t.get("parse") or {}).get("coord")
        face = env_result.get("revealed_face")
        if phase_after == "flip_second":
            st["coord1"], st["face1"] = coord_str, face
        elif phase_after == "flip_first":
            st["coord2"], st["face2"] = coord_str, face

    completed = sorted(r for r, st in round_state.items()
                       if "coord1" in st and "coord2" in st)
    if not completed:
        return None
    last_round = completed[-1]

    turns_to_keep = [t for t in turns if (t.get("round") or 0) <= last_round]

    # Replay applied turns through env, round-by-round so we can call
    # abort_round() for forfeited rounds (flip_first applied but flip_second
    # never applied — env stayed in flip_second phase mid-game). Mirrors the
    # `env.abort_round()` call in run_one_game when flip_second exhausts retries.
    turns_by_round: Dict[int, List[Dict[str, Any]]] = {}
    for t in turns_to_keep:
        turns_by_round.setdefault(t.get("round") or 0, []).append(t)
    for r in sorted(turns_by_round):
        f1_applied = f2_applied = False
        for t in turns_by_round[r]:
            env_result = t.get("env_result") or {}
            if not env_result.get("applied"):
                continue
            coord_str = (t.get("parse") or {}).get("coord")
            if not coord_str or len(coord_str) < 2:
                return None
            env.step(_str_to_coord(coord_str))
            pa = env_result.get("phase_after")
            if pa == "flip_second":
                f1_applied = True
            elif pa == "flip_first":
                f2_applied = True
        if f1_applied and not f2_applied:
            env.abort_round()

    # Reconstruct messages via snapshot diffing (each turn's messages_sent is a
    # superset of the previous turn's, plus any user msgs eval appended in
    # between).
    messages: List[Dict[str, Any]] = []
    for t in turns_to_keep:
        snapshot = t.get("messages_sent") or []
        if len(snapshot) > len(messages):
            messages.extend(snapshot[len(messages):])
        content = (t.get("response") or {}).get("content", "") or ""
        messages.append({"role": "assistant", "content": content})

    # Synthesize trailing outcome msg for last_round (eval appends it AFTER
    # the last applied turn was logged, so no snapshot captures it).
    obs = env.get_observation()
    st = round_state[last_round]
    coord1 = _str_to_coord(st["coord1"])
    coord2 = _str_to_coord(st["coord2"])
    matched = (st["face1"] == st["face2"])
    verdict = "Match!" if matched else "No match — cards flipped back."
    content_parts: List[Dict[str, Any]] = [_text("Both cards revealed:")]
    both_ref = br.store_both_flips(
        name=f"round_{last_round:03d}_both",
        coord1=coord1, coord2=coord2,
    )
    content_parts.append(br.ref_to_part(both_ref))
    content_parts.append(_text(
        f"{verdict} Score: {obs.score}. Remaining: {obs.remaining_pairs}."
    ))
    messages.append({"role": "user", "content": content_parts})

    return turns_to_keep, last_round, messages


# ── Main ────────────────────────────────────────────────────────────────

def run_one_game(
    *,
    model: str,
    rows: int, cols: int,
    num_cards: Optional[int],
    seed: int,
    render: str,                            # "text" | "image"
    theme: Optional[str],
    assets_dir: str,
    cell_size: int,
    out_dir: Path,
    max_responses: int,
    max_retries: int,
    label: Optional[str] = None,
    cot_enabled: bool = True,
    oracle_memory: bool = False,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_store: Optional[ImageStore] = None
    if render == "image":
        image_store = ImageStore(root=out_dir / "images", base_dir=out_dir)

    client = make_client(model, label=label)
    env = MatchingEnv(
        rows=rows, cols=cols, num_cards=num_cards, seed=seed,
        theme=theme if render == "image" else None,
        assets_dir=assets_dir, cell_size=(cell_size, cell_size),
    )
    obs = env.reset()
    br = BoardRenderer(env, render=render, image_store=image_store)

    # GT
    gt_part = br.ground_truth_part(name="ground_truth")
    gt_text = env.render_ground_truth(mode="text")
    optimal_greedy = compute_optimal_resp_times(env.board.cards)

    config = {
        "mode": MODE_NAME,
        "model": model, "label": label or model,
        "rows": rows, "cols": cols,
        "num_cards": env.board.num_cards,
        "total_pairs": env.total_pairs,
        "seed": seed,
        "render": render,
        "theme": theme if render == "image" else None,
        "cell_size": cell_size,
        "max_responses": max_responses,
        "max_retries": max_retries,
        "cot_enabled": cot_enabled,
        "oracle_memory": oracle_memory,
        "ground_truth": gt_part,
        "ground_truth_text": gt_text,
        "optimal_resp_times_greedy": optimal_greedy,
    }

    # Resume from a partial game.json if present (env replay + messages rebuild).
    # Oracle-memory runs start fresh (the `seen` set can't be reconstructed cheaply).
    resume = None if oracle_memory else _try_resume_from_partial(out_dir / "game.json", env, br, image_store)
    if resume is not None:
        resume_turns, round_idx, messages = resume
        obs = env.get_observation()
        tlog = TurnLogger(out_dir / "game.json", config=config, resume_turns=resume_turns)
        logger.info(f"[resume] {out_dir.name}: replayed {len(resume_turns)} turns, "
                    f"continuing from round {round_idx + 1}")
    else:
        tlog = TurnLogger(out_dir / "game.json", config=config)
        messages = [
            {"role": "system", "content": build_single_system(cot_enabled=cot_enabled)},
        ]
        round_idx = 0

    error: Optional[str] = None
    seen: set = set()  # all coords revealed so far (oracle-memory mode only)

    try:
        while not obs.done and tlog.response_count < max_responses:
            round_idx += 1

            # Flip 1
            coord1, _ = _run_flip(
                phase_label="flip_first",
                round_idx=round_idx,
                user_msg=_flip_first_user_msg(obs, br, round_idx, cot_enabled,
                                              oracle_seen=(seen if oracle_memory else None)),
                br=br, env=env, messages=messages, client=client,
                image_store=image_store, tlog=tlog,
                max_retries=max_retries, max_responses=max_responses,
                player=None, cot_enabled=cot_enabled,
            )
            if coord1 is None:
                messages.append({"role": "user", "content": [_text(
                    "Could not complete a valid first flip after retries "
                    "(parse error or invalid action each time). Round forfeited; "
                    "no cards flipped.")]})
                obs = env.get_observation()
                continue
            seen.add(coord1)

            obs = env.get_observation()

            # Flip 2
            coord2, both_ref = _run_flip(
                phase_label="flip_second",
                round_idx=round_idx,
                user_msg=_flip_second_user_msg(obs, br, round_idx, cot_enabled,
                                               oracle_seen=(seen if oracle_memory else None)),
                br=br, env=env, messages=messages, client=client,
                image_store=image_store, tlog=tlog,
                max_retries=max_retries, max_responses=max_responses,
                player=None, cot_enabled=cot_enabled,
            )
            if coord2 is None:
                env.abort_round()
                obs = env.get_observation()
                messages.append({"role": "user", "content": [
                    _text("Could not complete a valid second flip after retries "
                          "(parse error or invalid action each time). "
                          "Your first card was flipped back face-down. Round forfeited."),
                ]})
                continue
            seen.add(coord2)

            obs = env.get_observation()
            r = obs.last_result or {}
            verdict = "Match!" if r.get("matched") else "No match — cards flipped back."
            content_parts: List[Dict[str, Any]] = [
                _text("Both cards revealed:"),
            ]
            if both_ref is not None:
                content_parts.append(br.ref_to_part(both_ref))
            content_parts.append(_text(
                f"{verdict} Score: {obs.score}. Remaining: {obs.remaining_pairs}."
            ))
            messages.append({"role": "user", "content": content_parts})

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.error(f"Game aborted: {error}\n{traceback.format_exc()}")

    result = {
        "score": obs.score if env.board else 0,
        "total_pairs": env.total_pairs if env.board else None,
        "rounds_played": round_idx,
        "response_count": tlog.response_count,
        "done": obs.done if env.board else False,
        "error": error,
    }
    tlog.finalize(result)
    return result


def _run_flip(
    *,
    phase_label: str,
    round_idx: int,
    user_msg: Dict[str, Any],
    br: BoardRenderer,
    env: MatchingEnv,
    messages: List[Dict[str, Any]],
    client,
    image_store: Optional[ImageStore],
    tlog: TurnLogger,
    max_retries: int,
    max_responses: int,
    player: Optional[str],
    cot_enabled: bool,
) -> tuple:
    """Return (coord, both_ref) for flip_second, (coord, None) for flip_first, (None, None) on failure."""
    messages.append(user_msg)
    last_turn_id: Optional[int] = None
    last_call_type = phase_label

    for attempt in range(max_retries + 1):
        if tlog.response_count >= max_responses:
            return None, None

        materialized = materialize(messages, image_store) if image_store else messages
        resp = client.chat(materialized)
        content = resp.get("content", "")
        reasoning = resp.get("reasoning")

        coord = parse_coord(content, env.rows, env.cols)
        parse_ok = coord is not None
        parse_info = {
            "ok": parse_ok,
            "coord": coord_to_str(coord) if parse_ok else None,
            "error": None if parse_ok else "could not parse coordinate",
        }
        env_result: Dict[str, Any] = {"applied": False}
        applied = False
        both_ref: Optional[str] = None

        if parse_ok:
            try:
                first_coord = env.first_flip_coord if phase_label == "flip_second" else None
                env.step(coord)
                applied = True
                if phase_label == "flip_first":
                    revealed = env.first_flip_face
                else:
                    both_ref = br.store_both_flips(
                        name=f"round_{round_idx:03d}_both",
                        coord1=first_coord, coord2=coord,
                    )
                    revealed = (env.last_result or {}).get("face2")
                env_result = {
                    "applied": True,
                    "revealed_face": revealed,
                    "phase_after": env.phase,
                }
            except InvalidActionError as ie:
                parse_info["error"] = ie.reason
                env_result = {"applied": False, "error": ie.reason, "code": ie.code}

        last_turn_id = tlog.log_call(
            call_type=last_call_type, round_idx=round_idx, player=player,
            messages_sent=messages,
            response={"content": content, "reasoning": reasoning},
            parse=parse_info, env_result=env_result,
            retry_of=last_turn_id,
        )
        last_call_type = "retry"

        messages.append({"role": "assistant", "content": content})

        if applied:
            return coord, both_ref

        reason = (retry_parse_fail(cot_enabled) if not parse_ok
                  else retry_invalid_coord(parse_info["error"] or "invalid", cot_enabled=cot_enabled))
        messages.append(_retry_msg(reason))

    return None, None


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(_REPO / ".env")
    parser = argparse.ArgumentParser(description="Single player, show-action mode")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--grid", type=parse_grid_size, default=(6, 6))
    parser.add_argument("--num-cards", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", choices=["text", "image"], default="image")
    parser.add_argument("--theme", default=None, help="(image only) card asset theme")
    parser.add_argument("--assets-dir", default=str(_ROOT / "assets"))
    parser.add_argument("--cell-size", type=int, default=64)
    parser.add_argument("--max-responses", type=int, default=0,
                        help="Absolute cap on model replies (incl. retries). "
                             "0 = auto = total_pairs × max_resp_per_pair.")
    parser.add_argument("--max-resp-per-pair", type=int, default=7,
                        help="Multiplier for the auto-computed max_responses budget.")
    parser.add_argument("--max-retries", type=int, default=2)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cot", dest="cot_enabled", action="store_true",
                       help="Require Thought + Action output (default).")
    group.add_argument("--no-cot", dest="cot_enabled", action="store_false",
                       help="Require only Action output; no reasoning line.")
    parser.set_defaults(cot_enabled=True)
    parser.add_argument("--oracle-memory", action="store_true", default=False,
                        help="Memory-Gap oracle: append a 'memory aid' board (all cards "
                             "revealed so far, face-up) beside each observation. Off by default.")
    parser.add_argument("--out", required=True,
                        help="Output root; actual run dir is computed by common.output_layout.")
    parser.add_argument("--on-exists", choices=ON_EXISTS_CHOICES, default="overwrite",
                        help="If run dir already exists: overwrite (default, rm -rf first); "
                             "skip (exit if finalized game.json exists, else rm -rf + redo); "
                             "resume (skip if finalized; else replay env from partial "
                             "game.json and continue from last completed round).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    rows, cols = args.grid
    total_pairs = (rows * cols) // 2
    max_resp = args.max_responses or total_pairs * args.max_resp_per_pair

    run_dir = resolve_run_dir(
        out_root=Path(args.out), mode=MODE_NAME,
        leaf=render_leaf(args.render, args.theme),
        seed=args.seed,
        label=args.label or args.model,
        rows=rows, cols=cols,
        cot_enabled=args.cot_enabled,
    )

    prepared = prepare_run_dir(run_dir, args.on_exists)
    if prepared is None:
        logger.info(f"SKIPPED (exists): {run_dir}")
        return

    result = run_one_game(
        model=args.model, label=args.label,
        rows=rows, cols=cols,
        num_cards=args.num_cards, seed=args.seed,
        render=args.render,
        theme=args.theme, assets_dir=args.assets_dir,
        cell_size=args.cell_size,
        out_dir=run_dir,
        max_responses=max_resp, max_retries=args.max_retries,
        cot_enabled=args.cot_enabled,
        oracle_memory=args.oracle_memory,
    )
    logger.info(f"RUN_DIR: {run_dir}")
    logger.info(f"RESULT: {result}")


if __name__ == "__main__":
    main()
