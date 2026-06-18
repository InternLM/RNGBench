"""
Mode 2: single-player + no action (stateless rebuild, history = sequence of board snapshots).

Supports text / image rendering. Assistant text does not carry into the next turn; it is
only recorded by turn_logger.
History stores opaque refs (image path or text string); each turn calls ref_to_part when building messages.
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    FLIP_SECOND_INSTRUCTION_NOACTION,
    action_format_hint,
    describe_invalid,
)
from common.llm_call import call_llm_with_retry                              # noqa: E402
from common.optimal import compute_optimal_resp_times                        # noqa: E402
from model_presets import make_client, parse_grid_size                       # noqa: E402

logger = logging.getLogger(__name__)

MODE_NAME = "single_noaction"


def _text(t: str) -> Dict[str, Any]:
    return {"type": "text", "text": t}


def _describe_fail(f: Dict[str, Any]) -> str:
    """Coord-free description of a single failed attempt."""
    if f["kind"] == "parse":
        return "one reply could not be parsed"
    desc = describe_invalid(f.get("code", "invalid"), "was invalid")
    # describe_invalid returns "it selected ..." for known codes
    return "one reply " + desc[3:] if desc.startswith("it ") else "one reply " + desc


def _fail_note(label: str, fails: List[Dict[str, Any]], ultimately_ok: bool) -> Optional[str]:
    if not fails:
        return None
    total = len(fails) + (1 if ultimately_ok else 0)
    descs = "; ".join(_describe_fail(f) for f in fails)
    status = "then retry succeeded" if ultimately_ok else "all attempts failed"
    return f"  Note: {label} took {total} attempts ({descs}; {status})."


def _inflight_retry_feedback(last_fail: Dict[str, Any]) -> str:
    """Coord-free retry feedback for the *current* flip's previous failed attempts."""
    if last_fail["kind"] == "parse":
        detail = "it could not be parsed as a valid coordinate"
    else:
        detail = describe_invalid(last_fail.get("code", "invalid"), "it was invalid")
    return (f"Note: your previous reply for this flip was not executed because {detail}. "
            f"The board is unchanged. Try a different valid choice.")


def _build_stateless_user(
    history: List[Dict[str, Any]],
    current_board_part: Dict[str, Any],
    br: BoardRenderer,
    phase: str,
    remaining_pairs: int,
    score: int,
    first_flip_coord_str: Optional[str] = None,
    last_fail: Optional[Dict[str, Any]] = None,
    cot_enabled: bool = True,
) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = []

    if history:
        parts.append(_text("Game history (your previous rounds, in order):"))
        for entry in history:
            kind = entry["kind"]
            if kind == "normal":
                verdict = ("matched (pair removed)" if entry.get("matched")
                           else "no match (cards flipped back)")
                header = f"Round {entry['round']}: {verdict}"
            elif kind == "case_a":
                header = (f"Round {entry['round']}: forfeited (first card was revealed but the "
                          f"second flip could not be completed; the first card was flipped back)")
            else:  # case_b
                header = (f"Round {entry['round']}: forfeited (first flip could not be completed; "
                          f"no card was flipped)")
            parts.append(_text(header))
            n1 = _fail_note("first flip", entry.get("flip1_fails", []), kind in ("normal", "case_a"))
            n2 = _fail_note("second flip", entry.get("flip2_fails", []), kind == "normal")
            if n1: parts.append(_text(n1))
            if n2: parts.append(_text(n2))
            for ref in entry["refs"]:
                parts.append(br.ref_to_part(ref))
    else:
        parts.append(_text("Game history: (none — this is the first flip)"))

    parts.append(_text(f"Current board (score {score}, remaining {remaining_pairs} pairs):"))
    parts.append(current_board_part)

    if last_fail is not None:
        parts.append(_text(_inflight_retry_feedback(last_fail)))

    if phase == "flip_first":
        parts.append(_text(FLIP_FIRST_INSTRUCTION + "\n" + action_format_hint(cot_enabled)))
    else:
        parts.append(_text(
            FLIP_SECOND_INSTRUCTION_NOACTION + "\n" + action_format_hint(cot_enabled)
        ))

    return {"role": "user", "content": parts}


def run_one_game(
    *,
    model: str,
    rows: int, cols: int,
    num_cards: Optional[int],
    seed: int,
    render: str,
    theme: Optional[str],
    assets_dir: str,
    cell_size: int,
    out_dir: Path,
    max_responses: int,
    max_retries: int,
    label: Optional[str] = None,
    cot_enabled: bool = True,
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
        "ground_truth": gt_part,
        "ground_truth_text": gt_text,
        "optimal_resp_times_greedy": optimal_greedy,
    }
    tlog = TurnLogger(out_dir / "game.json", config=config)

    system_msg = {"role": "system", "content": build_single_system(noaction=True, cot_enabled=cot_enabled)}
    history: List[Dict[str, Any]] = []
    round_idx = 0
    error: Optional[str] = None

    try:
        while not obs.done and tlog.response_count < max_responses:
            round_idx += 1

            coord1, flip1_ref, flip1_fails = _run_flip(
                phase_label="flip_first",
                env=env, br=br, round_idx=round_idx, history=history,
                system_msg=system_msg, client=client,
                image_store=image_store, tlog=tlog,
                max_retries=max_retries, max_responses=max_responses,
                first_flip_str=None, player=None, cot_enabled=cot_enabled,
            )
            if coord1 is None:
                history.append({
                    "round": round_idx, "kind": "case_b",
                    "refs": [],
                    "flip1_fails": flip1_fails, "flip2_fails": [],
                })
                continue

            first_str = coord_to_str(coord1)
            coord2, both_ref, flip2_fails = _run_flip(
                phase_label="flip_second",
                env=env, br=br, round_idx=round_idx, history=history,
                system_msg=system_msg, client=client,
                image_store=image_store, tlog=tlog,
                max_retries=max_retries, max_responses=max_responses,
                first_flip_str=first_str, player=None, cot_enabled=cot_enabled,
            )

            if coord2 is None:
                env.abort_round()
                history.append({
                    "round": round_idx, "kind": "case_a",
                    "refs": [flip1_ref],
                    "flip1_fails": flip1_fails, "flip2_fails": flip2_fails,
                })
                obs = env.get_observation()
                continue

            obs = env.get_observation()
            matched = (obs.last_result or {}).get("matched", False)
            refs = [flip1_ref]
            if both_ref is not None:
                refs.append(both_ref)
            history.append({
                "round": round_idx, "kind": "normal", "matched": matched,
                "refs": refs,
                "flip1_fails": flip1_fails, "flip2_fails": flip2_fails,
            })

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
    env: MatchingEnv,
    br: BoardRenderer,
    round_idx: int,
    history: List[Dict[str, Any]],
    system_msg: Dict[str, Any],
    client,
    image_store: Optional[ImageStore],
    tlog: TurnLogger,
    max_retries: int,
    max_responses: int,
    first_flip_str: Optional[str],
    player: Optional[str],
    cot_enabled: bool,
) -> tuple:
    """Return (coord, ref_for_history, fail_reasons)."""
    last_turn_id: Optional[int] = None
    last_call_type = phase_label
    fail_reasons: List[Dict[str, Any]] = []

    for attempt in range(max_retries + 1):
        if tlog.response_count >= max_responses:
            return None, None, fail_reasons

        obs = env.get_observation()
        # current-board part (auto-shows first_flip during flip_second)
        if phase_label == "flip_first":
            board_name = f"round_{round_idx:03d}_start"
        else:
            board_name = f"round_{round_idx:03d}_flip1"
        current_part = br.board_part(name=board_name)

        user_msg = _build_stateless_user(
            history=history,
            current_board_part=current_part,
            br=br,
            phase=obs.phase,
            remaining_pairs=obs.remaining_pairs,
            score=obs.score,
            first_flip_coord_str=first_flip_str,
            last_fail=(fail_reasons[-1] if fail_reasons else None),
            cot_enabled=cot_enabled,
        )
        messages = [system_msg, user_msg]

        materialized = materialize(messages, image_store) if image_store else messages
        resp = call_llm_with_retry(client, materialized)
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
        history_ref: Optional[str] = None
        attempt_fail: Dict[str, Any] = {}

        if parse_ok:
            try:
                first_coord = env.first_flip_coord if phase_label == "flip_second" else None
                env.step(coord)
                applied = True
                if phase_label == "flip_first":
                    history_ref = br.store_board(name=f"round_{round_idx:03d}_flip1")
                    revealed = env.first_flip_face
                else:
                    history_ref = br.store_both_flips(
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
                attempt_fail = {"kind": "env", "code": ie.code, "reason": ie.reason}
        else:
            attempt_fail = {"kind": "parse"}

        last_turn_id = tlog.log_call(
            call_type=last_call_type, round_idx=round_idx, player=player,
            messages_sent=messages,
            response={"content": content, "reasoning": reasoning},
            parse=parse_info, env_result=env_result,
            retry_of=last_turn_id,
        )
        last_call_type = "retry"

        if applied:
            return coord, history_ref, fail_reasons

        fail_reasons.append(attempt_fail)

    return None, None, fail_reasons


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(_REPO / ".env")
    parser = argparse.ArgumentParser(description="Single player, no-action mode")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--grid", type=parse_grid_size, default=(6, 6))
    parser.add_argument("--num-cards", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", choices=["text", "image"], default="image")
    parser.add_argument("--theme", default=None)
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
    parser.add_argument("--out", required=True)
    parser.add_argument("--on-exists", choices=ON_EXISTS_CHOICES, default="overwrite",
                        help="If run dir already exists: overwrite/skip/resume (resume only fully implemented in single_normal; in this mode it behaves like overwrite-without-rm).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

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
    )
    logger.info(f"RUN_DIR: {run_dir}")
    logger.info(f"RESULT: {result}")


if __name__ == "__main__":
    main()
