"""
Mode 3: two-player + show action (accumulating multi-turn dialogue).

Each player has an independent conversation; the opponent's move is appended to one's own
context as result text + snapshot.
Rule: match -> same player goes again; unmatch -> switch player.

Supports per-player render: --render-a / --render-b can be text or image, and can be mixed.
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
from common.output_layout import resolve_run_dir, dual_render_leaf           # noqa: E402
from common.run_dir import prepare_run_dir, ON_EXISTS_CHOICES                # noqa: E402
from common.prompts import (                                                 # noqa: E402
    build_duel_system,
    FLIP_FIRST_INSTRUCTION,
    FLIP_SECOND_INSTRUCTION,
    action_format_hint,
    retry_parse_fail,
    retry_invalid_coord,
)
from common.llm_call import call_llm_with_retry                              # noqa: E402
from common.optimal import compute_optimal_resp_times                        # noqa: E402
from model_presets import make_client, parse_grid_size                       # noqa: E402

logger = logging.getLogger(__name__)

MODE_NAME = "dual_normal"


def _text(t: str) -> Dict[str, Any]:
    return {"type": "text", "text": t}


def _flip_first_user_msg(obs: Observation, br_me: BoardRenderer, round_idx: int,
                         scores: Dict[str, int], me: str,
                         cot_enabled: bool) -> Dict[str, Any]:
    return {"role": "user", "content": [
        _text(f"Your turn, Player {me}. Score A:{scores['A']} B:{scores['B']}. "
              f"Remaining pairs: {obs.remaining_pairs}. Round {round_idx} start — current board:"),
        br_me.board_part(name=f"round_{round_idx:03d}_{me}_start"),
        _text(FLIP_FIRST_INSTRUCTION + "\n" + action_format_hint(cot_enabled)),
    ]}


def _flip_second_user_msg(obs: Observation, br_me: BoardRenderer, round_idx: int,
                          me: str, cot_enabled: bool) -> Dict[str, Any]:
    return {"role": "user", "content": [
        _text("Board after your first flip:"),
        br_me.board_part(name=f"round_{round_idx:03d}_{me}_flip1"),
        _text(FLIP_SECOND_INSTRUCTION + "\n" + action_format_hint(cot_enabled)),
    ]}


def _self_result_msg(result: Dict[str, Any], br_me: BoardRenderer, round_idx: int,
                     me: str, scores: Dict[str, int], continue_turn: bool,
                     snap_both_me: Optional[str]) -> Dict[str, Any]:
    verdict = "Match!" if result["matched"] else "No match — cards flipped back."
    next_action = ("You get another turn." if continue_turn
                   else f"Turn passes to Player {'B' if me == 'A' else 'A'}.")
    parts: List[Dict[str, Any]] = [
        _text("Both cards revealed:"),
    ]
    if snap_both_me is not None:
        parts.append(br_me.ref_to_part(snap_both_me))
    parts.append(_text(
        f"{verdict} Score A:{scores['A']} B:{scores['B']}. {next_action}"
    ))
    return {"role": "user", "content": parts}


def _opponent_update_msg(opp: str, result: Dict[str, Any], br_opp: BoardRenderer,
                         round_idx: int, snap_both_opp: str,
                         scores: Dict[str, int]) -> Dict[str, Any]:
    verdict = "matched" if result["matched"] else "no match"
    return {"role": "user", "content": [
        _text(f"Player {opp}'s round {round_idx}: flipped {result['coord1']} then "
              f"{result['coord2']} ({verdict}). Board during reveal:"),
        br_opp.ref_to_part(snap_both_opp),
        _text(f"After resolution: score A:{scores['A']} B:{scores['B']}."),
    ]}


def _opponent_case_a_msg(opp: str, first_coord_str: str, snap_first_opp: str,
                         br_opp: BoardRenderer, round_idx: int) -> Dict[str, Any]:
    return {"role": "user", "content": [
        _text(f"Player {opp}'s round {round_idx}: flipped {first_coord_str} but could not "
              f"complete a valid second flip (parse error or invalid action each retry). "
              f"Card was flipped back. Snapshot when only first card was visible:"),
        br_opp.ref_to_part(snap_first_opp),
    ]}


def _opponent_case_b_msg(opp: str, br_opp: BoardRenderer, round_idx: int) -> Dict[str, Any]:
    return {"role": "user", "content": [
        _text(f"Player {opp}'s round {round_idx}: could not complete a valid first flip "
              f"(parse error or invalid action each retry). No card was flipped. "
              f"Turn passes to you."),
    ]}


def _retry_msg(reason: str) -> Dict[str, Any]:
    return {"role": "user", "content": [_text(reason)]}


def run_one_game(
    *,
    model_a: str, model_b: str,
    label_a: Optional[str], label_b: Optional[str],
    rows: int, cols: int,
    num_cards: Optional[int],
    seed: int,
    render_a: str, render_b: str,
    theme: Optional[str],
    assets_dir: str,
    cell_size: int,
    out_dir: Path,
    max_responses: int,
    max_retries: int,
    first_player: str = "A",
    cot_enabled: bool = True,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    any_image = render_a == "image" or render_b == "image"
    image_store: Optional[ImageStore] = None
    if any_image:
        image_store = ImageStore(root=out_dir / "images", base_dir=out_dir)

    clients = {
        "A": make_client(model_a, label=label_a),
        "B": make_client(model_b, label=label_b),
    }

    env = MatchingEnv(
        rows=rows, cols=cols, num_cards=num_cards, seed=seed,
        theme=theme if any_image else None,
        assets_dir=assets_dir, cell_size=(cell_size, cell_size),
    )
    obs = env.reset()
    br = {
        "A": BoardRenderer(env, render=render_a,
                           image_store=image_store if render_a == "image" else None),
        "B": BoardRenderer(env, render=render_b,
                           image_store=image_store if render_b == "image" else None),
    }
    # GT for config: prefer image side if mixed; else use A's.
    gt_side = "A" if render_a == "image" else ("B" if render_b == "image" else "A")
    gt_part = br[gt_side].ground_truth_part(name="ground_truth")
    gt_text = env.render_ground_truth(mode="text")
    optimal_greedy = compute_optimal_resp_times(env.board.cards)

    config = {
        "mode": MODE_NAME,
        "model_a": model_a, "label_a": label_a or model_a,
        "model_b": model_b, "label_b": label_b or model_b,
        "rows": rows, "cols": cols,
        "num_cards": env.board.num_cards,
        "total_pairs": env.total_pairs,
        "seed": seed,
        "render_a": render_a, "render_b": render_b,
        "theme": theme if any_image else None,
        "cell_size": cell_size,
        "max_responses": max_responses, "max_retries": max_retries,
        "cot_enabled": cot_enabled,
        "first_player": first_player,
        "ground_truth": gt_part,
        "ground_truth_text": gt_text,
        "optimal_resp_times_greedy": optimal_greedy,
    }
    tlog = TurnLogger(out_dir / "game.json", config=config)

    messages: Dict[str, List[Dict[str, Any]]] = {
        "A": [{"role": "system", "content": build_duel_system("A", cot_enabled=cot_enabled)}],
        "B": [{"role": "system", "content": build_duel_system("B", cot_enabled=cot_enabled)}],
    }
    scores = {"A": 0, "B": 0}
    current = first_player
    round_idx = 0
    error: Optional[str] = None

    try:
        while not obs.done and tlog.response_count < max_responses:
            round_idx += 1
            opp = "B" if current == "A" else "A"
            client = clients[current]

            coord1 = _run_flip_first(
                round_idx=round_idx,
                user_msg=_flip_first_user_msg(obs, br[current], round_idx, scores, current, cot_enabled),
                env=env, messages=messages[current], client=client,
                image_store=image_store, tlog=tlog,
                max_retries=max_retries, max_responses=max_responses,
                player=current, cot_enabled=cot_enabled,
            )
            if coord1 is None:
                messages[current].append({"role": "user", "content": [_text(
                    f"Could not complete a valid first flip after retries "
                    f"(parse error or invalid action each time). "
                    f"Round forfeited. Turn passes to Player {opp}.")]})
                messages[opp].append(_opponent_case_b_msg(current, br[opp], round_idx))
                obs = env.get_observation()
                current = opp
                continue

            obs = env.get_observation()
            first_str = coord_to_str(obs.first_flip_coord)

            coord2, snap_both_cur = _run_flip_second(
                round_idx=round_idx,
                user_msg=_flip_second_user_msg(obs, br[current], round_idx, current, cot_enabled),
                env=env, br_me=br[current], messages=messages[current], client=client,
                image_store=image_store, tlog=tlog,
                max_retries=max_retries, max_responses=max_responses,
                player=current, cot_enabled=cot_enabled,
            )

            if coord2 is None:
                # Case A: store first-flip snapshot for opponent too
                snap_first_opp = br[opp].store_board(
                    name=f"round_{round_idx:03d}_{current}_flip1_forfeit",
                )
                env.abort_round()
                obs = env.get_observation()
                messages[current].append({"role": "user", "content": [
                    _text(f"Could not complete a valid second flip after retries "
                          f"(parse error or invalid action each time). "
                          f"Your first card was flipped back. Round forfeited. Turn passes to "
                          f"Player {opp}."),
                ]})
                messages[opp].append(_opponent_case_a_msg(
                    current, first_str, snap_first_opp, br[opp], round_idx,
                ))
                current = opp
                continue

            # Store "both revealed" snapshot for opponent before env.step resolves.
            # Wait — env.step has already resolved inside _run_flip_second. Re-render "both" state.
            # br.store_both_flips works regardless of current env state (it reconstructs).
            snap_both_opp = br[opp].store_both_flips(
                name=f"round_{round_idx:03d}_{current}_both",
                coord1=obs.first_flip_coord, coord2=coord2,
            )

            obs = env.get_observation()
            r = obs.last_result or {}
            matched = r.get("matched", False)
            if matched:
                scores[current] += 1
            continue_turn = matched

            messages[current].append(_self_result_msg(
                r, br[current], round_idx, current, scores, continue_turn, snap_both_cur,
            ))
            messages[opp].append(_opponent_update_msg(
                current, r, br[opp], round_idx, snap_both_opp, scores,
            ))

            if not continue_turn:
                current = opp

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.error(f"Game aborted: {error}\n{traceback.format_exc()}")

    result = {
        "score_a": scores["A"], "score_b": scores["B"],
        "total_pairs": env.total_pairs if env.board else None,
        "rounds_played": round_idx,
        "response_count": tlog.response_count,
        "done": obs.done if env.board else False,
        "error": error,
    }
    tlog.finalize(result)
    return result


def _run_flip_first(
    *, round_idx, user_msg, env, messages, client, image_store, tlog,
    max_retries, max_responses, player, cot_enabled,
) -> Optional[tuple]:
    messages.append(user_msg)
    last_turn_id: Optional[int] = None
    last_call_type = "flip_first"

    for attempt in range(max_retries + 1):
        if tlog.response_count >= max_responses:
            return None

        materialized = materialize(messages, image_store) if image_store else messages
        resp = call_llm_with_retry(client, materialized)
        content = resp.get("content", "")
        reasoning = resp.get("reasoning")

        coord = parse_coord(content, env.rows, env.cols)
        parse_ok = coord is not None
        parse_info = {"ok": parse_ok,
                      "coord": coord_to_str(coord) if parse_ok else None,
                      "error": None if parse_ok else "could not parse coordinate"}
        env_result: Dict[str, Any] = {"applied": False}
        applied = False
        if parse_ok:
            try:
                env.step(coord)
                applied = True
                env_result = {"applied": True, "revealed_face": env.first_flip_face,
                              "phase_after": env.phase}
            except InvalidActionError as ie:
                parse_info["error"] = ie.reason
                env_result = {"applied": False, "error": ie.reason, "code": ie.code}

        last_turn_id = tlog.log_call(
            call_type=last_call_type, round_idx=round_idx, player=player,
            messages_sent=messages,
            response={"content": content, "reasoning": reasoning},
            parse=parse_info, env_result=env_result, retry_of=last_turn_id,
        )
        last_call_type = "retry"

        messages.append({"role": "assistant", "content": content})

        if applied:
            return coord

        reason = (retry_parse_fail(cot_enabled) if not parse_ok
                  else retry_invalid_coord(parse_info["error"] or "invalid", cot_enabled=cot_enabled))
        messages.append(_retry_msg(reason))

    return None


def _run_flip_second(
    *, round_idx, user_msg, env, br_me, messages, client, image_store, tlog,
    max_retries, max_responses, player, cot_enabled,
) -> tuple:
    messages.append(user_msg)
    last_turn_id: Optional[int] = None
    last_call_type = "flip_second"

    for attempt in range(max_retries + 1):
        if tlog.response_count >= max_responses:
            return None, None

        materialized = materialize(messages, image_store) if image_store else messages
        resp = call_llm_with_retry(client, materialized)
        content = resp.get("content", "")
        reasoning = resp.get("reasoning")

        coord = parse_coord(content, env.rows, env.cols)
        parse_ok = coord is not None
        parse_info = {"ok": parse_ok,
                      "coord": coord_to_str(coord) if parse_ok else None,
                      "error": None if parse_ok else "could not parse coordinate"}
        env_result: Dict[str, Any] = {"applied": False}
        applied = False
        snap_both_me: Optional[str] = None

        if parse_ok:
            try:
                first_coord = env.first_flip_coord
                env.step(coord)
                applied = True
                snap_both_me = br_me.store_both_flips(
                    name=f"round_{round_idx:03d}_{player}_both",
                    coord1=first_coord, coord2=coord,
                )
                env_result = {"applied": True,
                              "revealed_face": (env.last_result or {}).get("face2"),
                              "phase_after": env.phase}
            except InvalidActionError as ie:
                parse_info["error"] = ie.reason
                env_result = {"applied": False, "error": ie.reason, "code": ie.code}

        last_turn_id = tlog.log_call(
            call_type=last_call_type, round_idx=round_idx, player=player,
            messages_sent=messages,
            response={"content": content, "reasoning": reasoning},
            parse=parse_info, env_result=env_result, retry_of=last_turn_id,
        )
        last_call_type = "retry"

        messages.append({"role": "assistant", "content": content})

        if applied:
            return coord, snap_both_me

        reason = (retry_parse_fail(cot_enabled) if not parse_ok
                  else retry_invalid_coord(parse_info["error"] or "invalid", cot_enabled=cot_enabled))
        messages.append(_retry_msg(reason))

    return None, None


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(_REPO / ".env")
    parser = argparse.ArgumentParser(description="Dual player, show-action mode")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--grid", type=parse_grid_size, default=(6, 6))
    parser.add_argument("--num-cards", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-a", choices=["text", "image"], default="image")
    parser.add_argument("--render-b", choices=["text", "image"], default="image")
    parser.add_argument("--theme", default=None)
    parser.add_argument("--assets-dir", default=str(_ROOT / "assets"))
    parser.add_argument("--cell-size", type=int, default=64)
    parser.add_argument("--max-responses", type=int, default=0,
                        help="Absolute cap on total model replies (both players, incl. "
                             "retries). 0 = auto = total_pairs × max_resp_per_pair.")
    parser.add_argument("--max-resp-per-pair", type=int, default=7,
                        help="Multiplier for the auto-computed max_responses budget "
                             "(shared between the two players).")
    parser.add_argument("--max-retries", type=int, default=2)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cot", dest="cot_enabled", action="store_true",
                       help="Require Thought + Action output (default).")
    group.add_argument("--no-cot", dest="cot_enabled", action="store_false",
                       help="Require only Action output; no reasoning line.")
    parser.set_defaults(cot_enabled=True)
    parser.add_argument("--first-player", choices=["A", "B"], default="A")
    parser.add_argument("--out", required=True)
    parser.add_argument("--on-exists", choices=ON_EXISTS_CHOICES, default="overwrite",
                        help="If run dir already exists: overwrite/skip/resume (resume only fully implemented in single_normal; in this mode it behaves like overwrite-without-rm).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    rows, cols = args.grid
    total_pairs = (rows * cols) // 2
    max_resp = args.max_responses or total_pairs * args.max_resp_per_pair

    la = args.label_a or args.model_a
    lb = args.label_b or args.model_b

    run_dir = resolve_run_dir(
        out_root=Path(args.out), mode=MODE_NAME,
        leaf=dual_render_leaf(args.render_a, args.theme, args.render_b, args.theme),
        seed=args.seed,
        label=la, label_b=lb,
        rows=rows, cols=cols,
        cot_enabled=args.cot_enabled,
    )

    prepared = prepare_run_dir(run_dir, args.on_exists)
    if prepared is None:
        logger.info(f"SKIPPED (exists): {run_dir}")
        return

    result = run_one_game(
        model_a=args.model_a, model_b=args.model_b,
        label_a=args.label_a, label_b=args.label_b,
        rows=rows, cols=cols,
        num_cards=args.num_cards, seed=args.seed,
        render_a=args.render_a, render_b=args.render_b,
        theme=args.theme, assets_dir=args.assets_dir,
        cell_size=args.cell_size,
        out_dir=run_dir,
        max_responses=max_resp, max_retries=args.max_retries,
        first_player=args.first_player,
        cot_enabled=args.cot_enabled,
    )
    logger.info(f"RUN_DIR: {run_dir}")
    logger.info(f"RESULT: {result}")


if __name__ == "__main__":
    main()
