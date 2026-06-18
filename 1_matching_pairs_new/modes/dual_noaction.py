"""
Mode 4: two-player + no action (stateless rebuild, shared history).

Both players share one sequence of board snapshots; history keeps a set of refs per player
(same logical state, different rendering). Each call statelessly rebuilds
messages = [system(player identity), user(history + current board + request)].

Supports per-player render: --render-a / --render-b can be mixed.
Rule: match -> same player goes again; unmatch -> switch player.
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
    FLIP_SECOND_INSTRUCTION_NOACTION,
    action_format_hint,
    describe_invalid,
)
from common.llm_call import call_llm_with_retry                              # noqa: E402
from common.optimal import compute_optimal_resp_times                        # noqa: E402
from model_presets import make_client, parse_grid_size                       # noqa: E402

logger = logging.getLogger(__name__)

MODE_NAME = "dual_noaction"


def _text(t: str) -> Dict[str, Any]:
    return {"type": "text", "text": t}


def _describe_fail(f: Dict[str, Any]) -> str:
    """Coord-free description of a single failed attempt."""
    if f["kind"] == "parse":
        return "one reply could not be parsed"
    desc = describe_invalid(f.get("code", "invalid"), "was invalid")
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
    br_me: BoardRenderer,
    viewer: str,                      # "A" or "B" — whose view to use from history
    phase: str,
    me: str,
    scores: Dict[str, int],
    remaining_pairs: int,
    first_flip_coord_str: Optional[str] = None,
    last_fail: Optional[Dict[str, Any]] = None,
    cot_enabled: bool = True,
) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = []

    if history:
        parts.append(_text("Game history (both players' rounds, in order):"))
        for entry in history:
            who = entry["player"]
            rnd = entry["round"]
            kind = entry["kind"]
            if kind == "normal":
                verdict = ("matched (pair removed)" if entry.get("matched")
                           else "no match (cards flipped back)")
                parts.append(_text(f"Round {rnd} — Player {who}'s turn — {verdict}:"))
            elif kind == "case_a":
                parts.append(_text(
                    f"Round {rnd} — Player {who}'s turn — forfeited (first card was revealed "
                    f"but second flip failed; first card was flipped back, turn passed):"
                ))
            else:  # case_b
                parts.append(_text(
                    f"Round {rnd} — Player {who}'s turn — forfeited (first flip failed; "
                    f"no card was flipped, turn passed):"
                ))
            n1 = _fail_note("first flip", entry.get("flip1_fails", []), kind in ("normal", "case_a"))
            n2 = _fail_note("second flip", entry.get("flip2_fails", []), kind == "normal")
            if n1: parts.append(_text(n1))
            if n2: parts.append(_text(n2))
            for ref in entry["refs"][viewer]:
                parts.append(br_me.ref_to_part(ref))
    else:
        parts.append(_text("Game history: (none — first flip of the game)"))

    parts.append(_text(
        f"Current board (score A:{scores['A']} B:{scores['B']}, remaining {remaining_pairs} pairs):"
    ))
    parts.append(current_board_part)

    if last_fail is not None:
        parts.append(_text(_inflight_retry_feedback(last_fail)))

    if phase == "flip_first":
        parts.append(_text(
            f"Your turn, Player {me}. {FLIP_FIRST_INSTRUCTION}"
            + "\n" + action_format_hint(cot_enabled)
        ))
    else:
        parts.append(_text(
            f"Your turn, Player {me}. {FLIP_SECOND_INSTRUCTION_NOACTION}"
            + "\n" + action_format_hint(cot_enabled)
        ))

    return {"role": "user", "content": parts}


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

    system_msgs = {
        "A": {"role": "system", "content": build_duel_system("A", noaction=True, cot_enabled=cot_enabled)},
        "B": {"role": "system", "content": build_duel_system("B", noaction=True, cot_enabled=cot_enabled)},
    }
    history: List[Dict[str, Any]] = []
    scores = {"A": 0, "B": 0}
    current = first_player
    round_idx = 0
    error: Optional[str] = None

    try:
        while not obs.done and tlog.response_count < max_responses:
            round_idx += 1
            opp = "B" if current == "A" else "A"
            client = clients[current]

            coord1, flip1_refs, flip1_fails = _run_flip(
                phase_label="flip_first",
                env=env, br=br, round_idx=round_idx, history=history,
                system_msg=system_msgs[current], client=client,
                image_store=image_store, tlog=tlog,
                max_retries=max_retries, max_responses=max_responses,
                me=current, scores=scores, first_flip_str=None, cot_enabled=cot_enabled,
            )
            if coord1 is None:
                history.append({
                    "round": round_idx, "player": current,
                    "kind": "case_b", "matched": False,
                    "refs": {"A": [], "B": []},
                    "flip1_fails": flip1_fails, "flip2_fails": [],
                })
                current = opp
                continue

            first_str = coord_to_str(coord1)
            coord2, both_refs, flip2_fails = _run_flip(
                phase_label="flip_second",
                env=env, br=br, round_idx=round_idx, history=history,
                system_msg=system_msgs[current], client=client,
                image_store=image_store, tlog=tlog,
                max_retries=max_retries, max_responses=max_responses,
                me=current, scores=scores, first_flip_str=first_str, cot_enabled=cot_enabled,
            )

            if coord2 is None:
                env.abort_round()
                history.append({
                    "round": round_idx, "player": current,
                    "kind": "case_a", "matched": False,
                    "refs": {"A": [flip1_refs["A"]],
                             "B": [flip1_refs["B"]]},
                    "flip1_fails": flip1_fails, "flip2_fails": flip2_fails,
                })
                obs = env.get_observation()
                current = opp
                continue

            obs = env.get_observation()
            r = obs.last_result or {}
            matched = r.get("matched", False)
            if matched:
                scores[current] += 1

            entry_refs = {"A": [flip1_refs["A"]], "B": [flip1_refs["B"]]}
            if both_refs is not None:
                entry_refs["A"].append(both_refs["A"])
                entry_refs["B"].append(both_refs["B"])
            history.append({
                "round": round_idx, "player": current,
                "kind": "normal", "matched": matched,
                "refs": entry_refs,
                "flip1_fails": flip1_fails, "flip2_fails": flip2_fails,
            })

            if not matched:
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


def _run_flip(
    *,
    phase_label: str,
    env: MatchingEnv,
    br: Dict[str, BoardRenderer],
    round_idx: int,
    history: List[Dict[str, Any]],
    system_msg: Dict[str, Any],
    client,
    image_store: Optional[ImageStore],
    tlog: TurnLogger,
    max_retries: int,
    max_responses: int,
    me: str,
    scores: Dict[str, int],
    first_flip_str: Optional[str],
    cot_enabled: bool,
) -> tuple:
    """Returns (coord, refs_dict) where refs_dict = {"A": ref, "B": ref} storing
    snapshot as both players' views. For flip_first: per-player flip1 board.
    For flip_second: per-player 'both revealed' snapshot.

    Also returns fail_reasons (list of per-attempt failure reasons, most recent last)."""
    last_turn_id: Optional[int] = None
    last_call_type = phase_label
    fail_reasons: List[Dict[str, Any]] = []

    for attempt in range(max_retries + 1):
        if tlog.response_count >= max_responses:
            return None, None, fail_reasons

        obs = env.get_observation()
        board_name = (f"round_{round_idx:03d}_{me}_start" if phase_label == "flip_first"
                      else f"round_{round_idx:03d}_{me}_flip1")
        current_part = br[me].board_part(name=board_name)

        user_msg = _build_stateless_user(
            history=history,
            current_board_part=current_part,
            br_me=br[me], viewer=me, phase=obs.phase, me=me, scores=scores,
            remaining_pairs=obs.remaining_pairs,
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
        parse_info = {"ok": parse_ok,
                      "coord": coord_to_str(coord) if parse_ok else None,
                      "error": None if parse_ok else "could not parse coordinate"}
        env_result: Dict[str, Any] = {"applied": False}
        applied = False
        history_refs: Optional[Dict[str, str]] = None
        attempt_fail: Dict[str, Any] = {}

        if parse_ok:
            try:
                first_coord = env.first_flip_coord if phase_label == "flip_second" else None
                env.step(coord)
                applied = True
                if phase_label == "flip_first":
                    history_refs = {
                        "A": br["A"].store_board(name=f"round_{round_idx:03d}_{me}_flip1"),
                        "B": br["B"].store_board(name=f"round_{round_idx:03d}_{me}_flip1"),
                    }
                    revealed = env.first_flip_face
                else:
                    history_refs = {
                        "A": br["A"].store_both_flips(
                            name=f"round_{round_idx:03d}_{me}_both",
                            coord1=first_coord, coord2=coord),
                        "B": br["B"].store_both_flips(
                            name=f"round_{round_idx:03d}_{me}_both",
                            coord1=first_coord, coord2=coord),
                    }
                    revealed = (env.last_result or {}).get("face2")
                env_result = {"applied": True, "revealed_face": revealed,
                              "phase_after": env.phase}
            except InvalidActionError as ie:
                parse_info["error"] = ie.reason
                env_result = {"applied": False, "error": ie.reason, "code": ie.code}
                attempt_fail = {"kind": "env", "code": ie.code, "reason": ie.reason}
        else:
            attempt_fail = {"kind": "parse"}

        last_turn_id = tlog.log_call(
            call_type=last_call_type, round_idx=round_idx, player=me,
            messages_sent=messages,
            response={"content": content, "reasoning": reasoning},
            parse=parse_info, env_result=env_result, retry_of=last_turn_id,
        )
        last_call_type = "retry"

        if applied:
            return coord, history_refs, fail_reasons

        fail_reasons.append(attempt_fail)

    return None, None, fail_reasons


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(_REPO / ".env")
    parser = argparse.ArgumentParser(description="Dual player, no-action mode")
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
