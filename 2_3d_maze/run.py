"""
3D first-person maze — CLI entry point.

Usage examples:
    # single evaluation
    python run.py --model gpt-4o --seed 0

    # multi-seed batch evaluation
    python run.py --model gpt-4o --seeds 0,1,2,3,4

    # custom maze
    python run.py --model gpt-4o --maze-size 11 --vision-range 4 --max-steps 300

    # compare models (run separately, then aggregate with analyze.py)
    python run.py --model gpt-4o       --seeds 0,1,2,3,4
    python run.py --model claude-opus-4-6 --seeds 0,1,2,3,4
    python run.py --model gemini-2.0-flash --seeds 0,1,2,3,4
"""

import argparse
import logging
import sys
from pathlib import Path

# Repo root holds the shared model_presets.py (and framework/); add it so the
# import resolves regardless of cwd. `runner`/`game` still come from this dir.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv

load_dotenv(_REPO / ".env")  # API keys / endpoints (both games read the same .env)

from model_presets import make_client
from runner import MazeRunner3D, save_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="3D First-Person Maze — Multimodal LLM Evaluation"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name (e.g., gpt-4o, claude-opus-4-6, gemini-2.0-flash)",
    )
    parser.add_argument(
        "--maze-size", type=int, default=11,
        help="Maze cell dimension (odd number, default: 11)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for maze generation (default: 0)",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seeds for batch run, e.g., 0,1,2,3,4",
    )
    parser.add_argument(
        "--vision-range", type=int, default=4,
        help="Fog of war visibility in cells (default: 4)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="Max actions before game termination. 0 = auto (Nx optimal, see --max-steps-mul)",
    )
    parser.add_argument(
        "--max-steps-mul", type=int, default=4,
        help="Multiplier for auto max_steps: max(optimal * N, 80). Default: 4",
    )
    parser.add_argument(
        "--rollouts", type=int, default=1,
        help="Number of rollouts per seed (default: 1). For RL data collection use >1.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="Max retries per action on parse failure (default: 2)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for results JSON and logs (default: results)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Render and save a preview image of the initial state (no LLM call)",
    )
    parser.add_argument(
        "--save-screenshots", action="store_true",
        help="Save 3D + top-down screenshots for every step under output-dir/screenshots/",
    )
    parser.add_argument(
        "--minimap", action="store_true",
        help="Show minimap in the image sent to the model (default: hidden)",
    )
    parser.add_argument(
        "--obs-mode", type=str, default="scene",
        choices=["scene", "mm2d-local", "mm2d-cone", "text-symbolic"],
        help="Observation modality: 'scene' (3D first-person, default), "
             "'mm2d-local' (3x3 world-aligned 2D patch), "
             "'mm2d-cone' ((2v+1)x(2v+1) facing-cone 2D patch with FOV+LOS, aligned to 3D visibility), "
             "'text-symbolic' (4-directional text)",
    )
    parser.add_argument(
        "--no-hud", action="store_true",
        help="Disable the bottom HUD bar in the image sent to the model (default: HUD shown)",
    )
    parser.add_argument(
        "--action-space", type=str, default="v4", choices=["v4", "v5"],
        help="Action space: v4 (forward/backward/turn) or v5 (slide-forward/turn)",
    )
    parser.add_argument(
        "--history-window", type=int, default=0,
        help="Limit conversation history to last N steps (0 = full history, default: 0)",
    )
    parser.add_argument(
        "--loop-rate", type=float, default=0.15,
        help="Probability of removing each remaining wall to create loops (default: 0.15, 0=perfect maze)",
    )
    parser.add_argument(
        "--n-clearings", type=int, default=-1,
        help="Number of open clearings to add (-1 = auto: ~1 per 25 cells, default: -1)",
    )
    parser.add_argument(
        "--clearing-radius", type=int, default=1,
        help="Half-size of each clearing in cells; radius=1 → 3×3 open area (default: 1)",
    )
    parser.add_argument(
        "--maze-type", type=str, default="v5", choices=["v5", "v6"],
        help="Maze generation type: v5 (DFS+loops+clearings) or v6 (DFS+1-2 loops, no clearings)",
    )
    parser.add_argument(
        "--wall-style", type=str, default="plain",
        choices=["plain", "repetitive", "color_tag", "unique_poster"],
        help="Visual pattern variant for 3D wall rendering "
             "(plain=baseline, repetitive=non-distinctive complex texture, "
             "color_tag=per-face unique hue, unique_poster=per-face unique pattern)",
    )
    parser.add_argument(
        "--ask-map", action="store_true", default=False,
        help="Ask model to output an ASCII map each step (enables spatial awareness evaluation)",
    )
    parser.add_argument(
        "--ask-trajectory", action="store_true", default=False,
        help="Ask model to output a simple visited-cells ASCII map each step",
    )
    args = parser.parse_args()

    # Validate
    if args.maze_size < 3:
        logger.error("--maze-size must be >= 3")
        sys.exit(1)
    if args.maze_size % 2 == 0:
        logger.error("--maze-size must be odd (e.g., 7, 9, 11, 13, 15)")
        sys.exit(1)
    if args.ask_map and args.ask_trajectory:
        logger.error("--ask-map and --ask-trajectory cannot be used together")
        sys.exit(1)

    seeds = (
        [int(s.strip()) for s in args.seeds.split(",")]
        if args.seeds
        else [args.seed]
    )

    # Preview mode: just render the initial frame and exit
    if args.preview:
        from game import MazeGame3D
        env = MazeGame3D(
            maze_size=args.maze_size, seed=seeds[0], vision_range=args.vision_range,
            action_space=args.action_space, loop_rate=args.loop_rate,
            n_clearings=args.n_clearings, clearing_radius=args.clearing_radius,
            maze_type=args.maze_type, wall_style=args.wall_style,
        )
        img = env.render_frame()
        preview_path = Path(args.output_dir) / "preview"
        preview_path.mkdir(parents=True, exist_ok=True)
        out = preview_path / f"preview_{args.maze_size}x{args.maze_size}_seed{seeds[0]}.jpg"
        img.save(out, format="JPEG", quality=90)
        logger.info(f"Preview saved to {out}")
        logger.info(f"Optimal path length: {env.compute_optimal_path_length()} actions")
        logger.info(f"Maze top-down:\n{env.render_top_down_text(full=True)}")
        return

    client = make_client(args.model)
    results = []

    use_rollouts = args.rollouts > 1

    for seed in seeds:
        from game import MazeGame3D as _MazeGame3D
        _tmp_env = _MazeGame3D(
            maze_size=args.maze_size, seed=seed, vision_range=args.vision_range,
            action_space=args.action_space, loop_rate=args.loop_rate,
            n_clearings=args.n_clearings, clearing_radius=args.clearing_radius,
            maze_type=args.maze_type, wall_style=args.wall_style,
        )
        optimal = _tmp_env.compute_optimal_path_length()
        if args.max_steps > 0:
            max_steps = args.max_steps
        else:
            max_steps = max(optimal * args.max_steps_mul, 80)

        for rollout_idx in range(args.rollouts):
            # Build filename base for skip-check
            _check_base = (
                f"{args.model}__{args.maze_size}x{args.maze_size}"
                f"__v{args.vision_range}__seed{seed}"
            )
            if use_rollouts:
                _check_base += f"__r{rollout_idx}"
            _check_path = Path(args.output_dir) / f"{_check_base}.json"
            if _check_path.exists():
                logger.info(f"SKIP: {_check_path} already exists")
                continue

            rollout_tag = f" rollout={rollout_idx}" if use_rollouts else ""
            logger.info(f"{'=' * 60}")
            logger.info(
                f"Running: model={args.model} | maze={args.maze_size}×{args.maze_size} | "
                f"seed={seed}{rollout_tag} | optimal={optimal} | max_steps={max_steps}"
            )
            logger.info(f"{'=' * 60}")

            screenshot_dir = None
            if args.save_screenshots:
                safe_model = args.model.replace("/", "_").replace(":", "_")
                ss_name = f"{safe_model}_seed{seed}_size{args.maze_size}"
                if use_rollouts:
                    ss_name += f"_r{rollout_idx}"
                screenshot_dir = str(Path(args.output_dir) / "screenshots" / ss_name)

            runner = MazeRunner3D(
                client=client,
                maze_size=args.maze_size,
                seed=seed,
                vision_range=args.vision_range,
                max_steps=max_steps,
                max_retries=args.max_retries,
                screenshot_dir=screenshot_dir,
                show_minimap=args.minimap,
                show_hud=not args.no_hud,
                action_space=args.action_space,
                checkpoint_dir=str(Path(args.output_dir) / ".checkpoints"),
                history_window=args.history_window,
                loop_rate=args.loop_rate,
                n_clearings=args.n_clearings,
                clearing_radius=args.clearing_radius,
                ask_map=args.ask_map,
                ask_trajectory=args.ask_trajectory,
                maze_type=args.maze_type,
                obs_mode=args.obs_mode,
                wall_style=args.wall_style,
            )
            result = runner.run()
            save_result(result, output_dir=args.output_dir,
                        rollout=rollout_idx if use_rollouts else None)
            results.append(result)

            status = "REACHED" if result.reached_goal else "FAILED"
            logger.info(
                f"Seed {seed}{rollout_tag}: {status} | steps={result.total_steps} | "
                f"optimal={result.optimal_steps} | efficiency={result.path_efficiency:.3f} | "
                f"explored={result.exploration_rate:.1%} | turns={result.turn_count} | "
                f"bumps={result.wall_bump_count}"
            )

    # Summary for batch runs
    if len(results) > 1:
        sep = "=" * 60
        logger.info(f"\n{sep}\nBATCH SUMMARY\n{sep}")
        success_count = sum(1 for r in results if r.reached_goal)
        steps_all = [r.total_steps for r in results]
        logger.info(f"Model:        {args.model}")
        logger.info(f"Maze:         {args.maze_size}×{args.maze_size} | Vision: {args.vision_range}")
        logger.info(f"Seeds:        {seeds}")
        logger.info(f"Success rate: {success_count}/{len(results)}")
        logger.info(
            f"Avg steps:    {sum(steps_all)/len(steps_all):.1f} "
            f"(min={min(steps_all)}, max={max(steps_all)})"
        )
        eff = [r.path_efficiency for r in results if r.reached_goal]
        if eff:
            logger.info(f"Avg efficiency (successful): {sum(eff)/len(eff):.3f}")
        turns = [r.turn_count for r in results]
        bumps = [r.wall_bump_count for r in results]
        logger.info(f"Avg turns:    {sum(turns)/len(turns):.1f}")
        logger.info(f"Avg bumps:    {sum(bumps)/len(bumps):.1f}")
        logger.info(f"Total fallbacks: {sum(r.random_fallback_count for r in results)}")
        logger.info(sep)


if __name__ == "__main__":
    main()
