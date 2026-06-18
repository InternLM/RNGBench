# 3D Maze

Dynamic, spatial hidden state: the model navigates a maze from an egocentric
first-person view with a limited vision range (fog of war). Walls leave the
field of view as the agent moves, so reaching the goal requires building and
maintaining a **mental map** from a stream of partial observations — the
spatial, dynamic counterpart to Matching Pairs.

```
2_3d_maze/
├── game.py     # DFS maze generation + first-person raycast renderer
├── runner.py   # episode loop (binds to the shared framework LLM client)
├── run.py      # CLI entry point
└── scripts/    # example eval launcher
```

Models, endpoints and sampling come from the **repo-root** `model_presets.py`
(shared with Matching Pairs); `run.py` loads the repo-root `.env` for API
keys/endpoints. See the top-level README for the env-var table.

## Quick start

```bash
cd 2_3d_maze

# Single 11×11 maze, 3D first-person view
python run.py --model gpt-5.4 --maze-size 11 --seed 0

# Batch over five seeds on a 13×13 maze
python run.py --model gpt-5.4 --seeds 0,1,2,3,4 --maze-size 13

# Preview the initial state without calling any LLM
python run.py --model dummy --preview --seed 0 --maze-size 11
```

Output (default `results/`, override with `--output-dir`):
`<dir>/<model>__<N>x<N>__v<vision-range>__seed<S>.json`, with fields
`reached_goal`, `total_steps`, `optimal_steps`, `path_efficiency`,
`exploration_rate`, `wall_bump_count`, `maze_type`, `action_space`.

## Memory Gap (`--minimap` oracle)

Re-run the same config with `--minimap`: the true map is shown every step, so no
map-building from memory is needed (the oracle condition). The **Memory Gap** is
the score drop between the two runs — it isolates spatial recall from perception
/ decision-making.

```bash
python run.py --model gpt-5.4 --maze-size 13 --seed 0            # normal
python run.py --model gpt-5.4 --maze-size 13 --seed 0 --minimap  # oracle (map shown)
```

## Key flags

| Flag | Meaning |
|---|---|
| `--maze-size N` | Cell dimension (odd N → N×N maze). |
| `--seed S` / `--seeds 0,1,2` | Single seed, or a comma-separated batch. |
| `--vision-range K` | Fog-of-war radius in cells (default 4). |
| `--maze-type {v5,v6}` | `v5` = DFS + loops + clearings; `v6` = DFS + 1–2 loops, no clearings. |
| `--action-space {v4,v5}` | `v4` = forward/backward/turn; `v5` = slide-forward/turn. |
| `--obs-mode {scene,mm2d-local,mm2d-cone,text-symbolic}` | Observation modality (default `scene` = 3D first-person). |
| `--max-steps N` / `--max-steps-mul M` | Step cap; `0` = auto `max(optimal·M, 80)`. |
| `--minimap` | Show the true map every step (Memory-Gap oracle condition). |
| `--save-screenshots` | Dump 3D + top-down frames per step under `<dir>/screenshots/`. |

## Example sweep (`scripts/`)

`MODELS`/`SIZES`/`SEEDS`/`PARALLEL` env overrides; `PARALLEL` (via `xargs -P`)
sets concurrency.

```bash
MODELS="gpt-5.4 gemini-3.1-pro" SIZES="9 11 13" SEEDS="0 1 2 3 4" \
  PARALLEL=4 bash scripts/run_eval.sh
```
