# Matching Pairs

Static, categorical hidden state: the board is a grid of face-down cards. Each
turn the model flips two cells; matched pairs stay up, mismatches flip back. The
model must remember **identity→location** bindings it can no longer see and act
on them, turn after turn, in closed loop.

```
1_matching_pairs_new/
├── env/            # game logic + board rendering (text / image themes)
├── common/         # parsing, image store, renderer, turn logger, optimal policy
├── modes/          # the four eval entry points (run with `python -m modes.<name>`)
├── visualize/      # HTML replay + index builder
├── assets/         # card themes (noise, poker, textures, perlin, voronoi, ...)
└── scripts/        # example sweep launchers
```

Models, endpoints and sampling are defined once in the **repo-root**
`model_presets.py` and shared with 3D Maze. API keys/endpoints are read from the
repo-root `.env` (see the top-level README for the env-var table). Adding a model
there makes it available to every mode here.

## Modes

| Mode (`python -m modes.<name>`) | What it isolates |
|---|---|
| `single_normal`   | Main task: one model, must remember to act. |
| `single_noaction` | Ablation: the model's own actions are dropped from the history (only the sequence of board snapshots remains), so it must re-infer its flips from the visual diff. |
| `dual_normal`     | Duel: two models alternate on the *same* board (removes per-deal variance). |
| `dual_noaction`   | Duel variant under the no-action-feedback condition. |

## Quick start

```bash
cd 1_matching_pairs_new

# Single player, 8×10 image board, noise theme
python -m modes.single_normal \
  --model gpt-5.4 --grid 8x10 --render image --theme noise \
  --seed 0 --cot --max-resp-per-pair 5 --out results_demo

# Text mode (no images)
python -m modes.single_normal \
  --model gpt-5.4 --grid 8x10 --render text --seed 0 --out results_demo

# Duel: two models, same board
python -m modes.dual_normal \
  --model-a gpt-5.4 --model-b gemini-3.1-pro \
  --grid 8x10 --render-a image --render-b image --theme poker \
  --seed 0 --cot --out results_duel
```

Output: `<out>/<mode>/<model>/<render>-<theme>/<grid>/seed_<S>/game.json` plus
per-round renders under `images/`. The `game.json` carries `result`
(`score`, `total_pairs`, `response_count`) and a `turns` list (each with
`parse`, `env_result`, `response`).

## Key flags

| Flag | Meaning |
|---|---|
| `--grid RxC` | Board size, e.g. `8x10`. Must have an even cell count. |
| `--render {image,text}` | Observation modality. |
| `--theme NAME` | Card pattern for image mode: `noise`, `poker`, `textures`, `perlin`, `voronoi`, `abstract`, ... (see `assets/`). |
| `--cot / --no-cot` | Allow / forbid chain-of-thought before the action. |
| `--max-resp-per-pair N` | Response budget = `N × total_pairs` (caps cost on hard boards). |
| `--max-retries N` | Retries per turn on a parse failure. |
| `--on-exists {overwrite,resume,skip}` | Behaviour when the output already exists. |

For `single_normal`, `--on-exists resume` validates the saved configuration and
replays every recorded reply before making a new model request. Replies from
incomplete rounds and failed action attempts still count toward the original
budget. An incompatible or corrupt checkpoint is rejected without replacing
the saved log or images; finalized runs are skipped.

## Example sweeps (`scripts/`)

The launchers accept environment overrides as shown below. `PARALLEL`
(via `xargs -P`) controls concurrency; choose it according to your API rate limit.

`run_eval.sh` defaults to seeds `0 1 2`, with one episode per model/theme/grid/seed
and no outer try loop. These are example defaults; the complete seed list and
outer repetition count for the published single-player main table are not
specified in the released paper and scripts. The five-seed `0 1 2 3 4` default
belongs to the Maze launcher. See [Seeds and repeats](../README.md#seeds-and-repeats)
for the distinction between an environment seed and a full try.

For image duels, paper Table 3 specifies two seeds and both starting orders,
giving four games per model pair. `run_duel.sh` defaults to four seeds and model
A starting every game; it does not exchange the starting order. The example
below chooses seeds `0 1` and swaps the models in a second invocation. These
seed values are an example, since Table 3 does not name its two seeds.

```bash
MODELS="gpt-5.4 gemini-3.1-pro" GRIDS="8x10 10x10" SEEDS="0 1 2" PARALLEL=4 \
  bash scripts/run_eval.sh

# Two seeds, both starting orders: four games per pair
MODEL_A=gpt-5.4 MODEL_B=gemini-3.1-pro SEEDS="0 1" OUT_BASE=results_duel_a_first \
  bash scripts/run_duel.sh
MODEL_A=gemini-3.1-pro MODEL_B=gpt-5.4 SEEDS="0 1" OUT_BASE=results_duel_b_first \
  bash scripts/run_duel.sh
```

For repeated tries, use a fresh `OUT_BASE` for each full seed set. Both scripts
pass `--on-exists resume`, so reusing an output directory can resume or overwrite
existing games. Keep every try's outputs separate.

## Replays

```bash
python -m visualize.index <results_root>   # writes per-game replay.html + index.html
```
