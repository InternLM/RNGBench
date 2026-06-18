#!/bin/bash
# 3D Maze — evaluation matrix.
#
#   MODELS x SIZES x SEEDS, run in parallel.
#
# Each cell calls `run.py`, which loads ../.env for API keys/endpoints and
# resolves the model from the repo-root model_presets.py. Results land under
#   <OUT_BASE>/<model>/<N>x<N>/<model>__<N>x<N>__v<vision-range>__seed<S>.json
#
# Memory Gap: add --minimap to show the map every step (the oracle condition);
# the score drop vs. without it is the maze Memory Gap.
#
# Override via env vars, e.g.:
#   MODELS="gpt-5.4 gemini-3.1-pro" SIZES="9 11 13" SEEDS="0 1 2 3 4" \
#     PARALLEL=4 bash scripts/run_eval.sh
set -u
cd "$(dirname "$0")/.."                      # -> 2_3d_maze/
export PYTHONUNBUFFERED=1
PY=${PYTHON_BIN:-python}

MODELS=(${MODELS:-gpt-5.4})
SIZES=(${SIZES:-9 11 13})                    # maze cell dimension N (NxN), pass bare N
SEEDS=(${SEEDS:-0 1 2 3 4})
MAZE_TYPE=${MAZE_TYPE:-v6}                    # v5 (loops+clearings) | v6 (DFS+1-2 loops)
ACTION_SPACE=${ACTION_SPACE:-v5}             # v4 (fwd/back/turn) | v5 (slide-fwd/turn)
PARALLEL=${PARALLEL:-4}
OUT_BASE=${OUT_BASE:-results_demo}

LIST=$(mktemp)
for m in "${MODELS[@]}"; do for n in "${SIZES[@]}"; do for s in "${SEEDS[@]}"; do
  printf "%s\t%s\t%s\n" "$m" "$n" "$s" >>"$LIST"
done; done; done
TOTAL=$(wc -l <"$LIST")
echo "[$(date '+%F %T')] MAZE EVAL START total=$TOTAL parallel=$PARALLEL out=$OUT_BASE"

xargs -a "$LIST" -d '\n' -P "$PARALLEL" -I {} bash -c '
  IFS=$'"'"'\t'"'"' read -r model size seed <<<"{}"
  tag="${model}__${size}x${size}__s${seed}"
  start=$(date +%s)
  '"$PY"' run.py \
    --model "$model" --maze-size "$size" --seed "$seed" \
    --maze-type "'"$MAZE_TYPE"'" --action-space "'"$ACTION_SPACE"'" \
    --output-dir "'"$OUT_BASE"'/${model}/${size}x${size}" \
    >"/tmp/maze_eval_${tag}.log" 2>&1
  echo "[$(date +%T)] $tag rc=$? $(( $(date +%s)-start ))s"
'
echo "[$(date '+%F %T')] MAZE EVAL DONE ($OUT_BASE)"
rm -f "$LIST"
