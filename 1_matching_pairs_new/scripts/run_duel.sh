#!/bin/bash
# Matching Pairs — duel protocol (two models alternate on the SAME board).
#
# Removes per-instance board variance: A and B see identical layouts/seeds, so
# the score difference is attributable to the models, not the deal. Each cell
# calls `modes.dual_normal`.
#
#   MODEL_A vs MODEL_B  x  GRIDS x SEEDS, run in parallel.
#
# Override via env vars, e.g.:
#   MODEL_A=gpt-5.4 MODEL_B=gemini-3.1-pro GRIDS="8x10" SEEDS="0 1 2 3" \
#     bash scripts/run_duel.sh
set -u
cd "$(dirname "$0")/.."                      # -> 1_matching_pairs_new/
export PYTHONUNBUFFERED=1
PY=${PYTHON_BIN:-python}

MODEL_A=${MODEL_A:-gpt-5.4}
MODEL_B=${MODEL_B:-gemini-3.1-pro}
THEME=${THEME:-poker}
GRIDS=(${GRIDS:-8x10})
SEEDS=(${SEEDS:-0 1 2 3})
MR=${MR:-5}
PARALLEL=${PARALLEL:-4}
OUT_BASE=${OUT_BASE:-results_duel}

LIST=$(mktemp)
for g in "${GRIDS[@]}"; do for s in "${SEEDS[@]}"; do
  printf "%s\t%s\n" "$g" "$s" >>"$LIST"
done; done
TOTAL=$(wc -l <"$LIST")
echo "[$(date '+%F %T')] DUEL START ${MODEL_A} vs ${MODEL_B} total=$TOTAL parallel=$PARALLEL"

xargs -a "$LIST" -d '\n' -P "$PARALLEL" -I {} bash -c '
  IFS=$'"'"'\t'"'"' read -r grid seed <<<"{}"
  tag="'"$MODEL_A"'_vs_'"$MODEL_B"'__${grid}__s${seed}"
  start=$(date +%s)
  '"$PY"' -m modes.dual_normal \
    --model-a "'"$MODEL_A"'" --model-b "'"$MODEL_B"'" \
    --grid "$grid" --seed "$seed" \
    --render-a image --render-b image --theme "'"$THEME"'" --cot \
    --max-resp-per-pair '"$MR"' --max-retries 2 \
    --out "'"$OUT_BASE"'" --on-exists resume \
    >"/tmp/mp_duel_${tag}.log" 2>&1
  echo "[$(date +%T)] $tag rc=$? $(( $(date +%s)-start ))s"
'
echo "[$(date '+%F %T')] DUEL DONE ($OUT_BASE)"
rm -f "$LIST"
