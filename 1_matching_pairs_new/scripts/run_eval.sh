#!/bin/bash
# Matching Pairs — single-player evaluation matrix.
#
#   MODELS x THEMES x GRIDS x SEEDS, run in parallel.
#
# Every cell calls `modes.single_normal`, which loads ../.env for API keys/
# endpoints and resolves the model from the repo-root model_presets.py. Results
# land under  <OUT_BASE>/maxresp<MR>/single_normal/<model>/<render>-<theme>/<grid>/seed_<S>/game.json
#
# Override anything via env vars, e.g.:
#   MODELS="gpt-5.4 gemini-3.1-pro" GRIDS="8x10 10x10" PARALLEL=4 bash scripts/run_eval.sh
set -u
cd "$(dirname "$0")/.."                      # -> 1_matching_pairs_new/
export PYTHONUNBUFFERED=1
PY=${PYTHON_BIN:-python}

MODELS=(${MODELS:-gpt-5.4})
THEMES=(${THEMES:-noise})
GRIDS=(${GRIDS:-8x10 10x10})
SEEDS=(${SEEDS:-0 1 2})
MR=${MR:-5}                                   # --max-resp-per-pair
PARALLEL=${PARALLEL:-4}                       # concurrent games (raise per API rate limit)
OUT_BASE=${OUT_BASE:-results_demo}

LIST=$(mktemp)
for m in "${MODELS[@]}"; do for th in "${THEMES[@]}"; do for g in "${GRIDS[@]}"; do for s in "${SEEDS[@]}"; do
  printf "%s\t%s\t%s\t%s\n" "$m" "$th" "$g" "$s" >>"$LIST"
done; done; done; done
TOTAL=$(wc -l <"$LIST")
echo "[$(date '+%F %T')] EVAL START total=$TOTAL parallel=$PARALLEL mr=$MR out=$OUT_BASE"

# `xargs -P` is the parallelism knob: PARALLEL games run at once.
xargs -a "$LIST" -d '\n' -P "$PARALLEL" -I {} bash -c '
  IFS=$'"'"'\t'"'"' read -r model theme grid seed <<<"{}"
  tag="${model}__${theme}__${grid}__s${seed}"
  start=$(date +%s)
  '"$PY"' -m modes.single_normal \
    --model "$model" --grid "$grid" --seed "$seed" \
    --render image --theme "$theme" --cot \
    --max-resp-per-pair '"$MR"' --max-retries 2 \
    --out "'"$OUT_BASE"'/maxresp'"$MR"'" --on-exists resume \
    >"/tmp/mp_eval_${tag}.log" 2>&1
  echo "[$(date +%T)] $tag rc=$? $(( $(date +%s)-start ))s"
'
echo "[$(date '+%F %T')] EVAL DONE ($OUT_BASE)"
rm -f "$LIST"
