#!/usr/bin/env bash
# Main results on one seed, 2 GPUs:
#   GPU 0: SegNet       baseline -> Hyperband -> optimized + INT8 -> FedAvg (+INT8)
#   GPU 1: DenseNet121  baseline -> Hyperband -> optimized + INT8 -> FedAvg (+INT8)
# Re-running the script continues from where it stopped.
#   Usage: bash scripts/run_minimal.sh [seed]
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=${1:-0}
CONFIG=configs/minimal.yaml
RESULTS=${RESULTS:-results}
mkdir -p logs

# refuse to start if another copy of this script is already running (two copies corrupt the cache)
exec 9>"logs/.run.lock"
if ! flock -n 9; then
  echo "Another run is already in progress (logs/.run.lock). Check with: pgrep -af pannuke_fl" >&2
  exit 1
fi
# refuse to start if another run of this script is already active
exec 9>logs/.run.lock
flock -n 9 || { echo "Another run is already active (logs/.run.lock). Check with: pgrep -af pannuke_fl"; exit 1; }
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

python -u -m pannuke_fl.run prep --config "$CONFIG" --results "$RESULTS" 2>&1 | tee -a logs/prep.log

chain() {   # chain <gpu> <seg|cls> <baseline stage>
  local gpu=$1 task=$2 base=$3
  for stage in "$base" tune opt fl; do
    echo "[$(date '+%F %T')] GPU$gpu $task: $stage"
    # a stage that crashes on exit after saving its results must not stop the chain;
    # every stage checks for its own inputs and skips work that is already done
    if [ "$stage" = "$base" ]; then
      CUDA_VISIBLE_DEVICES=$gpu python -u -m pannuke_fl.run "$stage" --config "$CONFIG" \
        --seed "$SEED" --results "$RESULTS" || echo "[warn] GPU$gpu $task: $stage exited with code $?"
    else
      CUDA_VISIBLE_DEVICES=$gpu python -u -m pannuke_fl.run "$stage" --tasks "$task" \
        --config "$CONFIG" --seed "$SEED" --results "$RESULTS" || echo "[warn] GPU$gpu $task: $stage exited with code $?"
    fi
  done
  echo "[$(date '+%F %T')] GPU$gpu $task: finished"
}

chain 0 seg seg > logs/gpu0_segnet.log 2>&1 &
P0=$!
chain 1 cls cls > logs/gpu1_densenet121.log 2>&1 &
P1=$!
echo "running: GPU0 (SegNet) pid $P0, GPU1 (DenseNet121) pid $P1"
echo "progress: tail -f logs/gpu0_segnet.log logs/gpu1_densenet121.log"
wait $P0 $P1
python scripts/make_tables.py --results "$RESULTS" --out "$RESULTS/tables"
echo "done -> $RESULTS/tables/summary.md"
