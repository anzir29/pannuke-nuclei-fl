#!/usr/bin/env bash
# Full pipeline on a 2-GPU workstation (e.g. 2x RTX 4090 24 GB).
#
#   GPU 0: Hyperband tuning -> optimized/quantized models -> federated runs (need the tuned settings)
#   GPU 1: segmentation comparison (Tables 2-3) and classifier comparison (Table 6) for every seed
#
# Every stage resumes from where it stopped, so re-running this script after an interruption
# continues the work.  Usage:  bash scripts/run_all.sh [config] [seeds...]
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/default.yaml}
shift || true
SEEDS=${*:-0 1 2}
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

run() {   # run <gpu> <stage> <seed>
  local gpu=$1 stage=$2 seed=$3
  echo "[$(date '+%F %T')] GPU$gpu  $stage seed=$seed"
  CUDA_VISIBLE_DEVICES=$gpu python -u -m pannuke_fl.run "$stage" --config "$CONFIG" \
      --seed "$seed" --results "$RESULTS" >> "logs/${stage}_seed${seed}.log" 2>&1
}

# one-time preprocessing (CPU); must finish before the GPU queues start
python -u -m pannuke_fl.run prep --config "$CONFIG" --results "$RESULTS" 2>&1 | tee -a logs/prep.log

queue_gpu0() {
  run 0 tune 0
  for s in $SEEDS; do run 0 opt "$s"; done
  for s in $SEEDS; do run 0 fl "$s"; done
  run 0 flsweep 0
}

queue_gpu1() {
  for s in $SEEDS; do run 1 seg "$s"; run 1 cls "$s"; done
}

queue_gpu0 > logs/queue_gpu0.log 2>&1 &
P0=$!
queue_gpu1 > logs/queue_gpu1.log 2>&1 &
P1=$!
echo "started: GPU0 queue pid $P0, GPU1 queue pid $P1  (progress: tail -f logs/queue_gpu*.log)"
wait $P0 $P1
python scripts/make_tables.py --results "$RESULTS" --out "$RESULTS/tables"
echo "all done -> $RESULTS/tables"
