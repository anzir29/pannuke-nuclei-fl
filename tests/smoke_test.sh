#!/usr/bin/env bash
# End-to-end check on a tiny synthetic dataset (~15-30 min on CPU, a few minutes on GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
export TF_CPP_MIN_LOG_LEVEL=2
rm -rf /tmp/pfl_smoke
python tests/make_synthetic.py /tmp/pfl_smoke/raw
for st in prep seg cls tune opt fl; do
  echo "== $st"; python -m pannuke_fl.run $st --config tests/smoke.yaml --results /tmp/pfl_smoke/results
done
python scripts/make_tables.py --results /tmp/pfl_smoke/results --out /tmp/pfl_smoke/tables
echo "SMOKE TEST PASSED"
