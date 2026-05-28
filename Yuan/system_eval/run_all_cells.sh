#!/usr/bin/env bash
# Run all 5 cells in sequence on a given eval-set NPZ.
#
# Usage:
#   bash Yuan/system_eval/run_all_cells.sh <eval_set.npz> [out_dir]
#
# Cells B and D share the (expensive) diffusion+IK seed pool, so B writes a
# cache and D consumes it.
set -euo pipefail

EVAL_SET=${1:?"usage: $0 <eval_set.npz> [out_dir]"}
OUT_DIR=${2:-"Yuan/system_eval/runs/eval_10k_systematic"}
CONFIG=${CONFIG:-"Yuan/system_eval/config.yaml"}

mkdir -p "$OUT_DIR"

run_cell () {
  local cell="$1"; shift
  echo
  echo "================ Cell $cell ================"
  python -m Yuan.system_eval.run_cell \
    --config "$CONFIG" \
    --eval-set "$EVAL_SET" \
    --cell "$cell" \
    --out-dir "$OUT_DIR" \
    "$@"
}

t0=$(date +%s)
run_cell A
run_cell B --write-diffusion-cache
run_cell C
run_cell D --diffusion-cache "$OUT_DIR/diffusion_seeds_B.npz"
run_cell E

echo
echo "================ Aggregate ================"
python -m Yuan.system_eval.aggregate \
  --config "$CONFIG" \
  --in-dir "$OUT_DIR" \
  --require-all

t1=$(date +%s)
echo "[run_all] total wall time: $((t1 - t0)) s"
