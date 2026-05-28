#!/usr/bin/env bash
# Run all 6 cells in sequence on a given eval-set NPZ.
#
# Cell naming = <seed_source>_<controller>:
#   cls_cls     baseline                    (q0_seed       + classical)
#   diff_cls    seed-only ablation          (diffusion seed + classical)
#   cls_hyb     controller-only ablation    (q0_seed       + hybrid)
#   diff_hyb    full method                 (diffusion seed + hybrid)
#   oracle_cls  classical-label oracle      (label-argmax seed + hybrid)
#   oracle_hyb  controller-aware oracle     (max over SMM top-K' under hybrid)
#
# Usage:
#   bash Yuan/system_eval/run_all_cells.sh <eval_set.npz> [out_dir]
#
# diff_cls and diff_hyb share the (expensive) diffusion+IK seed pool, so
# diff_cls writes a cache and diff_hyb consumes it.
set -euo pipefail

EVAL_SET=${1:?"usage: $0 <eval_set.npz> [out_dir]"}
OUT_DIR=${2:-"Yuan/system_eval/runs/eval_10k_systematic"}
CONFIG=${CONFIG:-"Yuan/system_eval/config.yaml"}
PILOT_NPZ=${PILOT_NPZ:-"Yuan/seed_selection/runs/pilot_20k/pilot_20k.npz"}

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
run_cell cls_cls
run_cell diff_cls --write-diffusion-cache
run_cell cls_hyb
run_cell diff_hyb --diffusion-cache "$OUT_DIR/diffusion_seeds_diff_cls.npz"
run_cell oracle_cls

echo
echo "================ Cell oracle_hyb (controller-aware oracle) ================"
python -m Yuan.system_eval.run_oracle_prime \
  --config "$CONFIG" \
  --eval-set "$EVAL_SET" \
  --pilot-npz "$PILOT_NPZ" \
  --out-dir "$OUT_DIR"

echo
echo "================ Aggregate ================"
python -m Yuan.system_eval.aggregate \
  --config "$CONFIG" \
  --in-dir "$OUT_DIR" \
  --require-all

t1=$(date +%s)
echo "[run_all] total wall time: $((t1 - t0)) s"
