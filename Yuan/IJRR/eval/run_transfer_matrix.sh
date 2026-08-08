#!/usr/bin/env bash
# Does curvature have to be in the training distribution?
#
# Two policies, both vertex-action, both 30M steps, differing only in what they
# were trained on: straight rays, or the serpentine family. Each is scored on
# both task families. The environment is identical in all four cells -- same
# task controller, same lateral gain, same hybrid thresholds -- so the only
# things that vary are the training family and the curvature observation
# channel, which the policy dimensions force to move together.
#
# The classical arm is recomputed inside every cell, so every number is a ratio
# against the classical law on the very same tasks.
set -euo pipefail
cd /home/lqin/one
PY=/home/lqin/miniconda3/envs/one/bin/python
CFG=Yuan/IJRR/stage2_traj/config_vertex.yaml     # k_lateral 5.0, swing <= 30 deg
OUT=/home/lqin/one/Yuan/IJRR/runs/_transfer
N=${1:-2048}
mkdir -p "$OUT"

run () {   # name ckpt swing observe_curvature
  echo "=== $1 : ckpt=$2 swing=$3 observe_curvature=$4 ==="
  $PY -u -m Yuan.IJRR.eval.eval_curve \
      --config "$CFG" --ckpt "$2" --n-tasks "$N" \
      --swing-max-deg "$3" --observe-curvature "$4" \
      --out "$OUT/$1.json"
}

run line_on_straight Yuan/IJRR/runs/rl_vertex_line_30M 0  0
run line_on_curve    Yuan/IJRR/runs/rl_vertex_line_30M 30 0
run curve_on_straight Yuan/IJRR/runs/rl_vertex_30M     0  1
run curve_on_curve    Yuan/IJRR/runs/rl_vertex_30M     30 1
echo "ALL TRANSFER CELLS DONE"
