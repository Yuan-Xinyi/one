#!/usr/bin/env bash
# Wait for the xArm7 and Cobotta vertex trainings, then score each arm on
# 2048 of its own straight tasks: classical law / vertex policy / hybrid,
# reported as ratios to the classical law (the mechanism protocol used for
# every FR3 result in this line of work).
set -uo pipefail
cd /home/lqin/one
PY=/home/lqin/miniconda3/envs/one/bin/python
R=Yuan/IJRR/runs

wait_for () {
  while ps -eo cmd | grep -q "[s]tage2_traj.train --config.*$1"; do sleep 120; done
  echo "[queue] training $1 finished"
}

wait_for config_vertex_line_xarm7.yaml
wait_for config_vertex_line_cobotta.yaml

for ROBOT in xarm7 cobotta; do
  CK=$R/rl_vertex_line_${ROBOT}_30M
  if [ ! -f "$CK/agent.pt" ]; then echo "[queue] SKIP $ROBOT: no agent.pt"; continue; fi
  echo "=== eval $ROBOT ==="
  $PY -u -m Yuan.IJRR.eval.eval_curve \
      --config Yuan/IJRR/stage2_traj/config_vertex_line_${ROBOT}.yaml \
      --ckpt "$CK" --n-tasks 2048 --swing-max-deg 0 --observe-curvature 0 \
      --out "$CK/eval_straight.json" 2>&1 | grep -vE '^\[LineDist\]'
done
echo "ALL MULTIARM WORK DONE"
