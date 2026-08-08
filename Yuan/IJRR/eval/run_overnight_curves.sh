#!/usr/bin/env bash
# Overnight queue for the curve question.
#
# Two trainings are already in flight when this starts (the continuous-action
# curve baseline and the curvature-blind curve policy). This waits for them,
# scores them, then runs one replicate of each of the two policies that the
# transfer matrix compares, because that comparison is currently one training
# run against one training run and the paired bootstrap only covers task noise,
# not the variance between training runs.
#
# Everything is scored by eval_curve.py, which recomputes the classical arm
# inside each cell, so every number is a ratio on the same tasks.
set -uo pipefail
cd /home/lqin/one
PY=/home/lqin/miniconda3/envs/one/bin/python
R=Yuan/IJRR/runs
T=/home/lqin/one/Yuan/IJRR/runs/_transfer
LOG=/tmp/claude-1000/-home-lqin-one-Yuan--claude-worktrees-lucid-shtern-221bae/171093ae-57d7-467d-9173-1081856067dd/scratchpad
N=2048
mkdir -p "$T"

wait_for () {   # out-dir substring
  while ps -eo cmd | grep -q "[-]-out-dir $1\$"; do sleep 60; done
  echo "[queue] $1 no longer running"
}

score () {      # name ckpt swing observe_curvature
  if [ ! -f "$2/agent.pt" ]; then echo "[queue] SKIP $1 (no agent.pt)"; return; fi
  echo "=== score $1 : $2 swing=$3 obs_curv=$4 ==="
  $PY -u -m Yuan.IJRR.eval.eval_curve \
      --config Yuan/IJRR/stage2_traj/config_vertex.yaml \
      --ckpt "$2" --n-tasks "$N" \
      --swing-max-deg "$3" --observe-curvature "$4" \
      --out "$T/$1.json" 2>&1 | grep -vE '^\[LineDist\]'
}

train () {      # config out-dir logname
  echo "=== train $2 ==="
  $PY -u -m Yuan.IJRR.stage2_traj.train --config "$1" --out-dir "$2" \
      > "$LOG/$3.out" 2>&1
  echo "[queue] train $2 exit=$?"
}

echo "########## waiting for the two in-flight trainings ##########"
wait_for "$R/rl_curve_cont_30M"
wait_for "$R/rl_vertex_curve_noobs_30M"

echo "########## scoring them ##########"
score cont_on_curve       "$R/rl_curve_cont_30M"          30 1
score curveblind_on_curve "$R/rl_vertex_curve_noobs_30M"  30 0
score curveblind_on_straight "$R/rl_vertex_curve_noobs_30M" 0 0

echo "########## replicates ##########"
train Yuan/IJRR/stage2_traj/config_vertex_line.yaml "$R/rl_vertex_line_30M_s2" line_s2
score line_s2_on_straight "$R/rl_vertex_line_30M_s2"  0 0
score line_s2_on_curve    "$R/rl_vertex_line_30M_s2" 30 0

train Yuan/IJRR/stage2_traj/config_vertex.yaml "$R/rl_vertex_30M_s2" curve_s2
score curve_s2_on_straight "$R/rl_vertex_30M_s2"  0 1
score curve_s2_on_curve    "$R/rl_vertex_30M_s2" 30 1

echo "########## switching rate against the 20 Hz control rate ##########"
SW=Yuan/IJRR/stage2_traj/config_vertex.yaml
$PY -u -m Yuan.IJRR.eval.switching_rate --config $SW \
    --ckpt "$R/rl_vertex_line_30M" --n-tasks 2048 --swing-max-deg 0  --observe-curvature 0
$PY -u -m Yuan.IJRR.eval.switching_rate --config $SW \
    --ckpt "$R/rl_vertex_line_30M" --n-tasks 2048 --swing-max-deg 30 --observe-curvature 0
$PY -u -m Yuan.IJRR.eval.switching_rate --config $SW \
    --ckpt "$R/rl_vertex_30M"      --n-tasks 2048 --swing-max-deg 30 --observe-curvature 1

echo "ALL OVERNIGHT WORK DONE"
