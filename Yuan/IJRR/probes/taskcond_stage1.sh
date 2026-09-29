#!/bin/bash
# Stage 1 of the task-conditioned exploration: wait for run A (mixed cone),
# evaluate it, then train run C (generic curves + preview).
set -u
export MKL_THREADING_LAYER=GNU
PY=/home/lqin/miniconda3/envs/one/bin/python
WT=/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
cd $WT
LOG_A=Yuan/IJRR/runs/rl_dirfrac_e8kXXL_conemix.log
until grep -q 'done, ckpt' $LOG_A; do sleep 60; done
echo "[stage1] run A finished $(date)"
$PY Yuan/IJRR/probes/conemix_eval.py > Yuan/IJRR/runs/conemix_eval.log 2>&1
echo "[stage1] conemix_eval exit $? $(date)"
grep -v Warning Yuan/IJRR/runs/conemix_eval.log | grep -v '^$' | tail -30
mkdir -p Yuan/IJRR/runs/rl_dirfrac_e8kXXL_curves
$PY -m Yuan.IJRR.stage2_traj.train --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_curves.yaml \
    --out-dir Yuan/IJRR/runs/rl_dirfrac_e8kXXL_curves > Yuan/IJRR/runs/rl_dirfrac_e8kXXL_curves.log 2>&1
echo "[stage1] train C exit $? $(date)"
tail -2 Yuan/IJRR/runs/rl_dirfrac_e8kXXL_curves.log
echo STAGE1_DONE
