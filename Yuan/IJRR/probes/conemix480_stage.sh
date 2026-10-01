#!/bin/bash
# Budget test for the mixed-tolerance recipe: train the 480M-step variant of
# run A (config_line_cont_dirfrac_e8kXXL_conemix480), then evaluate it with
# the same protocol as run A (conemix_eval: 30 deg 10k paired with the
# flagship, 5 deg 10k c5 protocol paired with the 5-deg model, 15 deg 2000).
set -u
export MKL_THREADING_LAYER=GNU
PY=/home/lqin/miniconda3/envs/one/bin/python
WT=/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
cd $WT
OUT=Yuan/IJRR/runs/rl_dirfrac_e8kXXL_conemix480
mkdir -p $OUT
echo "[conemix480] train start $(date)"
$PY -m Yuan.IJRR.stage2_traj.train --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_conemix480.yaml \
    --out-dir $OUT > $OUT.log 2>&1
echo "[conemix480] train exit $? $(date)"; tail -2 $OUT.log
$PY Yuan/IJRR/probes/conemix_eval.py $OUT/agent.pt config_line_cont_dirfrac_e8kXXL_conemix480.yaml conemix480 \
    > Yuan/IJRR/runs/conemix480_eval.log 2>&1
echo "[conemix480] eval exit $? $(date)"
grep -v Warning Yuan/IJRR/runs/conemix480_eval.log | grep -v '^$' | tail -30
echo CONEMIX480_DONE
