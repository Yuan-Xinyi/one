#!/bin/bash
# Stage 2: evaluate run C (curves) in the background while run D (task
# conditioned: cone + tools + curves) trains; then evaluate run D.
set -u
export MKL_THREADING_LAYER=GNU
PY=/home/lqin/miniconda3/envs/one/bin/python
WT=/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
cd $WT
mkdir -p Yuan/IJRR/runs/rl_dirfrac_e8kXXL_taskcond
$PY -m Yuan.IJRR.stage2_traj.train --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_taskcond.yaml \
    --out-dir Yuan/IJRR/runs/rl_dirfrac_e8kXXL_taskcond > Yuan/IJRR/runs/rl_dirfrac_e8kXXL_taskcond.log 2>&1 &
TRAIN_PID=$!
echo "[stage2] train D started pid $TRAIN_PID $(date)"
PARTS=1234 $PY Yuan/IJRR/probes/curves_eval.py curves > Yuan/IJRR/runs/curves_eval.log 2>&1
echo "[stage2] curves_eval(C) exit $? $(date)"
grep -v Warning Yuan/IJRR/runs/curves_eval.log | grep -v '^$' | tail -40
wait $TRAIN_PID
echo "[stage2] train D exit $? $(date)"
tail -2 Yuan/IJRR/runs/rl_dirfrac_e8kXXL_taskcond.log
D=Yuan/IJRR/runs/rl_dirfrac_e8kXXL_taskcond/agent.pt
C=config_line_cont_dirfrac_e8kXXL_taskcond.yaml
$PY Yuan/IJRR/probes/conemix_eval.py $D $C taskcond > Yuan/IJRR/runs/taskcond_cone_eval.log 2>&1
echo "[stage2] cone eval(D) exit $? $(date)"
grep -v Warning Yuan/IJRR/runs/taskcond_cone_eval.log | grep -v '^$' | tail -30
PARTS=1234 $PY Yuan/IJRR/probes/curves_eval.py taskcond $C $D > Yuan/IJRR/runs/taskcond_curves_eval.log 2>&1
echo "[stage2] curves eval(D) exit $? $(date)"
grep -v Warning Yuan/IJRR/runs/taskcond_curves_eval.log | grep -v '^$' | tail -40
$PY Yuan/IJRR/probes/tool_sweep.py taskcond $C $D > Yuan/IJRR/runs/taskcond_tool_sweep.log 2>&1
echo "[stage2] tool sweep(D) exit $? $(date)"
grep -v Warning Yuan/IJRR/runs/taskcond_tool_sweep.log | tail -30
echo STAGE2_DONE
