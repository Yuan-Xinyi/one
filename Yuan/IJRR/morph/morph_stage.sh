#!/bin/bash
# Morphology-general queue: train on FR3 + xArm7 (+ variants), evaluate;
# then the leave-one-arm-out run (xArm7 family only), evaluate on FR3.
set -u
export MKL_THREADING_LAYER=GNU
PY=/home/lqin/miniconda3/envs/one/bin/python
WT=/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
cd $WT
mkdir -p Yuan/IJRR/runs/morph_all Yuan/IJRR/runs/morph_xarm_only
echo "[morph-stage] train all $(date)"
$PY -m Yuan.IJRR.morph.train_morph --config Yuan/IJRR/morph/config_morph_all.yaml \
    --out-dir Yuan/IJRR/runs/morph_all > Yuan/IJRR/runs/morph_all.log 2>&1
echo "[morph-stage] train all exit $? $(date)"; tail -2 Yuan/IJRR/runs/morph_all.log
$PY Yuan/IJRR/morph/morph_eval.py all Yuan/IJRR/runs/morph_all/agent.pt Yuan/IJRR/morph/config_morph_all.yaml \
    > Yuan/IJRR/runs/morph_all_eval.log 2>&1
echo "[morph-stage] eval all exit $? $(date)"; grep -v Warning Yuan/IJRR/runs/morph_all_eval.log | grep -v '^$' | tail -30
echo "[morph-stage] train xarm_only $(date)"
$PY -m Yuan.IJRR.morph.train_morph --config Yuan/IJRR/morph/config_morph_xarm_only.yaml \
    --out-dir Yuan/IJRR/runs/morph_xarm_only > Yuan/IJRR/runs/morph_xarm_only.log 2>&1
echo "[morph-stage] train xarm_only exit $? $(date)"; tail -2 Yuan/IJRR/runs/morph_xarm_only.log
$PY Yuan/IJRR/morph/morph_eval.py xarm_only Yuan/IJRR/runs/morph_xarm_only/agent.pt Yuan/IJRR/morph/config_morph_xarm_only.yaml \
    > Yuan/IJRR/runs/morph_xarm_only_eval.log 2>&1
echo "[morph-stage] eval xarm_only exit $? $(date)"; grep -v Warning Yuan/IJRR/runs/morph_xarm_only_eval.log | grep -v '^$' | tail -30
echo MORPH_STAGE_DONE
