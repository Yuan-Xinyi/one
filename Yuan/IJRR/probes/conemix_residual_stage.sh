#!/bin/bash
# Residual-gap candidates for the mixed-tolerance recipe, run one after the
# other at 480M (the budget where the curve flattened), each evaluated with
# conemix_eval.py (30 deg 10k, 5 deg 10k c5, 15 deg 2000):
#   grpnorm  per-tolerance-group return normalisation + log_std affine in the tolerance
#   w3072    actor/critic width 2048 -> 3072
set -u
export MKL_THREADING_LAYER=GNU
PY=/home/lqin/miniconda3/envs/one/bin/python
WT=/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
cd $WT
for tag in grpnorm w3072; do
  CFG=config_line_cont_dirfrac_e8kXXL_conemix480_$tag.yaml
  OUT=Yuan/IJRR/runs/rl_dirfrac_e8kXXL_conemix480_$tag
  TAG=conemix480_$tag
  mkdir -p $OUT
  echo "[$TAG] train start $(date)"
  $PY -m Yuan.IJRR.stage2_traj.train --config Yuan/IJRR/stage2_traj/$CFG --out-dir $OUT > $OUT.log 2>&1
  echo "[$TAG] train exit $? $(date)"; tail -2 $OUT.log
  $PY Yuan/IJRR/probes/conemix_eval.py $OUT/agent.pt $CFG $TAG > Yuan/IJRR/runs/${TAG}_eval.log 2>&1
  echo "[$TAG] eval exit $? $(date)"
  grep -v Warning Yuan/IJRR/runs/${TAG}_eval.log | grep -v '^$' | tail -30
  echo ${TAG}_DONE
done
echo RESIDUAL_STAGE_DONE
