#!/bin/bash
# Budget curve for the mixed-tolerance recipe, two runs launched together:
#   960M from scratch            (config_line_cont_dirfrac_e8kXXL_conemix960)
#   480M checkpoint + 240M       (config_line_cont_dirfrac_e8kXXL_conemix480c240, --resume-from-ckpt)
# Each run is evaluated with conemix_eval.py (30 deg 10k, 5 deg 10k c5, 15 deg 2000).
#   usage: conemix_budget_stage.sh 960 | c240
set -u
export MKL_THREADING_LAYER=GNU
PY=/home/lqin/miniconda3/envs/one/bin/python
WT=/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
cd $WT
case "$1" in
  960)  CFG=config_line_cont_dirfrac_e8kXXL_conemix960.yaml; OUT=Yuan/IJRR/runs/rl_dirfrac_e8kXXL_conemix960; TAG=conemix960; EXTRA="" ;;
  c240) CFG=config_line_cont_dirfrac_e8kXXL_conemix480c240.yaml; OUT=Yuan/IJRR/runs/rl_dirfrac_e8kXXL_conemix480c240; TAG=conemix480c240;
        EXTRA="--resume-from-ckpt Yuan/IJRR/runs/rl_dirfrac_e8kXXL_conemix480/agent.pt" ;;
  *) echo "usage: $0 960|c240"; exit 2 ;;
esac
mkdir -p $OUT
echo "[$TAG] train start $(date)"
$PY -m Yuan.IJRR.stage2_traj.train --config Yuan/IJRR/stage2_traj/$CFG --out-dir $OUT $EXTRA > $OUT.log 2>&1
echo "[$TAG] train exit $? $(date)"; tail -2 $OUT.log
$PY Yuan/IJRR/probes/conemix_eval.py $OUT/agent.pt $CFG $TAG > Yuan/IJRR/runs/${TAG}_eval.log 2>&1
echo "[$TAG] eval exit $? $(date)"
grep -v Warning Yuan/IJRR/runs/${TAG}_eval.log | grep -v '^$' | tail -30
echo ${TAG}_DONE
