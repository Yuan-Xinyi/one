#!/bin/bash
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
PY=/home/lqin/miniconda3/envs/one/bin/python
BEST_S=-1
for s in 0 1 2 3 4 5 6 7; do
  $PY -m Yuan.IJRR.eval.single_task_ppo --stage goexplore_env --run-dir single_task_ppo_v2_mf \
    --ge-generations 130 --ge-cell 0.025 --ge-k 30 --torch-seed $s \
    > Yuan/IJRR/runs/single_task_ppo_v2_mf/ge_try_s$s.log 2>&1
  F=$($PY -c "import numpy as np; print(int(np.load('Yuan/IJRR/runs/single_task_ppo_v2_mf/goexplore_env.npz')['frontier_depth']))")
  echo "TRY seed=$s frontier=$F"
  if [ "$F" -ge 45 ]; then BEST_S=$s; echo "SELECTED seed $s"; break; fi
done
if [ "$BEST_S" -lt 0 ]; then echo "NO-SEED-TOOK-OFF"; exit 1; fi
$PY -m Yuan.IJRR.eval.single_task_ppo --stage goexplore_env --run-dir single_task_ppo_v2_mf \
  --ge-generations 8000 --ge-stall 400 --ge-cell 0.025 --ge-k 30 --torch-seed $BEST_S \
  > Yuan/IJRR/runs/single_task_ppo_v2_mf/goexplore_env_final.log 2>&1
echo "LONG-RUN-DONE"
