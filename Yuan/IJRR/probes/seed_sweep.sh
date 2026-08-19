#!/bin/bash
# Basin lottery: 8 seeds x 2M steps of the vanilla recipe, 2 at a time.
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
PY=/home/lqin/miniconda3/envs/one/bin/python
for pair in "101 102" "103 104" "105 106" "107 108"; do
  for s in $pair; do
    d=Yuan/IJRR/runs/single_task_ppo_v2_seed$s
    mkdir -p $d && cp Yuan/IJRR/runs/single_task_ppo_v2/task.npz $d/
    $PY -m Yuan.IJRR.eval.single_task_ppo --stage train --run-dir single_task_ppo_v2_seed$s \
      --torch-seed $s --total-steps 2000000 --eval-every 200000 \
      > $d/train_stdout.log 2>&1 &
  done
  wait
  for s in $pair; do
    echo "seed $s: $(grep 'eval @' Yuan/IJRR/runs/single_task_ppo_v2_seed$s/train_stdout.log | awk '{print $5}' | sort -g | tail -1) (peak)"
  done
done
echo ALL-SEEDS-DONE
