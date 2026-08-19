#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
A=/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets
python -m Yuan.IJRR.eval.line_bound --robot xarm7 \
  --table "$A/fk_table_xarm7.npz" \
  --tasks "$A/tasks_pool_xarm7.npz" --n-tasks 10000 \
  --step 0.01 --n-dirs 96 --dir-pool 192 --n-try 32 --k-nn 400 \
  --chunk 4096 --witness "$A/witness_pool_xarm7.npz" \
  --out "$A/bound_pool_xarm7_v2.npz" > "$A/bound_pool_xarm7_v2.log" 2>&1
echo DONE > "$A/xarm7_bound_v2.done"
