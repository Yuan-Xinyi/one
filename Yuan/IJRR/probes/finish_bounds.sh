#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
A=/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets
python -m Yuan.IJRR.eval.line_bound --robot cobotta \
  --table "$A/fk_table_cobotta.npz" \
  --tasks "$A/tasks_pool_cobotta.npz" --n-tasks 10000 \
  --step 0.01 --n-dirs 96 --dir-pool 96 --n-try 16 --k-nn 200 \
  --chunk 4096 --witness "$A/witness_pool_cobotta.npz" \
  --out "$A/bound_pool_cobotta.npz" >> "$A/bound_pool_cobotta.log" 2>&1
echo DONE > "$A/all_bounds.done"
