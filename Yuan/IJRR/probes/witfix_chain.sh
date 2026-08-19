#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
A=/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets
SC=/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad
for S in pool_fr3 sel_serpentine sel_nonplanar; do
  if [ ! -f "$A/witarms_$S.done" ]; then
    python "$SC/witness_arms.py" "$S" > "$A/witarms_$S.log" 2>&1
    touch "$A/witarms_$S.done"
  fi
done
for S in pool_fr3 sel_serpentine sel_nonplanar; do
  if [ ! -f "$A/rebound2_$S.done" ]; then
    mv "$A/bound_$S.npz" "$A/bound_$S.prev.npz"
    T=(); NT=10000
    [ "$S" != "pool_fr3" ] && NT=2500
    python -m Yuan.IJRR.eval.line_bound --robot fr3 \
      --tasks "$A/tasks_$S.npz" --n-tasks $NT \
      --step 0.01 --n-dirs 96 --dir-pool 192 --n-try 32 --k-nn 400 \
      --chunk 4096 --witness "$A/witness_$S.npz" \
      --out "$A/bound_$S.npz" > "$A/bound_${S}_v5.log" 2>&1
    touch "$A/rebound2_$S.done"
  fi
done
touch "$A/witfix_all.done"
