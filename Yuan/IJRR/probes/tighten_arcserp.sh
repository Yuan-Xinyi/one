#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
A=/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets
SC=/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad
for S in sel_arc sel_serpentine; do
  mv "$A/witness_$S.npz" "$A/witness_$S.v3bak.npz"
  STARTS=3 python "$SC/witness_gen2.py" "$S" > "$A/wit_${S}_v4.log" 2>&1
  mv "$A/bound_$S.npz" "$A/bound_$S.v3bak.npz"
  python -m Yuan.IJRR.eval.line_bound --robot fr3 \
    --tasks "$A/tasks_$S.npz" --n-tasks 2500 \
    --step 0.01 --n-dirs 96 --dir-pool 192 --n-try 32 --k-nn 400 \
    --chunk 4096 --witness "$A/witness_$S.npz" \
    --out "$A/bound_$S.npz" > "$A/bound_${S}_v4.log" 2>&1
done
echo DONE > "$A/tighten_arcserp.done"
