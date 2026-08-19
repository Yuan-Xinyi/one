#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
A=/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets
SC=/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/witness_gen2.py
declare -A ROB=( [pool_fr3]=fr3 [pool_xarm7]=xarm7 [pool_cobotta]=cobotta [sel_straight]=fr3 [sel_arc]=fr3 [sel_serpentine]=fr3 [sel_nonplanar]=fr3 )
declare -A NT=( [pool_fr3]=10000 [pool_xarm7]=10000 [pool_cobotta]=10000 [sel_straight]=2500 [sel_arc]=2500 [sel_serpentine]=2500 [sel_nonplanar]=2500 )
for S in pool_fr3 sel_straight sel_arc sel_serpentine sel_nonplanar pool_xarm7 pool_cobotta; do
  if [ ! -f "$A/witness_$S.npz" ]; then
    python "$SC" "$S" > "$A/wit_$S.log" 2>&1
  fi
  if [ ! -f "$A/bound_$S.npz" ]; then
    TBL=()
    [ "${ROB[$S]}" = xarm7 ] && TBL=(--table "$A/fk_table_xarm7.npz")
    [ "${ROB[$S]}" = cobotta ] && TBL=(--table "$A/fk_table_cobotta.npz")
    python -m Yuan.IJRR.eval.line_bound --robot "${ROB[$S]}" "${TBL[@]}" \
      --tasks "$A/tasks_$S.npz" --n-tasks "${NT[$S]}" \
      --step 0.01 --n-dirs 96 --dir-pool 96 --n-try 16 --k-nn 200 \
      --chunk 4096 --witness "$A/witness_$S.npz" \
      --out "$A/bound_$S.npz" > "$A/bound_$S.log" 2>&1
  fi
done
echo DONE > "$A/all_bounds.done"
