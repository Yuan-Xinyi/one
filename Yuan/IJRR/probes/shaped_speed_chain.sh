#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
SC=/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad
FU=/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify
PIDS=()
for v in shaped speed; do
  python -m Yuan.IJRR.stage2_traj.train \
    --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_${v}.yaml \
    --out-dir Yuan/IJRR/runs/rl_dirfrac_${v}_30M > $FU/train_${v}.log 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"
for v in shaped speed; do
  python $SC/dirfrac_eval_any.py config_line_cont_dirfrac_${v}.yaml \
    Yuan/IJRR/runs/rl_dirfrac_${v}_30M/agent.pt dirfrac_${v}_10k.npz ${v} \
    >> $FU/eval_shaped_speed.log 2>&1
done
grep -E ": 0\.|improved" $FU/eval_shaped_speed.log
