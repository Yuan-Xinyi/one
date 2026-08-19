#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
SC=/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad
FU=/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify
python -m Yuan.IJRR.stage2_traj.train \
  --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_v3.yaml \
  --out-dir Yuan/IJRR/runs/rl_dirfrac_v3_30M > $FU/train_v3.log 2>&1
python $SC/dirfrac_eval_any.py config_line_cont_dirfrac_v3.yaml \
  Yuan/IJRR/runs/rl_dirfrac_v3_30M/agent.pt dirfrac_v3_10k.npz v3 \
  > $FU/eval_v3.log 2>&1
tail -4 $FU/eval_v3.log
