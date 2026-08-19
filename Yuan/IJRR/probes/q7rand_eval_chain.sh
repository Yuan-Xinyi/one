#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
FU=/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify
SC=/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad
while [ ! -f "$FU/q7rand_train.done" ]; do sleep 120; done
python "$SC/q7rand_eval.py" > "$FU/q7rand_eval.log" 2>&1
touch "$FU/q7rand_all.done"
