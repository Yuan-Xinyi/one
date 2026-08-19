#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
SC=/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad
FU=/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify
A=/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets
POOL_BATCH=1024 python "$SC/fam_unify.py" pool_straight > "$FU/pool_straight.log" 2>&1
touch "$FU/pool_straight_pass1.done"
# top-up cont arms for xarm7/cobotta once their trainings land
while [ ! -f "$A/all_cont.done" ]; do sleep 300; done
POOL_BATCH=1024 python "$SC/fam_unify.py" pool_straight >> "$FU/pool_straight.log" 2>&1
touch "$FU/pool_straight_all.done"
