#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
FU=/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify
SC=/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad
if [ ! -f Yuan/IJRR/runs/rl_vertex_line_bscale10_30M/agent.pt ]; then
  python -m Yuan.IJRR.stage2_traj.train \
    --config Yuan/IJRR/stage2_traj/config_vertex_line_bscale10.yaml \
    --out-dir Yuan/IJRR/runs/rl_vertex_line_bscale10_30M \
    > "$FU/train_bscale10.log" 2>&1
fi
python "$SC/bscale10_eval.py" > "$FU/bscale10_eval.log" 2>&1
touch "$FU/bscale10_all.done"
