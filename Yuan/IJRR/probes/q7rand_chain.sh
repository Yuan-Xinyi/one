#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
FU=/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify
if [ ! -f Yuan/IJRR/runs/rl_vertex_line_q7rand_30M/agent.pt ]; then
  python -m Yuan.IJRR.stage2_traj.train \
    --config Yuan/IJRR/stage2_traj/config_vertex_line_q7rand.yaml \
    --out-dir Yuan/IJRR/runs/rl_vertex_line_q7rand_30M \
    > "$FU/train_q7rand.log" 2>&1
fi
touch "$FU/q7rand_train.done"
