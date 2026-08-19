#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
FU=/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify
until [ -f Yuan/IJRR/runs/rl_vertex_line_amax2_30M/agent.pt ] && [ -f Yuan/IJRR/runs/rl_vertex_line_amax4_30M/agent.pt ]; do sleep 120; done
if [ ! -f Yuan/IJRR/runs/rl_cont_dirfrac_30M/agent.pt ]; then
  python -m Yuan.IJRR.stage2_traj.train \
    --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac.yaml \
    --out-dir Yuan/IJRR/runs/rl_cont_dirfrac_30M > "$FU/train_dirfrac.log" 2>&1
fi
touch "$FU/dirfrac_train.done"
