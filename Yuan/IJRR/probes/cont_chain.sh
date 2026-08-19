#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
A=/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets
SC=/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/eval_cont2.py
while [ ! -f "$A/all_evals.done" ]; do sleep 120; done
if [ ! -f Yuan/IJRR/runs/rl_cont_sqent_xarm7_30M/agent.pt ]; then
  python -m Yuan.IJRR.stage2_traj.train \
    --config Yuan/IJRR/stage2_traj/config_line_cont_sqent_xarm7.yaml \
    --out-dir Yuan/IJRR/runs/rl_cont_sqent_xarm7_30M \
    > "$A/train_cont_xarm7.log" 2>&1
fi
if [ ! -f Yuan/IJRR/runs/rl_cont_sqent_cobotta_30M/agent.pt ]; then
  python -m Yuan.IJRR.stage2_traj.train \
    --config Yuan/IJRR/stage2_traj/config_line_cont_sqent_cobotta.yaml \
    --out-dir Yuan/IJRR/runs/rl_cont_sqent_cobotta_30M \
    > "$A/train_cont_cobotta.log" 2>&1
fi
python "$SC" fr3 Yuan/IJRR/runs/rl_cont_sqent_30M > "$A/cont_fr3.log" 2>&1
python "$SC" xarm7 Yuan/IJRR/runs/rl_cont_sqent_xarm7_30M > "$A/cont_xarm7.log" 2>&1
python "$SC" cobotta Yuan/IJRR/runs/rl_cont_sqent_cobotta_30M > "$A/cont_cobotta.log" 2>&1
echo DONE > "$A/all_cont.done"
