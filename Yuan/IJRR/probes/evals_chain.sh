#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
A=/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets
while [ ! -f "$A/all_bounds.done" ]; do sleep 60; done
python -m Yuan.IJRR.eval.horizon_ladder --robot fr3 \
  --arms classical,myopic,vertex,vlook --n-tasks 1024 --sub 2 \
  --cfg-override Yuan/IJRR/stage2_traj/config_vertex_line_amax1.yaml \
  --ckpt-override Yuan/IJRR/runs/rl_vertex_line_amax1_30M \
  --vlook-ckpt Yuan/IJRR/runs/rl_vertex_line_amax1_30M \
  --out "$A/actgap_amax1.npz" > "$A/actgap_amax1.log" 2>&1
python -m Yuan.IJRR.eval.horizon_ladder --robot fr3 \
  --arms classical,myopic,vertex,vlook --n-tasks 1024 --sub 2 \
  --cfg-override Yuan/IJRR/stage2_traj/config_vertex_line_dt01.yaml \
  --ckpt-override Yuan/IJRR/runs/rl_vertex_line_dt01_30M \
  --vlook-ckpt Yuan/IJRR/runs/rl_vertex_line_dt01_30M \
  --out "$A/actgap_dt01.npz" > "$A/actgap_dt01.log" 2>&1
python -m Yuan.IJRR.eval.horizon_ladder --robot xarm7 \
  --arms classical,vlook --n-tasks 10000 --sub 2 \
  --vlook-ckpt Yuan/IJRR/runs/rl_vertex_line_30M \
  --out "$A/crosscritic_fr3_to_xarm7.npz" > "$A/cc_f2x.log" 2>&1
python -m Yuan.IJRR.eval.horizon_ladder --robot fr3 \
  --arms classical,vlook --n-tasks 10000 --sub 2 \
  --vlook-ckpt Yuan/IJRR/runs/rl_vertex_line_xarm7_30M \
  --out "$A/crosscritic_xarm7_to_fr3.npz" > "$A/cc_x2f.log" 2>&1
echo DONE > "$A/all_evals.done"
