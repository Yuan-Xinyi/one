#!/bin/bash
# Four-family unification master chain: strictly serial on the GPU.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
A=/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets
FU=/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify
V3=/home/lqin/one/Yuan/IJRR/runs/selector_ood/v3_k32_fixedgate
SC=/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad
mkdir -p "$FU"
export FAM_BATCH=${FAM_BATCH:-8192}
export FAM_CHUNK=${FAM_CHUNK:-32768}

# ---- 1. selector side runs immediately (no dependency on the bound march);
#         the cobotta pool bound (finish_bounds.sh) shares the GPU meanwhile.
# relabel -> retrain -> report -> refresh seeds
if [ ! -f "$V3/labels.done" ]; then
  python "$SC/fam_unify.py" relabel > "$FU/relabel.log" 2>&1
fi
if [ ! -f "$V3/rankers.pt" ]; then
  python "$SC/fam_unify.py" retrain > "$FU/retrain.log" 2>&1
fi
if [ ! -f "$FU/sel_report.npz" ]; then
  python "$SC/fam_unify.py" selreport > "$FU/selreport.log" 2>&1
fi
if [ ! -f "$FU/seltasks.done" ]; then
  python "$SC/fam_unify.py" seltasks > "$FU/seltasks.log" 2>&1
  touch "$FU/seltasks.done"
fi

# ---- re-witness + re-bound arc / serpentine (FR3 selector sets) -----------
while [ ! -f "$A/all_bounds.done" ]; do sleep 120; done
for S in sel_arc sel_serpentine; do
  if [ ! -f "$FU/rebound_$S.done" ]; then
    [ -f "$A/witness_$S.npz" ] && mv "$A/witness_$S.npz" "$A/witness_$S.v2bak.npz"
    python "$SC/witness_gen2.py" "$S" > "$A/wit_${S}_v3.log" 2>&1
    [ -f "$A/bound_$S.npz" ] && mv "$A/bound_$S.npz" "$A/bound_$S.v2bak.npz"
    python -m Yuan.IJRR.eval.line_bound --robot fr3 \
      --tasks "$A/tasks_$S.npz" --n-tasks 2500 \
      --step 0.01 --n-dirs 96 --dir-pool 96 --n-try 16 --k-nn 200 \
      --chunk 4096 --witness "$A/witness_$S.npz" \
      --out "$A/bound_$S.npz" > "$A/bound_${S}_v3.log" 2>&1
    touch "$FU/rebound_$S.done"
  fi
done

# ---- 4. FR3 curved controller rollouts ------------------------------------
python "$SC/fam_unify.py" fr3_curved > "$FU/fr3_curved.log" 2>&1

# ---- cross-critic straight 10k (unchanged straight protocol) --------------
if [ ! -f "$A/crosscritic_fr3_to_xarm7.npz" ]; then
  python -m Yuan.IJRR.eval.horizon_ladder --robot xarm7 \
    --arms classical,vlook --n-tasks 10000 --sub 2 \
    --vlook-ckpt Yuan/IJRR/runs/rl_vertex_line_30M \
    --out "$A/crosscritic_fr3_to_xarm7.npz" > "$A/cc_f2x.log" 2>&1
fi
if [ ! -f "$A/crosscritic_xarm7_to_fr3.npz" ]; then
  python -m Yuan.IJRR.eval.horizon_ladder --robot fr3 \
    --arms classical,vlook --n-tasks 10000 --sub 2 \
    --vlook-ckpt Yuan/IJRR/runs/rl_vertex_line_xarm7_30M \
    --out "$A/crosscritic_xarm7_to_fr3.npz" > "$A/cc_x2f.log" 2>&1
fi
touch "$A/all_evals.done"

# ---- continuous-PPO trainings + straight evals ------------------------------
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
[ ! -f "$A/cont_fr3_10k.npz" ] && \
  python "$SC/eval_cont2.py" fr3 Yuan/IJRR/runs/rl_cont_sqent_30M \
    > "$A/cont_fr3.log" 2>&1
[ ! -f "$A/cont_xarm7_10k.npz" ] && \
  python "$SC/eval_cont2.py" xarm7 Yuan/IJRR/runs/rl_cont_sqent_xarm7_30M \
    > "$A/cont_xarm7.log" 2>&1
[ ! -f "$A/cont_cobotta_10k.npz" ] && \
  python "$SC/eval_cont2.py" cobotta Yuan/IJRR/runs/rl_cont_sqent_cobotta_30M \
    > "$A/cont_cobotta.log" 2>&1
touch "$A/all_cont.done"

# ---- 6. xarm7 / cobotta curved sets + rollouts ------------------------------
python "$SC/fam_unify.py" xacb_tasks > "$FU/xacb_tasks.log" 2>&1
python "$SC/fam_unify.py" xacb_roll  > "$FU/xacb_roll.log" 2>&1
# cont arm may have been missing on FR3 curved runs before training finished
python "$SC/fam_unify.py" fr3_cont_curved > "$FU/fr3_cont_curved.log" 2>&1

# ---- 7. witnesses + bounds for the 6 new curved sets ------------------------
for R in xarm7 cobotta; do
  for F in arc serpentine nonplanar; do
    S="selx_${F}_${R}"
    if [ ! -f "$A/witness_$S.npz" ]; then
      python "$SC/witness_gen2.py" "$S" > "$A/wit_$S.log" 2>&1
    fi
    if [ ! -f "$A/bound_$S.npz" ]; then
      python -m Yuan.IJRR.eval.line_bound --robot "$R" \
        --table "$A/fk_table_${R}.npz" \
        --tasks "$A/tasks_$S.npz" --n-tasks 2500 \
        --step 0.01 --n-dirs 96 --dir-pool 96 --n-try 16 --k-nn 200 \
        --chunk 4096 --witness "$A/witness_$S.npz" \
        --out "$A/bound_$S.npz" > "$A/bound_$S.log" 2>&1
    fi
  done
done

touch "$FU/fam_unify_all.done"
echo ALL DONE
