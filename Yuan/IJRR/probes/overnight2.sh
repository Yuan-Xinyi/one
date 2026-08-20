#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
SC=/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad
FU=/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify

# wait for the two rhonorm ablations
until [ -f "$FU/dirfrac_rhonorm_10k.npz" ] && [ -f "$FU/dirfrac_rhonorm_aprev_10k.npz" ]; do sleep 120; done

# ---- WAVE 1: action-space metrics + ensemble seeds ----------------------
PIDS=()
for v in dvmetric dhmetric; do
  python -m Yuan.IJRR.stage2_traj.train \
    --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_${v}.yaml \
    --out-dir Yuan/IJRR/runs/rl_dirfrac_${v}_30M > $FU/train_${v}.log 2>&1 &
  PIDS+=($!)
done
for s in seedb seedc; do
  python -m Yuan.IJRR.stage2_traj.train \
    --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_margobs.yaml \
    --out-dir Yuan/IJRR/runs/rl_dirfrac_margobs_${s}_30M > $FU/train_margobs_${s}.log 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"
for v in dvmetric dhmetric; do
  python $SC/dirfrac_eval_any.py config_line_cont_dirfrac_${v}.yaml \
    Yuan/IJRR/runs/rl_dirfrac_${v}_30M/agent.pt dirfrac_${v}_10k.npz ${v} \
    >> $FU/eval_wave1.log 2>&1
done
for s in seedb seedc; do
  python $SC/dirfrac_eval_any.py config_line_cont_dirfrac_margobs.yaml \
    Yuan/IJRR/runs/rl_dirfrac_margobs_${s}_30M/agent.pt dirfrac_margobs_${s}_10k.npz margobs-${s} \
    >> $FU/eval_wave1.log 2>&1
done
touch $FU/WAVE1_DONE

# ---- WAVE 2: distribution + capacity + horizon --------------------------
python $SC/pool_weakness.py > $FU/pool_weakness.log 2>&1
PIDS=()
python $SC/train_weighted.py \
  --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_margobs.yaml \
  --out-dir Yuan/IJRR/runs/rl_dirfrac_oversample_30M > $FU/train_oversample.log 2>&1 &
PIDS+=($!)
for v in h1024 g995; do
  python -m Yuan.IJRR.stage2_traj.train \
    --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_${v}.yaml \
    --out-dir Yuan/IJRR/runs/rl_dirfrac_${v}_30M > $FU/train_${v}.log 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"
python $SC/dirfrac_eval_any.py config_line_cont_dirfrac_margobs.yaml \
  Yuan/IJRR/runs/rl_dirfrac_oversample_30M/agent.pt dirfrac_oversample_10k.npz oversample \
  >> $FU/eval_wave2.log 2>&1
for v in h1024 g995; do
  python $SC/dirfrac_eval_any.py config_line_cont_dirfrac_${v}.yaml \
    Yuan/IJRR/runs/rl_dirfrac_${v}_30M/agent.pt dirfrac_${v}_10k.npz ${v} \
    >> $FU/eval_wave2.log 2>&1
done
touch $FU/WAVE2_DONE

# ---- WAVE 3: calibrated ensemble ---------------------------------------
python $SC/cal_ensemble.py > $FU/cal_ensemble.log 2>&1
touch $FU/ALL_DONE
