#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate one
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
SC=/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad
FU=/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify
PIDS=()
python -m Yuan.IJRR.stage2_traj.train \
  --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_margobs.yaml \
  --out-dir Yuan/IJRR/runs/rl_dirfrac_margobs_30M > $FU/train_margobs.log 2>&1 &
PIDS+=($!)
for s in b c; do
  python -m Yuan.IJRR.stage2_traj.train \
    --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_aprev.yaml \
    --out-dir Yuan/IJRR/runs/rl_dirfrac_aprev_seed${s}_30M > $FU/train_aprev_${s}.log 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"
python $SC/dirfrac_eval_any.py config_line_cont_dirfrac_margobs.yaml \
  Yuan/IJRR/runs/rl_dirfrac_margobs_30M/agent.pt dirfrac_margobs_10k.npz margobs \
  >> $FU/eval_round3.log 2>&1
for s in b c; do
  python $SC/dirfrac_eval_any.py config_line_cont_dirfrac_aprev.yaml \
    Yuan/IJRR/runs/rl_dirfrac_aprev_seed${s}_30M/agent.pt dirfrac_aprev_seed${s}_10k.npz aprev-seed$s \
    >> $FU/eval_round3.log 2>&1
done
grep -E ": 0\.|improved" $FU/eval_round3.log
