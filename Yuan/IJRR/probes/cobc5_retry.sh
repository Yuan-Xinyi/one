PY=/home/lqin/miniconda3/envs/one/bin/python
export MKL_THREADING_LAYER=GNU
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
$PY -m Yuan.IJRR.stage2_traj.train   --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_cobotta_e8k_cone5.yaml   --out-dir Yuan/IJRR/runs/rl_dirfrac_cobotta_e8k_cone5   > Yuan/IJRR/runs/rl_dirfrac_cobotta_e8k_cone5.log 2>&1
if ! grep -q "\[train\] done" Yuan/IJRR/runs/rl_dirfrac_cobotta_e8k_cone5.log; then
  echo TRAIN_FAILED; tail -3 Yuan/IJRR/runs/rl_dirfrac_cobotta_e8k_cone5.log; exit 1
fi
echo TRAIN_OK
$PY /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/cob_branch.py > /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/cob_branch.log 2>&1
tail -8 /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/cob_branch.log
echo COBC5_RETRY_DONE
