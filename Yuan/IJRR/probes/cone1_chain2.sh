PY=/home/lqin/miniconda3/envs/one/bin/python
export MKL_THREADING_LAYER=GNU
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
rm -rf /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_cone1 Yuan/IJRR/runs/rl_dirfrac_e8k_cone1
$PY -m Yuan.IJRR.stage2_traj.train --config /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_cone1.yaml --out-dir /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_cone1 > /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_cone1.log 2>&1
if [ ! -f /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_cone1/agent.pt ]; then echo SMOKE_FAILED; tail -4 /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_cone1.log; exit 1; fi
EPLEN=$(grep -oE "episode/length_mean.: [0-9.]+" /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_cone1/train.log | tail -1 | grep -oE "[0-9.]+$")
echo "SMOKE_OK eplen=$EPLEN"
if $PY -c "exit(0 if float('$EPLEN') > 5 else 1)"; then true; else echo SPAWN_DEATH_STILL; exit 1; fi
$PY -m Yuan.IJRR.stage2_traj.train   --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8k_cone1.yaml   --out-dir Yuan/IJRR/runs/rl_dirfrac_e8k_cone1   > Yuan/IJRR/runs/rl_dirfrac_e8k_cone1.log 2>&1
if ! grep -q "\[train\] done" Yuan/IJRR/runs/rl_dirfrac_e8k_cone1.log; then
  echo TRAIN_FAILED; tail -3 Yuan/IJRR/runs/rl_dirfrac_e8k_cone1.log; exit 1
fi
echo TRAIN_OK
$PY /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/cone1_retrain_eval.py > /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/cone1_re.log 2>&1; tail -3 /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/cone1_re.log
$PY /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/cone1_sel.py > /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/cone1_sel.log 2>&1; tail -3 /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/cone1_sel.log
echo CONE1_DONE
