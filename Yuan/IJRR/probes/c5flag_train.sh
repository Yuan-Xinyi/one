PY=/home/lqin/miniconda3/envs/one/bin/python
export MKL_THREADING_LAYER=GNU
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
sed 's/total_timesteps: 240000000/total_timesteps: 2000000/'   Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_cone5.yaml > /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_c5flag.yaml
$PY -m Yuan.IJRR.stage2_traj.train --config /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_c5flag.yaml --out-dir /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_c5flag > /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_c5flag.log 2>&1
if [ ! -f /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_c5flag/agent.pt ]; then echo SMOKE_FAILED; tail -4 /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_c5flag.log; exit 1; fi
EPLEN=$(grep -oE "episode/length_mean.: [0-9.]+" /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_c5flag/train.log | tail -1 | grep -oE "[0-9.]+$")
echo "SMOKE_OK eplen=$EPLEN"
if $PY -c "exit(0 if float('$EPLEN') > 5 else 1)"; then true; else echo SPAWN_DEATH; exit 1; fi
$PY -m Yuan.IJRR.stage2_traj.train   --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_cone5.yaml   --out-dir Yuan/IJRR/runs/rl_dirfrac_e8kXXL_cone5   > Yuan/IJRR/runs/rl_dirfrac_e8kXXL_cone5.log 2>&1
if ! grep -q "\[train\] done" Yuan/IJRR/runs/rl_dirfrac_e8kXXL_cone5.log; then
  echo TRAIN_FAILED; tail -3 Yuan/IJRR/runs/rl_dirfrac_e8kXXL_cone5.log; exit 1
fi
echo FLAG_TRAIN_OK
