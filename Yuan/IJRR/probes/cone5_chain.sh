PY=/home/lqin/miniconda3/envs/one/bin/python
export MKL_THREADING_LAYER=GNU
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
sed 's/total_timesteps: 30000000/total_timesteps: 2000000/'   Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8k_cone5.yaml > /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_cone5.yaml
$PY -m Yuan.IJRR.stage2_traj.train --config /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_cone5.yaml --out-dir /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_cone5 > /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_cone5.log 2>&1
if [ ! -f /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_cone5/agent.pt ]; then echo SMOKE_FAILED; tail -5 /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_cone5.log; exit 1; fi
echo SMOKE_OK
$PY -m Yuan.IJRR.stage2_traj.train   --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8k_cone5.yaml   --out-dir Yuan/IJRR/runs/rl_dirfrac_e8k_cone5   > Yuan/IJRR/runs/rl_dirfrac_e8k_cone5.log 2>&1
echo "TRAIN_END $(tail -1 Yuan/IJRR/runs/rl_dirfrac_e8k_cone5.log)"
echo CONE5_DONE
