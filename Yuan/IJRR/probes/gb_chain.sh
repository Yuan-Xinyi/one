PY=/home/lqin/miniconda3/envs/one/bin/python
export MKL_THREADING_LAYER=GNU
cd /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05
for v in gate beta gb; do
  sed 's/total_timesteps: 30000000/total_timesteps: 2000000/'     Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8k_$v.yaml > /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_$v.yaml
  $PY -m Yuan.IJRR.stage2_traj.train --config /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_$v.yaml     --out-dir /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_gb_$v > /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_gb_$v.log 2>&1
  if [ ! -f /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_gb_$v/agent.pt ]; then echo "SMOKE_FAILED $v"; tail -4 /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/smoke_gb_$v.log; exit 1; fi
  echo "SMOKE_OK $v"
done
for v in gate beta gb; do
  echo "TRAIN_START $v $(date +%H:%M)"
  $PY -m Yuan.IJRR.stage2_traj.train     --config Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8k_$v.yaml     --out-dir Yuan/IJRR/runs/rl_dirfrac_e8k_$v     > Yuan/IJRR/runs/rl_dirfrac_e8k_$v.log 2>&1
  echo "TRAIN_END $v $(date +%H:%M) $(tail -1 Yuan/IJRR/runs/rl_dirfrac_e8k_$v.log)"
done
echo GB_CHAIN_DONE
