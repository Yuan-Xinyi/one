#!/bin/bash
SCR=/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad
while pgrep -f "onestroke_ompl_swee[p].py|onestroke_metric[s].py|redo_medi[a].sh" >/dev/null; do sleep 45; done
echo "ALL DONE"
echo "=== 5-deg OMPL ==="; grep -hE "pointwise" $SCR/ompl5T_*.log 2>/dev/null
echo "=== metrics ==="; grep -vE "^\[bound\]" $SCR/metrics.log | tail -16
echo "=== media ==="; cat $SCR/redo_media.log
