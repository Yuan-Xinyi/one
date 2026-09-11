#!/bin/bash
# Finish the print analysis once the bowl leg lands: recompute the field with
# the corrected (arm-only) bed clearance over the full z range, re-run the cube
# query, add the part-as-obstacle leg, redraw the figures.
set -u
cd /tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad
PY=/home/lqin/miniconda3/envs/one/bin/python
OUT=/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis

until [ -f "$OUT/bowl_query.npz" ]; do
  pgrep -f "bin/python print_[b]owl.py" > /dev/null || { echo "[chain] bowl died"; break; }
  sleep 20
done
echo "[chain] bowl leg done, refreshing the field"
$PY print_field.py   || { echo "[chain] field FAILED"; exit 1; }
$PY print_cube.py    || { echo "[chain] cube FAILED"; exit 1; }
$PY print_part.py    || echo "[chain] part FAILED (continuing)"
$PY print_figs.py    || echo "[chain] figs FAILED"
echo "[chain] done"
