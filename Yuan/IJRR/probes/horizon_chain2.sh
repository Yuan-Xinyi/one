#!/bin/bash
# Second-generation MPC worker for tab:horizon: runs the listed cells in
# order, skipping any cell that is done (three JSONs present or .done) or
# currently being run by another worker (pgrep). usage:
#   horizon_chain2.sh <wait_for_mppi:0|1> "<robot:family> <robot:family> ..."
PY=/home/lqin/miniconda3/envs/one/bin/python
cd /home/lqin/one
OUT=Yuan/IJRR/runs/paper_fill/horizon
WAIT=$1; shift
if [ "$WAIT" = 1 ]; then
  while pgrep -f "method mppi" >/dev/null; do sleep 120; done
fi
for cell in $1; do
  R=${cell%%:*}; F=${cell##*:}
  done=1; for H in 10 20 30; do [ -f $OUT/${R}_${F}_mpc$H.json ] || done=0; done
  [ -f $OUT/${R}_${F}_mpc.done ] && done=1
  if [ $done = 1 ]; then continue; fi
  if pgrep -f "robot $R --family $F --method mpc" >/dev/null; then
    echo "[$(date '+%F %T')] skip $R $F (running elsewhere)" >> $OUT/chain2_$$.log; continue; fi
  echo "[$(date '+%F %T')] start $R $F mpc" >> $OUT/chain2_$$.log
  $PY -m Yuan.IJRR.eval.mpc_mppi_baselines --robot $R --family $F \
      --method mpc --H 10,20,30 --batch 2500 --chunk 4096 > $OUT/${R}_${F}_mpc.log 2>&1 \
    && touch $OUT/${R}_${F}_mpc.done
  echo "[$(date '+%F %T')] end   $R $F mpc" >> $OUT/chain2_$$.log
done
