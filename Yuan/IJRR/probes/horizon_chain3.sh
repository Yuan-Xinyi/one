#!/bin/bash
# Generic tab:horizon worker: <method:mpc|mppi> "<robot:family> ...".
# Skips cells that are done (three JSONs) or running elsewhere (pgrep).
PY=/home/lqin/miniconda3/envs/one/bin/python
cd /home/lqin/one
OUT=Yuan/IJRR/runs/paper_fill/horizon
M=$1; shift
if [ "$M" = mpc ]; then HS="10 20 30"; HL="10,20,30"; else HS="16 32 64"; HL="16,32,64"; fi
for cell in $1; do
  R=${cell%%:*}; F=${cell##*:}
  done=1; for H in $HS; do [ -f $OUT/${R}_${F}_${M}$H.json ] || done=0; done
  [ $done = 1 ] && continue
  if pgrep -f "robot $R --family $F --method $M" >/dev/null; then
    echo "[$(date '+%F %T')] skip $R $F $M (running elsewhere)" >> $OUT/chain3_$$.log; continue; fi
  echo "[$(date '+%F %T')] start $R $F $M" >> $OUT/chain3_$$.log
  $PY -m Yuan.IJRR.eval.mpc_mppi_baselines --robot $R --family $F \
      --method $M --H $HL --batch 2500 --chunk 4096 > $OUT/${R}_${F}_${M}.log 2>&1 \
    && touch $OUT/${R}_${F}_${M}.done
  echo "[$(date '+%F %T')] end   $R $F $M" >> $OUT/chain3_$$.log
done
