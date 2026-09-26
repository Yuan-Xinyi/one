#!/bin/bash
# tab:horizon campaign: MPC (H=10,20,30) and MPPI (H=16,32,64) on the
# DirFrac interface, FR3 then xArm7, straight 10k then the two curved
# families (2.5k each). Two chains run concurrently (one per method).
# usage: horizon_chain.sh mpc|mppi
PY=/home/lqin/miniconda3/envs/one/bin/python
cd /home/lqin/one
M=$1
OUT=Yuan/IJRR/runs/paper_fill/horizon
if [ "$M" = mpc ]; then HS="10,20,30"; else HS="16,32,64"; fi
for R in fr3 xarm7; do
  for F in straight serpentine nonplanar; do
    LOG=$OUT/${R}_${F}_${M}.log
    if [ -f $OUT/${R}_${F}_${M}.done ]; then continue; fi
    echo "[$(date '+%F %T')] start $R $F $M" >> $OUT/chain_$M.log
    $PY -m Yuan.IJRR.eval.mpc_mppi_baselines --robot $R --family $F \
        --method $M --H $HS --batch 2500 --chunk 4096 > $LOG 2>&1 \
      && touch $OUT/${R}_${F}_${M}.done
    echo "[$(date '+%F %T')] end   $R $F $M rc=$?" >> $OUT/chain_$M.log
  done
done
