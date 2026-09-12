"""Completeness report: every cell of the two final tables, named if empty."""
import sys
from pathlib import Path
import numpy as np
OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis')
FIGS = ['circle', 'square', 'triangle', 'star', 'envelope', 'eight', 'spiral']
missing = []
prev = np.load(OUT / 'onestroke.npz')
met = np.load(OUT / 'onestroke_metrics.npz') if (OUT/'onestroke_metrics.npz').exists() else None
for cone in (30, 5):
    opt = None
    for f in (OUT / f'onestroke_opt_c{cone}.npz', OUT / 'onestroke_opt.npz'):
        if f.exists():
            opt = np.load(f); break
    for k in FIGS:
        kk = f'{k}_c{cone}'
        if kk not in prev.files: missing.append(f'{kk}: proposed size')
        if opt is None or f'{kk}_classical' not in opt.files:
            missing.append(f'{kk}: classical size')
        fo = OUT / f'onestroke_ompl_c{cone}_{k}.npz'
        if not fo.exists(): missing.append(f'{kk}: OMPL result')
        if met is None or f'{kk}_metrics' not in met.files:
            missing.append(f'{kk}: metrics row')
        else:
            m = met[f'{kk}_metrics']
            if np.isnan(m[0]): missing.append(f'{kk}: proposed metrics')
            if np.isnan(m[3]): missing.append(f'{kk}: classical metrics')
            if fo.exists() and float(np.load(fo)[kk][0]) > 0 and np.isnan(m[6]):
                missing.append(f'{kk}: OMPL metrics')
        if fo.exists() and float(np.load(fo)[kk][0]) > 0:
            for suf in ('pack.npz',):
                if not (OUT / 'ompl' / f'{k}_c{cone}_{suf}').exists():
                    missing.append(f'{kk}: media pack')
ft = [OUT / 'onestroke_fw_time_c30.npz', OUT / 'onestroke_fw_time_c5.npz']
for f in ft:
    if not f.exists(): missing.append(f'timing file {f.name}')
print('MISSING:' if missing else 'COMPLETE: every table cell has data')
for m in missing: print('  -', m)
