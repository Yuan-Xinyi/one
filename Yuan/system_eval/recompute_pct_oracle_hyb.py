"""Recompute every ablation table's % column with the controller-aware oracle.

Replaces the denominator from `max_label_L` (SMM classical oracle) with
`oracle_hyb` per-task max from cell_oracle_hyb_results.npz, and prints
new Mean/Std/Min/Max rows for every saved L array under sweeps/.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def stats(arr):
    m = np.isfinite(arr)
    a = arr[m]
    if a.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return float(a.mean()), float(a.std()), float(a.min()), float(a.max())


def fmt(t):
    return f'{t[0]:.2f} / {t[1]:.2f} / {t[2]:.2f} / {t[3]:.2f}'


def fmtl(t):
    return f'{t[0]:.3f} / {t[1]:.3f} / {t[2]:.3f} / {t[3]:.3f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sweeps',
                   default='Yuan/system_eval/runs/eval_10k_systematic/sweeps')
    p.add_argument('--oracle-npz',
                   default='Yuan/system_eval/runs/eval_10k_systematic/cell_oracle_hyb_results.npz')
    p.add_argument('--eval-set',
                   default='Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
    p.add_argument('--target-distance-m', type=float, default=1.5)
    args = p.parse_args()

    sweeps = Path(args.sweeps)
    tdm = float(args.target_distance_m)

    # ---- oracles: classical (old) and hybrid-aware (new) ----
    z_eval = np.load(args.eval_set, allow_pickle=False)
    l_oracle_classical = z_eval['max_label_L'].astype(np.float32) * tdm

    z_or = np.load(args.oracle_npz, allow_pickle=False)
    # cell_oracle_hyb_results.npz schema: same as cell_<name>_results.npz
    # L_best (n_tasks,) is the per-task max over K' candidates under hybrid.
    L_best = z_or['L_best'].astype(np.float32)
    l_oracle_hyb = L_best * tdm
    T = l_oracle_hyb.shape[0]
    print(f'Oracle stats (10k):')
    print(f'  classical  ℓ_oracle (m): mean={l_oracle_classical.mean():.3f} '
          f'median={np.median(l_oracle_classical):.3f}')
    print(f'  hybrid-aware ℓ_oracle (m): mean={np.nanmean(l_oracle_hyb):.3f} '
          f'median={np.nanmedian(l_oracle_hyb):.3f}')
    print(f'  hybrid / classical ratio: mean={np.nanmean(l_oracle_hyb/np.maximum(l_oracle_classical,1e-9)):.3f}')

    def pct(L_per_task):
        l = L_per_task.astype(np.float32) * tdm
        with np.errstate(invalid='ignore', divide='ignore'):
            return 100.0 * l / np.maximum(l_oracle_hyb, 1e-9)

    # Sets to process, name -> filename pattern
    rows = []

    # CFG table
    for w in ['0.0', '1.0', '1.5', '2.0', '3.0']:
        f = sweeps / f'cfg_only_w{w}.npz'
        if f.exists():
            L = np.load(f)['L']
            rows.append(('CFG', f'w={w}', L))

    # DDIM steps table
    for s in ['10', '20', '50', '100']:
        f = sweeps / f'steps_{s}.npz'
        if f.exists():
            L = np.load(f)['L']
            rows.append(('STEPS', f'ddim={s}', L))

    # Switching table
    for te, tx in [('0.85', '0.85'), ('0.9', '0.9'), ('0.95', '0.95'),
                   ('0.97', '0.97'), ('0.98', '0.98'), ('0.99', '0.99'),
                   ('0.98', '0.94'), ('0.99', '0.93')]:
        f = sweeps / f'switch_{te}_{tx}.npz'
        if f.exists():
            L = np.load(f)['L']
            rows.append(('SWITCH', f'(te={te},tx={tx})', L))

    # Gain rows that were saved (single-objective + best combo)
    for name in ['mu_only', 'jl_only', 'theta_only', 'best_combo']:
        f = sweeps / f'gain_{name}.npz'
        if f.exists():
            L = np.load(f)['L']
            rows.append(('GAIN', name, L))

    out = {}
    for table, label, L in rows:
        l_m = L.astype(np.float32) * tdm
        pct_arr = pct(L)
        out_l = stats(l_m)
        out_p = stats(pct_arr)
        out.setdefault(table, []).append({'label': label, 'l': out_l, 'pct': out_p})

    print()
    for table, items in out.items():
        print(f'==== {table} (% = vs hybrid-aware oracle) ====')
        for it in items:
            print(f'  {it["label"]:18s} l(m)={fmtl(it["l"])} | %={fmt(it["pct"])}')

    (sweeps / 'recompute_pct_oracle_hyb.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
