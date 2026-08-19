"""The corrected picture: explored fans (baseline + j7-drift) vs the pool vs
the winning FILAMENT, with the two gap notions contrasted."""
import numpy as np
import torch
from pathlib import Path

BASE = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05'
            '/Yuan/IJRR/runs')
ga = np.load(BASE / 'single_task_ppo_v2/goexplore_archive.npz')
gd = np.load(BASE / 'single_task_ppo_v2_drift/goexplore_archive.npz')
rb = np.load(BASE / 'single_task_ppo_v2/reachtree_bank.npz')
rt = np.load(BASE / 'single_task_ppo_v2/reachtree.npz')
deg = 180 / np.pi

qe, de = ga['q'], ga['depth']
qd, dd = gd['q'], gd['depth']
qo, do = rb['q'], rb['depth']
qt = rt['q']


def gap_curves(q_arch, d_arch):
    """min L-inf distance per depth: to the pool, and to the winning traj."""
    D = int(min(d_arch.max(), do.max(), len(qt) - 1))
    g_pool = np.full(D + 1, np.nan)
    g_fil = np.full(D + 1, np.nan)
    for d in range(D + 1):
        ei = np.abs(d_arch - d) <= 1
        if not ei.sum():
            continue
        a = torch.tensor(q_arch[ei])
        oi = do == d
        if oi.sum():
            b = torch.tensor(qo[oi])
            g_pool[d] = float((a[:, None] - b[None]).abs().amax(-1).min())
        g_fil[d] = float((a - torch.tensor(qt[d])[None]).abs()
                         .amax(-1).min())
    return g_pool, g_fil


gp_base, gf_base = gap_curves(qe, de)
_, gf_drift = gap_curves(qd, dd)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
s_e, s_d, s_o = de * 0.01, dd * 0.01, do * 0.01
s_t = np.arange(len(qt)) * 0.01

for ax, j in zip(axes[0], (6, 0)):
    ax.scatter(s_e, qe[:, j] * deg, s=2, c='tab:blue', alpha=0.12,
               rasterized=True, label='fan: baseline explore')
    ax.scatter(s_d, qd[:, j] * deg, s=2, c='tab:green', alpha=0.12,
               rasterized=True, label='fan: j7-drift explore')
    ax.scatter(s_o, qo[:, j] * deg, s=2, c='tab:orange', alpha=0.12,
               rasterized=True, label='reachtree pool')
    ax.plot(s_t, qt[:, j] * deg, c='darkred', lw=1.8,
            label='winning FILAMENT')
    ax.axvline(0.84, color='k', lw=0.8, ls='--', alpha=0.6)
    ax.set_title(f'joint {j + 1}', fontsize=11)
    ax.set_xlabel('arc length s (m)', fontsize=9)
    ax.set_ylabel('deg', fontsize=9)

ax = axes[1, 0]
x = np.arange(len(gp_base)) * 0.01
ax.plot(x, gp_base * deg, c='tab:orange', lw=1.6,
        label='to reachtree POOL (my earlier, misleading gap)')
ax.plot(x, gf_base * deg, c='darkred', lw=1.8,
        label='to winning FILAMENT (the real gap)')
ax.plot(np.arange(len(gf_drift)) * 0.01, gf_drift * deg, c='tab:green',
        lw=1.6, ls='--', label='to filament, j7-drift fan')
ax.axvline(0.84, color='k', lw=0.8, ls='--', alpha=0.6)
ax.set_title('min L$_\\infty$ distance from explored fan (deg)', fontsize=11)
ax.set_xlabel('arc length s (m)', fontsize=9)
ax.set_ylabel('deg', fontsize=9)
ax.legend(fontsize=8.5, frameon=False)

ax = axes[1, 1]
d80 = 80
ei = np.abs(de - d80) <= 2
di = np.abs(dd - d80) <= 2
oi = np.abs(do - d80) <= 2
ax.scatter(qe[ei][:, 0] * deg, qe[ei][:, 6] * deg, s=5, c='tab:blue',
           alpha=0.3, label='baseline fan')
ax.scatter(qd[di][:, 0] * deg, qd[di][:, 6] * deg, s=5, c='tab:green',
           alpha=0.3, label='j7-drift fan')
ax.scatter(qo[oi][:, 0] * deg, qo[oi][:, 6] * deg, s=5, c='tab:orange',
           alpha=0.4, label='pool')
ax.plot(qt[d80, 0] * deg, qt[d80, 6] * deg, '*', c='darkred', ms=16,
        label='filament @0.80')
ax.set_title('cross-section at s = 0.80 m: (j1, j7) plane', fontsize=11)
ax.set_xlabel('joint 1 (deg)', fontsize=9)
ax.set_ylabel('joint 7 (deg)', fontsize=9)
ax.legend(fontsize=8.5, frameon=False, markerscale=2)

h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4,
           fontsize=9.5, frameon=False, markerscale=5)
fig.suptitle('task 27: the fans never reach the filament — and pushing j7 '
             'grows the wrong part of the fan', y=1.04, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = BASE / 'single_task_ppo_v2/fan_vs_filament.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('filament gap @0.5/0.7/0.8 baseline:',
      [f"{gf_base[int(x*100)]*deg:.1f}" for x in (0.5, 0.7, 0.8)],
      ' drift:',
      [f"{gf_drift[int(x*100)]*deg:.1f}" for x in (0.5, 0.7, 0.8)])
print(f'wrote {out}')
