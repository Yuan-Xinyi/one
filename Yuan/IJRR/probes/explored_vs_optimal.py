"""Explored region (go-explore archive) vs the optimal corridor (reachtree
bank) in joint-trajectory coordinates, plus the per-arc gap between them."""
import numpy as np
import torch
from pathlib import Path

RUN = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05'
           '/Yuan/IJRR/runs/single_task_ppo_v2')
ga = np.load(RUN / 'goexplore_archive.npz')     # explored: q (N,7), depth
rb = np.load(RUN / 'reachtree_bank.npz')        # optimal corridor
rt = np.load(RUN / 'reachtree.npz')             # winning trajectory

qe, de = ga['q'], ga['depth']
qo, do = rb['q'], rb['depth']
qt = rt['q']
deg = 180 / np.pi

# per-depth minimal L-inf distance from explored set to optimal corridor
D = int(min(de.max(), do.max()))
gap = np.full(D + 1, np.nan)
gap_joint = np.full((D + 1, 7), np.nan)
for d in range(D + 1):
    ei = np.abs(de - d) <= 1
    oi = do == d
    if ei.sum() and oi.sum():
        a = torch.tensor(qe[ei])                # (ne,7)
        b = torch.tensor(qo[oi])                # (no,7)
        diff = (a[:, None, :] - b[None, :, :]).abs()   # (ne,no,7)
        linf = diff.amax(-1)
        i, j = np.unravel_index(int(linf.argmin()), linf.shape)
        gap[d] = float(linf[i, j])
        gap_joint[d] = diff[i, j].numpy()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
se, so = de * 0.01, do * 0.01
st = np.arange(len(qt)) * 0.01
for j in range(7):
    ax = axes.flat[j]
    ax.scatter(se, qe[:, j] * deg, s=2, c='tab:blue', alpha=0.10,
               rasterized=True, label='explored (go-explore archive)')
    ax.scatter(so, qo[:, j] * deg, s=2, c='tab:orange', alpha=0.15,
               rasterized=True, label='optimal corridor (reachtree)')
    ax.plot(st, qt[:, j] * deg, c='darkred', lw=1.4,
            label='winning trajectory')
    ax.axvline(0.84, color='k', lw=0.8, ls='--', alpha=0.6)
    ax.set_title(f'joint {j + 1}', fontsize=10)
    ax.set_ylabel('deg', fontsize=8)
    ax.tick_params(labelsize=8)
ax = axes.flat[7]
ax.plot(np.arange(D + 1) * 0.01, gap * deg, c='tab:purple', lw=1.6)
ax.axvline(0.84, color='k', lw=0.8, ls='--', alpha=0.6)
ax.set_title('gap: min L$_\\infty$ distance,\nexplored set → optimal '
             'corridor (deg)', fontsize=10)
ax.set_ylabel('deg', fontsize=8)
ax = axes.flat[8]
for j in (0, 2, 5, 6):
    ax.plot(np.arange(D + 1) * 0.01, gap_joint[:, j] * deg, lw=1.2,
            label=f'j{j + 1}')
ax.axvline(0.84, color='k', lw=0.8, ls='--', alpha=0.6)
ax.set_title('which joints carry the gap\n(at the closest pair)',
             fontsize=10)
ax.legend(fontsize=8, frameon=False, ncol=2)
for a2 in axes[-1]:
    a2.set_xlabel('arc length s (m)', fontsize=9)
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3,
           fontsize=10, frameon=False, markerscale=6)
fig.suptitle('task 27: what exploration reached vs where the optimum lives '
             '(dashed line = exploration frontier 0.84 m)', y=1.02,
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = RUN / 'explored_vs_optimal.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('gap at s=0.3/0.5/0.7/0.8:',
      [f"{gap[int(x*100)]*deg:.1f}deg" for x in (0.3, 0.5, 0.7, 0.8)])
print(f'wrote {out}')
