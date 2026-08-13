"""7-joint trajectory figure: RL local optimum (0.73 m) vs the WHOLE bundle
of search trajectories with progress > 0.70 m (4000 stratified of 120,927),
colored by final progress; the best (1.0614 m) highlighted."""
import matplotlib                       # must precede torch on this box
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np, yaml, sys
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import torch
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 1}), None, dev)
q_mid = env.q_mid.cpu().numpy(); q_half = env.q_half.cpu().numpy()

RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
tr = np.load(RUN / 'traj_compare.npz')          # PPO_q (71,7)
bu = np.load(RUN / 'reachtree_bundle.npz')
qp = tr['PPO_q']
flat, lens, progs = bu['q_flat'], bu['lens'], bu['progress']
offs = np.concatenate([[0], np.cumsum(lens)])
order = np.argsort(progs)                        # draw longest last (on top)
best = int(progs.argmax())
deg = 180 / np.pi
norm = Normalize(vmin=0.70, vmax=float(progs.max()))
cmap = plt.get_cmap('viridis')

fig, axes = plt.subplots(2, 4, figsize=(16.5, 8), sharex=True,
                         constrained_layout=True)
for j in range(7):
    ax = axes.flat[j]
    segs, cols = [], []
    for i in order:
        if i == best:
            continue
        qq = flat[offs[i]:offs[i + 1], j] * deg
        s = np.arange(lens[i]) * 0.01
        segs.append(np.column_stack([s, qq]))
        cols.append(cmap(norm(progs[i])))
    lc = LineCollection(segs, colors=cols, lw=0.4, alpha=0.10)
    ax.add_collection(lc)
    qq = flat[offs[best]:offs[best + 1], j] * deg
    ax.plot(np.arange(lens[best]) * 0.01, qq, c='tab:red', lw=2.0, zorder=5,
            label='best search trajectory (1.06 m)')
    ax.plot(np.arange(len(qp)) * 0.01, qp[:, j] * deg, c='tab:blue', lw=2.0,
            zorder=6, label='RL local optimum (PPO, 0.73 m)')
    ax.axvline(0.73, c='tab:blue', lw=0.8, ls=':', alpha=0.8)
    for sgn in (-1, 1):
        lim = (q_mid[j] + sgn * q_half[j]) * deg
        if abs(lim) < 400:
            ax.axhline(lim, color='k', lw=0.7, ls='--', alpha=0.55)
    ax.set_title(f'joint {j + 1}', fontsize=11)
    if j % 4 == 0:
        ax.set_ylabel('joint angle (deg)', fontsize=9)
    ax.autoscale_view()
    ax.grid(alpha=0.2)

ax = axes.flat[7]                                # final-progress histogram
ax.hist(progs, bins=36, color='tab:green', alpha=0.75)
ax.axvline(0.73, c='tab:blue', ls=':', lw=1.5, label='RL wall 0.73')
ax.axvline(float(progs.max()), c='tab:red', ls='-', lw=1.5,
           label='best 1.06')
ax.set_title('final progress of the 4000 drawn\n'
             '(from 120,927 leaves > 0.70 m)', fontsize=10)
ax.set_xlabel('final progress (m)', fontsize=10)
ax.legend(fontsize=8, frameon=False, loc='upper right')
ax.grid(alpha=0.2)

for a2 in axes[1][:3]:
    a2.set_xlabel('arc length s (m)', fontsize=10)
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=axes, location='bottom', fraction=0.035,
                    shrink=0.55, pad=0.02)
cbar.set_label('trajectory final progress (m)', fontsize=10)
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=2,
           fontsize=11, frameon=False)
fig.suptitle('task 27: every search trajectory longer than the RL wall '
             '(> 0.70 m), 7-joint view\n(thin lines: 4000 of 120,927 '
             'leaf trajectories; dashed = joint limits; dotted blue = '
             'where RL dies)', fontsize=12)
out = RUN / 'rl_vs_search_bundle.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('wrote', out)
