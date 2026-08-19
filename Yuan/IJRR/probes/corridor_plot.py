"""Visualize the corridor narrowness along the reachtree 1.06 m trajectory."""
import numpy as np
from pathlib import Path

RUN = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05'
           '/Yuan/IJRR/runs/single_task_ppo_v2')
cw = np.load(RUN / 'corridor_width.npz')
rt = np.load(RUN / 'reachtree.npz')
D, n_viable, act = cw['D'], cw['n_viable'], cw['act']
T = D.shape[0]
s = np.arange(T) * 0.01
q6 = rt['q'][:T, 5] * 180 / np.pi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(13, 8), sharex=True,
    gridspec_kw={'height_ratios': [2.2, 1.3]})

im = ax1.imshow(D.T, aspect='auto', origin='lower', cmap='viridis',
                extent=(0, T * 0.01, -0.5, 15.5), interpolation='nearest',
                vmin=0, vmax=40)
ax1.plot(s + 0.005, act, 'r.', ms=4, label='executed vertex')
ax1.set_ylabel('vertex action index (0-15)')
ax1.set_title('task 27, along the reachtree 1.06 m trajectory: '
              'survivable depth D(t, a) after each action\n'
              '(probe: width-1024 tree, horizon 40; dark = this action is a '
              'dead end)')
cb = fig.colorbar(im, ax=ax1, pad=0.01)
cb.set_label('steps survivable after action')
ax1.legend(loc='upper left', frameon=True, fontsize=9)

ax2.step(s, n_viable, where='mid', color='tab:purple', lw=1.6,
         label='n_viable(t): actions that stay on a max-length route')
ax2.set_ylabel('viable actions (of 16)', color='tab:purple')
ax2.set_ylim(-0.5, 16.5)
ax2.axhline(1, color='tab:purple', lw=0.6, ls=':')
ax2b = ax2.twinx()
ax2b.plot(s, q6, color='tab:orange', lw=1.4, label='joint 6 (deg)')
ax2b.set_ylabel('joint 6 (deg)', color='tab:orange')
ax2.set_xlabel('arc length s along the line (m)')
w0, w1 = 0.40, 0.90
ax2.axvspan(w0, w1, color='0.85', zorder=0)
ax2.text(0.65, 14, 'rescue window 0.40-0.90 m:\n'
         'P(random explorer stays viable) = 2.5e-51',
         ha='center', fontsize=9)
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc='lower left', fontsize=9, frameon=True)
fig.tight_layout()
out = RUN / 'corridor_width.png'
fig.savefig(out, dpi=160)
print('viable counts in window 40-90:',
      np.bincount(n_viable[40:90], minlength=6)[:6])
print(f'wrote {out}')
