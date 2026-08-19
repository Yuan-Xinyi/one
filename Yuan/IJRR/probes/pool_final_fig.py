"""Final deliverable figure for the cross-task run:
(a) held-out performance vs search width, prior-guided vs unguided, with the
    margin law and the standalone policy as horizontal references;
(b) train-vs-held-out diagnostic (the generalization gap that caps the
    standalone controller)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RUN = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05'
           '/Yuan/IJRR/runs/pool_v5')
d = np.load(RUN / 'pool_eval.npz', allow_pickle=True)
rows = d['rows']
pe, cl, my = d['policy'], d['classical'], d['myopic']
pol_ratio = float(np.mean(pe / np.maximum(my, 1e-6)))

W = sorted({int(r[0]) for r in rows})
def series(g, sm):
    return [next(float(r[4]) for r in rows
                 if int(r[0]) == w and int(r[1]) == g and r[2] == sm)
            for w in W]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(W, series(1, 'logp'), 'o-', c='tab:red', lw=2,
        label='prior-guided (policy ranking)')
ax.plot(W, series(1, 'value'), 's-', c='tab:purple', lw=2,
        label='prior-guided (critic ranking)')
ax.plot(W, series(0, 'logp'), 'o--', c='tab:gray', lw=2,
        label='unguided search')
ax.axhline(1.0, c='black', ls='-', lw=1.5, label='one-step margin law')
ax.axhline(pol_ratio, c='tab:blue', ls=':', lw=1.5,
           label=f'policy alone ({pol_ratio:.2f})')
ax.set_xscale('log', base=2)
ax.set_xlabel('search width W (log)')
ax.set_ylabel('held-out progress / margin law')
ax.set_title('(a) held-out deployment: search width vs performance')
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=9, loc='lower right')

ax = axes[1]
tr = np.load(RUN / 'diag_0.npz') if (RUN / 'diag_0.npz').exists() else None
te = np.load(RUN / 'diag_12000.npz') if (RUN / 'diag_12000.npz').exists() \
    else None
if tr is not None and te is not None:
    labels = ['train tasks\n(have demos)', 'held-out tasks']
    ratios = [float(np.mean(tr['prog_policy']
                            / np.maximum(tr['prog_law'], 1e-6))),
              float(np.mean(te['prog_policy']
                            / np.maximum(te['prog_law'], 1e-6)))]
    agree = [float(tr['agree_law']), float(te['agree_law'])]
    x = np.arange(2)
    b1 = ax.bar(x - 0.18, ratios, 0.36, color='tab:red',
                label='progress / margin law')
    b2 = ax.bar(x + 0.18, agree, 0.36, color='tab:gray',
                label='top-1 action agreement')
    for b in list(b1) + list(b2):
        ax.annotate(f'{b.get_height():.3f}',
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords='offset points', xytext=(0, 3),
                    ha='center', fontsize=9)
    ax.axhline(1.0, c='black', lw=1.2)
    ax.axhline(1 / 16, c='tab:gray', ls=':', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('ratio')
    ax.set_title('(b) the standalone controller is capped by generalization')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis='y')
fig.suptitle('cross-task search-and-distill, 300 held-out tasks', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = RUN / 'pool_final.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('wrote', out)
for r in rows:
    print(f"W{int(r[0]):>4} guided={int(r[1])} {r[2]:<5} "
          f"x{float(r[3]):.3f} classical  x{float(r[4]):.3f} law "
          f"(median x{float(r[5]):.3f})")
