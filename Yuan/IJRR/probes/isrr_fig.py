"""Full arm comparison on the 300 held-out tasks of the cross-task run,
grouped by how much model computation each arm spends per decision."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RUN = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05'
           '/Yuan/IJRR/runs/pool_v5')
groups = [
    ('reactive controller\n(no lookahead)', [
        ('classical null-space law', 1.000, 'tab:gray'),
        ('distilled policy (ours)', 1.146, 'tab:blue'),
        ('ISRR pure RL, 30M', 1.338, 'tab:orange'),
        ('ISRR hybrid B (RL/classical)', 1.388, 'tab:orange'),
    ]),
    ('one-step model lookahead\n(16 successor evaluations)', [
        ('alive filter + random', 1.295, 'tab:gray'),
        ('one-step margin law (deployed)', 1.685, 'black'),
        ('alive + softmin of all 4 margins', 1.745, 'tab:gray'),
        ('alive + distilled critic (ours)', 1.484, 'tab:blue'),
        ('alive + ISRR RL critic', 1.801, 'tab:red'),
    ]),
    ('prior-guided search\n(width 1 to 256)', [
        ('W=1, distilled prior (ours)', 1.616, 'tab:blue'),
        ('W=1, ISRR RL prior', 1.832, 'tab:red'),
        ('W=16, ISRR RL prior', 1.875, 'tab:red'),
        ('W=256, distilled prior (ours)', 1.766, 'tab:blue'),
    ]),
]

labels, vals, cols, seps = [], [], [], []
for gi, (gname, rows) in enumerate(groups):
    for nm, v, c in rows:
        labels.append(nm); vals.append(v); cols.append(c)
    seps.append(len(labels))
    if gi < len(groups) - 1:
        labels.append(''); vals.append(np.nan); cols.append('white')

fig, ax = plt.subplots(figsize=(11, 7.5))
ypos = np.arange(len(labels))[::-1]
ax.barh(ypos, vals, color=cols, height=0.72)
for yp, v in zip(ypos, vals):
    if v == v:
        ax.annotate(f'{v:.3f}', (v, yp), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=9)
ax.axvline(1.685, c='black', ls='--', lw=1.2)
ax.annotate('one-step margin law', (1.685, ypos.max() + 0.6),
            fontsize=9, ha='center')
ax.axvline(1.0, c='tab:gray', ls=':', lw=1)
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('held-out progress / classical null-space law (300 tasks)')
ax.set_xlim(0.9, 2.05)
ax.grid(alpha=0.25, axis='x')
y0 = 0
for gi, (gname, rows) in enumerate(groups):
    top = ypos[y0] + 0.45
    ax.annotate(gname.replace('\n', ' '), (1.99, top), fontsize=9.5,
                style='italic', color='dimgray', va='bottom', ha='right')
    if gi:
        ax.axhline(top + 0.15, color='lightgray', lw=1)
    y0 += len(rows) + 1
ax.set_title('all arms on the same tasks and protocol\n'
             '(SUB=1; the margin law and every lookahead arm use the same '
             'exact model)', fontsize=11)
fig.tight_layout()
out = RUN / 'isrr_compare.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('wrote', out)
