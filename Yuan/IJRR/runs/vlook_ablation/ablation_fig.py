"""Two ablations for the value-lookahead controller:
(a) does horizon help?  — on the learned scalar yes, on the margin scalar no
(b) which value works? — a PPO critic does; an accurately fitted regression
    value of a mediocre behaviour does not."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

D = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05'
         '/Yuan/IJRR/runs/vlook_ablation')
MYO = 0.5346                       # one-step margin law, same 1024 tasks

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

# (a) horizon
ax = axes[0]
steps = [1, 2, 3]
margin = [1.000, 0.5298 / MYO, 0.5312 / MYO]
val_w4 = [0.5678 / MYO, 0.5783 / MYO, 0.5800 / MYO]
val_w8 = [0.5678 / MYO, 0.5823 / MYO, 0.5872 / MYO]
ax.plot(steps, margin, 'o--', c='black', lw=2,
        label='handcrafted margin potential (beam, width 4)')
ax.plot(steps, val_w4, 's-', c='tab:red', lw=2,
        label='learned value (beam, width 4)')
ax.plot(steps, val_w8, '^-', c='darkred', lw=2,
        label='learned value (beam, width 8)')
ax.axhline(1.0, c='gray', ls=':', lw=1)
for x, v in zip(steps, val_w8):
    ax.annotate(f'{v:.3f}', (x, v), xytext=(0, 6),
                textcoords='offset points', ha='center', fontsize=9)
for x, v in zip(steps, margin):
    ax.annotate(f'{v:.3f}', (x, v), xytext=(0, -14),
                textcoords='offset points', ha='center', fontsize=9)
ax.set_xticks(steps); ax.set_xlabel('lookahead depth (steps)')
ax.set_ylabel('mean progress / one-step margin law')
ax.set_title('(a) horizon pays off on the learned scalar only')
ax.legend(frameon=False, fontsize=9, loc='lower right')
ax.grid(alpha=0.25)

# (b) value source
ax = axes[1]
rows = [('untrained network (floor)', 0.4186 / MYO, 'lightgray'),
        ('regression value of a\nmediocre behaviour (R2 0.989)', 0.4765 / MYO,
         'tab:blue'),
        ('PPO critic, heavy margin shaping', 0.5264 / MYO, 'lightcoral'),
        ('PPO critic, second seed', 0.5535 / MYO, 'tab:red'),
        ('PPO critic, gamma 0.90', 0.5584 / MYO, 'tab:red'),
        ('PPO critic, light shaping', 0.5640 / MYO, 'tab:red'),
        ('PPO critic, gamma 0.95', 0.5677 / MYO, 'tab:red'),
        ('PPO critic, deployed policy', 0.5678 / MYO, 'firebrick')]
y = np.arange(len(rows))[::-1]
ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.7)
for yy, r in zip(y, rows):
    ax.annotate(f'{r[1]:.3f}', (r[1], yy), xytext=(4, 0),
                textcoords='offset points', va='center', fontsize=9)
ax.axvline(1.0, c='black', ls='--', lw=1.2)
ax.annotate('one-step margin law', (1.0, y.max() + 0.6), fontsize=9,
            ha='center')
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.set_xlim(0.7, 1.13)
ax.set_xlabel('mean progress / one-step margin law')
ax.set_title('(b) the ranking value must estimate a GOOD continuation,\n'
             'not merely fit its labels well')
ax.grid(alpha=0.25, axis='x')

fig.suptitle('one-step value lookahead: what the gain does and does not '
             'depend on (FR3, 1024 tasks, paper protocol)', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = D / 'vlook_ablation.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('wrote', out)
