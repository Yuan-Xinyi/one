"""Three-arm ladder at the paper protocol (SUB=2): the value-lookahead arm
against the deployed controllers, parsed straight from the run logs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, re
from pathlib import Path

D = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05'
         '/Yuan/IJRR/runs/ladder_vlook')
FILES = [('FR3\n(2048 tasks)', 'fr3_2048.log'),
         ('xArm7\n(1024 tasks)', 'xarm7_1024.log'),
         ('Cobotta\n(1024 tasks)', 'cobotta_1024.log')]
ARMS = [('classical', 'classical null-space law', 'tab:gray'),
        ('sgngrad', 'margin law, zero privilege', 'darkgray'),
        ('vertex', 'RL policy (30M)', 'tab:orange'),
        ('hybrid', 'RL/classical hybrid', 'gold'),
        ('myopic', 'one-step margin law', 'black'),
        ('vlook', 'one-step value lookahead (new)', 'tab:red')]

data, cost = {}, {}
for rob, f in FILES:
    txt = (D / f).read_text()
    data[rob] = {m.group(1): float(m.group(2)) for m in
                 re.finditer(r'^(\w+)\s+ratio to classical\s+([\d.]+)',
                             txt, re.M)}
    cost[rob] = {m.group(1): float(m.group(2)) for m in
                 re.finditer(r'^(\w+)\s+mean progress.*compute\s+([\d.]+) ms',
                             txt, re.M)}

fig, ax = plt.subplots(figsize=(11, 5.6))
x = np.arange(len(FILES))
w = 0.13
for i, (key, lab, c) in enumerate(ARMS):
    vals = [data[r].get(key, np.nan) for r, _ in FILES]
    b = ax.bar(x + (i - 2.5) * w, vals, w, color=c, label=lab)
    for r_, v in zip(b, vals):
        ax.annotate(f'{v:.3f}', (r_.get_x() + w / 2, v), xytext=(0, 2),
                    textcoords='offset points', ha='center', fontsize=7.5,
                    rotation=90)
ax.axhline(1.0, c='gray', ls=':', lw=1)
ax.set_xticks(x); ax.set_xticklabels([r for r, _ in FILES])
ax.set_ylabel('mean progress / classical null-space law')
ax.set_ylim(0.9, 2.15)
ax.set_title('horizon ladder at the paper protocol (SUB=2): ranking the '
             'successors of the exact model\nby a learned value beats '
             'ranking them by the handcrafted margin, on all three arms',
             fontsize=11)
ax.legend(frameon=False, fontsize=9, ncol=3, loc='upper center')
ax.grid(alpha=0.2, axis='y')
fig.tight_layout()
out = D / 'ladder_vlook.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('wrote', out)

print(f"\n{'arm':<32s}" + ''.join(f"{r.splitlines()[0]:>12s}"
                                  for r, _ in FILES))
for key, lab, _ in ARMS:
    print(f"{lab:<32s}" + ''.join(f"{data[r].get(key, float('nan')):>12.4f}"
                                  for r, _ in FILES))
print(f"\n{'vlook / myopic':<32s}" +
      ''.join(f"{data[r]['vlook'] / data[r]['myopic']:>12.4f}"
              for r, _ in FILES))
print(f"{'vlook / hybrid':<32s}" +
      ''.join(f"{data[r]['vlook'] / data[r]['hybrid']:>12.4f}"
              for r, _ in FILES))
print(f"\nper-decision compute (ms, whole batch):")
for key, lab, _ in ARMS:
    print(f"{lab:<32s}" + ''.join(f"{cost[r].get(key, float('nan')):>12.1f}"
                                  for r, _ in FILES))
