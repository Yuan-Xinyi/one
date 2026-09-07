"""Per-task 2x4 joint-trajectory figure: classical (red) / DirFrac RL
(blue) / high-budget search (green), x = arc length, hardware limits
dashed black, full-range y-axis."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
MAIN = Path('/home/lqin/one/Yuan/IJRR')
OUT = MAIN/'runs/paper_fill/search_compare'
LMT_LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445,
                   -3.0159])
LMT_UP = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
TASKS = [int(x) for x in sys.argv[1:]]
for ti in TASKS:
    d = np.load(OUT/f't{ti}_three_way.npz')
    fig, axes = plt.subplots(2, 4, figsize=(32, 12))
    rng_pad = 0.15
    for j in range(7):
        ax = axes[j // 4, j % 4]
        ax.plot(d['cls_s'], d['cls_q'][:, j], color='crimson', lw=2.2,
                label=f"Classical gradient ({d['cls_s'][-1]:.2f} m)")
        ax.plot(d['rl_s'], d['rl_q'][:, j], color='royalblue', lw=2.2,
                label=f"DirFrac RL ({d['rl_s'][-1]:.2f} m)")
        ax.plot(d['search_s'], d['search_q'][:, j], color='forestgreen',
                lw=2.0, marker='o', ms=3.5,
                label=f"Layered search ({d['search_s'][-1]:.2f} m)")
        ax.axhline(LMT_LO[j], color='k', ls='--', lw=1.6)
        ax.axhline(LMT_UP[j], color='k', ls='--', lw=1.6)
        ax.set_ylim(min(LMT_LO[j], d['cls_q'][:, j].min(),
                        d['rl_q'][:, j].min()) - rng_pad,
                    max(LMT_UP[j], d['cls_q'][:, j].max(),
                        d['rl_q'][:, j].max()) + rng_pad)
        ax.set_title(f'Joint {j + 1}', fontsize=16)
        ax.set_xlabel('arc length s (m)', fontsize=13)
        ax.set_ylabel('q (rad)', fontsize=13)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=13, loc='best')
    ax = axes[1, 3]
    ax.axis('off')
    ax.text(0.05, 0.75, f"Task {ti}", fontsize=26, weight='bold')
    ax.text(0.05, 0.55,
            f"$\\ell^{{pw}}$ = {float(d['ref']):.2f} m", fontsize=20)
    ax.text(0.05, 0.38, f"Classical gradient: {d['cls_s'][-1]:.2f} m",
            fontsize=18, color='crimson')
    ax.text(0.05, 0.26, f"DirFrac RL: {d['rl_s'][-1]:.2f} m",
            fontsize=18, color='royalblue')
    ax.text(0.05, 0.14, f"Layered search (K=1024, 5 restarts): "
            f"{d['search_s'][-1]:.2f} m", fontsize=18, color='forestgreen')
    fig.tight_layout()
    fig.savefig(OUT/f't{ti}_three_way.png', dpi=70)
    plt.close(fig)
    print(f't{ti} plotted', flush=True)
