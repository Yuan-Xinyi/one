"""All eight components of the planner's state, against the framework's.

The planning state is (q1..q7, s): seven joints plus how far along the figure
the pen has got. Plotting the arc coordinate alongside the joints is the direct
check on whether the planner is doing something sensible -- a solution that
stalls, or that spends its states shuffling the arm while s barely moves, would
show up here as a flat stretch in the last panel, and the motion validator is
supposed to have forbidden exactly that.

Everything is drawn against the planner's own state index, not against arc
length, so the pacing of the returned path is visible rather than hidden by
re-sampling.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC',
                                          'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np

SCR = Path(__file__).resolve().parent
OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis')
PACK = OUT / 'ompl'
LMT_LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
LMT_UP = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
GREEN, RED = (0.10, 0.60, 0.45), (0.86, 0.16, 0.30)


def main(key='circle', S=0.70):
    raw = np.loadtxt(SCR / 'ompl_work' / f'sol_{key}_{int(S*100)}.txt',
                     skiprows=1).reshape(-1, 8)
    d = np.load(PACK / f'{key}_pack.npz', allow_pickle=False)
    need = float(d['need'])
    qb, u = d['q_b'], d['arc']            # framework, on the arc grid

    fig, axes = plt.subplots(2, 4, figsize=(32, 12))
    n = np.arange(len(raw))
    # framework re-indexed onto the planner's own abscissa by arc, so the two
    # curves refer to the same point of the figure at the same x
    xb = np.interp(np.clip(raw[:, 7], 0, u[-1]), u, np.arange(len(u)))
    xb_arc = np.interp(np.arange(len(u)), np.clip(
        np.interp(u, raw[:, 7], n, left=0, right=len(raw) - 1),
        0, len(raw) - 1), np.arange(len(u)))
    for j in range(7):
        ax = axes[j // 4, j % 4]
        ax.plot(n, raw[:, j], lw=2.4, color=GREEN, label='OMPL 规划（原始状态序列）')
        ax.plot(np.interp(u, raw[:, 7], n, left=0, right=len(raw) - 1),
                qb[:, j], lw=2.2, color=RED, alpha=0.9,
                label='学习框架（按弧长对齐到同一横坐标）')
        ax.axhline(LMT_LO[j], color='k', ls='--', lw=1.6)
        ax.axhline(LMT_UP[j], color='k', ls='--', lw=1.6)
        pad = 0.06 * (LMT_UP[j] - LMT_LO[j])
        ax.set_ylim(LMT_LO[j] - pad, LMT_UP[j] + pad)
        ax.set_xlim(0, len(raw) - 1)
        ax.set_title(f'J{j+1}   range [{LMT_LO[j]:.2f}, {LMT_UP[j]:.2f}] rad',
                     fontsize=15)
        ax.set_xlabel('规划器状态序号', fontsize=13)
        ax.set_ylabel('关节角 [rad]', fontsize=13)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=12, loc='lower left')

    ax = axes[1, 3]
    ax.plot(n, raw[:, 7], lw=2.6, color=GREEN)
    ax.axhline(need, color='0.45', ls=':', lw=1.6)
    ax.text(2, need, f'  图形全长 {need:.2f} m', va='bottom', fontsize=12,
            color='0.35')
    ds = np.diff(raw[:, 7])
    ax.set_title(f'第 8 维：沿图形的弧长 s\n'
                 f'最小步进 {ds.min()*1000:.2f} mm，零步进 {int((ds<=1e-12).sum())} 处，'
                 f'无倒退' if ds.min() >= -1e-12 else '存在倒退',
                 fontsize=15)
    ax.set_xlim(0, len(raw) - 1); ax.set_ylim(0, max(need, raw[:, 7].max()) * 1.05)
    ax.set_xlabel('规划器状态序号', fontsize=13)
    ax.set_ylabel('弧长 s [m]', fontsize=13)
    ax.grid(alpha=0.3)

    fig.suptitle(f'{key} {S*100:.0f} cm：OMPL 规划返回的完整八维状态（q1..q7, s），'
                 f'对照学习框架　（黑虚线＝真实硬件关节限，y 轴给满整个行程）',
                 fontsize=19, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    f = PACK / f'{key}_state8.png'
    fig.savefig(f, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {f}')
    print(f'planner states {len(raw)}, s from {raw[0,7]:.4f} to {raw[-1,7]:.4f}')
    print(f'ds: min {ds.min()*1000:.3f} mm, median {np.median(ds)*1000:.3f} mm, '
          f'max {ds.max()*1000:.3f} mm, zero-steps {int((ds<=1e-12).sum())}')
    dq = np.abs(np.diff(raw[:, :7], axis=0))
    print(f'per-state joint step: max {dq.max():.4f} rad, '
          f'median {np.median(dq.max(1)):.4f} rad')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'circle',
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.70)
