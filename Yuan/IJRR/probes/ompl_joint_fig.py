"""Joint trajectories of the planner and the framework on one figure.

Both are plotted against arc length along the drawing, not against time, so the
two curves are at the same point of the circle at the same abscissa -- that is
the only way the postures are comparable. Every panel spans the joint's full
hardware range with the limits drawn, so how much room each method left itself
is visible rather than inferred from a zoomed axis.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis')
PACK = OUT / 'ompl'
LMT_LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
LMT_UP = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
COL = {'a': (0.10, 0.60, 0.45), 'b': (0.86, 0.16, 0.30)}


def main(key='circle'):
    d = np.load(PACK / f'{key}_pack.npz', allow_pickle=False)
    u = d['arc']
    qa, qb = d['q_a'], d['q_b']
    ea, eb = [float(v) for v in d['ends']]
    S = float(d['size'][0])
    need = float(d['need'])

    fig, axes = plt.subplots(2, 4, figsize=(32, 12))
    for j in range(7):
        ax = axes[j // 4, j % 4]
        ax.plot(u, qa[:, j], lw=2.4, color=COL['a'], label='OMPL 规划')
        ax.plot(u, qb[:, j], lw=2.4, color=COL['b'], label='学习框架')
        ax.axhline(LMT_LO[j], color='k', ls='--', lw=1.6)
        ax.axhline(LMT_UP[j], color='k', ls='--', lw=1.6)
        ax.axvline(need, color='0.45', ls=':', lw=1.4)
        pad = 0.06 * (LMT_UP[j] - LMT_LO[j])
        ax.set_ylim(LMT_LO[j] - pad, LMT_UP[j] + pad)
        ax.set_xlim(0, u[-1])
        ax.set_title(f'J{j+1}   range [{LMT_LO[j]:.2f}, {LMT_UP[j]:.2f}] rad',
                     fontsize=15)
        ax.set_xlabel('沿图形的弧长 [m]', fontsize=13)
        ax.set_ylabel('关节角 [rad]', fontsize=13)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=13, loc='upper left')

    ax = axes[1, 3]
    ha = np.min(np.minimum(qa - LMT_LO, LMT_UP - qa), axis=1)
    hb = np.min(np.minimum(qb - LMT_LO, LMT_UP - qb), axis=1)
    ax.plot(u, ha, lw=2.4, color=COL['a'])
    ax.plot(u, hb, lw=2.4, color=COL['b'])
    ax.axhline(0.0, color='k', ls='--', lw=1.6)
    ax.axvline(need, color='0.45', ls=':', lw=1.4)
    ax.set_xlim(0, u[-1]); ax.set_ylim(bottom=-0.05)
    ax.set_title('最紧的关节限余量（越大越安全）', fontsize=15)
    ax.set_xlabel('沿图形的弧长 [m]', fontsize=13)
    ax.set_ylabel('到最近关节限的距离 [rad]', fontsize=13)
    ax.grid(alpha=0.3)

    fig.suptitle(f'{key} {S*100:.0f} cm 一笔画：关节角随弧长的对照　'
                 f'（黑虚线＝真实硬件关节限，灰点线＝图形全长 {need:.2f} m；'
                 f'两条轨迹重采到同一条 5 mm 弧长网格，同一横坐标＝图形上同一点）',
                 fontsize=19, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    f = PACK / f'{key}_joints.png'
    fig.savefig(f, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {f}')
    print(f'min joint-limit headroom over the stroke: '
          f'OMPL {ha.min():.3f} rad, framework {hb.min():.3f} rad')
    print(f'max |q| travel per joint (rad): OMPL '
          f'{np.round(qa.max(0)-qa.min(0), 2)}')
    print(f'                               framework '
          f'{np.round(qb.max(0)-qb.min(0), 2)}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'circle')
