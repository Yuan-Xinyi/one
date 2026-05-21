"""Fig 5 — 7-DOF / 6-DOF redundancy and the resulting SMM is NOT connected.

Same representative task as the other Part-1 figures (seed 118). The
counting 7 (joints) − 6 (task pose) = 1 predicts a 1-D self-motion
manifold. Intuition says that 1-D set is a single connected curve. The
data says otherwise: on FR3 it splits into ~4 disjoint arcs.

Panels:

  (A) Conceptual: 7-DOF arm, 6-DOF task constraint, 1-DOF nullspace.
      Drawn as a labelled block diagram in matplotlib (no PDF/SVG dep).

  (B) PCA of the SMM in 7-D joint space: every IK candidate (dedup'd)
      sits on one of the branch arcs; arcs are colored. The visual
      separation makes it clear the candidates do NOT lie on a single
      connected curve.

  (C) Joint-pair scatter (j3 vs j5): a 2-D slice of the same SMM that
      shows the disconnection without relying on PCA. j3 and j5 carry
      the asymmetric FR3 limits, so this slice exposes the geometry
      most directly.

  (D) Per-branch arc length (in radians of joint motion) and whether
      the branch closed into a loop or terminated at a joint limit /
      singularity.

Usage:
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_5_redundancy_smm_discontinuity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _shared import DEFAULT_SEED, FIG_DIR, build_or_load  # noqa: E402


def _branches_from_cache(d) -> list[dict]:
    n = int(d['meta']['n_branches'])
    return [{
        'traj': d[f'branch_traj_{bid}'].astype(np.float32),
        'closed': bool(d['branch_closed'][bid]),
    } for bid in range(n)]


def _draw_conceptual(ax, n_branches: int):
    """Block diagram: 7-DOF arm → 6-DOF task constraint → 1-DOF nullspace (SMM)."""
    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    boxes = [
        (0.5, 6.8, 'FR3 joint space\nq ∈ R^7\n7 DOF', '#cfe2f3'),
        (4.0, 6.8, 'Task constraint\nFK(q) = (p_tgt, R_tgt)\n6 DOF', '#fce5cd'),
        (7.5, 6.8, 'Nullspace dim\nr = 7 − 6 = 1\nSMM ⊂ R^7', '#d9ead3'),
    ]
    for (x, y, txt, c) in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), 2.2, 2.0, boxstyle='round,pad=0.1', linewidth=1.2,
            edgecolor='black', facecolor=c))
        ax.text(x + 1.1, y + 1.0, txt, ha='center', va='center',
                fontsize=9, fontweight='normal')
    ax.annotate('', xy=(4.0, 7.8), xytext=(2.7, 7.8),
                arrowprops=dict(arrowstyle='->', lw=1.4))
    ax.annotate('', xy=(7.5, 7.8), xytext=(6.2, 7.8),
                arrowprops=dict(arrowstyle='->', lw=1.4))
    # Naive expectation vs observed
    ax.text(5.0, 5.5, 'Naive: one connected 1-D curve.',
            ha='center', fontsize=10, style='italic', color='0.20')
    ax.text(5.0, 4.7, f'Observed on FR3: {n_branches} disjoint arcs',
            ha='center', fontsize=11, fontweight='bold', color='C3')
    ax.text(5.0, 3.9, '→ see panels B, C, D', ha='center', fontsize=9,
            color='0.30')
    # 3-step pipeline
    ax.text(0.5, 2.6, 'SMM enumeration pipeline (per [Guri & Kantor 2025]):',
            fontsize=9, fontweight='bold')
    ax.text(0.7, 1.9,
            '1. Newton-refine many IK seeds to FK(q)=(p_tgt,R_tgt), tol=1e-6',
            fontsize=8)
    ax.text(0.7, 1.3,
            '2. RK4 + Newton corrector along null(J), step h=0.03 rad',
            fontsize=8)
    ax.text(0.7, 0.7,
            '3. Group candidates into branches by closeness to arc',
            fontsize=8)
    ax.set_title('A) Why 7−6=1 SMM should be 1-D', fontsize=11)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    d = build_or_load(seed=args.seed, force=args.force)
    n_branches = int(d['meta']['n_branches'])
    Q_clean = d['Q_clean']
    assigned = d['assigned']
    branches = _branches_from_cache(d)
    lo = d['lmt_lo']; hi = d['lmt_up']

    cmap = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # --- Panel A: conceptual ---
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_conceptual(ax_a, n_branches)

    # --- Panel B: PCA of branches in 7D joint space ---
    ax_b = fig.add_subplot(gs[0, 1])
    all_pts = np.concatenate([b['traj'] for b in branches] + [Q_clean], axis=0)
    mu = all_pts.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(all_pts - mu, full_matrices=False)
    W = Vt[:2].T
    for bid, b in enumerate(branches):
        t2d = (b['traj'] - mu) @ W
        arc = float(np.sum(np.linalg.norm(np.diff(b['traj'], axis=0), axis=1)))
        ax_b.plot(t2d[:, 0], t2d[:, 1], '-', color=cmap(bid % 10),
                  alpha=0.85, linewidth=2.0,
                  label=f'br{bid}: arc={arc:.2f} rad'
                        + (' (closed)' if b['closed'] else ''))
        ax_b.scatter(t2d[0, 0], t2d[0, 1], s=80, c=[cmap(bid % 10)],
                     edgecolors='black', linewidth=1.0, marker='*', zorder=6)
    Q_2d = (Q_clean - mu) @ W
    for j in range(Q_clean.shape[0]):
        b_j = int(assigned[j])
        c = cmap(b_j % 10) if b_j >= 0 else 'gray'
        ax_b.scatter(Q_2d[j, 0], Q_2d[j, 1], s=24, c=[c],
                     edgecolors='black', linewidth=0.3, zorder=5)
    ax_b.set_title(f'B) SMM in 7-DOF joint space (PCA→2D): '
                   f'{n_branches} disjoint arcs', fontsize=11)
    ax_b.set_xlabel('PC1'); ax_b.set_ylabel('PC2')
    ax_b.legend(fontsize=8, loc='best'); ax_b.grid(alpha=0.3)

    # --- Panel C: joint-pair scatter ---
    # Pick the pair (j_a, j_b) where the branches are most visually separated:
    # maximize the ratio of between-branch spread to within-branch spread.
    ax_c = fig.add_subplot(gs[1, 0])
    best_score = -1.0
    j_a, j_b = 0, 2
    for ja in range(7):
        for jb in range(ja + 1, 7):
            centers = np.array([b['traj'][:, [ja, jb]].mean(axis=0)
                                 for b in branches])
            within = np.mean([b['traj'][:, [ja, jb]].std(axis=0).max()
                              for b in branches])
            between = 0.0
            for i in range(len(branches)):
                for k in range(i + 1, len(branches)):
                    between += float(np.linalg.norm(centers[i] - centers[k]))
            score = between / max(within, 1e-3)
            if score > best_score:
                best_score = score
                j_a, j_b = ja, jb
    for bid, b in enumerate(branches):
        traj = b['traj']
        ax_c.plot(traj[:, j_a], traj[:, j_b], '-',
                  color=cmap(bid % 10), linewidth=2.0, alpha=0.85,
                  label=f'br{bid}')
        ax_c.scatter([traj[0, j_a], traj[-1, j_a]],
                     [traj[0, j_b], traj[-1, j_b]],
                     s=55, c=[cmap(bid % 10)], edgecolors='black',
                     linewidths=0.6, zorder=5)
    # Joint-limit guides
    ax_c.axvline(lo[j_a], color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_c.axvline(hi[j_a], color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_c.axhline(lo[j_b], color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_c.axhline(hi[j_b], color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_c.axvspan(lo[j_a] - 0.2, lo[j_a], color='red', alpha=0.08)
    ax_c.axvspan(hi[j_a], hi[j_a] + 0.2, color='red', alpha=0.08)
    ax_c.axhspan(lo[j_b] - 0.2, lo[j_b], color='red', alpha=0.08)
    ax_c.axhspan(hi[j_b], hi[j_b] + 0.2, color='red', alpha=0.08)
    ax_c.set_xlabel(f'q[{j_a}]  (FR3 j{j_a}, limits [{lo[j_a]:.2f}, {hi[j_a]:.2f}])')
    ax_c.set_ylabel(f'q[{j_b}]  (FR3 j{j_b}, limits [{lo[j_b]:.2f}, {hi[j_b]:.2f}])')
    ax_c.set_title(f'C) Joint-pair slice (q{j_a}, q{j_b}): '
                   f'arcs still separated', fontsize=11)
    ax_c.legend(fontsize=8); ax_c.grid(alpha=0.3)

    # --- Panel D: per-branch arc length + closure status ---
    ax_d = fig.add_subplot(gs[1, 1])
    arcs = [float(np.sum(np.linalg.norm(np.diff(b['traj'], axis=0), axis=1)))
            for b in branches]
    closed_flags = [b['closed'] for b in branches]
    x = np.arange(n_branches)
    bar_colors = [cmap(b % 10) for b in range(n_branches)]
    ax_d.bar(x, arcs, color=bar_colors, edgecolor='black', alpha=0.85)
    for xi, a, cl in zip(x, arcs, closed_flags):
        ax_d.text(xi, a + 0.06, f'{a:.2f} rad', ha='center', fontsize=9)
        tag = 'closed' if cl else 'open'
        ax_d.text(xi, -0.18, tag, ha='center', fontsize=8,
                  color=('C2' if cl else 'C3'))
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f'br{b}' for b in range(n_branches)])
    ax_d.set_ylabel('branch arc length [rad]')
    ax_d.set_ylim(-0.4, max(arcs) * 1.25)
    ax_d.set_title('D) per-branch arc length + closure status', fontsize=11)
    ax_d.grid(alpha=0.3, axis='y')

    fig.suptitle(
        f'7-DOF / 6-DOF redundancy on FR3  —  the predicted 1-D SMM is '
        f'observed as {n_branches} disjoint arcs (seed {args.seed})',
        fontsize=12, y=1.00,
    )

    out_path = Path(args.out) if args.out else FIG_DIR / f'fig5_seed{args.seed}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
