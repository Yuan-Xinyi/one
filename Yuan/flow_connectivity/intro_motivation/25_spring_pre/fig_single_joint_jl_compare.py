"""Single-joint joint-trajectory figure: with-JL vs without-JL side-by-side.

A focused companion to `fig_no_jl_smm_closure.py`. Pulls one joint
(default j1) out into its own panel and plots it twice:

  Left  — with FR3 joint limits: each branch only traces the
          FR3-feasible slice of its SMM (branch_traj_*). Outside the
          red JL bands the curve does not exist.
  Right — without joint limits: each branch traces a full closed
          period of its SMM (≈685 steps for seed 17). The curve may
          leave the red bands; the JL-feasible slice is still drawn
          thicker for reference.

X-axis on both panels is arc length along the SMM (rad), measured
from the branch's own start posture.

Usage:
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_single_joint_jl_compare.py
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_single_joint_jl_compare.py --joint 3
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_single_joint_jl_compare.py --seed 89 --joint 1
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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _shared import DEFAULT_SEED, FIG_DIR, build_or_load  # noqa: E402


def _anchor_in_no_jl(no_jl_traj: np.ndarray, branch_start: np.ndarray) -> int:
    return int(np.argmin(
        np.linalg.norm(no_jl_traj - branch_start[None, :], axis=1)))


def _find_period(traj: np.ndarray, q0: np.ndarray, min_step: int = 30
                 ) -> tuple[int, float, np.ndarray]:
    diffs = traj - q0[None, :]
    k_int = np.round(diffs / (2.0 * np.pi))
    err = np.linalg.norm(diffs - 2.0 * np.pi * k_int, axis=1)
    err[:min_step] = np.inf
    t_star = int(np.argmin(err))
    return t_star, float(err[t_star]), k_int[t_star].astype(int)


def _one_period_slice(no_jl_traj: np.ndarray, branch_traj: np.ndarray,
                       min_step: int = 30
                       ) -> tuple[np.ndarray, int, int, np.ndarray, float]:
    i_anchor = _anchor_in_no_jl(no_jl_traj, branch_traj[0])
    fwd = no_jl_traj[i_anchor:]
    q0 = fwd[0]
    t_star, err, k_int = _find_period(fwd, q0, min_step=min_step)
    slc = fwd[:t_star + 1]
    d_a = np.linalg.norm(slc - branch_traj[0][None, :], axis=1)
    d_b = np.linalg.norm(slc - branch_traj[-1][None, :], axis=1)
    i_a = int(np.argmin(d_a))
    i_b = int(np.argmin(d_b))
    if i_a > i_b:
        i_a, i_b = i_b, i_a
    return slc, i_a, i_b, k_int, err


def _draw_jl_bands(ax, lo_j: float, hi_j: float):
    ax.axhspan(lo_j - 1, lo_j, color='red', alpha=0.10)
    ax.axhspan(hi_j, hi_j + 1, color='red', alpha=0.10)
    ax.axhline(lo_j, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(hi_j, color='red', linestyle='--', linewidth=0.8, alpha=0.7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--joint', type=int, default=2,
                        help='which joint to plot (1..7, default 2)')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    if not (1 <= args.joint <= 7):
        raise SystemExit(f'--joint must be in [1, 7], got {args.joint}')
    # --joint is 1-based for display; convert to 0-based array index.
    j = args.joint - 1

    d = build_or_load(seed=args.seed, force=args.force)
    n_branches = int(d['meta']['n_branches'])
    lo = d['lmt_lo']; hi = d['lmt_up']
    branches = [d[f'branch_traj_{b}'].astype(np.float32)
                for b in range(n_branches)]
    no_jl = [d[f'no_jl_traj_{b}'].astype(np.float32)
             for b in range(n_branches)]

    cmap = plt.get_cmap('tab10')
    fig, (ax_with, ax_no) = plt.subplots(1, 2, figsize=(20, 8.5),
                                          sharey=True)

    _draw_jl_bands(ax_with, float(lo[j]), float(hi[j]))
    _draw_jl_bands(ax_no, float(lo[j]), float(hi[j]))

    y_lo_all = float(lo[j])
    y_hi_all = float(hi[j])

    # ---- Left panel: with joint limits ----
    for bid in range(n_branches):
        rgb = cmap(bid % 10)
        br = branches[bid]
        arc_diffs = np.linalg.norm(np.diff(br, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(arc_diffs)]).astype(np.float32)
        ax_with.plot(arc, br[:, j], '-', color=rgb, alpha=0.95, linewidth=2.0,
                     label=f'branch {bid}')
        ax_with.scatter([arc[0], arc[-1]], [br[0, j], br[-1, j]],
                        s=45, c=[rgb], edgecolors='black', linewidths=0.5,
                        marker='*', zorder=6)
        y_lo_all = min(y_lo_all, float(br[:, j].min()))
        y_hi_all = max(y_hi_all, float(br[:, j].max()))

    # ---- Right panel: without joint limits (one closed period) ----
    period_info = []
    for bid in range(n_branches):
        rgb = cmap(bid % 10)
        slc, i_a, i_b, k_int, err = _one_period_slice(no_jl[bid], branches[bid])
        arc_diffs = np.linalg.norm(np.diff(slc, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(arc_diffs)]).astype(np.float32)
        q = slc[:, j]

        # Faded outside the FR3-feasible slice.
        if i_a > 0:
            ax_no.plot(arc[:i_a + 1], q[:i_a + 1], '-', color=rgb,
                       alpha=0.30, linewidth=1.0)
        if i_b < len(arc) - 1:
            ax_no.plot(arc[i_b:], q[i_b:], '-', color=rgb,
                       alpha=0.30, linewidth=1.0)
        # Opaque FR3-feasible slice.
        ax_no.plot(arc[i_a:i_b + 1], q[i_a:i_b + 1], '-', color=rgb,
                   alpha=0.95, linewidth=2.0, label=f'branch {bid}')

        # Closure level: q0[j] + 2π·k_j.
        k_j = int(k_int[j])
        closure_level = float(q[0] + 2.0 * np.pi * k_j)
        ax_no.axhline(closure_level, color=rgb, linestyle=':',
                      linewidth=1.0, alpha=0.6)
        ax_no.scatter([arc[0], arc[-1]], [q[0], q[-1]],
                      s=55, c=[rgb], edgecolors='black', linewidths=0.5,
                      marker='*', zorder=6)
        ax_no.scatter([arc[i_a], arc[i_b]], [q[i_a], q[i_b]],
                      s=22, c=[rgb], edgecolors='black', linewidths=0.4,
                      zorder=6)

        y_lo_all = min(y_lo_all, float(q.min()))
        y_hi_all = max(y_hi_all, float(q.max()))
        period_info.append(
            f'br{bid}: T*={len(slc) - 1}, period={float(arc[-1]):.2f} rad, '
            f'k_j{j + 1}={k_j:+d}, wrap-err={err:.4f}')

    # Common y-range so the two panels are visually comparable.
    y_pad = 0.3
    ax_with.set_ylim(y_lo_all - y_pad, y_hi_all + y_pad)
    ax_no.set_ylim(y_lo_all - y_pad, y_hi_all + y_pad)

    ax_with.set_title(f'with joint limits  —  FR3-feasible slice only',
                      fontsize=24)
    ax_no.set_title(
        f'without joint limits  —  one closed SMM period',
        fontsize=24)
    for ax in (ax_with, ax_no):
        ax.set_xlabel('arc length along SMM (rad)', fontsize=24)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=18, loc='best')
        ax.tick_params(labelsize=18)
    ax_with.set_ylabel(f'q[{j + 1}] (rad)', fontsize=20)

    fig.suptitle(
        f'Joint j{j + 1}  —  with vs. without joint limits  '
        f'(seed={args.seed}, FR3 limits [{lo[j]:.2f}, {hi[j]:.2f}])\n'
        + '  |  '.join(period_info),
        fontsize=24, y=1.02,
    )
    fig.tight_layout()

    out_path = (Path(args.out) if args.out else
                FIG_DIR / f'fig_j{j + 1}_jl_compare_seed{args.seed}.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
