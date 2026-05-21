"""Per-joint no-JL closed-period figures — one PNG per joint.

A focused companion to `fig_single_joint_jl_compare.py`: drops the
with-JL panel and emits one figure per joint, showing all branches'
SMM trajectories over a single closed period (joint limits removed).

Layout per figure:
  - X-axis: arc length along one closed SMM period (rad), measured
    from each branch's start posture.
  - Y-axis: q_j (rad).
  - Red dashed lines + bands: FR3 joint limits for joint j.
  - Thick opaque segment: the FR3-feasible slice of each branch.
  - Faded segments: the rest of the closed SMM period.
  - Dotted horizontal line: closure level q0[j] + 2π·k_j per branch.
  - ★ markers: SMM period endpoints.

Usage (default emits 7 figures, j0..j6):
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_no_jl_per_joint.py
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_no_jl_per_joint.py --joint 3
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_no_jl_per_joint.py --seed 89
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


def _draw_one_joint(ax, j: int, lo_j: float, hi_j: float,
                    branches, no_jl, cmap):
    ax.axhspan(lo_j - 1, lo_j, color='red', alpha=0.10)
    ax.axhspan(hi_j, hi_j + 1, color='red', alpha=0.10)
    ax.axhline(lo_j, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(hi_j, color='red', linestyle='--', linewidth=0.8, alpha=0.7)

    y_lo = lo_j
    y_hi = hi_j
    period_info = []
    for bid in range(len(branches)):
        rgb = cmap(bid % 10)
        slc, i_a, i_b, k_int, err = _one_period_slice(
            no_jl[bid], branches[bid])
        arc_diffs = np.linalg.norm(np.diff(slc, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(arc_diffs)]).astype(np.float32)
        q = slc[:, j]

        if i_a > 0:
            ax.plot(arc[:i_a + 1], q[:i_a + 1], '-', color=rgb,
                    alpha=0.30, linewidth=1.0)
        if i_b < len(arc) - 1:
            ax.plot(arc[i_b:], q[i_b:], '-', color=rgb,
                    alpha=0.30, linewidth=1.0)
        ax.plot(arc[i_a:i_b + 1], q[i_a:i_b + 1], '-', color=rgb,
                alpha=0.95, linewidth=2.0, label=f'branch {bid}')

        k_j = int(k_int[j])
        closure_level = float(q[0] + 2.0 * np.pi * k_j)
        ax.axhline(closure_level, color=rgb, linestyle=':',
                   linewidth=1.0, alpha=0.6)
        ax.scatter([arc[0], arc[-1]], [q[0], q[-1]],
                   s=55, c=[rgb], edgecolors='black', linewidths=0.5,
                   marker='*', zorder=6)
        ax.scatter([arc[i_a], arc[i_b]], [q[i_a], q[i_b]],
                   s=22, c=[rgb], edgecolors='black', linewidths=0.4,
                   zorder=6)

        y_lo = min(y_lo, float(q.min()))
        y_hi = max(y_hi, float(q.max()))
        period_info.append(
            f'br{bid}: period={float(arc[-1]):.2f} rad, '
            f'k_j{j + 1}={k_j:+d}, wrap-err={err:.4f}')
    return y_lo, y_hi, period_info


def _emit_one(seed: int, j: int, branches, no_jl, lo, hi,
              cmap, fig_dir: Path, out_override: str | None):
    fig, ax = plt.subplots(figsize=(10, 8))
    y_lo, y_hi, period_info = _draw_one_joint(
        ax, j, float(lo[j]), float(hi[j]), branches, no_jl, cmap)
    ax.set_ylim(y_lo - 0.3, y_hi + 0.3)
    # ax.set_title(
    #     f'Joint j{j + 1} — without joint limits — one closed SMM period\n'
    #     f'(seed={seed}, FR3 limits [{lo[j]:.2f}, {hi[j]:.2f}])\n'
    #     + '  |  '.join(period_info),
    #     fontsize=20)
    ax.set_xlabel('arc length along SMM (rad)', fontsize=22)
    ax.set_ylabel(f'q[{j + 1}] (rad)', fontsize=22)
    ax.tick_params(labelsize=18)
    ax.legend(fontsize=18, loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = (Path(out_override) if out_override else
                fig_dir / f'fig_no_jl_j{j + 1}_seed{seed}.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--joint', type=int, default=-1,
                        help='which joint to plot (1..7); -1 = all 7 (default)')
    parser.add_argument('--out', type=str, default=None,
                        help='only used when --joint is a single index')
    args = parser.parse_args()

    if args.joint != -1 and not (1 <= args.joint <= 7):
        raise SystemExit(f'--joint must be in [1, 7] or -1, got {args.joint}')
    # --joint is 1-based for display; convert to 0-based array index.
    j_arg = (args.joint - 1) if args.joint != -1 else -1

    d = build_or_load(seed=args.seed, force=args.force)
    n_branches = int(d['meta']['n_branches'])
    lo = d['lmt_lo']; hi = d['lmt_up']
    branches = [d[f'branch_traj_{b}'].astype(np.float32)
                for b in range(n_branches)]
    no_jl = [d[f'no_jl_traj_{b}'].astype(np.float32)
             for b in range(n_branches)]
    cmap = plt.get_cmap('tab10')

    js = [j_arg] if j_arg != -1 else list(range(7))
    for j in js:
        _emit_one(args.seed, j, branches, no_jl, lo, hi, cmap,
                  FIG_DIR, args.out if (len(js) == 1) else None)


if __name__ == '__main__':
    main()
