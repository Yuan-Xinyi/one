"""Joint-trajectory figure for the no-JL SMM re-walk — closed-loop view.

The cached `no_jl_traj_*` walk is a bi-directional integration of the
SMM nullspace with FR3 joint limits replaced by ±5π. In R^7 it never
returns to q0 because joints j0 and j6 are continuous rotational
axes whose unwrapped winding accumulates. BUT in T^7 (each joint mod
2π) the curve closes: there is a step T* such that
        q(T*) − q0  ≈  2π · k,    k ∈ Z^7,
and that step delimits exactly one full revolution of the SMM.

This figure finds T* for each branch (smallest step after `min_step`
where the wrap-error ||q − q0 − 2π·round((q−q0)/2π)|| is minimum),
then plots the JL-feasible branch slice and the no-JL extension as
two cycles of that period — so each subplot ends visually at a
q0 + 2π·k_j offset that exposes the winding number per joint.

Panels:
  - 7 per-joint subplots: q_j vs arc length over one full period.
    Red dashed lines + bands = FR3 limits. Thick opaque = the FR3
    feasible slice. Faded = the rest of the closed SMM loop.
    Magenta dashed horizontal line = q0[j] + 2π·k_j (closure level).
  - Info panel: per-branch period (rad), winding-vector k, and
    wrap-error at T*.

Usage:
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_no_jl_smm_closure.py
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_no_jl_smm_closure.py --seed 89
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_no_jl_smm_closure.py --raw  # old unwrapped view
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
    """Index in no_jl_traj closest to branch_traj[0]."""
    return int(np.argmin(
        np.linalg.norm(no_jl_traj - branch_start[None, :], axis=1)))


def _branch_slice_in_no_jl(no_jl_traj: np.ndarray, branch_traj: np.ndarray
                            ) -> tuple[int, int]:
    d0 = np.linalg.norm(no_jl_traj - branch_traj[0][None, :], axis=1)
    d1 = np.linalg.norm(no_jl_traj - branch_traj[-1][None, :], axis=1)
    i_a = int(np.argmin(d0))
    i_b = int(np.argmin(d1))
    if i_a > i_b:
        i_a, i_b = i_b, i_a
    return i_a, i_b


def _find_period(traj: np.ndarray, q0: np.ndarray, min_step: int = 30
                 ) -> tuple[int, float, np.ndarray]:
    """Smallest t > min_step where (traj[t] − q0)/2π is nearest an integer
    vector. Returns (t, wrap_err, k_int)."""
    diffs = traj - q0[None, :]
    k_int = np.round(diffs / (2.0 * np.pi))
    err = np.linalg.norm(diffs - 2.0 * np.pi * k_int, axis=1)
    err[:min_step] = np.inf
    t_star = int(np.argmin(err))
    return t_star, float(err[t_star]), k_int[t_star].astype(int)


def _one_period_slice(no_jl_traj: np.ndarray, branch_traj: np.ndarray,
                       min_step: int = 30
                       ) -> tuple[np.ndarray, int, int, np.ndarray, float]:
    """Pull one closed-period slice out of the bi-directional no-JL walk.

    The cached no-JL walk is `[bwd_reversed, fwd[1:]]` concatenated, with
    branch[0] sitting in the middle at an anchor index. We re-walk forward
    from the anchor and find the first T* where q ≈ q0 + 2π·k. The slice
    returned is `no_jl_traj[anchor : anchor + T* + 1]` so its endpoints
    differ exactly by 2π·k in joint space.

    Returns: (slice_traj, i_a_in_slice, i_b_in_slice, k_int, wrap_err)
        where (i_a, i_b) are the branch_traj endpoints' indices inside
        the returned slice.
    """
    i_anchor = _anchor_in_no_jl(no_jl_traj, branch_traj[0])
    fwd = no_jl_traj[i_anchor:]
    q0 = fwd[0]
    t_star, err, k_int = _find_period(fwd, q0, min_step=min_step)
    slc = fwd[:t_star + 1]
    # Branch endpoints in slice frame.
    d_a = np.linalg.norm(slc - branch_traj[0][None, :], axis=1)
    d_b = np.linalg.norm(slc - branch_traj[-1][None, :], axis=1)
    i_a = int(np.argmin(d_a))
    i_b = int(np.argmin(d_b))
    if i_a > i_b:
        i_a, i_b = i_b, i_a
    return slc, i_a, i_b, k_int, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--raw', action='store_true',
                        help='Plot the full bi-directional no-JL walk (unwrapped). '
                             'Default is the closed-period view.')
    args = parser.parse_args()

    d = build_or_load(seed=args.seed, force=args.force)
    n_branches = int(d['meta']['n_branches'])
    lo = d['lmt_lo']; hi = d['lmt_up']
    no_jl_limit_mult = float(d['meta'].get('no_jl_limit_mult', 5.0))
    branches = [d[f'branch_traj_{b}'].astype(np.float32)
                for b in range(n_branches)]
    no_jl = [d[f'no_jl_traj_{b}'].astype(np.float32)
             for b in range(n_branches)]

    # Build per-branch trajectory + branch-slice + (closed only) winding k.
    plot_data = []
    for b in range(n_branches):
        if args.raw:
            traj = no_jl[b]
            arc_diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(arc_diffs)]).astype(np.float32)
            i_anchor = _anchor_in_no_jl(traj, branches[b][0])
            arc = cum - cum[i_anchor]
            i_a, i_b = _branch_slice_in_no_jl(traj, branches[b])
            plot_data.append({
                'traj': traj, 'arc': arc, 'i_a': i_a, 'i_b': i_b,
                'k': None, 'wrap_err': None,
            })
        else:
            slc, i_a, i_b, k_int, err = _one_period_slice(no_jl[b], branches[b])
            arc_diffs = np.linalg.norm(np.diff(slc, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(arc_diffs)]).astype(np.float32)
            plot_data.append({
                'traj': slc, 'arc': cum, 'i_a': i_a, 'i_b': i_b,
                'k': k_int, 'wrap_err': err,
            })

    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    axes = axes.flatten()

    for j in range(7):
        ax = axes[j]
        ax.axhspan(lo[j] - 1, lo[j], color='red', alpha=0.10)
        ax.axhspan(hi[j], hi[j] + 1, color='red', alpha=0.10)
        ax.axhline(lo[j], color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(hi[j], color='red', linestyle='--', linewidth=0.8, alpha=0.7)

        y_lo_all = float(lo[j]); y_hi_all = float(hi[j])
        for bid in range(n_branches):
            rgb = cmap(bid % 10)
            arc = plot_data[bid]['arc']
            q = plot_data[bid]['traj'][:, j]
            i_a = plot_data[bid]['i_a']
            i_b = plot_data[bid]['i_b']

            # Faded curve outside branch slice.
            if i_a > 0:
                ax.plot(arc[:i_a + 1], q[:i_a + 1], '-', color=rgb,
                        alpha=0.30, linewidth=1.0)
            if i_b < len(arc) - 1:
                ax.plot(arc[i_b:], q[i_b:], '-', color=rgb,
                        alpha=0.30, linewidth=1.0)
            # Opaque JL-feasible slice.
            ax.plot(arc[i_a:i_b + 1], q[i_a:i_b + 1], '-', color=rgb,
                    alpha=0.95, linewidth=2.0,
                    label=(f'br{bid}' if j == 0 else None))
            # Closure level (q0 + 2π·k_j) for closed view.
            if not args.raw:
                k_j = int(plot_data[bid]['k'][j])
                closure_level = float(q[0] + 2.0 * np.pi * k_j)
                ax.axhline(closure_level, color=rgb, linestyle=':',
                           linewidth=1.0, alpha=0.6)
                # Endpoint marker at the closure point.
                ax.scatter([arc[0], arc[-1]], [q[0], q[-1]],
                           s=55, c=[rgb], edgecolors='black',
                           linewidths=0.5, marker='*', zorder=6)
            else:
                ax.scatter([arc[0], arc[-1]], [q[0], q[-1]],
                           s=45, c=[rgb], edgecolors='black',
                           linewidths=0.5, marker='*', zorder=5)
            ax.scatter([arc[i_a], arc[i_b]], [q[i_a], q[i_b]],
                       s=22, c=[rgb], edgecolors='black', linewidths=0.4,
                       zorder=6)

            y_lo_all = min(y_lo_all, float(q.min()))
            y_hi_all = max(y_hi_all, float(q.max()))

        ax.set_title(f'j{j + 1}  limits [{lo[j]:.2f}, {hi[j]:.2f}]', fontsize=10)
        ax.set_xlabel('arc length along one SMM period (rad)'
                      if not args.raw
                      else 'signed arc length along no-JL walk (rad)',
                      fontsize=8)
        ax.set_ylabel('q [rad]', fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.set_ylim(y_lo_all - 0.3, y_hi_all + 0.3)
        if j == 0:
            ax.legend(fontsize=8, loc='best')

    # Info panel.
    ax_info = axes[7]
    ax_info.axis('off')
    if args.raw:
        header = [
            f'seed = {args.seed}',
            f'{n_branches} SMM branches  (no-JL limit = ±{no_jl_limit_mult:.1f}π)',
            'mode: raw (bi-directional unwrapped no-JL walk)',
            '',
        ]
        body = []
        for bid in range(n_branches):
            nj = no_jl[bid]
            arc_total = float(np.sum(np.linalg.norm(np.diff(nj, axis=0), axis=1)))
            arc_feasible = float(np.sum(np.linalg.norm(
                np.diff(branches[bid], axis=0), axis=1)))
            end_to_start = float(np.linalg.norm(nj[-1] - nj[0]))
            body.append(f'  br{bid}: feasible {arc_feasible:.2f} rad / '
                        f'no-JL {arc_total:.2f} rad   '
                        f'Δ_end={end_to_start:.2f}')
    else:
        header = [
            f'seed = {args.seed}',
            f'{n_branches} SMM branches  (no-JL limit = ±{no_jl_limit_mult:.1f}π)',
            'mode: closed-period (one revolution of the SMM)',
            '',
            'thick opaque = FR3-feasible branch slice',
            'faded line   = rest of the closed SMM period',
            'dotted line  = closure level q0[j] + 2π·k_j',
            '★            = SMM start & period-end points',
            '',
        ]
        body = []
        for bid in range(n_branches):
            slc = plot_data[bid]['traj']
            k = plot_data[bid]['k']
            err = plot_data[bid]['wrap_err']
            T_star = len(slc) - 1
            period_arc = float(plot_data[bid]['arc'][-1])
            k_str = ' '.join(f'{ki:+d}' for ki in k)
            body.append(f'  br{bid}: T*={T_star} steps, period={period_arc:.2f} rad')
            body.append(f'         k=[{k_str}]  wrap-err={err:.4f} rad')
    info = header + body + [
        '',
        'reading the figure:',
        '  for each joint j, the trajectory ENDS at q0[j] + 2π·k_j',
        '  (the colored dotted line). On j0 and j6 (continuous',
        '  rotational axes) k is nonzero, so the SMM curve winds',
        '  several turns; on j1..j5 it returns to q0[j] (k=0).',
        '  This is the meaning of "the SMM closes in T^7".',
    ]
    ax_info.text(0.0, 1.0, '\n'.join(info), fontsize=9,
                 family='monospace', verticalalignment='top')

    mode_tag = 'raw' if args.raw else 'closed'
    fig.suptitle(
        f'SMM joint trajectories with JL removed — {mode_tag} view  '
        f'(seed={args.seed}, {n_branches} branches)',
        fontsize=12, y=1.005)
    fig.tight_layout()

    out_path = (Path(args.out) if args.out else
                FIG_DIR / f'fig_no_jl_smm_{mode_tag}_seed{args.seed}.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
