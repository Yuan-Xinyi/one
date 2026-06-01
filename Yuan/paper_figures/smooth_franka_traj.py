"""Smooth the captured joint trajectories so they replay cleanly on the
real Franka (no libfranka velocity/acceleration-discontinuity reflexes).

The sim trajectories in `fig04_traj_task<T>.npz` contain C1 kinks -- most
visibly in `hybrid`, where the RL<->classical switch flips the joint
velocity direction in a single step. Ruckig faithfully drives through
those kinks, producing high instantaneous acceleration/jerk that trips
franky's 1 kHz reflex. This script low-pass-filters each `<mode>_q`
trajectory (Savitzky-Golay by default, or a C2 smoothing spline) so the
velocity/acceleration stay continuous, writes a new NPZ with the same
keys (drop-in for replay_franka_traj.py), and emits a raw-vs-smoothed
comparison figure per mode (position + joint velocity).

Usage:
    python -m Yuan.paper_figures.smooth_franka_traj \\
        --npz Yuan/paper_figures/fig04_traj_task7199.npz
    python -m Yuan.paper_figures.smooth_franka_traj \\
        --npz ...npz --method spline --resample 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import make_smoothing_spline

# Shared 7-joint palette (q1..q7), same as the per-joint paper figures.
JOINT_COLORS = [
    '#F27970', '#BB9727', '#54B345', '#32B897',
    '#05B9E2', '#8983BF', '#C76DA2',
]
MODES = ('classical', 'rl', 'hybrid')


def _odd_window(win: int, n: int) -> int:
    """Clamp a Savitzky-Golay window to be odd and < n."""
    win = min(int(win), n - (1 - n % 2))   # < n, keep parity room
    if win % 2 == 0:
        win -= 1
    return max(win, 3)


def smooth_savgol(q: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Per-joint Savitzky-Golay filter over the step axis (length-preserving)."""
    n = len(q)
    win = _odd_window(window, n)
    p = min(int(poly), win - 1)
    return savgol_filter(q, win, p, axis=0, mode='interp').astype(np.float32)


def smooth_spline(q: np.ndarray, lam, n_out: int) -> np.ndarray:
    """Per-joint C2 smoothing spline; resampled onto n_out points.

    lam=None lets scipy pick the penalty via generalized cross-validation.
    """
    n = len(q)
    t = np.arange(n, dtype=np.float64)
    t_out = np.linspace(0.0, n - 1, n_out)
    out = np.empty((n_out, q.shape[1]), dtype=np.float32)
    for j in range(q.shape[1]):
        spl = make_smoothing_spline(t, q[:, j].astype(np.float64), lam=lam)
        out[:, j] = spl(t_out).astype(np.float32)
    return out


def resample_savgol(q_smooth: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly resample an already-smoothed, length-preserving array."""
    n = len(q_smooth)
    if n_out == n:
        return q_smooth
    t = np.arange(n, dtype=np.float64)
    t_out = np.linspace(0.0, n - 1, n_out)
    return np.stack([np.interp(t_out, t, q_smooth[:, j])
                     for j in range(q_smooth.shape[1])], axis=1).astype(np.float32)


def discontinuity_stats(q: np.ndarray) -> tuple[float, float]:
    """Max |1st diff| (velocity proxy) and max |2nd diff| (accel proxy), rad."""
    v = np.abs(np.diff(q, axis=0)).max() if len(q) > 1 else 0.0
    a = np.abs(np.diff(q, n=2, axis=0)).max() if len(q) > 2 else 0.0
    return float(v), float(a)


def plot_compare(raw, smooth, mode, out_path, dpi):
    t_raw = np.arange(len(raw))
    t_sm = np.linspace(0, len(raw) - 1, len(smooth))
    fig, (ax_p, ax_v) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for j in range(7):
        c = JOINT_COLORS[j]
        ax_p.plot(t_raw, raw[:, j], '--', color=c, lw=1.0, alpha=0.45)
        ax_p.plot(t_sm, smooth[:, j], '-', color=c, lw=2.0,
                  label=fr'$q_{{{j + 1}}}$')
        # Joint velocity proxy = per-step finite difference.
        ax_v.plot(t_raw[1:], np.diff(raw[:, j]), '--', color=c, lw=1.0, alpha=0.45)
        ax_v.plot(t_sm[1:], np.diff(smooth[:, j]), '-', color=c, lw=2.0)

    ax_p.set_ylabel('q  (rad)', fontsize=13)
    ax_v.set_ylabel(r'$\Delta q$ / step  (rad)', fontsize=13)
    ax_v.set_xlabel('step', fontsize=13)
    ax_p.set_title(f'{mode}: raw (dashed) vs smoothed (solid)', fontsize=13)
    for ax in (ax_p, ax_v):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=11)
    ax_p.legend(ncol=7, fontsize=9, loc='upper center',
                bbox_to_anchor=(0.5, 1.18), frameon=False, columnspacing=1.0,
                handlelength=1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved figure: {out_path}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', default='Yuan/paper_figures/fig04_traj_task7199.npz')
    p.add_argument('--out', default=None,
                   help='output NPZ (default: <input>_smooth.npz)')
    p.add_argument('--method', choices=['savgol', 'spline'], default='savgol')
    p.add_argument('--sg-window', type=int, default=11,
                   help='Savitzky-Golay window (odd, clamped < len)')
    p.add_argument('--sg-poly', type=int, default=3,
                   help='Savitzky-Golay polynomial order')
    p.add_argument('--lam', type=float, default=None,
                   help='spline smoothing penalty (default: GCV auto)')
    p.add_argument('--resample', type=int, default=0,
                   help='resample each trajectory to N points (0 = keep length)')
    p.add_argument('--out-dir', default=None,
                   help='where to write comparison figures (default: NPZ dir)')
    p.add_argument('--dpi', type=int, default=180)
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.npz)
    data = dict(np.load(in_path, allow_pickle=False))
    task = int(data['task'])
    print(f'loaded {in_path}  (task {task})')

    out = dict(data)   # copy every original key; overwrite <mode>_q below
    fig_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    fig_dir.mkdir(parents=True, exist_ok=True)

    method_str = (f'savgol(win={args.sg_window},poly={args.sg_poly})'
                  if args.method == 'savgol'
                  else f'spline(lam={args.lam})')

    for mode in MODES:
        key = f'{mode}_q'
        if key not in data:
            continue
        raw = data[key].astype(np.float32)
        n_out = args.resample if args.resample > 0 else len(raw)

        if args.method == 'savgol':
            sm = smooth_savgol(raw, args.sg_window, args.sg_poly)
            sm = resample_savgol(sm, n_out)
        else:
            sm = smooth_spline(raw, args.lam, n_out)

        v0, a0 = discontinuity_stats(raw)
        v1, a1 = discontinuity_stats(sm)
        print(f'{mode:9s}: {len(raw)}->{len(sm)} pts | '
              f'max|Δq| {v0:.4f}->{v1:.4f} rad | '
              f'max|Δ²q| {a0:.4f}->{a1:.4f} rad')

        out[key] = sm.astype(np.float32)
        out[f'{mode}_q_raw'] = raw                       # keep original for ref
        if n_out != len(raw):
            # using_rl no longer index-aligns after resampling; nearest-map it.
            urlk = f'{mode}_using_rl'
            if urlk in data:
                idx = np.round(np.linspace(0, len(raw) - 1, n_out)).astype(int)
                out[urlk] = data[urlk][idx]

        plot_compare(raw, sm, mode, fig_dir / f'fig04_smooth_compare_{mode}.png',
                     args.dpi)

    out['smoothing'] = np.array(method_str)
    out_path = Path(args.out) if args.out else \
        in_path.with_name(in_path.stem + '_smooth.npz')
    np.savez(out_path, **out)
    print(f'\nwrote smoothed NPZ: {out_path}')
    print(f'replay with:\n  python Yuan/paper_figures/replay_franka_traj.py '
          f'--npz {out_path} --modes hybrid --dry-run')


if __name__ == '__main__':
    main()
