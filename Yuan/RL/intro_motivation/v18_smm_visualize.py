"""Visualize SMM branches (from v18_smm_enumerate) in the ONE viewer.

Loads data/smm_branches.jsonl and renders each branch as an arm that
sweeps along its 1D self-motion manifold. All arms share base origin
(physically overlapping), each tinted a distinct color. The TCP stays
fixed at p_tgt throughout — that's what makes them all valid IK
solutions for the same pose, just on different SMM components.

Modes:
  --mode animate (default): one arm per branch, each ping-pongs along
    its own SMM curve. Watch the joints move while the pen tip stays
    locked at the yellow target sphere.
  --mode static : K transparent snapshots per branch overlaid in one
    static scene, showing the "fan" of valid postures.

Usage:
    python -m Yuan.RL.intro_motivation.v18_smm_visualize
    python -m Yuan.RL.intro_motivation.v18_smm_visualize --mode static
    python -m Yuan.RL.intro_motivation.v18_smm_visualize --layout side
"""
from __future__ import annotations

import argparse
import builtins
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from Yuan.RL.fr3_with_pen import attach_pen_visual, make_fr3_with_pen


PLAYBACK_DT = 0.04


def draw_target_marker(base, p_tgt: np.ndarray, R_tgt: np.ndarray):
    """Yellow sphere + RGB axes at p_tgt to show the shared TCP target."""
    ossop.sphere(pos=tuple(p_tgt.astype(np.float32)),
                 radius=0.015,
                 rgb=(0.98, 0.85, 0.10),
                 alpha=0.95).attach_to(base.scene)
    axis_colors = [(0.95, 0.20, 0.20),
                   (0.20, 0.85, 0.20),
                   (0.20, 0.30, 0.95)]
    for k in range(3):
        tip = (p_tgt + 0.08 * R_tgt[:, k]).astype(np.float32)
        segs = np.stack([p_tgt.astype(np.float32), tip], axis=0)[None, ...]
        ossop.linsegs(segs=segs, radius=0.003,
                      srgbs=np.array(axis_colors[k], dtype=np.float32),
                      alpha=0.95).attach_to(base.scene)


def load_branches(jsonl_path: Path):
    meta = None
    branches = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get('type') == 'meta':
                meta = d
            else:
                branches.append(d)
    if meta is None:
        raise RuntimeError(f'no meta entry in {jsonl_path}')
    p_tgt = np.array(meta['p_tgt'], dtype=np.float32)
    R_tgt = np.array(meta['R_tgt'], dtype=np.float32)
    return meta, p_tgt, R_tgt, branches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', type=str,
                        default='Yuan/RL/intro_motivation/data/smm_branches.jsonl')
    parser.add_argument('--mode', choices=['animate', 'static'], default='animate')
    parser.add_argument('--snapshots', type=int, default=6,
                        help='static mode: snapshots per branch')
    parser.add_argument('--layout', choices=['stack', 'side'], default='stack',
                        help='stack=arms overlap at origin; side=side-by-side per branch')
    parser.add_argument('--spacing', type=float, default=1.4,
                        help='side layout: spacing between arms (m)')
    parser.add_argument('--steps-per-tick', type=float, default=1.0,
                        help='animate: how many subsampled traj steps per frame')
    parser.add_argument('--play-mode', choices=['sequential', 'simultaneous'],
                        default='sequential',
                        help='animate mode: sequential plays branches one at a time '
                             '(default); simultaneous plays them all at once')
    args = parser.parse_args()

    meta, p_tgt, R_tgt, branches = load_branches(Path(args.jsonl))
    print(f'loaded {len(branches)} branches from {args.jsonl}')
    print(f'  p_tgt = {p_tgt}')
    print(f'  z_tgt = R[:,2] = {R_tgt[:, 2]}')
    for b in branches:
        print(f'  branch {b["branch_id"]}: '
              f'{"CLOSED" if b["closed"] else "OPEN"}, '
              f'T={b["n_steps"]}, arc={b["arc_length_rad"]:.2f} rad, '
              f'{b["n_members"]} members')

    # Camera: aim at p_tgt from front-up.
    cam_focus = (float(p_tgt[0]), float(p_tgt[1]), float(p_tgt[2]))
    cam_pos = (cam_focus[0] + 1.0, cam_focus[1] - 1.6, cam_focus[2] + 0.9)
    base = ovw.World(cam_pos=cam_pos, cam_lookat_pos=cam_focus,
                     toggle_auto_cam_orbit=False)
    builtins.base = base

    # World axis frame at origin (base of robot).
    ossop.frame(length_scale=0.18, radius_scale=0.7).attach_to(base.scene)

    cmap = plt.get_cmap('tab10')

    if args.layout == 'side':
        # spread branches along +y of base
        offsets = [np.array([0.0, k * args.spacing, 0.0], dtype=np.float32)
                   for k in range(len(branches))]
        # draw a target marker at each shifted location
        for k, off in enumerate(offsets):
            draw_target_marker(base, p_tgt + off, R_tgt)
    else:
        offsets = [np.zeros(3, dtype=np.float32) for _ in branches]
        draw_target_marker(base, p_tgt, R_tgt)

    arms = []
    trajs = []
    for bid, b in enumerate(branches):
        traj = np.array(b['traj_subsampled'], dtype=np.float32)
        rgb = tuple(float(c) for c in cmap(bid % 10)[:3])

        if args.mode == 'animate':
            arm, _ = make_fr3_with_pen(pos=offsets[bid])
            arm.attach_to(base.scene)
            arm.rgb = rgb
            arm.alpha = 0.85 if args.layout == 'side' else 0.55
            attach_pen_visual(arm, rgb=rgb, alpha=0.95)
            arm.fk(traj[0])
            arms.append(arm)
            trajs.append(traj)
            print(f'  → animating branch {bid} in color '
                  f'{tuple(round(c, 2) for c in rgb)}')

        else:  # static
            n_snap = min(args.snapshots, traj.shape[0])
            idxs = np.linspace(0, traj.shape[0] - 1, n_snap).astype(int)
            for k, idx in enumerate(idxs):
                arm, _ = make_fr3_with_pen(pos=offsets[bid])
                arm.attach_to(base.scene)
                arm.rgb = rgb
                # gradient: start of branch translucent, end opaque
                alpha = 0.18 + 0.55 * (k / max(n_snap - 1, 1))
                arm.alpha = alpha
                attach_pen_visual(arm, rgb=rgb, alpha=alpha)
                arm.fk(traj[idx])
            print(f'  → branch {bid}: {n_snap} static snapshots in color '
                  f'{tuple(round(c, 2) for c in rgb)}')

    if args.mode == 'animate':
        # park every arm at its branch's start frame so static arms look
        # reasonable while another branch is the only thing moving.
        for i, arm in enumerate(arms):
            arm.fk(trajs[i][0])

        if args.play_mode == 'simultaneous':
            state = {'t_float': 0.0}

            def animate(_dt, *_args, **_kwargs):
                t = state['t_float']
                for i, arm in enumerate(arms):
                    T = trajs[i].shape[0]
                    period = 2.0 * (T - 1)
                    phase = t % period
                    idx_f = phase if phase < T - 1 else (period - phase)
                    idx = int(round(max(0.0, min(float(T - 1), idx_f))))
                    arm.fk(trajs[i][idx])
                state['t_float'] += float(args.steps_per_tick)
        else:
            state = {'t_float': 0.0, 'active_bid': 0, 'just_switched': True}
            print(f'\n  sequential playback: cycling through {len(arms)} branches')
            # Try to hide inactive arms by alpha=0. ONE viewer may or may
            # not honor dynamic alpha updates; if you still see ghosts,
            # use --layout side.
            ACTIVE_ALPHA = 0.95
            HIDDEN_ALPHA = 0.0

            def animate(_dt, *_args, **_kwargs):
                bid = state['active_bid']
                T = trajs[bid].shape[0]
                period = 2.0 * (T - 1)

                if state['just_switched']:
                    for i, arm in enumerate(arms):
                        if i != bid:
                            arm.fk(trajs[i][0])
                            arm.alpha = HIDDEN_ALPHA
                        else:
                            arm.alpha = ACTIVE_ALPHA
                    print(f'  → now playing branch {bid} '
                          f'(T={T}, period={period:.0f} steps)')
                    state['just_switched'] = False

                t = state['t_float']
                if t >= period:
                    arms[bid].fk(trajs[bid][0])
                    state['active_bid'] = (bid + 1) % len(arms)
                    state['t_float'] = 0.0
                    state['just_switched'] = True
                    return

                idx_f = t if t < T - 1 else (period - t)
                idx = int(round(max(0.0, min(float(T - 1), idx_f))))
                arms[bid].fk(trajs[bid][idx])
                state['t_float'] += float(args.steps_per_tick)

        base.schedule_interval(animate, PLAYBACK_DT)

    base.run()


if __name__ == '__main__':
    main()
