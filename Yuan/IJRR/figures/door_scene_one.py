#!/usr/bin/env python3
"""The door rollouts inside ``one``'s own real-time viewer.

Same task, same rollouts and same numbers as ``fig_door_opening.py`` — this file
only draws them: it builds the door in ``one``'s scene graph, spawns one FR3 per
sampled pose with rising opacity (the transparent overlay), and hands the window
to you. Orbit with the mouse and take the screenshot yourself.

    conda activate one
    cd /home/lqin/one
    python Yuan/IJRR/figures/door_scene_one.py --scenario init --variant 0
    python Yuan/IJRR/figures/door_scene_one.py --scenario redundancy --variant 3
    python Yuan/IJRR/figures/door_scene_one.py --scenario base --all

One case at a time by default (``--variant k``), drawn twice: the FR3 meshes and,
``--gap`` metres away, the same poses as a flat **top view** link diagram in the
variant's colour (arm, leaf, wall and base projected onto one horizontal plane —
look straight down on it), so each can be screenshotted on its own. ``--all`` shows every case of the
scenario side by side instead (meshes only), ``--no-stick`` drops the link
diagram, ``--animate`` adds an opaque arm that plays the rollout in a loop on top
of the overlay, and ``--door-alpha`` / ``--ghost-alpha`` / ``--wall-alpha`` set
how transparent the leaf, the ghost arms and the wall are.
"""
from __future__ import annotations

# fig_door_opening imports matplotlib before torch on purpose; keep it first so
# that one's own matplotlib import (one.utils.constant) is already satisfied.
import Yuan.IJRR.figures.fig_door_opening as F      # noqa: E402

import argparse                                     # noqa: E402
import builtins                                     # noqa: E402
import math                                         # noqa: E402
import pathlib                                      # noqa: E402

import numpy as np                                  # noqa: E402

import one.scene.scene_object_primitive as ossop    # noqa: E402
import one.utils.math as oum                        # noqa: E402,F401
import one.viewer.world as ovw                      # noqa: E402
from one.robots.manipulators.franka.fr3.fr3 import fr3_with_hand   # noqa: E402

PANEL_SPACING = 2.6          # metres between the variants of one scenario
WOOD = (0.72, 0.55, 0.35)
WALL = (0.80, 0.81, 0.84)
FRAME = (0.55, 0.52, 0.48)
STEEL = (0.55, 0.56, 0.60)
PEDESTAL = (0.30, 0.31, 0.34)


def _rotz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float32)


class DoorLeaf:
    """The swinging leaf plus its handle, as one movable group."""

    def __init__(self, scene, door: F.DoorSpec, offset: np.ndarray, alpha: float):
        self.door, self.offset = door, offset
        hw, hh = 0.5 * door.width, 0.5 * door.height
        self.hinge = door.hinge + offset
        bar_a = min(1.0, alpha + 0.25)
        self.panel = ossop.box(half_extents=(hw, 0.018, hh), rgb=WOOD, alpha=alpha)
        # local geometry, placed by set_rotmat_pos below
        self.bar = ossop.cylinder(spos=(0.0, 0.0, -0.10), epos=(0.0, 0.0, 0.10),
                                  radius=door.handle_radius, segments=16,
                                  rgb=STEEL, alpha=bar_a)
        self.stub = ossop.cylinder(spos=(0.0, 0.0, 0.0),
                                   epos=(0.0, -door.handle_offset, 0.0),
                                   radius=0.009, segments=12, rgb=STEEL, alpha=bar_a)
        for o in (self.panel, self.bar, self.stub):
            o.attach_to(scene)
        self.set_angle(0.0)

    def set_angle(self, theta: float):
        d = self.door
        R = _rotz(math.radians(d.leaf_dir_deg) + d.swing * theta)
        self.panel.set_rotmat_pos(
            rotmat=R, pos=self.hinge + R @ np.array([0.5 * d.width, 0.0,
                                                     0.5 * d.height], np.float32))
        self.bar.set_rotmat_pos(
            rotmat=R, pos=self.hinge + R @ np.array([d.handle_r, -d.handle_offset,
                                                     d.handle_z], np.float32))
        # the little stub that carries the bar, lying along the leaf normal
        self.stub.set_rotmat_pos(
            rotmat=R, pos=self.hinge + R @ np.array([d.handle_r, 0.0, d.handle_z],
                                                    np.float32))


def build_static(scene, door: F.DoorSpec, base: F.BasePose, offset: np.ndarray,
                 wall_alpha: float):
    """Wall slabs, door frame and the robot pedestal — the parts that never move."""
    h = door.hinge + offset
    d0 = door.d0
    R = _rotz(math.radians(door.leaf_dir_deg))
    hh = 0.5 * door.height
    for s0 in (-0.60, door.width):
        c = h + (s0 + 0.30) * d0 + np.array([0.0, 0.0, hh])
        ossop.box(half_extents=(0.30, 0.04, hh), rotmat=R, pos=c,
                  rgb=WALL, alpha=wall_alpha).attach_to(scene)
    for s0 in (-0.03, door.width + 0.03):
        c = h + s0 * d0 + np.array([0.0, 0.0, hh])
        ossop.box(half_extents=(0.03, 0.06, hh), rotmat=R, pos=c,
                  rgb=FRAME, alpha=1.0).attach_to(scene)
    c = h + 0.5 * door.width * d0 + np.array([0.0, 0.0, door.height + 0.03])
    ossop.box(half_extents=(0.5 * door.width + 0.06, 0.06, 0.03), rotmat=R, pos=c,
              rgb=FRAME, alpha=1.0).attach_to(scene)
    bx, by, bz = base.t + offset
    ossop.box(half_extents=(0.115, 0.115, 0.5 * bz), pos=(bx, by, 0.5 * bz),
              rgb=PEDESTAL, alpha=1.0).attach_to(scene)


SCENARIO_NO = {'init': 1, 'redundancy': 2, 'base': 3, 'height': 4}
LINK_COLOR = '#1f1f1f'       # link diagrams are one neutral colour, not per-case


def save_link_topview(path, arm_kin, v, ghosts: int, ghost_alpha: float,
                      dpi: int = 300, figsize=(6.0, 6.0),
                      color: str = LINK_COLOR):
    """Write the link diagram of one case: a flat top view, poses overlaid.

    Pure 2-D line drawing — arm polyline, joint dots, TCP, the leaf as it swings,
    the wall with its doorway and the base. No viewer involved.
    """
    import matplotlib.pyplot as plt
    import torch

    plt.switch_backend('Agg')
    door, base = v.door, v.base
    h, d0 = door.hinge, door.d0
    fig, ax = plt.subplots(figsize=figsize)
    for s0, s1 in ((-0.60, 0.0), (door.width, door.width + 0.60)):
        p0, p1 = h + s0 * d0, h + s1 * d0
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='#6f7276', lw=6.0,
                solid_capstyle='butt', zorder=1)
    th = np.linspace(0.0, math.radians(door.theta_end_deg), 160)
    arc, _ = door.grasp_path(th)
    ax.plot(arc[:, 0], arc[:, 1], color='#b5b5b5', lw=1.0, ls='--', zorder=1)
    ax.plot([base.x], [base.y], 's', color='#3a3d42', ms=10, zorder=2)

    idx = np.linspace(0, v.roll.n_ok - 1, max(1, ghosts)).round().astype(int)
    for j, ti in enumerate(idx):
        a = 1.0 if len(idx) == 1 else (
            ghost_alpha + (1.0 - ghost_alpha) * (j / (len(idx) - 1)) ** 2.4)
        d, _ = door.leaf_frame(np.array([v.roll.theta[ti]]))
        e = h + door.width * d[0]
        ax.plot([h[0], e[0]], [h[1], e[1]], color='#8a5a2b', lw=3.5,
                alpha=min(1.0, 0.30 + 0.70 * a), solid_capstyle='round', zorder=2)
        qt = torch.as_tensor(v.roll.q[ti], dtype=F.DTYPE).reshape(1, 7)
        pts = arm_kin.joints_world(qt, base)[0]
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=3.2, alpha=a,
                solid_capstyle='round', zorder=3)
        ax.plot(pts[1:8, 0], pts[1:8, 1], 'o', color=color, ms=5.0,
                mfc='white', mew=1.3, alpha=a, zorder=4)
        p_tcp = arm_kin.tcp_world(qt, base)[0][0]
        ax.plot([p_tcp[0]], [p_tcp[1]], 'o', color=color, ms=7.0, alpha=a,
                zorder=5)

    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout(pad=0.15)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def spawn_arm(scene, base: F.BasePose, offset: np.ndarray, q: np.ndarray,
              alpha: float, jaw_width: float = 0.026):
    """One FR3 with the Franka Hand closed on the handle, at a fixed pose."""
    pos = np.asarray(base.t + offset, np.float32)
    arm, hand = fr3_with_hand(rotmat=_rotz(math.radians(base.yaw_deg)), pos=pos,
                              jaw_width=jaw_width)
    arm.attach_to(scene)
    arm.fk(np.asarray(q, np.float32))
    if alpha < 1.0:
        for lnk in list(arm.runtime_lnks) + list(hand.runtime_lnks):
            lnk.alpha = alpha
    return arm, hand


CAND_RGB = (0.78, 0.78, 0.80)      # the other IK solutions
GOOD_RGB = (0.12, 0.40, 0.85)      # the one the successful trajectory uses


def _tint(arm, hand, rgb, alpha):
    for lnk in list(arm.runtime_lnks) + list(hand.runtime_lnks):
        lnk.rgba = (rgb[0], rgb[1], rgb[2], alpha)


def build_ik_fan(world, arm_kin, door, base, roll, way_deg: float, gap: float,
                 wall_alpha: float, cand_alpha: float, good_alpha: float,
                 n_restart: int, want: int, jaw: float):
    """Two points of one rollout, each with its IK candidates fanned out.

    Left cluster: the closed-door pose. Right cluster: the same trajectory
    ``way_deg`` further along. Every collision-free IK solution of that pose is
    drawn faint; the one the successful trajectory actually uses is blue.
    """
    theta_way = math.radians(way_deg)
    ti = int(np.argmin(np.abs(roll.theta[:roll.n_ok] - theta_way)))
    clusters = [(0.0, roll.q[0], np.zeros(3, np.float32)),
                (float(roll.theta[ti]), roll.q[ti],
                 np.array([0.0, gap, 0.0], np.float32))]
    for theta, q_good, off in clusters:
        build_static(world.scene, door, base, off, wall_alpha)
        leaf = DoorLeaf(world.scene, door, off, 0.22)
        leaf.set_angle(theta)
        p_w, R_w = door.grasp_path(np.array([theta]))
        qs, sw = F.ik_solutions(arm_kin, p_w[0], R_w[0], base, door,
                                n_restart=n_restart, dedup=0.5, want=want,
                                theta=theta)
        far = [q for q in qs if np.abs(np.asarray(q) - q_good).max() > 0.3]
        for q in far:
            a, h = spawn_arm(world.scene, base, off, q, 1.0, jaw_width=jaw)
            _tint(a, h, CAND_RGB, cand_alpha)
        a, h = spawn_arm(world.scene, base, off, q_good, 1.0, jaw_width=jaw)
        _tint(a, h, GOOD_RGB, good_alpha)
        print(f'  theta = {math.degrees(theta):5.1f} deg: {len(far)} other IK '
              f'solutions + the one on the trajectory')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenario', default='init', choices=list(F.SCENARIOS))
    ap.add_argument('--variant', type=int, default=0,
                    help='which case of the scenario to show (0 = first)')
    ap.add_argument('--all', action='store_true',
                    help='show every case of the scenario side by side instead')
    ap.add_argument('--start', type=int, default=None, metavar='K',
                    help='pick the K-th start posture directly (see --list-starts) '
                         'instead of a scenario case')
    ap.add_argument('--law', default=None, choices=list(F.LAW_BY_KEY),
                    help='resolution law to use with --start')
    ap.add_argument('--name', default=None,
                    help='basename of the link jpg, e.g. "a" -> a-link.jpg')
    ap.add_argument('--list-starts', action='store_true',
                    help='print the start postures of the closed-door pose and exit')
    ap.add_argument('--ik-fan', action='store_true',
                    help='instead of the pose overlay: the IK candidates at the '
                         'start and at a waypoint of the chosen rollout')
    ap.add_argument('--way-deg', type=float, default=60.0,
                    help='door angle of the waypoint cluster [deg]')
    ap.add_argument('--gap', type=float, default=3.0,
                    help='distance between the two IK-candidate clusters [m]')
    ap.add_argument('--cand-alpha', type=float, default=0.08)
    ap.add_argument('--good-alpha', type=float, default=0.45)
    ap.add_argument('--n-restart', type=int, default=600,
                    help='IK restarts per pose in --ik-fan')
    ap.add_argument('--want', type=int, default=8,
                    help='how many distinct IK solutions to keep per pose')
    ap.add_argument('--outdir', default=str(pathlib.Path(__file__).parent / 'out'),
                    help='where the link-diagram jpgs go '
                         '(default: Yuan/IJRR/figures/out)')
    ap.add_argument('--no-link', action='store_true',
                    help='do not write the link-diagram jpg')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--link-color', default=LINK_COLOR,
                    help='colour of the link diagram (one colour for all cases)')
    ap.add_argument('--save-only', action='store_true',
                    help='write the link-diagram jpg(s) and exit, no viewer')
    ap.add_argument('--ghosts', type=int, default=6,
                    help='how many poses of each rollout to overlay')
    ap.add_argument('--ghost-alpha', type=float, default=0.16,
                    help='opacity of the earliest ghost (the last one is opaque)')
    ap.add_argument('--jaw', type=float, default=None,
                    help='override the gripper opening [m]')
    ap.add_argument('--door-alpha', type=float, default=0.14)
    ap.add_argument('--wall-alpha', type=float, default=0.06)
    ap.add_argument('--animate', action='store_true',
                    help='also play an opaque arm through the rollout, in a loop')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--dtheta', type=float, default=0.5, help='door step [deg]')
    ap.add_argument('--door-height', type=float, default=None,
                    help='leaf height [m], purely a framing choice')
    args = ap.parse_args()

    arm_kin = F.Arm()
    door, base = F.DoorSpec(), F.BasePose()
    if args.door_height is not None:
        from dataclasses import replace
        door = replace(door, height=args.door_height)
    if args.list_starts:
        qs, sw = F._start_configs(arm_kin, door, base)
        print('start postures at the closed-door pose (--start K):')
        for k, (q, s_) in enumerate(zip(qs, sw)):
            print(f'  K={k}  elbow swivel {s_:+6.1f} deg')
        return

    if args.start is not None:
        qs, sw = F._start_configs(arm_kin, door, base)
        if not 0 <= args.start < len(qs):
            raise SystemExit(f'--start must be 0..{len(qs) - 1}')
        law = F.LAW_BY_KEY[args.law or 'hybrid']
        variants = [F.Variant(label=f'start {sw[args.start]:+.0f} deg, {law.name}',
                              door=door, base=base, q0=qs[args.start], law=law,
                              color=F.PALETTE[0])]
    else:
        variants = F.SCENARIOS[args.scenario](arm_kin, door, base)
    if args.start is None and not args.all:
        if not 0 <= args.variant < len(variants):
            raise SystemExit(f'--variant must be 0..{len(variants) - 1} '
                             f'for scenario "{args.scenario}"')
        variants = [variants[args.variant]]
    print(f'scenario "{args.scenario}": {len(variants)} case(s)')
    for v in variants:
        v.roll = F.rollout_door(arm_kin, v.door, v.base, v.q0, law=v.law,
                                dtheta_deg=args.dtheta)
        print(f'  {v.label:46s} theta_max = {v.roll.theta_max_deg:6.1f}°   '
              f'stop: {v.roll.reason_str}')

    if args.ik_fan:
        v = variants[0]
        lo, hi = F.scene_bounds(variants)
        look = np.array([0.5 * (lo[0] + hi[0]),
                         0.5 * (lo[1] + hi[1]) + 0.5 * args.gap, 0.85])
        n = v.door.n0
        eye = look - (2.8 + 0.9 * args.gap) * np.array([n[0], n[1], -0.42])
        world = ovw.World(cam_pos=tuple(eye), cam_lookat_pos=tuple(look))
        builtins.base = world
        build_ik_fan(world, arm_kin, v.door, v.base, v.roll, args.way_deg,
                     args.gap, args.wall_alpha, args.cand_alpha, args.good_alpha,
                     args.n_restart, args.want, args.jaw or v.door.jaw_width)
        print('one viewer: drag to orbit, scroll to zoom, close the window to quit')
        world.run()
        return

    if not args.no_link:                  # the link diagram is a file, not a body
        outdir = pathlib.Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        sc_no = SCENARIO_NO[args.scenario]
        for k, v in enumerate(variants):
            if args.name:
                stem = args.name if len(variants) == 1 else f'{args.name}{k + 1}'
            elif args.start is not None:
                stem = f's{args.start}-{args.law or "hybrid"}'
            else:
                stem = f'{sc_no}-{(args.variant if not args.all else k) + 1}'
            out = outdir / f'{stem}-link.jpg'
            save_link_topview(out, arm_kin, v, args.ghosts, args.ghost_alpha,
                              dpi=args.dpi, color=args.link_color)
            print(f'  wrote {out.resolve()}')
    if args.save_only:
        return

    lo, hi = F.scene_bounds(variants)
    span = 0.5 * (len(variants) - 1) * PANEL_SPACING
    look = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]) + span, 0.85])
    # stand on the robot's side of the wall, looking into the door
    n = variants[0].door.n0
    eye = look - (2.6 + 1.1 * span) * np.array([n[0], n[1], -0.42])
    world = ovw.World(cam_pos=tuple(eye), cam_lookat_pos=tuple(look))
    builtins.base = world          # one's demos keep the world here, so do we

    live = []
    for k, v in enumerate(variants):
        off = np.array([0.0, k * PANEL_SPACING, 0.0], np.float32)
        build_static(world.scene, v.door, v.base, off, args.wall_alpha)
        idx = np.linspace(0, v.roll.n_ok - 1, max(1, args.ghosts)).round().astype(int)
        for j, ti in enumerate(idx):
            a = 1.0 if len(idx) == 1 else (
                args.ghost_alpha + (1.0 - args.ghost_alpha)
                * (j / (len(idx) - 1)) ** 2.4)
            jaw = args.jaw or v.door.jaw_width
            spawn_arm(world.scene, v.base, off, v.roll.q[ti], a, jaw_width=jaw)
            leaf = DoorLeaf(world.scene, v.door, off,
                            args.door_alpha * (0.6 + 0.4 * a))
            leaf.set_angle(float(v.roll.theta[ti]))
        if args.animate:
            arm, _ = spawn_arm(world.scene, v.base, off, v.roll.q[0], 1.0,
                               jaw_width=args.jaw or v.door.jaw_width)
            live.append((v, arm, DoorLeaf(world.scene, v.door, off, 0.85)))

    if live:
        state = {'t': 0}
        n_hold = int(1.0 * args.fps)

        def tick(_dt):
            t = state['t']
            for v, arm, leaf in live:
                ti = min(t, v.roll.n_ok - 1)
                arm.fk(np.asarray(v.roll.q[ti], np.float32))
                leaf.set_angle(float(v.roll.theta[ti]))
            longest = max(v.roll.n_ok for v, _, _ in live)
            state['t'] = 0 if t > longest + n_hold else t + 1

        world.schedule_interval(tick, interval=1.0 / args.fps)

    print('one viewer: drag to orbit, scroll to zoom, close the window to quit')
    world.run()


if __name__ == '__main__':
    main()
