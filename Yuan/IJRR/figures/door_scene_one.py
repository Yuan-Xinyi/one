#!/usr/bin/env python3
"""The door rollouts inside ``one``'s own real-time viewer.

Same task, same rollouts and same numbers as ``fig_door_opening.py`` — this file
only draws them: it builds the door in ``one``'s scene graph, spawns one FR3 per
sampled pose with rising opacity (the transparent overlay), and hands the window
to you. Orbit with the mouse and take the screenshot yourself.

    conda activate one
    cd /home/lqin/one
    python Yuan/IJRR/figures/door_scene_one.py --scenario init
    python Yuan/IJRR/figures/door_scene_one.py --scenario base --ghosts 8
    python Yuan/IJRR/figures/door_scene_one.py --scenario init --variant 0 --animate

``--variant k`` shows one rollout alone instead of the whole comparison;
``--animate`` adds an opaque arm that plays the rollout in a loop on top of the
overlay. ``--door-alpha`` / ``--ghost-alpha`` / ``--wall-alpha`` set how
transparent the leaf, the ghost arms and the wall are.
"""
from __future__ import annotations

# fig_door_opening imports matplotlib before torch on purpose; keep it first so
# that one's own matplotlib import (one.utils.constant) is already satisfied.
import Yuan.IJRR.figures.fig_door_opening as F      # noqa: E402

import argparse                                     # noqa: E402
import builtins                                     # noqa: E402
import math                                         # noqa: E402

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
                                  radius=0.011, segments=16, rgb=STEEL, alpha=bar_a)
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


def spawn_arm(scene, base: F.BasePose, offset: np.ndarray, q: np.ndarray,
              alpha: float):
    """One FR3 with the Franka Hand, at a fixed pose and opacity."""
    pos = np.asarray(base.t + offset, np.float32)
    arm, hand = fr3_with_hand(rotmat=_rotz(math.radians(base.yaw_deg)), pos=pos)
    arm.attach_to(scene)
    arm.fk(np.asarray(q, np.float32))
    if alpha < 1.0:
        for lnk in list(arm.runtime_lnks) + list(hand.runtime_lnks):
            lnk.alpha = alpha
    return arm, hand


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenario', default='init', choices=list(F.SCENARIOS))
    ap.add_argument('--variant', type=int, default=None,
                    help='show only this variant of the scenario')
    ap.add_argument('--ghosts', type=int, default=6,
                    help='how many poses of each rollout to overlay')
    ap.add_argument('--ghost-alpha', type=float, default=0.30,
                    help='opacity of the earliest ghost (the last one is opaque)')
    ap.add_argument('--door-alpha', type=float, default=0.25)
    ap.add_argument('--wall-alpha', type=float, default=0.12)
    ap.add_argument('--animate', action='store_true',
                    help='also play an opaque arm through the rollout, in a loop')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--dtheta', type=float, default=0.5, help='door step [deg]')
    args = ap.parse_args()

    arm_kin = F.Arm()
    door, base = F.DoorSpec(), F.BasePose()
    variants = F.SCENARIOS[args.scenario](arm_kin, door, base)
    if args.variant is not None:
        variants = [variants[args.variant]]
    print(f'scenario "{args.scenario}": {len(variants)} variant(s)')
    for v in variants:
        v.roll = F.rollout_door(arm_kin, v.door, v.base, v.q0, law=v.law,
                                dtheta_deg=args.dtheta)
        print(f'  {v.label:46s} theta_max = {v.roll.theta_max_deg:6.1f}°   '
              f'stop: {v.roll.reason_str}')

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
                * (j / (len(idx) - 1)) ** 1.6)
            spawn_arm(world.scene, v.base, off, v.roll.q[ti], a)
            leaf = DoorLeaf(world.scene, v.door, off,
                            args.door_alpha * (0.6 + 0.4 * a))
            leaf.set_angle(float(v.roll.theta[ti]))
        if args.animate:
            arm, _ = spawn_arm(world.scene, v.base, off, v.roll.q[0], 1.0)
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
