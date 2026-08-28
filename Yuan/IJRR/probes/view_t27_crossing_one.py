#!/usr/bin/env python3
"""The task-27 full-line crossing (no-gate policy, true reach 1.72 m) inside
``one``'s real-time viewer.

Draws the seam line with the three door positions marked, a few transparent
ghost arms at key stages, the pen-tip trace, and one opaque arm that replays
the 171-step trajectory in a loop. Interactive window: drag to orbit, scroll
to zoom, close to quit.

    conda activate one
    cd /home/lqin/one
    DISPLAY=:1 python Yuan/IJRR/probes/view_t27_crossing_one.py
"""
from __future__ import annotations

# matplotlib before anything that pulls torch/one internals (CXXABI order).
import matplotlib                                   # noqa: E402
matplotlib.use('Agg')                               # noqa: E402

import builtins                                     # noqa: E402
import numpy as np                                  # noqa: E402

import one.scene.scene_object_primitive as ossop    # noqa: E402
import one.viewer.world as ovw                      # noqa: E402
from one.robots.manipulators.franka.fr3.fr3 import fr3_with_hand   # noqa: E402

PACK = ('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/'
        't27_render_pack.npz')
DOORS = (0.68, 0.85, 1.58)
SEAM = (0.85, 0.35, 0.25)
DOOR_C = (0.95, 0.55, 0.10)
TRACE = (0.15, 0.35, 0.75)
PEN = (0.10, 0.10, 0.12)
PEN_LEN = 0.10
FPS = 20.0


def _rot_with_z(z: np.ndarray) -> np.ndarray:
    """Any rotation whose third column is z (for placing the pen cylinder)."""
    z = z / (np.linalg.norm(z) + 1e-9)
    h = np.array([1.0, 0.0, 0.0], np.float32)
    if abs(float(h @ z)) > 0.95:
        h = np.array([0.0, 1.0, 0.0], np.float32)
    x = h - (h @ z) * z
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _pen_pose(tip: np.ndarray, zax: np.ndarray):
    """Rotation + midpoint position for a pen ENDING exactly at the tip.

    The cylinder is defined symmetrically about its own origin, so this
    works whether the primitive keeps the given endpoints or re-centres
    the geometry at their midpoint."""
    z = zax / (np.linalg.norm(zax) + 1e-9)
    return _rot_with_z(z), (tip - 0.5 * PEN_LEN * z).astype(np.float32)


def spawn_pen(scene, tip: np.ndarray, zax: np.ndarray, alpha: float):
    """The pen as a cylinder ending at the TCP tip, aligned with the tool."""
    pen = ossop.cylinder(spos=(0.0, 0.0, -0.5 * PEN_LEN),
                         epos=(0.0, 0.0, 0.5 * PEN_LEN),
                         radius=0.006, rgb=PEN, alpha=alpha)
    R, pos = _pen_pose(tip, zax)
    pen.set_rotmat_pos(R, pos)
    pen.attach_to(scene)
    return pen


def main():
    d = np.load(PACK)
    Q, TIP = d['q'].astype(np.float32), d['tip'].astype(np.float32)
    p0, ray = d['p0'].astype(np.float32), d['d'].astype(np.float32)

    mid = p0 + 0.85 * ray
    world = ovw.World(cam_pos=tuple(mid + np.array([1.6, -2.2, 1.4])),
                      cam_lookat_pos=tuple(mid))
    builtins.base = world
    scene = world.scene

    # the seam: full pointwise-feasible extent, doors marked as spheres
    ossop.cylinder(spos=tuple(p0), epos=tuple(p0 + 1.72 * ray), radius=0.004,
                   rgb=SEAM, alpha=0.9).attach_to(scene)
    for s0 in DOORS:
        ossop.sphere(pos=tuple(p0 + s0 * ray), radius=0.016, rgb=DOOR_C,
                     alpha=0.95).attach_to(scene)

    # pen-tip trace of the executed crossing
    for i in range(0, len(TIP) - 1, 2):
        ossop.cylinder(spos=tuple(TIP[i]), epos=tuple(TIP[i + 2 if i + 2
                       < len(TIP) else -1]), radius=0.002, rgb=TRACE,
                       alpha=0.55).attach_to(scene)

    ZAX = d['zax'].astype(np.float32)

    # ghost arms at key stages (rising opacity), each with its pen
    keys = [0, 55, 75, 120, len(Q) - 1]        # ~s = 0, D1, D2, 1.2 m, end
    for j, ti in enumerate(keys):
        arm, hand = fr3_with_hand(jaw_width=0.0)
        arm.attach_to(scene)
        arm.fk(Q[ti])
        a = 0.10 + 0.16 * j
        for lnk in list(arm.runtime_lnks) + list(hand.runtime_lnks):
            lnk.alpha = a
        spawn_pen(scene, TIP[ti], ZAX[ti], min(1.0, a + 0.2))

    # the live arm with its pen
    arm, _ = fr3_with_hand(jaw_width=0.0)
    arm.attach_to(scene)
    arm.fk(Q[0])
    live_pen = spawn_pen(scene, TIP[0], ZAX[0], 1.0)
    state = {'t': 0}
    n_hold = int(1.5 * FPS)

    def tick(_dt):
        t = state['t']
        ti = min(t, len(Q) - 1)
        arm.fk(Q[ti])
        live_pen.set_rotmat_pos(*_pen_pose(TIP[ti], ZAX[ti]))
        state['t'] = 0 if t > len(Q) + n_hold else t + 1

    world.schedule_interval(tick, interval=1.0 / FPS)
    print('one viewer: drag to orbit, scroll to zoom, close the window to '
          'quit. Orange spheres = doors D1/D2/D3; blue trace = pen tip.')
    world.run()


if __name__ == '__main__':
    main()
