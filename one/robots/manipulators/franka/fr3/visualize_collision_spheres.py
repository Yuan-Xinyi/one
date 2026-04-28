"""Visualize FR3 collision spheres in WRS World.

Usage:
    python -m one.robots.manipulators.franka.fr3.visualize_collision_spheres
    python -m one.robots.manipulators.franka.fr3.visualize_collision_spheres --static
"""
from __future__ import annotations

import argparse
import builtins

import numpy as np

from one.robots.manipulators.franka.fr3.fr3 import FR3
from one.robots.manipulators.franka.fr3.sphere_collision import load_link_spheres


LINK_COLORS = (
    (0.90, 0.15, 0.12),
    (0.95, 0.55, 0.10),
    (0.95, 0.85, 0.10),
    (0.25, 0.75, 0.25),
    (0.10, 0.65, 0.90),
    (0.25, 0.35, 0.95),
    (0.60, 0.25, 0.90),
    (0.95, 0.30, 0.70),
)


def attach_collision_spheres(arm: FR3, scene, alpha: float):
    import one.scene.scene_object_primitive as ossop

    centers, radii, link_indices = load_link_spheres()
    centers = centers.cpu().numpy()
    radii = radii.cpu().numpy()
    link_indices = link_indices.cpu().numpy()

    spheres = []
    for center, radius, link_idx in zip(centers, radii, link_indices):
        rgb = LINK_COLORS[int(link_idx) % len(LINK_COLORS)]
        sphere = ossop.sphere(pos=tuple(center),
                              radius=float(radius),
                              segments=16,
                              rgb=rgb,
                              alpha=alpha)
        sphere.attach_to(arm.runtime_lnks[int(link_idx)])
        spheres.append(sphere)

    print(f'loaded {len(spheres)} collision spheres')
    for link_idx in range(8):
        print(f'  link{link_idx}: {(link_indices == link_idx).sum()} spheres')
    return spheres


def schedule_demo_motion(base, arm: FR3, fps: float = 30.0):
    """Play a gentle sinusoidal joint motion so attached spheres can be checked."""
    q0 = arm.home_qs.copy()
    amp = np.array([0.35, 0.22, 0.35, 0.18, 0.45, 0.25, 0.45],
                   dtype=np.float32)
    phase = np.linspace(0.0, np.pi, arm.ndof).astype(np.float32)
    t = [0.0]

    def tick(_dt):
        t[0] += 0.03
        q = q0 + amp * np.sin(t[0] + phase)
        q = np.clip(q, arm._chain.lmt_lo, arm._chain.lmt_up)
        arm.fk(q)

    base.schedule_interval(tick, interval=1.0 / fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.28)
    parser.add_argument('--static', action='store_true',
                        help='Do not animate the robot.')
    parser.add_argument('--show-tcp', action='store_true')
    parser.add_argument('--fps', type=float, default=30.0)
    args = parser.parse_args()

    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop

    base = ovw.World(cam_pos=(1.4, 1.0, 0.9),
                     cam_lookat_pos=(0.35, 0.0, 0.35))
    arm = FR3()
    builtins.base = base
    builtins.arm = arm
    arm.attach_to(base.scene)
    ossop.frame(length_scale=0.2, radius_scale=0.8).attach_to(base.scene)
    if args.show_tcp:
        arm.toggle_tcp(length_scale=0.15, radius_scale=0.6)
    attach_collision_spheres(arm, base.scene, alpha=args.alpha)

    if not args.static:
        schedule_demo_motion(base, arm, fps=args.fps)

    base.run()


if __name__ == '__main__':
    main()
