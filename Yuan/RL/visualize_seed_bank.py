"""Visualize the deterministic IK seed bank from batched_rollout.

Renders each seed posture as a translucent FR3, color-cycled by group:
  - canonical (3)             : white
  - shoulder rotated (6)      : cyan / blue
  - wrist-flipped (4)         : magenta
  - extreme reach (3)         : orange / red

Usage:
    python -m Yuan.RL.visualize_seed_bank
    python -m Yuan.RL.visualize_seed_bank --n 9          # only first 9
    python -m Yuan.RL.visualize_seed_bank --alpha 0.25
"""
from __future__ import annotations
import argparse, builtins
import numpy as np


# Hard-copied from batched_rollout._branch_seed_bank to keep this script
# free of torch / kinematics deps.
SEEDS = np.array([
    # canonical (FR3 home, elbow up/down ±)
    [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397],
    [0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.0],
    [0.0,  0.4, 0.0, -2.2, 0.0, 1.8, 0.0],
    # shoulder rotated ±, with various elbow
    [ 1.0, 0.8,  1.0, -2.1,  1.2, 1.0,  0.5],
    [-1.0, 0.8, -1.0, -2.1, -1.2, 1.0, -0.5],
    [ 1.0, 1.2, -1.0, -2.2,  1.5, 1.0,  0.5],
    [-1.0, 1.2,  1.0, -2.2, -1.5, 1.0, -0.5],
    [ 0.0, 1.2,  1.2, -2.0,  1.2, 1.2,  0.0],
    [ 0.0, 1.2, -1.2, -2.0, -1.2, 1.2,  0.0],
    # wrist-flipped (J6 negative -> tool pointing UP)
    [ 0.0, -0.4,  0.0, -2.2,  0.0, -1.5,  0.0],
    [ 0.0,  0.4,  0.0, -2.2,  0.0, -1.5,  0.0],
    [ 1.5, -0.4,  0.0, -2.2,  0.0, -1.5,  0.0],
    [-1.5, -0.4,  0.0, -2.2,  0.0, -1.5,  0.0],
    # extreme back / side reaches
    [ 2.5,  0.5, -2.0, -1.0,  0.5, 1.5,  0.0],
    [-2.5,  0.5,  2.0, -1.0, -0.5, 1.5,  0.0],
    [ 0.0, -1.0,  0.0, -1.5,  0.0,  2.5,  0.0],
], dtype=np.float32)

GROUPS = [
    (0,  3, "canonical",        (0.95, 0.95, 0.95)),
    (3,  9, "shoulder-rotated", (0.20, 0.55, 0.95)),
    (9, 13, "wrist-flipped",    (0.95, 0.30, 0.85)),
    (13, 16, "extreme-reach",    (0.95, 0.55, 0.15)),
]


def _group_for(idx: int):
    for lo, hi, name, rgb in GROUPS:
        if lo <= idx < hi:
            return name, rgb
    return "unknown", (0.5, 0.5, 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(SEEDS),
                    help="how many seeds to render (1..16, default=all)")
    ap.add_argument("--alpha", type=float, default=0.30,
                    help="per-arm transparency")
    ap.add_argument("--show-tcp", action="store_true",
                    help="overlay each arm's TCP frame")
    args = ap.parse_args()

    n = max(1, min(args.n, len(SEEDS)))
    seeds = SEEDS[:n]

    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw
    from Yuan.RL.fr3_with_pen import make_fr3_with_pen, attach_pen_visual

    base = ovw.World(cam_pos=(1.6, 1.4, 1.2),
                     cam_lookat_pos=(0.2, 0.0, 0.4),
                     toggle_auto_cam_orbit=False)
    builtins.base = base
    ossop.frame(length_scale=0.25, radius_scale=1.0).attach_to(base.scene)

    print(f"rendering {n} seed configs:")
    for i, q_active in enumerate(seeds):
        name, rgb = _group_for(i)
        arm, _ = make_fr3_with_pen()
        arm.attach_to(base.scene)
        full_q = np.zeros(arm.qs.shape[0], dtype=np.float32)
        full_q[arm._chain.active_mask] = q_active
        arm.fk(full_q)
        arm.rgb = rgb
        arm.alpha = args.alpha
        attach_pen_visual(arm, rgb=rgb, alpha=min(1.0, args.alpha + 0.3))
        if args.show_tcp:
            arm.toggle_tcp(length_scale=0.10, radius_scale=0.4)
        print(f"  seed {i:2d} [{name:>16s}]  q={np.array2string(q_active, precision=2, suppress_small=True)}")

    base.run()


if __name__ == "__main__":
    main()
