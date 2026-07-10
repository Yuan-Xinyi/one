"""Sample a surface point cloud of the simulated xArm7 + XHand at a given joint
configuration, expressed in the robot *base* frame.

Runs under the **wrs** conda env (needs panda3d):

    /home/lqin/miniconda3/envs/wrs/bin/python sim_xarm7_xhand_cloud.py \
        --jnts-deg 20,-90,120,30,0,40,0 --n 20000 --out sim_cloud.npz

The XHand fingers stay at the model default (open / zero) pose -- put the real
hand in the same open pose before capturing.

Output npz keys: ``points`` (N,3 float32, base frame), ``jnts_rad`` (7,).
"""
import argparse

import numpy as np


def sample_robot_cloud(jnts_rad, n_total=20000):
    import wrs.robot_sim.robots.xarm7_dual.xarm7_xhand as rbt

    robot = rbt.XArm7XHR(enable_cc=False)
    robot.goto_given_conf(jnt_values=np.asarray(jnts_rad, dtype=float))
    cm_list = [c for c in robot.gen_meshmodel().cm_list
               if c.trm_mesh is not None and len(c.trm_mesh.vertices) > 0]
    # distribute samples by surface area so big links do not get sparse
    try:
        areas = np.array([float(c.trm_mesh.area) for c in cm_list])
    except Exception:
        areas = np.ones(len(cm_list))
    areas = np.maximum(areas, 1e-9)
    clouds = []
    for c, a in zip(cm_list, areas):
        n = max(150, int(round(n_total * a / areas.sum())))
        # sample_surface returns points already in the world (= base) frame
        clouds.append(np.asarray(c.sample_surface(n_samples=n), dtype=np.float32))
    return np.vstack(clouds)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--jnts-deg", help="7 comma-separated joint angles in degrees")
    g.add_argument("--jnts-rad", help="7 comma-separated joint angles in radians")
    ap.add_argument("--n", type=int, default=20000, help="approx. total sample points")
    ap.add_argument("--out", required=True, help="output .npz path")
    args = ap.parse_args()

    if args.jnts_deg is not None:
        jnts = np.radians([float(x) for x in args.jnts_deg.split(",")])
    else:
        jnts = np.array([float(x) for x in args.jnts_rad.split(",")])
    if jnts.shape != (7,):
        ap.error(f"expected 7 joint values, got {jnts.shape}")

    pts = sample_robot_cloud(jnts, n_total=args.n)
    np.savez_compressed(args.out, points=pts, jnts_rad=jnts)
    print(f"saved {pts.shape[0]} pts -> {args.out}")


if __name__ == "__main__":
    main()
