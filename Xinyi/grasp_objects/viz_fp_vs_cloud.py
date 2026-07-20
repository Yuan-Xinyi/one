"""Overlay the FoundationPose-reconstructed object on the real D435 point cloud,
in the ``one`` sim world, to see whether the detected pose actually lands on the
object -- i.e. to tell an EXTRINSIC error from a FoundationPose error.

Both are placed in the SAME world frame with the SAME chain grasp_tool uses:

    world = ROBOT_BASE_POS + ( T_base_cam @ camera_T_object )[:3,3]     (mesh)
    world = ROBOT_BASE_POS +   T_base_cam @ p_cam                       (cloud)

Read the overlay like this:
  * mesh sits ON its own cloud, but the pair is shifted off the real object
        -> EXTRINSIC (T_base_cam) error -- the point-cloud registration's
           rotation residual amplified by the object's distance from the robot.
  * mesh does NOT sit on the object cloud
        -> FoundationPose error -- mesh local origin not at the object centre,
           or a camera-intrinsics / optical-frame-convention mismatch.

The mesh is shown at the RAW FP pose (NO snap-to-table), so what you see is
exactly what the camera + extrinsic report.

Run (conda activate one):
    python viz_fp_vs_cloud.py                      # live D435 capture + FP mesh
    python viz_fp_vs_cloud.py --save obj_cloud.npz # also dump the captured cloud
    python viz_fp_vs_cloud.py --cloud obj_cloud.npz# reuse a saved cloud (no camera)
    python viz_fp_vs_cloud.py --no-window          # print the numeric offset only
Env: ONE_FP_POSE (camera_T_object npy), ONE_CAM_YAML (extrinsics yaml).
"""
import argparse
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS))
for _p in (_PROJECT_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import one.scene.scene_object_primitive as ossop            # noqa: E402
import one.viewer.world as ovw                               # noqa: E402
from one.camera.RS435.detect_cube import load_extrinsics     # noqa: E402

import grasp_tool as gt                                      # noqa: E402
from scene import (build_robot, build_static_objects,        # noqa: E402
                   ROBOT_BASE_POS)
import grasp_cube as gc                                      # noqa: E402


def _raw_fp_world_pose():
    """RAW FP world (pos, rot) -- gt._tool_pose_from_fp WITHOUT the snap-to-table
    lift, so the mesh shows exactly what the camera + extrinsic report."""
    npy = os.environ.get('ONE_FP_POSE', gt.FP_POSE_NPY)
    if not os.path.exists(npy):
        raise SystemExit(f"[fp] pose file not found: {npy}. Run foundationpose_tool.py "
                         "first, or point ONE_FP_POSE at a camera_T_object npy.")
    return gt._tool_pose_from_fp(npy)


def _get_cloud_world(args):
    """(pts_world(N,3), colors(N,3)) of the real scene cloud. Live D435 capture by
    default (base frame -> world = base + ROBOT_BASE_POS), or a saved npz."""
    base = np.asarray(ROBOT_BASE_POS, np.float64)
    if args.cloud:
        d = np.load(args.cloud)
        pts, cols = d['pts_world'], d['colors']
        print(f"[cloud] loaded {len(pts)} pts <- {args.cloud}")
        return pts, cols
    if args.fake_cloud:                       # self-test only: a blob at the FP pose
        pos, _ = _raw_fp_world_pose()
        rng = np.random.default_rng(0)
        pts = pos + rng.normal(0, 0.03, size=(2000, 3)).astype(np.float32)
        return pts, np.tile([0.2, 0.6, 0.9], (len(pts), 1)).astype(np.float32)
    from one.camera.RS435.detect_cube import capture_base_cloud
    yaml_path = os.environ.get('ONE_CAM_YAML')
    T_base_cam, _ = (load_extrinsics(yaml_path) if yaml_path else load_extrinsics())
    pts_base, cols = capture_base_cloud(T_base_cam, z_near=args.z_min, z_far=args.z_max)
    pts_world = (pts_base.astype(np.float64) + base).astype(np.float32)
    print(f"[cloud] captured {len(pts_world)} pts (z in [{args.z_min}, {args.z_max}] m)")
    if args.save:
        np.savez_compressed(args.save, pts_world=pts_world, colors=cols)
        print(f"[cloud] saved -> {args.save} (reuse with --cloud {args.save})")
    return pts_world, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cloud', help='load a saved cloud npz (pts_world,colors); skip camera')
    ap.add_argument('--save', help='save the live-captured cloud to this npz')
    ap.add_argument('--z-min', type=float, default=0.15)
    ap.add_argument('--z-max', type=float, default=1.5)
    ap.add_argument('--crop', type=float, default=0.0,
                    help='if >0, keep only cloud pts within this radius (m) of the FP pose')
    ap.add_argument('--no-window', action='store_true', help='print offset only, no viewer')
    ap.add_argument('--fake-cloud', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args()

    pos, rot = _raw_fp_world_pose()
    snapped = gt._snap_to_table(pos, rot)
    print(f"[fp] RAW world pos   = [{pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}]")
    print(f"[fp] snap-to-table lift = {float(snapped[2]-pos[2])*1000:.1f} mm "
          "(NOT applied here; mesh shown at RAW pose)")

    pts, cols = _get_cloud_world(args)

    # numeric hint: centroid of the cloud NEAR the FP pose vs the FP translation.
    r = args.crop if args.crop > 0 else 0.15
    near = np.linalg.norm(pts - pos, axis=1) < r
    if near.sum() >= 20:
        c = pts[near].mean(axis=0)
        off = c - pos
        print(f"[offset] cloud centroid within {r*100:.0f}cm of FP pose: "
              f"[{c[0]:+.3f} {c[1]:+.3f} {c[2]:+.3f}]")
        print(f"[offset] FP_pose - cloud_centroid = "
              f"[{-off[0]*1000:+.0f} {-off[1]*1000:+.0f} {-off[2]*1000:+.0f}] mm "
              f"(|.|={np.linalg.norm(off)*1000:.0f} mm)")
    else:
        print(f"[offset] <20 cloud pts within {r*100:.0f}cm of FP pose -- "
              "mesh likely lands OFF the object (big error) or wrong crop.")

    if args.crop > 0:
        pts, cols = pts[near], cols[near]

    if args.no_window:
        return

    base = ovw.World(cam_pos=(1.6, 0.4, 1.6), cam_lookat_pos=(0.45, -0.1, 0.95))
    robot = build_robot()
    qs = robot.qs.astype(np.float64).copy()
    qs[robot.chain(gc.CHAIN).active_jnt_ids] = np.deg2rad(gc.HOME_DEG)
    robot.fk(qs=qs)
    statics = build_static_objects()
    tool = gt.build_tool(pos, rot)                 # RAW FP pose, gold
    pcd = ossop.point_cloud(pts, cols, alpha=1.0)  # real D435 colours

    ossop.frame().attach_to(base.scene)
    for e in [robot] + statics + [tool, pcd]:
        e.attach_to(base.scene)
    print("[viz] gold mesh = FoundationPose pose;  points = real D435 cloud. "
          "Orbit to check overlap.")
    base.run()


if __name__ == '__main__':
    main()
