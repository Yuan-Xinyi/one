#!/usr/bin/env python3
"""FoundationPose 6D detection for the grasp_tool object -- the same flow as the
cube's ``RealExperiments/foundationpose_then_play.py --no_play``, but standalone
and for THIS object's mesh.

Captures one RealSense RGB-D frame, you drag a box around the object (GrabCut ->
mask), FoundationPose registers + briefly tracks, and the resulting
``camera_T_object`` (4x4, camera colour-optical frame) is saved where
``grasp_tool.py``'s ONE_FP path reads it (``/tmp/foundationpose_tool_pose.npy``).

Run in the FoundationPose env (has torch / nvdiffrast / weights), NOT the `one` env:
    conda activate env_isaaclab
    python foundationpose_tool.py                     # drag a box on frame 1
    python foundationpose_tool.py --roi 220 140 110 95  # pass a known ROI (x y w h)
    python foundationpose_tool.py --preview            # show the fitted-pose overlay

Then, in the `one` env:
    ONE_FP=1 python grasp_tool.py            # sim preview at the detected pose
    ONE_FP=1 ONE_REAL=1 python grasp_tool.py # + real xArm7 + XHand
"""
import argparse
import os
import sys
import tempfile

import cv2
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
FP_ROOT = os.environ.get('FOUNDATIONPOSE_ROOT', '/disk2/FoundationPose')
# The cube bridge's ROI helpers (cv2 GUI -> browser page -> text prompt) work when
# OpenCV is headless (as in env_isaaclab). Reused so mask selection matches cube.
CUBE_BRIDGE_ROOT = os.environ.get(
    'CUBE_BRIDGE_ROOT', '/disk2/xhand_inhand/xhand_inhand/RealExperiments')


def load_live_demo():
    """Import FoundationPose's cube/live_demo (build_estimator / select_mask /
    annotate + the estimater stack). Same module the cube capture reuses."""
    for p in (FP_ROOT, os.path.join(FP_ROOT, 'cube')):
        if p not in sys.path:
            sys.path.insert(0, p)
    import live_demo
    return live_demo


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mesh_file', default=os.path.join(_THIS, 'textured_mesh.obj'),
                    help='object model FoundationPose registers against')
    ap.add_argument('--pose_out',
                    default=os.path.join(tempfile.gettempdir(),
                                         'foundationpose_tool_pose.npy'),
                    help='where camera_T_object is written (grasp_tool ONE_FP_POSE)')
    ap.add_argument('--image_out',
                    default=os.path.join(tempfile.gettempdir(),
                                         'foundationpose_tool_init.png'))
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--est_refine_iter', type=int, default=5)
    ap.add_argument('--track_refine_iter', type=int, default=2)
    ap.add_argument('--track_frames', type=int, default=5,
                    help='tracking frames to refine after registration')
    ap.add_argument('--roi', type=int, nargs=4, default=None,
                    metavar=('X', 'Y', 'W', 'H'),
                    help='known ROI box instead of dragging one interactively')
    ap.add_argument('--preview', action='store_true',
                    help='show/save the registered-pose overlay')
    return ap.parse_args()


def _load_bridge():
    """The cube bridge's ROI helpers (cv2 GUI, else a local browser ROI page, else
    a text prompt) -- it imports cheaply (no torch/Isaac at module level)."""
    if CUBE_BRIDGE_ROOT not in sys.path:
        sys.path.insert(0, CUBE_BRIDGE_ROOT)
    try:
        import foundationpose_then_play as fpp
        return fpp
    except Exception as exc:  # noqa: BLE001
        print(f"[FoundationPose] cube ROI helpers unavailable ({exc}); "
              "use --roi X Y W H (read from the saved init image).")
        return None


def _select_mask(live_demo, color_rgb, args, fpp):
    """Boolean object mask. Prefer the cube bridge's headless-safe selection
    (drag box -> browser -> prompt); else --roi, else the cv2 drag box."""
    if fpp is not None:
        return fpp._select_mask(live_demo, color_rgb, args)
    if args.roi is not None:
        from make_mask import grabcut
        return grabcut(color_rgb[..., ::-1].copy(), tuple(args.roi)) > 0
    return live_demo.select_mask(color_rgb)


def main():
    args = parse_args()
    import pyrealsense2 as rs
    live_demo = load_live_demo()
    fpp = _load_bridge()

    if not os.path.exists(args.mesh_file):
        raise FileNotFoundError(f"mesh not found: {args.mesh_file}")
    live_demo.set_logging_format()
    live_demo.set_seed(0)
    print(f"[FoundationPose] building estimator from {args.mesh_file} ...")
    est, mesh, to_origin, bbox, mt = live_demo.build_estimator(args.mesh_file)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    for _ in range(15):
        pipeline.wait_for_frames()

    def grab():
        frames = align.process(pipeline.wait_for_frames())
        c = frames.get_color_frame()
        d = frames.get_depth_frame()
        color = np.asarray(c.get_data())
        depth = np.asarray(d.get_data()).astype(np.float32) * depth_scale
        intr = c.profile.as_video_stream_profile().intrinsics
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
        return color, depth, K

    try:
        print("[FoundationPose] capturing init frame...")
        color, depth, K = grab()
        cv2.imwrite(args.image_out, color[..., ::-1])
        print(f"[FoundationPose] saved init image: {args.image_out}")
        print("[FoundationPose] select the object (drag box / browser / --roi).")
        mask = _select_mask(live_demo, color, args, fpp)
        if mask is None:
            print("[FoundationPose] no mask selected, aborting.")
            return

        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask,
                            iteration=args.est_refine_iter)
        for _ in range(max(0, args.track_frames)):
            color, depth, K = grab()
            pose = est.track_one(rgb=color, depth=depth, K=K,
                                 iteration=args.track_refine_iter)

        pose = np.asarray(pose, dtype=np.float32)
        np.save(args.pose_out, pose)
        t = pose[:3, 3]
        print(f"[FoundationPose] saved pose: {args.pose_out}")
        print(f"[FoundationPose] camera_T_object xyz(m): "
              f"{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}")

        if args.preview:
            vis = live_demo.annotate(color, pose, K, to_origin, bbox, mt)
            try:
                cv2.imshow("FoundationPose pose - press any key", vis[..., ::-1])
                cv2.waitKey(0)
            except cv2.error:
                p = str(os.path.splitext(args.pose_out)[0] + '.preview.png')
                cv2.imwrite(p, vis[..., ::-1])
                print(f"[FoundationPose] no GUI; saved preview image: {p}")
    finally:
        pipeline.stop()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass                       # headless OpenCV has no GUI to destroy


if __name__ == '__main__':
    main()
