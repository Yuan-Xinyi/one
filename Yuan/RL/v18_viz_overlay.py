"""Static transparent overlay of K branch arms in the one-world viewer.

All K sampled q's at one checkpoint are shown simultaneously, each arm
colored by its branch signature. No animation - just the overlaid snapshot
so you can see branch diversity in 3D.

Usage:
    python -m Yuan.RL.v18_viz_overlay --task-idx 4
    python -m Yuan.RL.v18_viz_overlay --task-idx 9 --ckpt 0  --alpha 0.35
    python -m Yuan.RL.v18_viz_overlay --task-idx 9 --ckpt -1               # at goal
    python -m Yuan.RL.v18_viz_overlay --task-idx 9 --all-ckpts             # K x T overlay
"""
from __future__ import annotations
import argparse, builtins, pickle
import numpy as np

from Yuan.RL.v18_viz_world import branch_color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="Yuan/RL/data/v18_eval_viz.pkl")
    ap.add_argument("--task-idx", type=int, default=4)
    ap.add_argument("--ckpt", type=int, default=0,
                    help="which checkpoint to show (0=start, -1=goal). ignored if --all-ckpts")
    ap.add_argument("--all-ckpts", action="store_true",
                    help="overlay every (k, ckpt) pose simultaneously")
    ap.add_argument("--alpha", type=float, default=0.35,
                    help="arm transparency (lower = more transparent)")
    args = ap.parse_args()

    with open(args.pkl, 'rb') as f:
        tasks = pickle.load(f)
    task = tasks[args.task_idx]
    q_trajs = task['q_trajs']                          # (K, T_co, 7)
    sigs    = task['sigs_at_start']
    path_pts = task['path_pts']
    plane_normal = task['plane_normal']
    plane_point  = task['plane_point']
    direction    = task['direction']
    K, T_co, _ = q_trajs.shape

    if args.all_ckpts:
        which_ckpts = list(range(T_co))
    else:
        which_ckpts = [args.ckpt % T_co]
    print(f"task {args.task_idx}: L={task['L']:.2f}m  tilt={task['tilt_deg']:.1f}°  "
          f"K={K}  ckpts shown={which_ckpts}  total arms={K * len(which_ckpts)}")
    print(f"branch sigs: {sigs}")

    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop
    from Yuan.RL.fr3_with_pen import make_fr3_with_pen, attach_pen_visual

    cam_lookat = (plane_point + 0.5 * task['L'] * direction).tolist()
    base = ovw.World(cam_pos=(1.4, 1.0, 0.9),
                     cam_lookat_pos=cam_lookat,
                     toggle_auto_cam_orbit=False)
    builtins.base = base

    # one arm per (branch k, checkpoint ci) - all static, all transparent
    for k in range(K):
        rgb = branch_color(sigs[k])
        for ci in which_ckpts:
            arm, _ = make_fr3_with_pen()
            arm.attach_to(base.scene)
            arm.rgb = rgb
            arm.alpha = float(args.alpha)
            attach_pen_visual(arm, rgb=rgb, alpha=0.75)
            arm.fk(q_trajs[k, ci])

    # base coord frame
    ossop.frame(length_scale=0.20, radius_scale=0.8).attach_to(base.scene)

    # task plane
    ossop.plane(pos=tuple(plane_point), normal=tuple(plane_normal),
                size=(0.5, 0.5),
                rgb=(0.55, 0.55, 0.6), alpha=0.18).attach_to(base.scene)

    # TCP target path checkpoints
    for ci in range(T_co):
        ossop.sphere(pos=tuple(path_pts[ci]), radius=0.012,
                     rgb=(0.05, 0.05, 0.05), alpha=0.95).attach_to(base.scene)

    # plane normal arrow
    ossop.arrow(spos=tuple(plane_point),
                epos=tuple(plane_point + 0.18 * plane_normal),
                shaft_radius=0.005, head_radius=0.012, head_length=0.025,
                rgb=(0.95, 0.20, 0.85), alpha=0.85).attach_to(base.scene)

    # path direction arrow
    ossop.arrow(spos=tuple(plane_point),
                epos=tuple(plane_point + task['L'] * direction),
                shaft_radius=0.003, head_radius=0.008, head_length=0.018,
                rgb=(0.10, 0.80, 0.85), alpha=0.65).attach_to(base.scene)

    print("\n[control]: spin viewer with mouse, close window to exit")
    base.run()


if __name__ == "__main__":
    main()
