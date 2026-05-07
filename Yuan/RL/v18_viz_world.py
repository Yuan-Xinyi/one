"""Visualize v18 sampled q-trajectories in the one viewer world.

K instances of FR3+pen arms, each playing one of the K sampled q_traj's.
Different IK branches → different colors.

Usage:
    python -m Yuan.RL.v18_viz_world --task-idx 4
    python -m Yuan.RL.v18_viz_world --pkl ... --task-idx 9 --fps 20
"""
from __future__ import annotations
import argparse, builtins, pickle
import numpy as np


BRANCH_COLORS = {
    (+1, -1, +1): (0.20, 0.50, 0.95),     # blue
    (+1, -1, -1): (1.00, 0.55, 0.10),     # orange
    (-1, -1, +1): (0.20, 0.75, 0.30),     # green
    (-1, -1, -1): (0.95, 0.20, 0.30),     # red
    (+1, +1, +1): (0.65, 0.30, 0.85),     # purple
    (+1, +1, -1): (0.55, 0.40, 0.30),     # brown
    (-1, +1, +1): (0.95, 0.50, 0.75),     # pink
    (-1, +1, -1): (0.75, 0.75, 0.20),     # olive
}


def branch_color(sig):
    return BRANCH_COLORS.get(tuple(int(s) for s in sig), (0.5, 0.5, 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="Yuan/RL/data/v18_eval_viz.pkl")
    ap.add_argument("--task-idx", type=int, default=4)
    ap.add_argument("--n-arms", type=int, default=None,
                    help="how many sampled q_traj's to show (default = all K)")
    ap.add_argument("--fps", type=float, default=4.0,
                    help="animation FPS (low because there are few checkpoints)")
    ap.add_argument("--alpha", type=float, default=0.30,
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
    K_use = K if args.n_arms is None else min(args.n_arms, K)

    print(f"task {args.task_idx}: L={task['L']:.2f}m  tilt={task['tilt_deg']:.1f}°  "
          f"K_arms={K_use}  T_co={T_co}")
    print(f"branch sigs: {sigs[:K_use]}")

    # build the world scene
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop
    from Yuan.RL.fr3_with_pen import make_fr3_with_pen, attach_pen_visual

    cam_lookat = (plane_point + 0.5 * task['L'] * direction).tolist()
    base = ovw.World(cam_pos=(1.4, 1.0, 0.9),
                     cam_lookat_pos=cam_lookat,
                     toggle_auto_cam_orbit=False)
    builtins.base = base

    # K arm instances, each colored by its branch signature
    arms = []
    for k in range(K_use):
        arm, _ = make_fr3_with_pen()
        rgb = branch_color(sigs[k])
        arm.attach_to(base.scene)
        arm.rgb = rgb
        arm.alpha = float(args.alpha)
        attach_pen_visual(arm, rgb=rgb, alpha=0.85)
        arm.toggle_tcp(length_scale=0.10, radius_scale=0.4)
        arms.append(arm)

    # base coord frame
    ossop.frame(length_scale=0.20, radius_scale=0.8).attach_to(base.scene)

    # task plane (translucent disc)
    ossop.plane(pos=tuple(plane_point), normal=tuple(plane_normal),
                size=(0.5, 0.5),
                rgb=(0.55, 0.55, 0.6), alpha=0.18).attach_to(base.scene)

    # TCP target path: line of small spheres at each checkpoint
    for ci in range(T_co):
        ossop.sphere(pos=tuple(path_pts[ci]), radius=0.012,
                     rgb=(0.05, 0.05, 0.05), alpha=0.95).attach_to(base.scene)

    # plane normal arrow at start
    ossop.arrow(spos=tuple(plane_point),
                epos=tuple(plane_point + 0.18 * plane_normal),
                shaft_radius=0.005, head_radius=0.012, head_length=0.025,
                rgb=(0.95, 0.20, 0.85), alpha=0.85).attach_to(base.scene)

    # path direction arrow
    ossop.arrow(spos=tuple(plane_point),
                epos=tuple(plane_point + task['L'] * direction),
                shaft_radius=0.003, head_radius=0.008, head_length=0.018,
                rgb=(0.10, 0.80, 0.85), alpha=0.65).attach_to(base.scene)

    # initialize all arms at ckpt 0
    for k, arm in enumerate(arms):
        arm.fk(q_trajs[k, 0])

    idx = [0]
    direction_anim = [+1]                              # play forward then backward

    def tick(_dt):
        for k, arm in enumerate(arms):
            arm.fk(q_trajs[k, idx[0]])
        # ping-pong through checkpoints
        idx[0] += direction_anim[0]
        if idx[0] >= T_co - 1:
            idx[0] = T_co - 1
            direction_anim[0] = -1
        elif idx[0] <= 0:
            idx[0] = 0
            direction_anim[0] = +1

    base.schedule_interval(tick, interval=1.0 / args.fps)
    print("\n[control]: spin viewer with mouse, close window to exit")
    base.run()


if __name__ == "__main__":
    main()
