"""Ad-hoc scene viewer: one world + FR3-with-pen + task plane + TCP frame.

Pure visualization (no controller, no rollout). Samples one task spec
(q0, line_dir, n_target) from the same LineDistribution used by
train/eval, then renders a static interactive scene so the user can
orbit and inspect.

Usage:
    python -m Yuan.RL_controller.viz_scene \\
        --config Yuan/RL_controller/config.yaml
"""
from __future__ import annotations

import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    if __spec__ is not None and __spec__.name != "__main__":
        argv = [sys.executable, "-m", __spec__.name] + sys.argv[1:]
    else:
        argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, argv, new_env)

import argparse
import builtins

import numpy as np
import torch
import yaml

from one import ovw, ossop
from one.robots.manipulators.franka.fr3_pen.fr3_with_pen import (
    make_fr3_with_pen, attach_pen_visual,
)

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import LineDistribution


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="Yuan/RL_controller/config.yaml")
parser.add_argument("--seed", type=int, default=3,
                    help="line_distribution seed; defaults to config train_seed")
parser.add_argument("--ray-len", type=float, default=1.5,
                    help="line ray length (m)")
parser.add_argument("--plane-size", type=float, default=0.6,
                    help="task plane patch edge length (m)")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg_yaml = yaml.safe_load(f)

env_cfg = EnvConfig(**{**cfg_yaml["env"], "n_envs": 1})
seed_val = args.seed if args.seed is not None else cfg_yaml["line_distribution"]["train_seed"]

env = NSRLBatchedEnv(env_cfg, line_dist=None, device=torch.device("cpu"))
env.line_dist = LineDistribution(
    kin=env.kin, collision=env.collision,
    n_pool=500,
    n_target_noise_deg=cfg_yaml["line_distribution"]["n_target_noise_deg"],
    seed=seed_val,
)
env.reset()

q = env.q[0].cpu().numpy().astype(np.float32)
u_hat = env.line_dir[0].cpu().numpy().astype(np.float32)
n_target = env.n_target[0].cpu().numpy().astype(np.float32)


# Scene -----------------------------------------------------------------------

base = ovw.World(cam_pos=(1.5, 1.2, 1.2),
                 cam_lookat_pos=(0.0, 0.0, 0.4),
                 toggle_auto_cam_orbit=False)
ossop.frame().attach_to(base.scene)  # world origin frame

robot, _hand = make_fr3_with_pen(use_pen_tcp=True)
robot.attach_to(base.scene)
attach_pen_visual(robot, rgb=(0.15, 0.15, 0.15), alpha=1.0)
robot.fk(qs=q)

p_tip = robot.gl_tcp_tf[:3, 3].copy().astype(np.float32)
R_flange = robot.gl_flange_tf[:3, :3].copy().astype(np.float32)

# Task plane patch: starts at the pen tip, extends along u_hat, normal =
# n_target. Built as a thin box with an explicit rotmat so the long edge
# is guaranteed to be along u_hat (ossop.plane picks an arbitrary
# in-plane basis from `normal`, which would leave one short edge floating
# off the tip in general).
y_axis = np.cross(n_target, u_hat)
y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
plane_rotmat = np.stack([u_hat, y_axis, n_target], axis=1).astype(np.float32)
plane_center = p_tip + u_hat * (args.ray_len * 0.5)
plane = ossop.box(
    pos=plane_center,
    half_extents=(args.ray_len * 0.5, args.plane_size * 0.5, 5e-4),
    rotmat=plane_rotmat,
    rgb=(0.40, 0.70, 1.00), alpha=0.25,
)
plane.attach_to(base.scene)

# Reference line ray from pen tip along u_hat (the trajectory target).
ray = ossop.dashed_cylinder(
    spos=p_tip, epos=p_tip + u_hat * args.ray_len,
    radius=0.003, rgb=(0.2, 0.4, 1.0), alpha=0.9,
)
ray.attach_to(base.scene)

# u_hat (task direction, blue) and n_target (cone normal, green) arrows.
u_arrow = ossop.arrow(
    spos=p_tip, epos=p_tip + u_hat * 0.30, rgb=(0.2, 0.4, 1.0))
u_arrow.attach_to(base.scene)
n_arrow = ossop.arrow(
    spos=p_tip, epos=p_tip + n_target * 0.30, rgb=(0.2, 0.9, 0.2))
n_arrow.attach_to(base.scene)

# TCP coordinate axes (R = pen flange rotation, origin = pen tip).
tcp_frame = ossop.frame(pos=p_tip, rotmat=R_flange, length_scale=0.7)
tcp_frame.attach_to(base.scene)

print(f"[viz_scene] q0       = {q.tolist()}")
print(f"[viz_scene] u_hat    = {u_hat.tolist()}")
print(f"[viz_scene] n_target = {n_target.tolist()}")
print(f"[viz_scene] p_tip    = {p_tip.tolist()}")
print("[viz_scene] orbit / zoom; close window or Ctrl-C to exit.")

builtins.base = base
builtins.robot = robot
base.run()
