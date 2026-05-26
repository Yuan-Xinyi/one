"""Static-overlay rendering of one task's joint trajectory in the `one` viewer.

Loads a cached rollout (rollouts.npz from diagnose_p0_vs_baseline) and places K
ghost robots per controller at evenly-spaced timesteps along the trajectory,
all in one scene. RL ghosts in blue, Classical baseline ghosts in red. The
viewer launches interactive so the user can orbit and screenshot.

Earlier ghosts are more transparent; final pose is most opaque so the failure
configuration is visually obvious.

Usage:
    python -m Yuan.RL_controller.viz_static_overlay \\
        --diag-dir Yuan/RL_controller/runs/p0_progress_only_30M_0520/diag_10000_classical \\
        --task 1310 --n-ghosts 5
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
from pathlib import Path

import numpy as np

from one import ovw, ossop
from one.robots.manipulators.franka.fr3_pen.fr3_with_pen import (
    make_fr3_with_pen, attach_pen_visual,
)


parser = argparse.ArgumentParser()
parser.add_argument("--diag-dir", required=True,
                    help="dir with rollouts.npz (from diagnose_p0_vs_baseline)")
parser.add_argument("--task", type=int, default=1310,
                    help="task index in the rollouts npz to render")
parser.add_argument("--n-ghosts", type=int, default=5,
                    help="number of ghost poses per controller (evenly spaced over each trajectory)")
parser.add_argument("--alpha-min", type=float, default=0.20,
                    help="alpha of the earliest ghost; latest ghost gets alpha-max")
parser.add_argument("--alpha-max", type=float, default=0.90)
parser.add_argument("--ray-len", type=float, default=1.5,
                    help="reference ray length in meters")
parser.add_argument("--plane-size", type=float, default=2.0,
                    help="ground plane edge length in meters; 0 to skip")
parser.add_argument("--separate-y", type=float, default=0.0,
                    help="if non-zero, shift Classical ghosts (and their ref ray) "
                         "by this many meters along +Y so the two controllers no "
                         "longer overlap. 0 keeps the original overlaid layout.")
args = parser.parse_args()

diag_dir = Path(args.diag_dir)
npz = np.load(diag_dir / "rollouts.npz")
q_traj_rl = npz["q_traj_rl"]       # (T+1, N, 7)
q_traj_base = npz["q_traj_base"]
rl_len = npz["episode_len_rl"]
base_len = npz["episode_len_base"]
line_dir = npz["line_dir"]
n_target = npz["n_target"]

i = int(args.task)
T_rl = int(rl_len[i])
T_base = int(base_len[i])
print(f"[viz] task {i}: T_rl={T_rl}  T_base={T_base}")


def _pick_timesteps(T: int, K: int) -> list[int]:
    """K evenly-spaced timesteps along [0, T]. Always includes 0 and T."""
    if K <= 1 or T == 0:
        return [T]
    return list({int(round(k * T / (K - 1))) for k in range(K)})


steps_rl = sorted(_pick_timesteps(T_rl, args.n_ghosts))
steps_base = sorted(_pick_timesteps(T_base, args.n_ghosts))
print(f"[viz]   RL ghost steps      : {steps_rl}")
print(f"[viz]   Classical ghost steps: {steps_base}")


# Camera: position-tweaked from visualize.py defaults. When the two
# controllers are split along Y, recenter the camera lookat on the midpoint
# so the framing stays symmetric.
y_mid = 0.5 * float(args.separate_y)
base = ovw.World(cam_pos=(1.5, 1.2 + y_mid, 1.2),
                 cam_lookat_pos=(0.0, y_mid, 0.4),
                 toggle_auto_cam_orbit=False)
ossop.frame().attach_to(base.scene)
if args.plane_size > 0:
    # Plane is square; widen Y so it spans both robots and still extends as
    # far in X as before. Center it at the midpoint between the two bases.
    plane_y = args.plane_size + abs(args.separate_y)
    ground = ossop.plane(
        pos=(0.0, y_mid, -1e-3),
        size=(args.plane_size, plane_y),
        rgb=(0.82, 0.82, 0.85), alpha=1.0)
    ground.attach_to(base.scene)


def _alpha_at(idx: int, n: int) -> float:
    """Linear interpolation alpha_min → alpha_max over the n ghosts."""
    if n <= 1:
        return args.alpha_max
    frac = idx / (n - 1)
    return args.alpha_min + frac * (args.alpha_max - args.alpha_min)


def _add_ghost(q: np.ndarray, body_rgb, pen_rgb, alpha: float,
               base_pos=None):
    kw = {"use_pen_tcp": True}
    if base_pos is not None:
        kw["pos"] = np.asarray(base_pos, dtype=np.float32)
    arm, _hand = make_fr3_with_pen(**kw)
    arm.attach_to(base.scene)
    attach_pen_visual(arm, rgb=pen_rgb, alpha=alpha)
    arm.rgb = body_rgb
    arm.alpha = alpha
    arm.fk(qs=q.astype(np.float32))
    return arm


RL_BODY = (0.45, 0.60, 0.95)
RL_PEN = (0.10, 0.25, 0.95)
CLS_BODY = (0.95, 0.55, 0.55)
CLS_PEN = (0.95, 0.10, 0.10)


# RL ghosts (blue)
for k, t in enumerate(steps_rl):
    q = q_traj_rl[t, i]
    a = _alpha_at(k, len(steps_rl))
    _add_ghost(q, body_rgb=RL_BODY, pen_rgb=RL_PEN, alpha=a)
print(f"[viz] placed {len(steps_rl)} RL ghosts (blue)")

# Classical ghosts (red) — optionally shifted along +Y so they don't overlap
# the RL cluster.
cls_offset = np.array([0.0, float(args.separate_y), 0.0], dtype=np.float32)
for k, t in enumerate(steps_base):
    q = q_traj_base[t, i]
    a = _alpha_at(k, len(steps_base))
    _add_ghost(q, body_rgb=CLS_BODY, pen_rgb=CLS_PEN, alpha=a,
               base_pos=cls_offset if args.separate_y != 0.0 else None)
print(f"[viz] placed {len(steps_base)} Classical ghosts (red)"
      + (f"  [Y-shifted by {args.separate_y:+.2f} m]"
         if args.separate_y != 0.0 else ""))


# Reference line ray (from p_start along u_hat). Compute p_start via FK on q0.
arm_tmp, _ = make_fr3_with_pen(use_pen_tcp=True)
arm_tmp.fk(qs=q_traj_rl[0, i].astype(np.float32))
p_start = arm_tmp.gl_tcp_tf[:3, 3].copy()
u_hat = line_dir[i].astype(np.float32)
n_tgt = n_target[i].astype(np.float32)
def _draw_task_anchor(p0):
    ray = ossop.dashed_cylinder(
        spos=p0, epos=p0 + u_hat * args.ray_len,
        radius=0.003, rgb=(0.2, 0.4, 1.0), alpha=0.6)
    ray.attach_to(base.scene)
    u_arrow = ossop.arrow(
        spos=p0, epos=p0 + u_hat * 0.20, rgb=(0.2, 0.4, 1.0))
    u_arrow.attach_to(base.scene)
    n_arrow = ossop.arrow(
        spos=p0, epos=p0 + n_tgt * 0.15, rgb=(0.2, 0.9, 0.2))
    n_arrow.attach_to(base.scene)


_draw_task_anchor(p_start)
if args.separate_y != 0.0:
    _draw_task_anchor(p_start + cls_offset)


print(f"[viz] task {i} static overlay ready — blue=RL, red=Classical, "
      f"transparent→opaque = start→end. Orbit / screenshot in the viewer.")
builtins.base = base
base.run()
