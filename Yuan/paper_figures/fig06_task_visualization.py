"""Figure 6: task visualisation in the One viewer.

Displays one task's geometry:
  - starting point     : small black sphere at p0
  - motion direction d : black arrow from p0 along d
  - plane normal n     : black arrow from p0 along n
  - task plane         : thin transparent disk centred at p0, normal = n
  - rotation cone      : transparent wire-frame cone (half-angle 30 deg) with
                         apex at p0 and axis n -- the orientation tolerance
                         constraint of Eq.(orientation) in the paper
  - FR3 arm (optional) : at the task's generating configuration q0_seed

Usage:
    python -m Yuan.paper_figures.fig06_task_visualization
    python -m Yuan.paper_figures.fig06_task_visualization --task 5721 --no-arm
    python -m Yuan.paper_figures.fig06_task_visualization --no-cone
"""
from __future__ import annotations

# Conda lib bootstrap (so the One viewer can find shared libraries).
import os, sys
_conda_lib = os.path.join(sys.prefix, 'lib')
if _conda_lib not in os.environ.get('LD_LIBRARY_PATH', ''):
    new_env = dict(os.environ)
    new_env['LD_LIBRARY_PATH'] = _conda_lib + ':' + new_env.get('LD_LIBRARY_PATH', '')
    if __spec__ is not None and __spec__.name != '__main__':
        argv = [sys.executable, '-m', __spec__.name] + sys.argv[1:]
    else:
        argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, argv, new_env)

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

from one import ovw, ossop
from one.robots.manipulators.franka.fr3_pen.fr3_with_pen import (
    make_fr3_with_pen, attach_pen_visual,
)

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig


DEFAULT_EVAL_NPZ = 'Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz'

# Colours.
COLOR_BLACK = (0.05, 0.05, 0.05)
COLOR_PLANE = (0.55, 0.65, 0.75)   # cool grey-blue
COLOR_CONE  = (0.85, 0.55, 0.30)   # warm orange
COLOR_ARM   = None   # None -> FR3 default renderer colour

CONE_HALF_ANGLE_DEG = 30.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=int, default=4351,
                   help='eval-set task index')
    p.add_argument('--arm', dest='show_arm', action='store_true', default=True,
                   help='Visualise the FR3 arm at q0_seed (default ON).')
    p.add_argument('--no-arm', dest='show_arm', action='store_false',
                   help='Suppress the FR3 arm rendering.')
    p.add_argument('--cone', dest='show_cone', action='store_true', default=True,
                   help='Visualise the 30-deg rotation cone (default ON).')
    p.add_argument('--no-cone', dest='show_cone', action='store_false',
                   help='Suppress the rotation cone.')
    p.add_argument('--cone-extent', type=float, default=0.25,
                   help='Length of cone rays from apex (m).')
    p.add_argument('--cone-rays', type=int, default=16,
                   help='Number of azimuthal rays drawn on the cone surface.')
    p.add_argument('--cone-alpha', type=float, default=0.35,
                   help='Cone transparency.')
    p.add_argument('--plane-size', type=float, default=0.30,
                   help='Diameter of the task-plane disk (m).')
    p.add_argument('--plane-thickness', type=float, default=0.003,
                   help='Disk thickness along n (m).')
    p.add_argument('--plane-alpha', type=float, default=0.35,
                   help='Plane transparency.')
    p.add_argument('--p0-radius', type=float, default=0.012,
                   help='Sphere radius at p0 (m).')
    p.add_argument('--arrow-d-len', type=float, default=0.18,
                   help='Length of the d-direction arrow (m).')
    p.add_argument('--arrow-n-len', type=float, default=0.15,
                   help='Length of the n-direction arrow (m).')
    p.add_argument('--config', default='Yuan/system_eval/config.yaml')
    p.add_argument('--eval-set', default=DEFAULT_EVAL_NPZ)
    p.add_argument('--arm-alpha', type=float, default=0.95,
                   help='Arm body alpha.')
    return p.parse_args()


def _build_kin_env(env_yaml, device):
    """We only need the FR3 kinematics; build a minimal n_envs=1 env."""
    with open(env_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    valid = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in cfg['env'].items() if k in valid}
    return NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': 1}), line_dist=None,
                          device=device)


def orthonormal_perp(n: np.ndarray):
    """Return two orthonormal vectors (u, v) perpendicular to ``n``."""
    n = n / np.linalg.norm(n)
    if abs(n[2]) < 0.9:
        a = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u = a - float(np.dot(a, n)) * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u.astype(np.float32), v.astype(np.float32)


# ---------------------------------------------------------------------------
# Scene drawing
# ---------------------------------------------------------------------------
def draw_p0_sphere(base, p0, radius):
    """Try ossop.sphere; fall back to a small box if unavailable."""
    if hasattr(ossop, 'sphere'):
        ossop.sphere(pos=p0, radius=radius,
                     rgb=COLOR_BLACK).attach_to(base.scene)
    else:
        s = float(radius * 2)
        ossop.box(pos=p0, extent=np.array([s, s, s], dtype=np.float32),
                  rgb=COLOR_BLACK).attach_to(base.scene)


def draw_task_plane(base, p0, n, size, thickness, alpha):
    """Thin disk perpendicular to n, centred at p0."""
    radius = float(size) / 2.0
    half_thick = float(thickness) / 2.0
    ossop.cylinder(
        spos=p0 - n * half_thick,
        epos=p0 + n * half_thick,
        radius=radius,
        rgb=COLOR_PLANE, alpha=float(alpha),
    ).attach_to(base.scene)


def draw_cone(base, p0, n, half_angle_deg, extent, n_rays, alpha):
    """Wire-frame transparent cone: apex = p0, axis = n.
    Draws ``n_rays`` rays from p0 to the rim circle, plus ``n_rays`` rim
    segments connecting consecutive rim points."""
    u, v = orthonormal_perp(n)
    half = np.deg2rad(float(half_angle_deg))
    cos_h, sin_h = float(np.cos(half)), float(np.sin(half))
    theta = np.linspace(0.0, 2 * np.pi, n_rays, endpoint=False)
    # rim points at distance ``extent`` from apex along the cone surface
    dir_lateral = (cos_h * n[None, :] +
                   sin_h * (np.cos(theta)[:, None] * u[None, :] +
                            np.sin(theta)[:, None] * v[None, :]))
    rim = p0[None, :] + extent * dir_lateral
    rim = rim.astype(np.float32)
    # rays from apex
    for r in rim:
        ossop.cylinder(spos=p0, epos=r, radius=0.0012,
                       rgb=COLOR_CONE, alpha=float(alpha)
                       ).attach_to(base.scene)
    # rim segments
    for i in range(n_rays):
        a = rim[i]
        b = rim[(i + 1) % n_rays]
        ossop.cylinder(spos=a, epos=b, radius=0.0012,
                       rgb=COLOR_CONE, alpha=float(alpha)
                       ).attach_to(base.scene)


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Task spec --------------------------------------------------------
    es = np.load(args.eval_set, allow_pickle=False)
    T = int(args.task)
    p0 = es['cs_p0'][T].astype(np.float32)
    d  = es['cs_line_dir'][T].astype(np.float32)
    nt = es['cs_n_target'][T].astype(np.float32)
    q0_seed = es['q0_seed'][T].astype(np.float32)
    print(f'[fig06] task={T}')
    print(f'        p0={p0}  d={d}  n={nt}')
    print(f'        q0_seed={q0_seed}')

    # ---- Kinematics (for FK on the arm) ----------------------------------
    env = _build_kin_env(cfg['env']['config_yaml'], dev)

    # ---- Scene -----------------------------------------------------------
    base = ovw.World(cam_pos=(1.5, -1.2, 1.2),
                     cam_lookat_pos=(p0[0], p0[1], p0[2]))
    ossop.frame().attach_to(base.scene)

    # FR3 arm at q0_seed (optional)
    if args.show_arm:
        arm, _ = make_fr3_with_pen(use_pen_tcp=True)
        arm.attach_to(base.scene)
        if COLOR_ARM is not None:
            attach_pen_visual(arm, rgb=COLOR_ARM, alpha=float(args.arm_alpha))
            arm.rgb = COLOR_ARM
        else:
            attach_pen_visual(arm, alpha=float(args.arm_alpha))
        arm.alpha = float(args.arm_alpha)
        arm.fk(qs=q0_seed)

    # Task plane (thin disk perpendicular to n, centred at p0)
    draw_task_plane(base, p0, nt, args.plane_size, args.plane_thickness,
                    args.plane_alpha)

    # Starting-point sphere at p0
    draw_p0_sphere(base, p0, args.p0_radius)

    # Direction arrows
    ossop.arrow(spos=p0, epos=p0 + d * float(args.arrow_d_len),
                radius=0.005, rgb=COLOR_BLACK).attach_to(base.scene)
    ossop.arrow(spos=p0, epos=p0 + nt * float(args.arrow_n_len),
                radius=0.005, rgb=COLOR_BLACK).attach_to(base.scene)

    # Rotation cone (optional)
    if args.show_cone:
        draw_cone(base, p0, nt, CONE_HALF_ANGLE_DEG,
                  args.cone_extent, args.cone_rays, args.cone_alpha)

    print(f'[fig06] viewer ready (show_arm={args.show_arm}, '
          f'show_cone={args.show_cone}). Close the window to exit.')
    base.run()


if __name__ == '__main__':
    main()
